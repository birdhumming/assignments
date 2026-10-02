"""Hyperball Adam for linear weights, with ordinary Adam for other parameters."""
import torch

from optimizers import is_norm_module

ADAMH_DEFAULT_ADAM_LR_RATIO = 0.000656 / 0.00630

def should_use_adamh(module, parameter_name, parameter):
    if parameter_name != "weight":
        return False
    if parameter.ndim < 2:
        return False
    if isinstance(module, torch.nn.Embedding):
        return False
    if is_norm_module(module):
        return False
    return isinstance(module, torch.nn.Linear)


def build_adamh_parameter_groups(model, learning_rate, adamh_adam_lr=None):
    if adamh_adam_lr is None:
        adamh_adam_lr = learning_rate * ADAMH_DEFAULT_ADAM_LR_RATIO

    adamh_parameters = []
    adam_parameters = []
    adamh_names = []
    adam_names = []
    seen_parameter_ids = set()

    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in seen_parameter_ids:
                continue
            seen_parameter_ids.add(parameter_id)

            full_name = (
                f"{module_name}.{parameter_name}"
                if module_name
                else parameter_name
            )
            if should_use_adamh(module, parameter_name, parameter):
                adamh_parameters.append(parameter)
                adamh_names.append(full_name)
            else:
                adam_parameters.append(parameter)
                adam_names.append(full_name)

    if not adamh_parameters:
        raise ValueError("AdamH requested, but no Linear weight tensors were routed to AdamH.")

    parameter_groups = [
        {"params": adamh_parameters, "mode": "adamh", "lr": learning_rate},
    ]
    if adam_parameters:
        parameter_groups.append({"params": adam_parameters, "mode": "adam", "lr": adamh_adam_lr})

    print(
        "Using AdamH optimizer: "
        f"{len(adamh_parameters)} hyperball tensors, "
        f"{len(adam_parameters)} Adam fallback tensors; "
        f"adamh_lr={learning_rate:g}, adam_lr={adamh_adam_lr:g}."
    )

    return parameter_groups, {
        "weight_decay_style": "ignored_for_adamh",
        "masked_weight_decay": False,
        "optimizer_weight_decay_effective": 0.0,
        "adamh_adam_lr": adamh_adam_lr,
        "adamh_default_adam_lr_ratio": ADAMH_DEFAULT_ADAM_LR_RATIO,
        "adamh_tensor_count": len(adamh_parameters),
        "adam_fallback_tensor_count": len(adam_parameters),
        "adamh_tensor_examples": adamh_names[:16],
        "adam_fallback_tensor_examples": adam_names[:16],
    }


class AdamH(torch.optim.Optimizer):
    """Adam with Hyperball projection for matrix weights.

    This mirrors Marin/Levanter's AdamH update: first form an Adam update, then
    move each selected matrix along that direction with update norm proportional
    to its current Frobenius norm, and project back to the original sphere.
    Non-hyperball parameter groups use plain Adam.
    """

    def __init__(self, params, betas=(0.9, 0.95), eps=1e-8):
        defaults = {"betas": betas, "eps": eps, "mode": "adam"}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            mode = group.get("mode", "adam")

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamH does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"].add_(1.0)
                step = int(state["step"].item())

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                update = (exp_avg / bias_correction1) / (
                    exp_avg_sq.sqrt() / (bias_correction2**0.5) + eps
                )

                if mode == "adamh":
                    apply_hyperball_update_(parameter, update, lr)
                elif mode == "adam":
                    parameter.add_(update, alpha=-lr)
                else:
                    raise ValueError(f"Invalid AdamH parameter group mode: {mode!r}")

        return loss


def apply_hyperball_update_(parameter, update, learning_rate):
    if parameter.ndim == 2:
        p_norm = torch.linalg.vector_norm(parameter)
        u_norm = torch.linalg.vector_norm(update)
        candidate = parameter - learning_rate * update * p_norm / u_norm.clamp_min(1e-10)
        candidate_norm = torch.linalg.vector_norm(candidate).clamp_min(1e-10)
        parameter.copy_(candidate * (p_norm / candidate_norm))
        return

    axes = tuple(range(1, parameter.ndim))
    p_norm = torch.sqrt(torch.sum(parameter.square(), dim=axes, keepdim=True))
    u_norm = torch.sqrt(torch.sum(update.square(), dim=axes, keepdim=True))
    candidate = parameter - learning_rate * update * p_norm / u_norm.clamp_min(1e-10)
    candidate_norm = torch.sqrt(torch.sum(candidate.square(), dim=axes, keepdim=True))
    parameter.copy_(candidate * p_norm / candidate_norm.clamp_min(1e-10))
