"""Numerical measurements for the handout-defined update-alignment ratio."""
import math

import torch


def training_steps(config):
    """Match the shared trainer's accumulation and epoch accounting."""
    microbatch = config.batch_size // config.num_micro_batches
    batches = (config.num_train_sequences + microbatch - 1) // microbatch
    return int(config.num_epochs * (batches // config.num_micro_batches))


def alignment(numerator_rms, update_rms, input_rms, fan_in):
    denominator = update_rms * input_rms
    if denominator == 0:
        return {'alpha': None, 'status': 'undefined_denominator', 'numerator_rms': numerator_rms,
                    'update_rms': update_rms, 'input_rms': input_rms, 'fan_in': fan_in}
    if numerator_rms == 0:
        return {'alpha': None, 'status': 'zero_action', 'numerator_rms': 0.,
                    'update_rms': update_rms, 'input_rms': input_rms, 'fan_in': fan_in}
    return {'alpha': math.log(numerator_rms / denominator, fan_in), 'status': 'finite',
                'numerator_rms': numerator_rms, 'update_rms': update_rms,
                'input_rms': input_rms, 'fan_in': fan_in}


@torch.no_grad()
def measure_delta(before, after, inputs, tile_rows=256):
    """Full weight RMS and full fixed-probe action; bound GPU workspace by rows."""
    x=inputs.to(device=after.device,dtype=torch.float32).reshape(-1,after.shape[1])
    weight_sq=torch.zeros((),device=after.device,dtype=torch.float64)
    action_sq=weight_sq.clone();changed=0
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        for start in range(0,after.shape[0],tile_rows):
            delta=after[start:start+tile_rows].float()-before[start:start+tile_rows].float()
            action=x @ delta.T
            weight_sq+=delta.double().square().sum()
            action_sq+=action.double().square().sum()
            changed+=torch.count_nonzero(delta).item()
        result=alignment((action_sq/(len(x)*after.shape[0])).sqrt().item(),
                         (weight_sq/after.numel()).sqrt().item(),
                         x.double().square().mean().sqrt().item(),after.shape[1])
        result.update(changed_fraction=changed/after.numel(),probe_positions=len(x),
                      output_coordinates=after.shape[0])
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf32
