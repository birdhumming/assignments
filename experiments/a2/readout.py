"""Readout/feature RMS ratios, using PyTorch weights shaped [classes, width]."""
import math
import torch


@torch.no_grad()
def measure_readout(weight, features, tile_rows=256):
    """Pool all probe positions and coordinates; exclude the forward multiplier.

    Compute the action in FP32 even inside a mixed-precision training scope.
    Keep initial weights on CPU and transfer only one output tile at a time.
    """
    if weight.ndim != 2 or features.shape[-1] != weight.shape[1]:
        raise ValueError('Expected weight [classes, width] and features [..., width]')
    n = weight.shape[1]
    if n <= 1 or tile_rows <= 0:
        raise ValueError('Alignment exponent requires width > 1 and positive tile_rows')
    x = features.detach().float().reshape(-1, n)
    device = x.device
    weight_sq = torch.zeros((), device=device, dtype=torch.float64)
    action_sq = weight_sq.clone()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.autocast(device.type, enabled=False):
            for start in range(0, weight.shape[0], tile_rows):
                w = weight[start:start + tile_rows].to(device=device, dtype=torch.float32)
                weight_sq += w.double().square().sum()
                action_sq += (x @ w.T).double().square().sum()
            wrms = (weight_sq / weight.numel()).sqrt().item()
            xrms = x.double().square().mean().sqrt().item()
            yrms = (action_sq / (len(x) * weight.shape[0])).sqrt().item()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    denominator = wrms * xrms
    ratio = yrms / denominator if denominator else None
    omega = math.log(ratio, n) if ratio is not None and ratio > 0 else None
    return dict(omega=omega, ratio=ratio,
                status='undefined_denominator' if denominator == 0 else
                       'zero_action' if yrms == 0 else 'finite',
                weight_rms=wrms, feature_rms=xrms, action_rms=yrms,
                fan_in=n, output_coordinates=weight.shape[0], probe_positions=len(x))


@torch.no_grad()
def readout_alignment(weight, features, initial_weight, initial_features):
    """Current S(V_t,h_t) and initial-readout S(V_0,h_t-h_0)."""
    h = features.detach().float()
    return dict(current=measure_readout(weight, h),
                movement=measure_readout(initial_weight,
                    h - initial_features.to(device=h.device, dtype=torch.float32)))
