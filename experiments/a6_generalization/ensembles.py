"""P2(b)(iii): dropout-member and mixed ensembles against the supplied no-dropout bank, evaluated on Modal CPU.
    uv run python -m experiments.a6_generalization.ensembles <run names...>   (checkpoints under ckpts/a6/<name>/model.pt)"""
import sys, json, itertools, modal
import numpy as np
from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, MODAL_DATA_DIR, timestamped_modal_app_name

def logprobs_of(logits, ids):
    import torch
    lp = torch.log_softmax(torch.as_tensor(np.asarray(logits), dtype=torch.float32), -1)
    return torch.gather(lp, -1, torch.as_tensor(ids[:, 1:, None], dtype=torch.long))[..., 0].numpy()  # (rows, T-1)

def ensemble_nll(member_logits, ids, ks):
    """Mean NLL of probability-averaged ensembles for the first k members."""
    import torch
    probs = None; out = {}
    for k, lg in enumerate(member_logits, 1):
        p = torch.softmax(torch.as_tensor(np.asarray(lg), dtype=torch.float32), -1)
        probs = p if probs is None else probs + p
        if k in ks:
            lp = torch.log(probs / k)
            out[k] = float(-torch.gather(lp, -1, torch.as_tensor(ids[:, 1:, None], dtype=torch.long))[..., 0].mean())
    return out

def compute(names):
    import torch
    from pathlib import Path
    from .common import Config, data
    from .models import build
    from .training import evaluate
    from .p2_regularization import ensemble_bank
    torch.set_num_threads(16)
    _, val = data(Config())
    ids = np.asarray(val)
    ours = {}
    for n in names:
        ck = torch.load(Path(MODAL_DATA_DIR) / 'ckpts/a6' / n / 'model.pt', map_location='cpu', weights_only=False)
        c = Config(**ck['config']); m = build(c); m.load_state_dict(ck['model'])
        pred = np.zeros((len(ids), ids.shape[1] - 1, len(ck['model']['lm_head.weight'])), dtype=np.float32)
        nll = evaluate(m, ids, batch=8, device='cpu', predictions=pred)
        ours[n] = (nll, pred); print(n, nll, flush=True)
    bank = ensemble_bank()
    ks = set(range(1, 6))
    res = {'single': {n: v[0] for n, v in ours.items()}}
    res['no_dropout'] = ensemble_nll(bank, ids, ks)
    res['dropout_members'] = ensemble_nll([ours[n][1] for n in names], ids, ks)
    res['mixed_first_dropout'] = ensemble_nll([ours[names[0]][1]] + list(bank[:4]), ids, ks)
    res['mixed_all_dropout_then_bank'] = ensemble_nll([ours[n][1] for n in names] + list(bank[:5 - len(names)]), ids, ks)
    for n in names:  # each dropout member paired with one supplied member
        res[f'pair_{n}_bank0'] = ensemble_nll([ours[n][1], bank[0]], ids, {2})
    res['pair_bank0_bank1'] = ensemble_nll(list(bank[:2]), ids, {2})
    return res

if __name__ == '__main__':
    names = sys.argv[1:]
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, cpu=16, memory=65536, timeout=4 * 3600)(compute)
    with modal.enable_output(), app.run(name=timestamped_modal_app_name('a6-ens'), environment_name=MODAL_ENVIRONMENT):
        res = remote.remote(names)
    out = 'results/a6_generalization/ensembles.json'
    json.dump(res, open(out, 'w'), indent=1); print(json.dumps(res, indent=1))
