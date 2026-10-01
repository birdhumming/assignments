"""P4.1: launch five-step stress runs for every (policy, width/depth, LR) in one detached Modal app.

Width test: depth 2, head dim 64, reference width 512, FP32, policies kaiming / mup.
Depth test: width 64, one head, reference depth 2, BF16 autocast, policies mup / depth_mup / completep
(at depth 2 the three depth policies coincide, so that point is shared).
Outputs: /a2-stress/<name>.json on the user volume + W&B run tagged a2, a2-p41.
"""
import argparse
from pathlib import PurePosixPath

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS, MODAL_ENVIRONMENT,
                         MODAL_SHARED_DATASETS_DIR, timestamped_modal_app_name, secrets)

WIDTH_LRS = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
DEPTH_LRS = (2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3)
WIDTHS = (640, 2560, 5120)
DEPTHS = (2, 100, 1000)
OUTPUT_DIR = '/root/data/a2-stress'


def name_of(spec):
    return (f"p41-{spec['test']}-{spec['policy']}-w{spec['width']}-d{spec['depth']}"
            f"-lr{spec['lr']:g}-{spec['precision']}")


def specs(points=None):
    width_lrs = WIDTH_LRS if points is None else WIDTH_LRS[-points:] if points < len(WIDTH_LRS) else WIDTH_LRS
    depth_lrs = DEPTH_LRS if points is None else DEPTH_LRS[-points:] if points < len(DEPTH_LRS) else DEPTH_LRS
    out, seen = [], set()
    for policy in ('kaiming', 'mup'):
        for w in WIDTHS:
            for lr in width_lrs:
                out.append(dict(test='width', policy=policy, width=w, depth=2, head_dim=64, precision='fp32',
                                lr=lr, reference_width=512, reference_depth=2))
    for policy in ('mup', 'depth_mup', 'completep'):
        for d in DEPTHS:
            for lr in depth_lrs:
                pol = 'mup' if d == 2 else policy  # all depth policies coincide at the reference depth
                s = dict(test='depth', policy=pol, width=64, depth=d, head_dim=64, precision='mp',
                         lr=lr, reference_width=64, reference_depth=2)
                if name_of(s) not in seen:
                    seen.add(name_of(s)); out.append(s)
    return out


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100', retries=0,
              max_containers=2, timeout=3600)
def _run_spec(spec, use_wandb=True):
    import json
    from pathlib import Path
    from experiments.a2.stress import StressConfig, load_tokens, run
    from experiments.a2.p41_policies import make_policy
    name = name_of(spec)
    output = Path(OUTPUT_DIR) / f'{name}.json'
    if output.exists():
        print('exists, skipping', output); return str(output)
    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    c = StressConfig(width=spec['width'], depth=spec['depth'], head_dim=spec['head_dim'],
                     microbatch=8, seed=42, precision=spec['precision'])
    c.validate()
    train = load_tokens(str(data / 'train'), c.steps * c.batch, c.context)
    val = load_tokens(str(data / 'val'), c.batch, c.context)
    init, groups = make_policy(spec['policy'], reference_width=spec['reference_width'],
                               reference_depth=spec['reference_depth'])
    try:
        result = run(c, train, val, base_lr=spec['lr'], initialize_fn=init, groups_fn=groups,
                     device='cuda', alignment=True, output=str(output))
    finally:
        user_volume.commit()
    print(name, 'final val', result['history'][-1]['val_loss'])
    if use_wandb:
        import wandb
        from utils import WANDB_ENTITY, WANDB_PROJECT
        from experiments.a2.wandb_diagnostics import log_records
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=name,
                        tags=['a2', 'a2-p41', f"a2-p41-{spec['test']}"], reinit=True,
                        config={**result['config'], **spec, 'parameter_groups': result['parameter_groups'],
                                'data_sha256': result['data_sha256']}) as wb:
            wandb.define_metric('optimizer_step'); wandb.define_metric('*', step_metric='optimizer_step')
            for row in result['history']:
                wandb.log({'optimizer_step': row['step'],
                           **{k: row[k] for k in ('val_loss', 'train_loss', 'logit_rms') if k in row}})
            log_records(wandb, result['alignment'], result['history'])
            wandb.save(str(output), base_path=OUTPUT_DIR, policy='now')
    return str(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--points', type=int, default=None, help='Use only the largest N LRs of each grid')
    p.add_argument('--only', choices=('width', 'depth'), default=None)
    a = p.parse_args()
    todo = [s for s in specs(a.points) if a.only is None or s['test'] == a.only]
    for s in todo:
        print(' ', name_of(s))
    print(len(todo), 'stress runs')
    if not a.execute:
        return
    import modal
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name('a2-p41-stress'), detach=True,
                     environment_name=MODAL_ENVIRONMENT):
            fn = _run_spec.with_options(secrets=secrets(include_wandb=True))
            for s in todo:
                call = fn.spawn(s)
                print(call.object_id, name_of(s))


if __name__ == '__main__':
    main()
