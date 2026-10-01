"""Run one completed P4.1 student configuration from your laptop on Modal."""
import argparse
from pathlib import PurePosixPath

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (app, build_image, user_volume, VOLUME_MOUNTS,
                         MODAL_ENVIRONMENT, MODAL_SHARED_DATASETS_DIR,
                         timestamped_modal_app_name, secrets)


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu='H100',
              retries=0, max_containers=1, timeout=3600)
def _run(arguments):
    from experiments.a2.p31_student import main
    try:
        main(arguments)
    finally:
        user_volume.commit()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--precision', choices=('fp32', 'mp'), default='fp32')
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-wandb', action='store_true', help='Save diagnostics locally without W&B')
    p.add_argument('--output', required=True, help='Unique JSON filename inside your private volume')
    a = p.parse_args(argv)
    if PurePosixPath(a.output).name != a.output or not a.output.endswith('.json'):
        p.error('--output must be a JSON filename, without directory components')
    from experiments.a2.stress import StressConfig
    import math
    StressConfig(width=a.width, depth=a.depth, head_dim=a.head_dim,
                 microbatch=a.microbatch, seed=a.seed, precision=a.precision).validate()
    if not math.isfinite(a.lr) or a.lr <= 0:
        p.error('--lr must be finite and positive')
    data = MODAL_SHARED_DATASETS_DIR / DEFAULT_DATASET_DIR_NAME
    arguments = ['--train-path', str(data / 'train'), '--val-path', str(data / 'val'),
                 '--output', '/root/data/a2-stress/' + a.output]
    for key in ('lr', 'width', 'depth', 'head_dim', 'precision', 'microbatch', 'seed'):
        arguments += ['--' + key.replace('_', '-'), str(getattr(a, key))]
    if a.no_wandb:
        arguments.append('--no-wandb')
    import modal
    with modal.enable_output():
        with app.run(name=timestamped_modal_app_name('a2-stress'), environment_name=MODAL_ENVIRONMENT):
            _run.with_options(secrets=secrets(include_wandb=not a.no_wandb)).remote(arguments)
    print('Saved to your Modal volume: /a2-stress/' + a.output)


if __name__ == '__main__':
    main()
