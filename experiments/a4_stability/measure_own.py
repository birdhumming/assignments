"""Measure our own A4 P3/P4 runs (init / early / final) on Modal, same probes as measure_supplied.

    A4_MEASURE_GPU=A100-80GB uv run python -m experiments.a4_stability.measure_own a4-d8-attention-lr0.003 ...

Run names are `a4-<variant>-lr<lr>`; checkpoints live under /root/data/ckpts/<run>/ on the user volume.
Results: results/a4_stability/<run>/{init,early,final}.json (same layout as the supplied runs).
"""
import os
import sys
from pathlib import Path

from .measure_supplied import RESULTS, measure_runs

OWN_ROOT = Path(os.environ.get('A4_OWN_ROOT', '/root/data/ckpts'))


def job(run):
    variant, lr = run[3:].rsplit('-lr', 1)
    return run, variant, float(lr)


if __name__ == '__main__':
    import modal

    from modal_utils import MODAL_ENVIRONMENT, VOLUME_MOUNTS, app, build_image

    from .diagnostics import save
    jobs = [job(r) for r in sys.argv[1:]]
    print(len(jobs), 'runs:', [j[0] for j in jobs])
    probe = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu=os.environ.get('A4_MEASURE_GPU', 'H100'), timeout=3*3600, secrets=[modal.Secret.from_dict({'TORCHDYNAMO_DISABLE': '1'})])(measure_runs)
    with modal.enable_output(), app.run(environment_name=MODAL_ENVIRONMENT):
        results = probe.remote(jobs, str(OWN_ROOT), early='step_100.pt')
    for run, stages in results.items():
        for stage, stats in stages.items():
            save(RESULTS/run/f'{stage}.json', stats)
    print('wrote', RESULTS)
