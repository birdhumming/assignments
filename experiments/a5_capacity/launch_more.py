"""Second-day batch: LR trials, seed reruns, grid completion, compositions, own routers.
    uv run python -m experiments.a5_capacity.launch_more [containers] [names...]"""
import sys, time, modal
from . import launch_plan as lp
from .sweeps import config
from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets, timestamped_modal_app_name

BATCH = [  # priority order
    config('dense', model_seed=1, run_name='a5-dense-seed1-d8-w512-lr0.003'),
    config('loop-r6', .006), config('moe-learned', .006),
    config('softmax-r256'), lp.loop('loop-1block-r9', passes=9, core_size=1),
    lp.loop('loop-1block-r18-norm-recall', passes=18, core_size=1, boundary_norm=True, recall=1.),
    lp.moe('moe-balanced-noisy', balance=.01, noise=.05), lp.own('moe-st', own='st'),
    config('loop-r6', .0015), config('moe-learned', .0015),
    config('mos-r256'), config('cplm-r256'), config('sigsoftmax-r128'),
    config('moe-learned', model_seed=1, model_builder_kwargs=dict(router=dict(balance=0.)),
           run_name='a5-moe-learned-seed1-d8-w512-lr0.003'),
    lp.moe('moe-cloned-balanced', clone=True, balance=.01), lp.own('moe-zloss', own='zloss', zloss=1e-3),
    config('loop-1block-r1'), config('loop-boundary-norm', .001),
    config('softmax-r32', .006), lp.moe('moe-noisy01', noise=.1),
    lp.loop('loop-1block-r18-recall05', passes=18, core_size=1, recall=.5),
    lp.moe('moe-quantile-noisy', quantile=True, noise=.05),
    config('loop-norm-recall', .006), config('mos-r64'), config('cplm-r64'),
]

if __name__ == '__main__':
    containers = int(sys.argv[1]) if sys.argv[1:] else 4
    names = sys.argv[2:]
    todo = [c for c in BATCH if not names or c.run_name in names]
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, secrets=secrets(include_wandb=True), gpu=lp.GPU,
                          cpu=8, memory=32768, timeout=24*3600, max_containers=containers,
                          retries=modal.Retries(initial_delay=0., max_retries=2))(lp.run)
    with modal.enable_output(), app.run(name=timestamped_modal_app_name('a5-day2'), environment_name=MODAL_ENVIRONMENT, detach=True):
        for c in todo:
            print(time.strftime('%H:%M'), remote.spawn(c).object_id, c.run_name, flush=True)

# P4(e) controls: second phase from random weights, sharing pattern kept (needs the switch_to='random' code).
BATCH2 = [config('loop-random'), config('unique-random')]
