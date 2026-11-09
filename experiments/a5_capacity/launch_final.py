import time, modal
from . import launch_plan as lp
from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets, timestamped_modal_app_name
NAMES = ['a5-loop-to-unique-d8-w512-lr0.003', 'a5-moe-hash-d8-w512-lr0.003', 'a5-moe-balanced001-d8-w512-lr0.003',
         'a5-moe-balanced1-d8-w512-lr0.003', 'a5-loop-1block-r12-d8-w512-lr0.003', 'a5-moe-noisy02-d8-w512-lr0.003',
         'a5-loop-1block-r4-d8-w512-lr0.003', 'a5-moe-norm-balanced-d8-w512-lr0.003', 'a5-loop-recall05-d8-w512-lr0.003',
         'a5-moe-all-d8-w512-lr0.003', 'a5-mos-r256-d8-w512-lr0.003', 'a5-cplm-r256-d8-w512-lr0.003', 'a5-cplm-d8-w512-lr0.003']
by_name = {c.run_name: c for lane in lp.LANES.values() for c in lane}
remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, secrets=secrets(include_wandb=True), gpu=lp.GPU, cpu=8,
                      memory=32768, timeout=24*3600, max_containers=3, retries=modal.Retries(initial_delay=0., max_retries=2))(lp.run)
with modal.enable_output(), app.run(name=timestamped_modal_app_name('a5-final'), environment_name=MODAL_ENVIRONMENT, detach=True):
    for n in NAMES:
        print(time.strftime('%H:%M'), remote.spawn(by_name[n]).object_id, n, flush=True)
