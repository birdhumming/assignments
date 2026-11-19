"""A6 run queue: one detached Modal app, FIFO over `containers` GPUs, list order = priority.
    uv run python -m experiments.a6_generalization.launch_queue [containers] [names...]"""
import sys, time, modal
from dataclasses import replace
from .common import Config
from .training import run
from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, timestamped_modal_app_name

INITS = ('scratch', 'related', 'dclm', 'noise')
def ft(init, passes, lr=.003, ntrain=30000, **kw):
    tag = f"ft-{init}-n{ntrain//1000}k-p{passes}-lr{lr:g}" + ''.join(f'-{k}{v}' for k, v in kw.items())
    return Config(name=tag, initialization='random' if init == 'scratch' else init, epochs=passes, lr=lr, ntrain=ntrain, **kw)

QUEUE = (
    [ft(i, p) for p in (1, 4) for i in INITS]                                   # P4(a) cheap passes
    + [Config(name=f'hyperball-lr{lr:g}', optimizer='hyperball', wd=0., lr=lr) for lr in (.003, .001, .01)]  # P2(a)(ii)
    + [Config(name=f'dropout{d:g}', attention_dropout=d) for d in (.15, .05, .25)]  # P2(b)(i)
    + [Config(name='moe-lr0.003', architecture='moe'), Config(name='loop-lr0.003', architecture='loop')]  # P1(b,c)
    + [Config(name='sam-rho0.05', optimizer='sam', rho=.05)]                       # P3(c)
    + [ft(i, 16) for i in INITS]                                                 # P4(a) 16 passes
    + [ft(i, 16, ntrain=n) for n in (1000, 3000, 10000) for i in INITS]          # P4(b)
    + [ft('related', 16, adaptation=a) for a in ('toward_initialization', 'frozen_head', 'frozen_body')]
    + [ft('related', 16, wd=.1)]                                                 # P4(c)
    + [ft(i, 16, lr=.001) for i in INITS]                                        # P4(a) LR trial
    + [Config(name=f'dropout0.15-seed{s}', attention_dropout=.15, seed=s, data_seed=s) for s in (43, 44)]  # P2(b)(iii)
    + [Config(name='baseline-seed43', seed=43, data_seed=43)]                     # paired second seed
    + [Config(name=f'{a}-lr{lr:g}', architecture=a, lr=lr) for a in ('moe', 'loop') for lr in (.01, .001)]  # P1 LR trials
    + [Config(name=f'dropout0.15-lr{lr:g}', attention_dropout=.15, lr=lr) for lr in (.01,)]
)

if __name__ == '__main__':
    containers = int(sys.argv[1]) if sys.argv[1:] else 4
    names = sys.argv[2:]
    todo = [c for c in QUEUE if not names or c.name in names]
    assert len({c.name for c in QUEUE}) == len(QUEUE), 'duplicate names'
    if names == ['dry']:
        todo = QUEUE
    for c in todo: print(c.name)
    if names == ['dry']: sys.exit()
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, gpu=['H100', 'H200', 'A100-80GB'], cpu=4, memory=32768,
                          timeout=24*3600, max_containers=containers, retries=modal.Retries(initial_delay=0., max_retries=2))(run)
    with modal.enable_output(), app.run(name=timestamped_modal_app_name('a6-queue'), environment_name=MODAL_ENVIRONMENT, detach=True):
        for c in todo:
            print(time.strftime('%H:%M'), remote.spawn(c).object_id, c.name, flush=True)
