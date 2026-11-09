"""A5 run queue: lanes of sequential Modal runs, at most MAX_APPS containers at once.

    uv run python -m experiments.a5_capacity.launch_plan dry          # list lanes
    uv run python -m experiments.a5_capacity.launch_plan LANE         # launch one lane now
    uv run python -m experiments.a5_capacity.launch_plan watch L1 L2  # launch lanes as GPU slots free up
    uv run python -m experiments.a5_capacity.launch_plan queue 3 L1 L2 # one app, 3 GPUs, lanes in order

Each lane is one detached Modal app whose function has max_containers=1, so its runs execute in
order; the watcher starts the next lane whenever fewer than MAX_APPS apps are ephemeral.
Runs go through sweeps.run (quantile bias hook, sharing switch, SpamLang regeneration).
"""
import subprocess
import sys
import time

from .sweeps import config, run

ENV = 'cs312-aleyang'
GPU = ['H100', 'H200', 'A100-80GB']  # priority list; the course workspace caps concurrent H100s


def moe(name, lr=.003, **router):
    base = dict(balance=0.)
    base.update(router)
    return config('moe-learned', lr, model_builder_kwargs=dict(router=base),
                  run_name=f'a5-{name}-d8-w512-lr{lr:g}')


def loop(name, lr=.003, **kwargs):
    return config('loop-r6', lr, model_builder_kwargs=kwargs, run_name=f'a5-{name}-d8-w512-lr{lr:g}')


def own(name, lr=.003, **router):
    base = dict(balance=0.)
    base.update(router)
    return config('moe-learned', lr, model_builder='experiments.a5_capacity.p3_own:OwnMoELM',
                  model_builder_kwargs=dict(router=base), run_name=f'a5-{name}-d8-w512-lr{lr:g}')


def spam(vocab, width, lr=.001):
    from .p2_readout import training_config
    return training_config(vocab, width, lr)


LANES = {
    # Wave 1 (A-moe already launched): central comparisons at the reference LR .003.
    'A-moe': [config(n) for n in ('moe-learned', 'moe-random', 'moe-frozen', 'moe-unit', 'moe-full',
                                  'moe-balanced', 'moe-norm', 'moe-noisy', 'moe-cloned', 'moe-quantile')],
    'B1-loop': [config(n) for n in ('dense', 'loop-r6', 'loop-r3', 'loop-r2', 'loop-norm-recall', 'loop-boundary-norm')],
    'B2-loop': [config(n) for n in ('loop-recall', 'loop-1block-r1', 'loop-1block-r2', 'loop-1block-r3',
                                    'loop-1block-r4', 'loop-1block-r18')],
    'B3-switch': [config(n) for n in ('loop-to-unique', 'unique-to-loop', 'loop-reset', 'unique-reset')],
    'C1-readout': [config(n) for n in ('softmax-r32', 'softmax-r64', 'softmax-r128', 'softmax-r256',
                                       'sigsoftmax', 'sigsoftmax-r32', 'sigsoftmax-r128')],
    'C2-cplm': [config(n) for n in ('cplm', 'cplm-r32', 'cplm-r64', 'cplm-r128', 'cplm-r256')],
    'C3-mos': [config(n) for n in ('mos', 'mos-r32', 'mos-r64', 'mos-r128', 'mos-r256')],
    'F-spam': [spam(v, 64) for v in (64, 256, 1024, 4096)] + [spam(4096, w) for w in (128, 256, 512)],
    # Wave 2: LR trials, sweeps and compositions, own variants.
    'D1-moe2': [config('moe-learned', .006), moe('moe-balanced001', balance=.001), moe('moe-balanced1', balance=.1),
                moe('moe-noisy02', noise=.2), moe('moe-norm-balanced', norm=True, balance=.01),
                moe('moe-balanced-noisy', balance=.01, noise=.05)],
    'D2-moe2': [moe('moe-cloned-balanced', clone=True, balance=.01), moe('moe-all', norm=True, balance=.01, noise=.05, clone=True),
                own('moe-hash', own='hash'), own('moe-st', own='st'), own('moe-zloss', own='zloss', zloss=1e-3)],
    'E1-loop2': [config('loop-r6', .006), config('loop-norm-recall', .006), config('dense', .006),
                 loop('loop-recall05', recall=.5), loop('loop-1block-r12', passes=12, core_size=1),
                 loop('loop-1block-r18-norm-recall', passes=18, core_size=1, boundary_norm=True, recall=1.),
                 loop('loop-1block-r9', passes=9, core_size=1)],
}
ORDER = ['B1-loop', 'C1-readout', 'F-spam', 'B2-loop', 'C3-mos', 'C2-cplm', 'B3-switch', 'D1-moe2', 'E1-loop2', 'D2-moe2']
MAX_APPS = 4


def ephemeral_apps():
    out = subprocess.run(['uv', 'run', 'modal', 'app', 'list', '--env', ENV],
                         capture_output=True, text=True, timeout=120).stdout
    return out.count('ephemeral')


def launch_lane(lane, gpu=GPU):
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets, timestamped_modal_app_name
    configs = LANES[lane]
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, secrets=secrets(include_wandb=True),
                          gpu=gpu, cpu=8, memory=32768, timeout=24*3600, max_containers=1,
                          retries=modal.Retries(initial_delay=0., max_retries=3))(run)
    name = timestamped_modal_app_name(f'a5-{lane}')
    print(time.strftime('%H:%M'), 'launching lane', lane, len(configs), 'runs as app', name, flush=True)
    with modal.enable_output(), app.run(name=name, environment_name=MODAL_ENVIRONMENT, detach=True):
        for cfg in configs:
            call = remote.spawn(cfg)
            print('  ', call.object_id, cfg.run_name, flush=True)


def launch_queue(lanes, containers=3, gpu=GPU):
    """One ephemeral app for several lanes (the workspace caps ephemeral apps); inputs run FIFO on
    `containers` GPUs, so the lane order is the priority order."""
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets, timestamped_modal_app_name
    configs = [c for lane in lanes for c in LANES[lane]]
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, secrets=secrets(include_wandb=True),
                          gpu=gpu, cpu=8, memory=32768, timeout=24*3600, max_containers=containers,
                          retries=modal.Retries(initial_delay=0., max_retries=3))(run)
    name = timestamped_modal_app_name('a5-queue')
    print(time.strftime('%H:%M'), 'launching queue', lanes, len(configs), 'runs on', containers, 'x', gpu, 'as app', name, flush=True)
    with modal.enable_output(), app.run(name=name, environment_name=MODAL_ENVIRONMENT, detach=True):
        for cfg in configs:
            call = remote.spawn(cfg)
            print('  ', call.object_id, cfg.run_name, flush=True)


def watch(pending):
    while pending:
        try:
            n = ephemeral_apps()
        except Exception as e:
            print('app list failed', e, flush=True)
            n = MAX_APPS
        if n < MAX_APPS:
            launch_lane(pending.pop(0))
            time.sleep(150)
        else:
            time.sleep(120)
    print(time.strftime('%H:%M'), 'all lanes launched', flush=True)



def cancel(patterns, log='logs/a5_queue.log'):
    """Cancel queued calls whose run name contains any pattern (ids come from the launch log)."""
    import re
    import modal
    from modal_utils import MODAL_ENVIRONMENT
    calls = re.findall(r'(fc-\w+) (a5-\S+)', open(log).read())
    for call_id, name in calls:
        if any(p in name for p in patterns):
            modal.FunctionCall.from_id(call_id).cancel()
            print('cancelled', call_id, name)



# Runs cut from the queue on 11/08 (workspace GPU cap); re-added, most important first, when GPUs free up.
EXPANSION = ['a5-softmax-r64-d8-w512-lr0.003', 'a5-softmax-r256-d8-w512-lr0.003', 'a5-loop-1block-r1-d8-w512-lr0.003',
             'a5-loop-1block-r3-d8-w512-lr0.003', 'a5-unique-reset-d8-w512-lr0.003', 'a5-mos-r128-d8-w512-lr0.003',
             'a5-cplm-r128-d8-w512-lr0.003', 'a5-sigsoftmax-r128-d8-w512-lr0.003', 'a5-moe-hash-d8-w512-lr0.003',
             'a5-moe-zloss-d8-w512-lr0.003', 'a5-moe-balanced001-d8-w512-lr0.003', 'a5-moe-balanced1-d8-w512-lr0.003',
             'a5-moe-noisy02-d8-w512-lr0.003', 'a5-loop-1block-r9-d8-w512-lr0.003', 'a5-loop-1block-r12-d8-w512-lr0.003',
             'a5-moe-norm-balanced-d8-w512-lr0.003', 'a5-moe-balanced-noisy-d8-w512-lr0.003',
             'a5-moe-cloned-balanced-d8-w512-lr0.003', 'a5-moe-all-d8-w512-lr0.003', 'a5-loop-recall05-d8-w512-lr0.003',
             'a5-dense-d8-w512-lr0.006', 'a5-loop-r6-d8-w512-lr0.006', 'a5-loop-norm-recall-d8-w512-lr0.006',
             'a5-mos-r64-d8-w512-lr0.003', 'a5-cplm-r64-d8-w512-lr0.003', 'a5-mos-r256-d8-w512-lr0.003',
             'a5-cplm-r256-d8-w512-lr0.003', 'a5-moe-learned-d8-w512-lr0.006']


def app_tasks(prefix):
    """(running task count, still-live) for the newest Modal app whose description starts with prefix."""
    import json, subprocess
    out = subprocess.run(['uv', 'run', 'modal', 'app', 'list', '--env', ENV, '--json'], capture_output=True, text=True).stdout
    apps = [a for a in json.loads(out[out.index('['):]) if a['description'].startswith(prefix)]
    if not apps:
        return 0, False
    a = max(apps, key=lambda a: a['created_at'])
    return int(a['tasks'] or 0), a['state'].startswith('ephemeral') and a['stopped_at'] is None


def expand(max_runs=14, containers=2, poll=600, gpu=GPU):
    """Spawn cut runs one at a time whenever the main queue holds >=2 GPUs (or has finished)."""
    import modal
    from modal_utils import app, build_image, VOLUME_MOUNTS, MODAL_ENVIRONMENT, secrets, timestamped_modal_app_name
    by_name = {c.run_name: c for lane in LANES.values() for c in lane}
    todo = [by_name[n] for n in EXPANSION[:max_runs]]
    remote = app.function(image=build_image(), volumes=VOLUME_MOUNTS, secrets=secrets(include_wandb=True),
                          gpu=gpu, cpu=8, memory=32768, timeout=24*3600, max_containers=containers,
                          retries=modal.Retries(initial_delay=0., max_retries=3))(run)
    live = []
    with modal.enable_output(), app.run(name=timestamped_modal_app_name('a5-expand'), environment_name=MODAL_ENVIRONMENT, detach=True):
        while todo:
            for call in list(live):
                try:
                    call.get(timeout=0)
                    live.remove(call)
                except TimeoutError:
                    pass
                except Exception as e:
                    print(time.strftime('%H:%M'), 'expansion run failed:', e, flush=True)
                    live.remove(call)
            tasks, queue_live = app_tasks('a5-queue')
            if len(live) < containers and (tasks >= 2 or not queue_live):
                cfg = todo.pop(0)
                live.append(remote.spawn(cfg))
                print(time.strftime('%H:%M'), f'queue tasks={tasks} live={queue_live}; spawned', cfg.run_name, flush=True)
            else:
                print(time.strftime('%H:%M'), f'queue tasks={tasks} live={queue_live}; expansion running={len(live)} todo={len(todo)}', flush=True)
            time.sleep(poll)
        print(time.strftime('%H:%M'), 'expansion list exhausted', flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    if args[:1] == ['dry'] or not args:
        for lane, items in LANES.items():
            print(lane, len(items))
            for c in items:
                print('   ', c.run_name)
    elif args[:1] == ['watch']:
        watch(args[1:] or ORDER)
    elif args[:1] == ['expand']:
        expand(*map(int, args[1:]))
    elif args[:1] == ['cancel']:
        cancel(args[1:])
    elif args[:1] == ['queue']:
        launch_queue(args[2:] or ORDER, containers=int(args[1]))
    else:
        for lane in args:
            launch_lane(lane)
