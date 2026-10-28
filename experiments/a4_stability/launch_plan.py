"""A4 run queue: three lanes, at most two Modal containers at once.

    uv run python -m experiments.a4_stability.launch_plan            # launch A1 now, then watch
    uv run python -m experiments.a4_stability.launch_plan watch      # only watch/launch remaining lanes

Each lane is one Modal app with max_parallel_runs=1 (sequential). The watcher launches the next
lane whenever fewer than two apps are ephemeral. Calibration (H200): GRU-type cell ~1.7 h/run,
tanh RNN ~0.5 h/run (est.), d8 ~0.25 h/run (est.).
"""
import subprocess
import sys
import time

from experiments.a4_stability.sweeps import config

ENV = 'cs312-aleyang'
GPU = 'H200'


def runs(names, rates):
    return [(n, lr) for n in names for lr in rates]


D8_P3 = ['d8-attention', 'd8-half', 'd8-ema09', 'd8-uniform', 'd8-block4', 'd8-ema01', 'd8-hybrid']
RNN_P3 = ['rnn-attention', 'rnn-half', 'rnn-ema09', 'rnn-uniform', 'rnn-block2', 'rnn-block4', 'rnn-ema01', 'rnn-hybrid', 'rnn-grouped']
def ordered(names, rates):
    """All variants at the first rate, then the second rate: stop the lane early and still have a full grid."""
    return [(n, lr) for lr in rates for n in names]


LANES = {
    'A1-d8-p3': ordered(D8_P3 + ['d8-block2'], (.003, .006)),
    'B-p4': [('gru', .0015), ('gru', .006), ('normgate', .003), ('gru-no-reset', .003),
             ('gru-reset-only', .003), ('normgate', .0015), ('gru-input-gate', .003), ('normgate-b0', .003)],
    'A3-d8-grouped': [('d8-grouped', .003), ('d8-grouped', .006)],
    'A2-rnn-p3': ordered(RNN_P3, (.001, .002)),
}
ORDER = ['A1-d8-p3', 'measure', 'B-p4', 'A2-rnn-p3']


def ephemeral_apps():
    out = subprocess.run(['uv', 'run', 'modal', 'app', 'list', '--env', ENV], capture_output=True, text=True, timeout=120).stdout
    return out.count('ephemeral')


def launch_lane(lane):
    if lane == 'measure':  # blocking: holds one GPU slot until the supplied-checkpoint probes finish
        print(time.strftime('%H:%M'), 'measuring supplied checkpoints', flush=True)
        subprocess.run(['uv', 'run', 'python', '-m', 'experiments.a4_stability.measure_supplied'], check=False)
        return
    from modal_train import launch_training_jobs
    configs = [config(name, lr) for name, lr in LANES[lane]]
    print(time.strftime('%H:%M'), 'launching lane', lane, len(configs), 'runs', flush=True)
    launch_training_jobs(configs, max_parallel_runs=1, gpu=GPU)


def watch(pending):
    while pending:
        try:
            n = ephemeral_apps()
        except Exception as e:
            print('app list failed', e, flush=True)
            n = 2
        if n < 2:
            launch_lane(pending.pop(0))
            time.sleep(300)
        else:
            time.sleep(120)
    print(time.strftime('%H:%M'), 'all lanes launched', flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['watch']:
        watch(sys.argv[2:] or ORDER[1:])
    elif sys.argv[1:] == ['dry']:
        for lane, items in LANES.items():
            print(lane, len(items), [config(n, lr).run_name for n, lr in items][:3], '...')
    else:
        launch_lane(ORDER[0])
        watch(ORDER[1:])
