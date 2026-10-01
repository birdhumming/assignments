"""P1(c) target experiments; P1(a,b) source sweeps are supplied."""
import argparse
import math

EXPERIMENT_KEY = 'a2-p1-D4915m'
LEARNING_RATES = (.0015, .003, .006)
TARGET_TOKENS = 4_915_200_000


def runs(predicted_lrs):
    """Combine the original target grid with the student's two predictions."""
    if len(predicted_lrs) != 2 or any(not math.isfinite(x) or x <= 0 for x in predicted_lrs):
        raise ValueError('Supply two positive finite predicted LRs from your source fits.')
    from experiments.a2.modal_launcher import config
    return [config(tokens=TARGET_TOKENS, learning_rate=lr,
                   run_name_suffix=EXPERIMENT_KEY, wandb_tags=(EXPERIMENT_KEY,))
            for lr in dict.fromkeys((*LEARNING_RATES, *predicted_lrs))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predicted-lrs', type=float, nargs=2, required=True,
                        metavar=('ALL_SIX', 'LARGER_THREE'))
    parser.add_argument('--execute', action='store_true', help='Submit the target runs to Modal.')
    args = parser.parse_args(argv)
    configs = runs(args.predicted_lrs)
    for c in configs:
        print(f'tokens={TARGET_TOKENS} lr={c.learning_rate:g} wd={c.weight_decay:g}')
    if args.execute:
        from experiments.a2.modal_launcher import launch_training_jobs
        launch_training_jobs(configs)
    else:
        print('Preview only. Add --execute to train these target configurations.')


if __name__ == '__main__':
    main()
