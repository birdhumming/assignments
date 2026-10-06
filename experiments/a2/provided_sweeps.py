"""Load supplied source measurements without launching training or contacting W&B."""
import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / 'worksheets/hparam_invariants/data'
PARTS = ('P1a', 'P1b', 'P1d', 'P1e', 'P2a')


def load(part):
    """Return only the requested part, so P1(b) can be withheld until prediction."""
    if part not in PARTS:
        raise ValueError(f'part must be one of {PARTS}')
    with (DATA_DIR / 'provided_sweeps.csv').open(newline='') as f:
        rows = [row for row in csv.DictReader(f) if part in row['parts'].split('|')]
    for row in rows:
        for key in ('tokens', 'batch_size', 'seed'):
            row[key] = int(row[key])
        for key in ('learning_rate', 'weight_decay', 'final_val_loss', 'beta1', 'beta2'):
            row[key] = float(row[key])
    return rows


def reference_diagnostics():
    """Raw width-512 reference measurements reused in P4.2."""
    return json.loads((DATA_DIR / 'reference_diagnostics.json').read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--part', choices=PARTS, required=True)
    parser.add_argument('--links', action='store_true',
                        help='List configurations and direct W&B links instead of measurements.')
    args = parser.parse_args(argv)
    rows = load(args.part)
    if args.links:
        print('Tokens\tOptimizer\tSchedule\tLR\tWD\tW&B run / local ID')
        for row in rows:
            print(f"{row['tokens']:,}\t{row['optimizer']}\t{row['lr_schedule']}\t{row['learning_rate']:g}\t"
                  f"{row['weight_decay']:g}\t{row['run_url'] or row['run_id']}")
    else:
        print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
