"""Student entry point for P4.1. Complete the two functions below."""
import argparse

from experiments.a2.stress import StressConfig, load_tokens, run


def initialize(model):
    """TODO: initialize all weights/gains for the policy you are testing.

    Set model.output_multiplier and each block.residual_multiplier if needed.
    This function runs under torch.no_grad() with the requested seed.
    """
    raise NotImplementedError('Fill in initialization and forward multipliers from your derivation.')


def parameter_groups(model, base_lr):
    """TODO: return a list of dicts with params, lr, and eps.

    Every model parameter must occur exactly once. Construct the groups using
    model.named_parameters(); the runner supplies Adam, fixed betas, and WD=0.
    """
    raise NotImplementedError('Fill in the per-group learning rates and Adam stabilizers.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-path', required=True)
    p.add_argument('--val-path', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--head-dim', type=int, default=64)
    p.add_argument('--precision', choices=('fp32', 'mp'), default='fp32',
                   help='fp32 for width comparisons; mp for depth comparisons')
    p.add_argument('--microbatch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--no-alignment', action='store_true')
    p.add_argument('--no-wandb', action='store_true', help='Save diagnostics locally without W&B')
    a = p.parse_args(argv)
    c = StressConfig(width=a.width, depth=a.depth, head_dim=a.head_dim,
                     microbatch=a.microbatch, seed=a.seed, precision=a.precision)
    c.validate()
    train = load_tokens(a.train_path, c.steps * c.batch, c.context)
    val = load_tokens(a.val_path, c.batch, c.context)
    result = run(c, train, val, base_lr=a.lr, initialize_fn=initialize,
                 groups_fn=parameter_groups, device=a.device,
                 alignment=not a.no_alignment, output=a.output)
    if not a.no_wandb:
        from pathlib import Path

        import wandb

        from experiments.a2.wandb_diagnostics import log_records
        from utils import WANDB_ENTITY, WANDB_PROJECT
        with wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                        name=Path(a.output).stem, tags=['a2', 'p4.1'],
                        config={**result['config'], 'base_lr': a.lr,
                                'parameter_groups': result['parameter_groups'],
                                'data_sha256': result['data_sha256']}) as wb:
            wandb.define_metric('optimizer_step')
            wandb.define_metric('*', step_metric='optimizer_step')
            for row in result['history']:
                wandb.log({'optimizer_step': row['step'],
                           **{key: row[key] for key in ('val_loss', 'train_loss', 'logit_rms') if key in row}})
            log_records(wandb, result['alignment'], result['history'])
            wandb.save(a.output, base_path=str(Path(a.output).parent), policy='now')
            print('WANDB_RUN_URL=' + wb.url)
    print('Final validation loss:', result['history'][-1]['val_loss'])


if __name__ == '__main__':
    main()
