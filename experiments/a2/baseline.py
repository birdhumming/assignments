"""The shared language-model baseline for Problems 1, 2, 3.2, and 4.2."""
from data import TokenDatasetConfig
from model_config import LMConfig
from train import TrainConfig
from metric_logging import MetricLogger, AFTER_BACKWARD, AFTER_TRAIN_STEP


# A1's supplied baseline uses 64 sequences in one microbatch.
A1_BATCH_SIZE = TrainConfig.batch_size

def config(train_path, val_path, *, tokens=153_600_000, batch=64, diagnostics=True):
    if tokens <= 0 or tokens % 1024:
        raise ValueError('tokens must be a positive multiple of 1024')
    if batch <= 0:
        raise ValueError('batch must be positive')
    microbatch = min(batch, A1_BATCH_SIZE)
    if batch % microbatch:
        raise ValueError(f'batches above {A1_BATCH_SIZE} must be multiples of {A1_BATCH_SIZE}')
    return TrainConfig(
        model_config=LMConfig(
            'a2-d8', 4096, 1024, 512, 1792, 8, 8, 8,
        ),
        metric_loggers=(
            MetricLogger(AFTER_TRAIN_STEP, 'experiments.a2.probes:features'),
            MetricLogger(AFTER_BACKWARD, 'experiments.a2.alignment:before_update'),
            MetricLogger(AFTER_BACKWARD, 'experiments.a2.probes:gradients'),
            MetricLogger(AFTER_TRAIN_STEP, 'experiments.a2.alignment:after_update'),
        ) if diagnostics else (),
        learning_rate=.003, weight_decay=.1, beta1=.9, beta2=.95,
        lr_schedule='linear', warmup_percent=.01, qk_norm=True,
        batch_size=batch, num_micro_batches=batch // microbatch,
        num_epochs=1, num_train_sequences=tokens//1024,
        train_dataset=TokenDatasetConfig('train', 'a2', path=str(train_path)),
        val_dataset=TokenDatasetConfig('val', 'a2', sequence_limit=1000, path=str(val_path)),
    )
