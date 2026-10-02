"""Launch A2 training with automatic preparation of data prefixes."""
import errno
import os
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath

from data import DEFAULT_DATA_SEED, DEFAULT_DATASET_DIR_NAME, PreprocessedTokenDataset
from experiments.a2.baseline import config as baseline_config
from experiments.a2.data import stage_prefix
from modal_utils import (
    MODAL_ENVIRONMENT,
    MODAL_SHARED_DATASETS_DIR,
    MODAL_USER_DATASETS_DIR,
    VOLUME_MOUNTS,
    app,
    build_image,
    timestamped_modal_app_name,
    user_volume,
)


# Container paths come from the same mounts used by A1; no student-specific paths.
def config(*, tokens=153_600_000, batch=64, diagnostics=True, **overrides):
    seed = overrides.pop('data_seed', DEFAULT_DATA_SEED)
    if seed != DEFAULT_DATA_SEED:
        raise ValueError('The A2 starter fixes data_seed=42; extend preparation explicitly for other seeds.')
    train_path = prefix_path(MODAL_USER_DATASETS_DIR, tokens // 1024)
    val_path = PurePosixPath(MODAL_SHARED_DATASETS_DIR) / DEFAULT_DATASET_DIR_NAME / 'val'
    c = baseline_config(str(train_path), str(val_path), tokens=tokens, batch=batch, diagnostics=diagnostics)
    if overrides.get('optimizer_name') in {'adamh', 'hyperball', 'muon'}:
        overrides.setdefault('optimizer_builder', 'experiments.a2.optimizers:build_optimizer')
        if overrides['optimizer_name'] in {'adamh', 'hyperball'}:
            overrides.setdefault('weight_decay', 0.)
    return replace(c, **overrides)


def prefix_path(user_datasets, sequences):
    return PurePosixPath(user_datasets) / 'a2-global-prefixes' / f'n{sequences}-seed42' / 'train'


def _validate_prefix(destination, source, n):
    try:
        cached = PreprocessedTokenDataset(destination)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f'Incomplete A2 prefix at {destination}; remove this cache directory and retry') from error
    expected = {'num_sequences': n, 'seq_len': 1024, 'data_seed': 42,
                    'shuffle': 'global_shuffle_then_prefix', 'prefix_sequences': n,
                    'shuffle_source_num_sequences': len(PreprocessedTokenDataset(source)),
                    'source_path': str(source)}
    meta = cached.metadata
    if any(meta.get(k) != v for k, v in expected.items()) or not meta.get('tokens_sha256'):
        raise ValueError(f'Existing A2 prefix does not match the requested data: {destination}')


def prepare_prefixes(shared_datasets, user_datasets, sequence_counts):
    """CPU-only preparation; publish complete prefixes without overwriting caches."""
    source = Path(shared_datasets) / DEFAULT_DATASET_DIR_NAME / 'train'
    for n in sorted(set(sequence_counts)):
        destination = Path(prefix_path(user_datasets, n))
        if destination.exists():
            _validate_prefix(destination, source, n)
            continue
        original = PreprocessedTokenDataset(source)
        if original.seq_len != 1024:
            raise ValueError('A2 language-model training requires length-1024 data')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=destination.parent) as folder:
            staged = Path(folder) / 'train'
            stage_prefix(source, staged, n, seed=42)
            _validate_prefix(staged, source, n)
            try:
                os.rename(staged, destination)
            except OSError as error:
                if error.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.exists():
                    raise
                # Another preparer published first; accept only matching complete data.
                _validate_prefix(destination, source, n)
    return [str(prefix_path(user_datasets, n)) for n in sorted(set(sequence_counts))]


@app.function(image=build_image(), volumes=VOLUME_MOUNTS, max_containers=1,
              retries=0, timeout=3600)
def _prepare_prefixes(counts):
    paths = prepare_prefixes(str(MODAL_SHARED_DATASETS_DIR), str(MODAL_USER_DATASETS_DIR), counts)
    user_volume.commit()
    return paths


def launch_training_jobs(configs, *, max_parallel_runs=None):
    """Prepare the requested budgets on CPU, then invoke A1's GPU launcher."""
    from modal_train import launch_training_jobs as shared_launch
    if not configs:
        raise ValueError('Supply at least one configuration')
    for c in configs:
        expected = config(tokens=c.num_train_sequences * 1024, batch=c.batch_size)
        if (c.train_dataset.path != expected.train_dataset.path or
                c.val_dataset.path != expected.val_dataset.path or c.data_seed != 42):
            raise ValueError('Use this module\'s config() for automatic data preparation')

    import modal
    with modal.enable_output(), app.run(name=timestamped_modal_app_name('a2-data'), environment_name=MODAL_ENVIRONMENT):
        _prepare_prefixes.remote([c.num_train_sequences for c in configs])
    return shared_launch(configs, max_parallel_runs=max_parallel_runs)
