"""Prepare a prefix of globally shuffled training data."""
import hashlib
import json
from pathlib import Path
import numpy as np
from data import PreprocessedTokenDataset, global_shuffle_prefix_indices


def stage_prefix(source_path, output_path, sequences, seed=42):
    source = PreprocessedTokenDataset(source_path)
    if source.data_seed is not None or not 0 < sequences <= len(source):
        raise ValueError('Use an unshuffled source and a valid prefix length')
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    permutation = global_shuffle_prefix_indices(len(source), sequences, seed)
    tokens = np.memmap(output/'tokens.bin', dtype=source.dtype, mode='w+',
                       shape=(sequences, source.seq_len))
    digest = hashlib.sha256()
    for start in range(0, sequences, 8192):
        rows = source.tokens[permutation[start:start+8192]]
        tokens[start:start+len(rows)] = rows
        digest.update(rows.tobytes())
    tokens.flush()
    info = dict(source.metadata, token_file='tokens.bin', num_sequences=sequences,
                data_seed=seed, shuffle='global_shuffle_then_prefix', prefix_sequences=sequences,
                shuffle_source_num_sequences=len(source),
                source_path=str(source_path), tokens_sha256=digest.hexdigest())
    (output/'metadata.json').write_text(json.dumps(info, indent=2)+'\n')
    return output
