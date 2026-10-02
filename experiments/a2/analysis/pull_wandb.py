"""Pull every `a2`-tagged run (config, summary, full history) into $A2_DATA_DIR/runs/<id>.json."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import wandb

DATA_DIR = os.environ.get('A2_DATA_DIR', os.path.expanduser('~/a2'))
os.makedirs(os.path.join(DATA_DIR, 'runs'), exist_ok=True)
api = wandb.Api(timeout=300)
states = set(sys.argv[1:]) or {'finished'}
runs = [r for r in api.runs('aleyang-stanford-university/assignments', filters={'tags': 'a2'}, per_page=500)
        if 'a2-corrupted' not in r.tags and r.state in states]
print(len(runs), 'runs', flush=True)


def _plain(o):
    if hasattr(o, 'items'):
        return dict(o.items())
    if hasattr(o, '__iter__'):
        return list(o)
    return str(o)


def pull(r):
    path = os.path.join(DATA_DIR, 'runs', f'{r.id}.json')
    if os.path.exists(path) and r.state == 'finished':
        return r.name, 'cached'
    t = time.time()
    rows = list(r.scan_history(page_size=20000))
    record = {'name': r.name, 'id': r.id, 'state': r.state, 'tags': r.tags, 'config': dict(r.config),
                  'summary': dict(r.summary.items()), 'history': rows, 'created_at': r.created_at}
    with open(path, 'w') as fh:
        json.dump(record, fh, default=_plain)
    return r.name, f'{len(rows)} rows {time.time() - t:.0f}s'


with ThreadPoolExecutor(8) as ex:
    for name, msg in ex.map(pull, runs):
        print(name, msg, flush=True)
print('DONE')
