"""Collect finished a6/<name>/history.json files from the user volume into results/a6_generalization/runs.json."""
import json, subprocess, pathlib, tempfile
OUT = pathlib.Path('results/a6_generalization/runs.json')
ENV = 'cs312-aleyang'; VOL = 'volume-dl_alchemy'

def main():
    ls = subprocess.run(['uv', 'run', 'modal', 'volume', 'ls', VOL, 'ckpts/a6', '--env', ENV, '--json'], capture_output=True, text=True, timeout=300).stdout
    names = [e['filename'].rstrip('/').split('/')[-1] for e in json.loads(ls or '[]')]
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        for n in names:
            if prev.get(n, {}).get('final') is not None:
                continue
            r = subprocess.run(['uv', 'run', 'modal', 'volume', 'get', VOL, f'ckpts/a6/{n}/history.json', f'{tmp}/{n}.json', '--env', ENV, '--force'],
                               capture_output=True, text=True, timeout=300)
            p = pathlib.Path(f'{tmp}/{n}.json')
            if not p.exists():
                prev.setdefault(n, {}).update(state='running'); continue
            try:
                hist = json.loads(p.read_text())
            except ValueError:
                prev.setdefault(n, {}).update(state='running'); continue
            cfg = subprocess.run(['uv', 'run', 'modal', 'volume', 'get', VOL, f'ckpts/a6/{n}/config.json', f'{tmp}/{n}.cfg.json', '--env', ENV, '--force'],
                                 capture_output=True, text=True, timeout=300)
            done = pathlib.Path(f'{tmp}/{n}.cfg.json').exists()
            prev[n] = dict(history=hist, final=hist[-1]['validation_nll'] if done and hist else None,
                           state='finished' if done else 'running', last=hist[-1] if hist else None,
                           config=json.loads(pathlib.Path(f'{tmp}/{n}.cfg.json').read_text()) if done else None)
    OUT.write_text(json.dumps(prev, indent=1) + '\n')
    for n, v in sorted(prev.items()):
        print(f"{n:60s} {v.get('state'):9s} {v.get('final') or (v.get('last') or {}).get('validation_nll')}")

if __name__ == '__main__':
    main()
