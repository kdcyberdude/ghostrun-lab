"""One row per ROM in the ROM folder: what we know about running it on the GPU emulator (NeSLE).

Merges reports/romlib/gpu_check_*.jsonl (later files win), calibrate.jsonl, fceux_check.jsonl and the genre table
reports/rom_trainability.csv into reports/romlib/status.csv, and prints counts.

gpu verdict:
  pass       NeSLE's RAM matches fceumm within timing scratch for 60 s of seeded play (gpu_check.py)
  natural    NeSLE drifts from fceumm, but fceumm and FCEUX drift from each other as early (calibrate.py): the game
             amplifies tiny timing differences; nothing says NeSLE plays it wrong
  matches_ref NeSLE matches FCEUX or fceumm above the pair's noise floor (calibrate.py)
  suspect    the two reference emulators agree longer than NeSLE agrees with them: possibly wrong on NeSLE
  fault      NeSLE stops on an opcode the CPU cannot run (the game crashed: mapper or timing problem)
  no_mapper  the cartridge board is not implemented
usage: python status.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / 'reports/romlib'
ROM_DIR = Path('/home/kd/Desktop/roms')
ORDER = ['gpu_check_baseline', 'gpu_check_header', 'gpu_check_ops', 'gpu_check_m1', 'gpu_check_m2', 'gpu_check_m3',
         'gpu_check_pad', 'gpu_check_m4', 'gpu_check_m4b', 'gpu_check_m5', 'gpu_check_m6', 'gpu_check_m7', 'gpu_check_m8', 'gpu_check_m9', 'gpu_check_m10', 'gpu_check_m11', 'gpu_check_m12', 'gpu_check_m13', 'gpu_check_m14']


def norm(s):
    s = re.sub(r'^ZZZ_UNK_', '', s).lower()
    s = re.sub(r'^(.*), the\b', r'the \1', s)
    return re.sub(r'\bthe\b|[^a-z0-9]', '', s)


def base(f):
    return re.split(r'\s*[\(\[]', f[:-4])[0].strip()


def main():
    gpu = {}
    rest = sorted(p.stem for p in REP.glob('gpu_check_*.jsonl') if p.stem not in ORDER)
    full = [n for n in rest if n.startswith('gpu_check_full_')]   # whole-folder runs win over per-batch runs
    for name in ORDER + [n for n in rest if n not in full] + full:
        p = REP / f'{name}.jsonl'
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                gpu[r['rom']] = r
    cal = {json.loads(l)['rom']: json.loads(l) for l in open(REP / 'calibrate.jsonl')} if (REP / 'calibrate.jsonl').exists() else {}
    fx = {json.loads(l)['rom']: json.loads(l) for l in open(REP / 'fceux_check.jsonl')}
    genre = {}
    for r in csv.DictReader(open(ROOT / 'reports/rom_trainability.csv')):
        genre[norm(r['title'])] = (r['category'], r['fit'])
    rows = []
    for f in sorted(p.name for p in ROM_DIR.iterdir() if p.suffix == '.nes'):
        g, c, x = gpu.get(f, {}), cal.get(f), fx.get(f, {})
        msg = g.get('nesle_msg', '')
        if g.get('pass'):
            v = 'pass'
        elif 'unsupported mapper' in msg or 'NeSLE emulates' in msg:
            v = 'no_mapper'
        elif 'fault' in msg:
            v = 'fault'
        elif c and 'error' not in c and g.get('nesle_ok'):
            a, b, d = c['fceumm_fceux'], c['nesle_fceumm'], c['nesle_fceux']
            if b is None or d is None:
                v = 'matches_ref'
            elif a is not None and max(b, d) >= a - 120:
                v = 'natural'
            else:
                v = 'suspect'
        elif g.get('nesle_ok'):
            v = 'diverges_uncalibrated'
        else:
            v = 'unknown' if not g else 'no_load'
        cat, fit = genre.get(norm(base(f)), ('', ''))
        rows.append(dict(rom=f, mapper=g.get('mapper', ''), gpu=v, fceux=('runs' if x.get('screens', 0) > 1 else
                                                                         'blank' if x else ''),
                         category=cat, fit=fit))
    with open(REP / 'status.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('ROMs', len(rows))
    print('gpu', Counter(r['gpu'] for r in rows).most_common())
    ok = {'pass', 'natural', 'matches_ref'}
    by = Counter((r['fit'] or '?', r['gpu'] in ok) for r in rows)
    for fit in sorted({k[0] for k in by}):
        print(f'{fit:22s} runs on GPU {by[(fit, True)]:5d}   not yet {by[(fit, False)]:5d}')


if __name__ == '__main__':
    main()
