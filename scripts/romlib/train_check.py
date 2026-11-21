"""Per game, on NeSLE itself (the emulator training runs on): does the readers' boot (recorded on fceumm) reach a state
the player controls?

Play readers.jsonl boot_pads from power-on on NeSLE (CPU host lib). Control: from that state, the button pair that
moved a byte in stage 1 (Right vs Left, Up vs Down, A vs nothing, ...) held PROBE frames each pushes one of the stage-1
candidate bytes >= 2 apart (readers.Play.probe, on NeSLE). Checked at the boot end and again after SETTLE idle frames
(some games give control a moment after the boot ends), then every 60 frames up to LATE: control_after = idle
frames after the boot until control (None: never).

usage: NESLE_HOST_LIB=... python train_check.py OUT.jsonl [--readers readers.jsonl] [--only NAME...] [--workers 24]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/romlib'))
sys.path.insert(0, str(ROOT / 'scripts/nesle_check'))
ROM_DIR = Path('/home/kd/Desktop/roms')
R = ROOT / 'reports/romlib'
STAGE1 = [R / 'ramfind_fixed.jsonl', R / 'ramfind_restored_fixed.jsonl', R / 'ramfind_deep_ok.jsonl',
          R / 'ramfind_menu_ok.jsonl', R / 'ramfind_stable_ok.jsonl', R / 'ramfind_new_ok.jsonl']
NAMES = ['A', 'B', 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']
PAIRS = {'RIGHT / LEFT': (('RIGHT',), ('LEFT',)), 'UP / DOWN': (('UP',), ('DOWN',)), 'A / none': (('A',), ()),
         'B / none': (('B',), ())}
KEYS = {'RIGHT / LEFT': 'x_candidates', 'UP / DOWN': 'y_candidates'}
PROBE = int(__import__("os").environ.get("TRAIN_CHECK_PROBE", 16))
SETTLE, LATE = 120, int(__import__("os").environ.get("TRAIN_CHECK_LATE", 1200))


def buttons(p):
    return tuple(n for k, n in enumerate(NAMES) if p >> k & 1)


def control(nes, pair, pos):
    s = nes.save()
    a, b = pair
    nes.step(a, PROBE)
    ra = nes.ram()
    nes.load(s)
    nes.step(b, PROBE)
    rb = nes.ram()
    nes.load(s)
    return any(abs(int(ra[p]) - int(rb[p])) >= 2 for p in pos)


def check(row, st1):
    import neslenes
    out = dict(rom=row['rom'])
    moved = (st1.get('moved_by') or ['RIGHT / LEFT'])[0]
    cands = st1.get(KEYS.get(moved, 'a_candidates')) or st1.get('x_candidates') or st1.get('y_candidates') or []
    pos = [int(c[0][1:], 16) for c in cands[:3]]
    out['moved_by'] = moved
    if not pos:
        out['control_boot'] = out['control_settled'] = False
        out['why'] = 'no stage-1 candidate byte'
        return out
    pair = PAIRS.get(moved, PAIRS['RIGHT / LEFT'])
    nes = neslenes.NesleNes(str(ROM_DIR / row['rom']))
    try:
        for p in bytes.fromhex(row['boot_pads']):
            nes.step(buttons(p), 1)
        out['control_boot'] = control(nes, pair, pos)
        nes.step((), SETTLE)
        out['control_settled'] = control(nes, pair, pos)
        # later control (an intro, a stage card): idle frames after the boot until the first probe that works
        out['control_after'] = 0 if out['control_boot'] else SETTLE if out['control_settled'] else None
        t = SETTLE
        while out['control_after'] is None and t < LATE:
            nes.step((), 60)
            t += 60
            if control(nes, pair, pos):
                out['control_after'] = t
    except neslenes.EmulatorFault as e:
        out['fault'] = str(e)[:200]
        out['control_boot'] = out['control_settled'] = False
    return out


def _w(args):
    try:
        return check(*args)
    except Exception as e:  # noqa: BLE001
        return dict(rom=args[0]['rom'], error=repr(e)[:300])


def stage1_rows():
    rows = {}
    for p in STAGE1:
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                if r.get('controllable') or r['rom'] not in rows:
                    rows[r['rom']] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--readers', default=str(R / 'readers.jsonl'))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    st1 = stage1_rows()
    rows = [json.loads(l) for l in open(a.readers)]
    rows = [r for r in rows if r.get('boot_pads') and (not a.only or r['rom'] in a.only)]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    jobs = [(r, st1.get(r['rom'], {})) for r in rows if r['rom'] not in done]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=16) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, jobs):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
