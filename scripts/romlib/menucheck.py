"""Tell a stage-1 boot that reached gameplay from one that stopped on a menu whose cursor follows the d-pad, and
continue the boot past such menus.

Check (at the boot end, with the stage-1 button pair, e.g. Right vs Left):
  - cursor jumps: holding the button 60 frames moves the best "position" byte in steps of >= CURSOR_STEP (median
    over the frames it changes) through <= CURSOR_VALUES values; a walking player moves 1-3 px per frame (Tetris'
    level grid: 16 px between 2 values; Excitebike's lane change jumps 7 px but every frame)
  - screen ignores it: the screens after 30 frames of Right and of Left are identical, so the byte the d-pad moves
    is not drawn (Punch-Out's dark screen with a hidden cursor)
  either one = menu, and again LATER frames on with no input (a race countdown holds the player at first). A pause is
  never accepted as a new boot end: the d-pad moves nothing there. Start/A scene-change and frozen-screen tests were measured too and dropped: pausing (Gradius,
  Castlevania) and Final Mission's Start screen look the same as a menu advancing.
Continue: from the boot end, the ramfind.py press sequence (Start, A, Start, Right, ...) with its waits and probes; the
first state where the pair moves something again AND the check says gameplay is the new boot end. New position
candidates come from ramfind.follows at that state.

usage: python menucheck.py OUT.jsonl [--workers 16] [--only ROM...]   (stage-1 rows: ramfind_merged.jsonl)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
sys.path.insert(0, str(ROOT / 'scripts/romlib'))
ROM_DIR = Path('/home/kd/Desktop/roms')
CURSOR_STEP = 6
CURSOR_VALUES = 8     # ... and stops on a few values (Excitebike's lane changes jump 7 px but every frame)
HOLD = 60
MAX_MORE = 16          # presses after the stage-1 boot
LATER = 300


class Adapter:
    """ramfind.follows wants .nes and .run(buttons, frames)."""

    def __init__(self, nes):
        self.nes = nes

    def run(self, b, frames):
        out = []
        self.nes.step(tuple(b), frames, record=out)
        return np.stack(out)


def check(nes, pair):
    """Menu or gameplay at the current state (restored afterwards)."""
    import ramfind
    s = nes.save()
    score, _ = ramfind.follows(Adapter(nes), *pair)
    nes.load(s)
    good = np.flatnonzero(score)
    if not len(good):
        return dict(controllable=False)
    pos = [int(a) for a in good[np.argsort(-score[good])][:3]]
    tr = []
    for _ in range(HOLD):
        nes.step(pair[0], 1)
        tr.append(int(nes.ram()[pos[0]]))
    nes.load(s)
    d = np.abs(((np.diff(tr) + 128) % 256) - 128)
    step = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    nes.step(pair[0], 30)
    sa = nes.screen()
    nes.load(s)
    nes.step(pair[1], 30)
    sb = nes.screen()
    nes.load(s)
    drawn = float((sa != sb).any(2).mean())
    why = [w for w, c in (('cursor jumps', step >= CURSOR_STEP and len(set(tr)) <= CURSOR_VALUES), ('screen ignores the d-pad', drawn == 0)) if c]
    return dict(controllable=True, menu=bool(why), why=why, step=step, nvals=len(set(tr)), drawn=round(drawn, 4),
                pos=[f'${a:04X}' for a in pos])


def run(row):
    import ramfind
    from readers import PAIRS, buttons, BIT
    from nesram import Nes
    nes = Nes(str(ROM_DIR / row['rom']))
    pads = list(bytes.fromhex(row['boot_pads']))
    for p in pads:
        nes.step(buttons(p), 1)
    moved = (row.get('moved_by') or ['RIGHT / LEFT'])[0]
    pair = PAIRS[moved]
    res = dict(rom=row['rom'], moved_by=moved, check=check(nes, pair))
    if not res['check'].get('menu'):
        return res
    # a race or stage may hold the player for a countdown (Excitebike): look again LATER frames on, untouched
    s = nes.save()
    nes.step((), LATER)
    res['check_later'] = check(nes, pair)
    nes.load(s)
    if res['check_later'].get('controllable') and not res['check_later']['menu']:
        res['check']['menu'] = False
        return res
    rec = Adapter(nes)
    for k in range(MAX_MORE):
        b = ramfind.MENU[k % len(ramfind.MENU)]
        nes.step((b,), ramfind.PRESS)
        pads += [BIT[b]] * ramfind.PRESS
        for _ in range(ramfind.WAIT // ramfind.PROBE_EVERY + 1):
            sc, _ = ramfind.follows(rec, *pair)
            if sc.any():
                c = check(nes, pair)
                if c.get('controllable') and not c['menu']:
                    new = dict(row, boot_pads=bytes(pads).hex(), boot_frames=len(pads),
                               presses=row.get('presses', 0) + k + 1, moved_by=[moved])
                    for i, key in ((0, 'x_candidates'), (1, 'y_candidates'), (2, 'a_candidates')):
                        score, _ = ramfind.follows(rec, *ramfind.PAIRS[i])
                        good = np.flatnonzero(score)
                        new[key] = [(f'${a:04X}', float(score[a])) for a in good[np.argsort(-score[good])][:8]]
                    res.update(fixed=True, after=c, presses_more=k + 1, row=new)
                    return res
            nes.step((), ramfind.PROBE_EVERY)          # still a menu (or nothing moves): wait out the jingle
            pads += [0] * ramfind.PROBE_EVERY
    res['fixed'] = False
    return res


def _w(row):
    try:
        return run(row)
    except Exception as e:  # noqa: BLE001
        return dict(rom=row['rom'], error=repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--stage1', default=str(ROOT / 'reports/romlib/ramfind_merged.jsonl'))
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.stage1)]
    rows = [r for r in rows if r.get('boot_pads') and (not a.only or r['rom'] in a.only)]
    done = {json.loads(l)['rom'] for l in open(a.out)} if Path(a.out).exists() else set()
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for r in pool.imap_unordered(_w, [r for r in rows if r['rom'] not in done]):
            fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
