"""Ground truth for the RAM readers: the reward10 games (scripts/reward10/games/*.py, hand-verified RAM maps).

truth_signals(slug, ram[E, F, 2048]) runs the game's own Reader over recorded RAM and returns per-frame arrays
(in_play, level, dying, lives, score, game_over; -1 where the game has no such signal).
TRUTH_ADDR: the addresses the hand maps read for lives, score and level (for address-level comparisons).

usage: python truth.py collect     (records the readers.py play of every truth game into .cache/romlib/truth/)
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
sys.path.insert(0, str(ROOT / 'scripts/romlib'))
OUT = ROOT / '.cache/romlib/truth'
SLUGS = ['smb', 'castlevania', 'contra', 'gradius', 'pacman', 'excitebike', 'finalmission', 'kage', 'punchout',
         'tecmobowl', 'tetris', 'zelda']
# from the module docstrings / Readers: lives byte, score bytes, level byte (None = the game has none)
TRUTH_ADDR = {
    'smb': dict(lives=[0x75A], score=list(range(0x7DD, 0x7E3)), level=[0x75F, 0x75C]),
    'castlevania': dict(lives=[0x2A], score=[0x7FC, 0x7FD, 0x7FE], level=[0x28]),
    'contra': dict(lives=[0x32], score=[0x7E2, 0x7E3], level=[0x30]),
    'gradius': dict(lives=[0x20], score=[0x7E4, 0x7E5, 0x7E6], level=[0x19]),
    'pacman': dict(lives=[0x67], score=list(range(0x70, 0x76)), level=[0x68]),
    'excitebike': dict(lives=None, score=None, level=[0x43, 0x46]),
    'finalmission': dict(lives=None, score=[0x8A, 0x8B, 0x8C], level=[0x1E]),
    'kage': dict(lives=None, score=None, level=[0x40]),
    'punchout': dict(lives=None, score=list(range(0x3E8, 0x3EE)), level=[0x01]),
    'tecmobowl': dict(lives=None, score=None, level=[0x8F]),
    'tetris': dict(lives=None, score=[0x53, 0x54, 0x55], level=[0x64]),
    'zelda': dict(lives=None, score=None, level=None),
}


def rom_of(slug):
    src = (ROOT / f'scripts/reward10/games/{slug}.py').read_text()
    m = re.search(r"/ '([^']+)'\)|/ \"([^\"]+)\"\)", src)
    return m.group(1) or m.group(2)


def stage1_rows():
    rows = {}
    for f in ('ramfind_v1.jsonl', 'ramfind_v2.jsonl'):
        p = ROOT / 'reports/romlib' / f
        if p.exists():
            for l in open(p):
                r = json.loads(l)
                if r.get('controllable') or r['rom'] not in rows:
                    rows[r['rom']] = r
    return rows


def truth_signals(slug, ram):
    import os
    os.environ.setdefault('NES_ROM_DIR', '/home/kd/Desktop/roms')
    m = importlib.import_module(f'games.{slug}')
    keys = ('in_play', 'level', 'dying', 'lives', 'score', 'game_over')
    out = {k: np.full(ram.shape[:2], -1, np.int64) for k in keys}
    for e in range(ram.shape[0]):
        rd = m.Reader()
        rd.reset()
        for f in range(ram.shape[1]):
            s = rd.observe(ram[e, f])
            for k in keys:
                v = getattr(s, k)
                if v is not None:
                    out[k][e, f] = int(v)
    return out


def _collect(slug):
    from readers import Play
    try:
        rom = rom_of(slug)
        row = stage1_rows()[rom]
        if not row.get('controllable'):
            return slug, 'not controllable in stage 1'
        p = Play(str(Path('/home/kd/Desktop/roms') / rom), row)
        p.save_npz(OUT / f'{slug}.npz')
        t = truth_signals(slug, np.stack(p.ram))
        np.savez_compressed(OUT / f'{slug}_truth.npz', **t)
        return slug, 'ok'
    except Exception as e:  # noqa: BLE001
        return slug, repr(e)[:300]



def events(x):
    """Rising edges of a boolean trace."""
    x = np.asarray(x, bool)
    return np.flatnonzero(x[1:] & ~x[:-1]) + 1


def match_events(pred, true, before=200, after=200):
    """(true events matched by a predicted one within [t - before, t + after], predicted events with no true one)."""
    hit = sum(any(t - before <= p <= t + after for p in pred) for t in true)
    false = sum(not any(t - before <= p <= t + after for t in true) for p in pred)
    return hit, false


def number(ram, a, n, kind):
    import ramfind2
    return ramfind2.number(ram, a, n, kind)[0]


def evaluate(slug, rd, ram, t):
    """Compare a reader (readers.py row) with the truth signals t on the recorded play ram [E, F, 2048].
    Returns {field: (verdict, note)}; verdict 'ok', 'wrong', 'missed', 'unknown-ok' (no such signal, none
    reported), 'false' (reported for a game without it) or 'n/a' (the truth cannot judge it on this play)."""
    ta = TRUTH_ADDR[slug]
    out = {}
    in_game = (t['in_play'] > 0).any()
    if not in_game:
        return {k: ('n/a', 'boot does not reach the game the truth map describes') for k in
                ('lives', 'score', 'level', 'death', 'in_play', 'game_over')}
    for k in ('lives', 'level'):
        r = rd.get(k)
        got = None if not isinstance(r, dict) else int(r['addr'][1:], 16)
        if ta[k] is None:
            out[k] = ('unknown-ok', '') if got is None else ('false', f'${got:04X}')
        elif got is None:
            out[k] = ('missed', '')
        else:
            out[k] = ('ok' if got in ta[k] else 'wrong', f'${got:04X}')
    r = rd.get('score')
    if ta['score'] is None:
        out['score'] = ('unknown-ok', '') if not isinstance(r, dict) else ('false', r['addr'])
    elif not isinstance(r, dict):
        out['score'] = ('missed', '')
    else:
        a = int(r['addr'][1:], 16)
        ours = np.concatenate([number(x, a, r['n'], r['kind']) for x in ram])
        tru = t['score'].reshape(-1)
        m = (tru >= 0) & (t['in_play'].reshape(-1) > 0)
        nz = m & (ours > 0)
        k = np.median(tru[nz] / ours[nz]) if nz.any() else 0
        good = nz.any() and (np.abs(tru[m] - k * ours[m]) < 1e-6).mean() > 0.99 and tru[m].max() > 0
        out['score'] = ('ok' if good else 'wrong', f"{r['addr']} {r['n']} {r['kind']} x{k:g}")
    # events: death and game over; in_play per frame
    for k, key, win in (('death', 'dying', (60, 400)), ('game_over', 'game_over', (300, 600))):
        r = rd.get(k)
        cuts = []
        for e in range(len(ram)):
            go = events(t['game_over'][e] > 0)
            cuts.append(go[0] + 600 if len(go) else ram.shape[1])   # after a game over: title and attract demo
        tr = [[x for x in events(t[key][e] > 0) if x < cuts[e]] for e in range(len(ram))]
        if k == 'game_over':
            tr = [x[:1] for x in tr]
        n_true = sum(len(x) for x in tr)
        if not isinstance(r, dict):
            out[k] = ('unknown-ok', '') if n_true == 0 else ('missed', f'{n_true} true events')
            continue
        a, v = int(r['addr'][1:], 16), r['value']
        hit = false = 0
        for e in range(len(ram)):
            pred = [x for x in events(ram[e][:, a] == v) if x < cuts[e]]
            if k == 'game_over':                  # one event per game: the first edge (the title/demo may follow)
                pred, tr[e] = pred[:1], tr[e][:1]
            h, f = match_events(pred, tr[e], win[0], win[1])
            hit, false = hit + h, false + f
        if n_true == 0:
            out[k] = ('false' if false else 'n/a', f'{false} events, none true')
        else:
            out[k] = ('ok' if hit == n_true and false == 0 else 'wrong', f'{hit}/{n_true} found, {false} false')
    r = rd.get('in_play')
    if not isinstance(r, dict):
        out['in_play'] = ('missed', '')
    else:
        a = int(r['addr'][1:], 16)
        pred = np.isin(ram[:, :, a], r['values']).reshape(-1)
        tru = t['in_play'].reshape(-1) > 0
        acc = (pred == tru).mean()
        bal = 0.5 * (pred[tru].mean() + (~pred[~tru]).mean()) if (~tru).any() else pred.mean()
        out['in_play'] = ('ok' if bal >= 0.9 else 'wrong', f'acc {acc:.3f} balanced {bal:.3f}')
    return out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context('spawn').Pool(12, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_collect, sys.argv[2:] or SLUGS):
            print(*r, flush=True)
