"""Every table game (rl/games_table.json) in the environment's CPU build with random legal moves: does it run, does the
reward see the player move (new cells), and how do episodes end? One env per game, batches of BATCH games per process.

Per game: decisions, fault (an emulator fault event), return, progress / death / level / score / stall reward parts,
episodes ended by game over / stall, cells (progress part * progress_unit / W_PROGRESS = new cells).
A game whose random play never finds a new cell, or that faults, is reported as not ready.

usage: RL_CPU_LIB=... NES_ROM_DIR=... python rl/scripts/table_sweep.py OUT.jsonl [--decisions 400] [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('RL_TABLE_GAMES', str(ROOT / 'rl/games_table.json'))
BATCH = 16
EV_DEATH, EV_GAME_OVER, EV_STALL = 4, 16, 32


def sweep(games, decisions, seed=3):
    sys.path[:0] = [str(ROOT / 'rl')]
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import torch
    from rlenv import RlEnv, TABLE, PART_NAMES
    from actions import random_legal
    env = RlEnv({g: 1 for g in games}, device='cpu', seed=seed, sticky=True, idle_max=0, dropout=False)
    env.start()
    gen = torch.Generator().manual_seed(seed)
    n = env.n
    ret = torch.zeros(n, dtype=torch.float64)
    parts = torch.zeros(n, len(PART_NAMES), dtype=torch.float64)
    ends = {k: torch.zeros(n, dtype=torch.long) for k in ('game_over', 'stall', 'death')}
    for _ in range(decisions):
        env.actions.copy_(random_legal(env.masks.clone(), gen))
        env.step()
        ret += env.reward.double()
        parts += env.parts.double()
        ev = env.events.long()
        ends['game_over'] += (ev & EV_GAME_OVER) != 0
        ends['stall'] += (ev & EV_STALL) != 0
        ends['death'] += (ev & EV_DEATH) != 0
    out = []
    for i, g in enumerate(env.games):
        t = TABLE[g]['table']
        p = dict(zip(PART_NAMES, parts[i].tolist()))
        out.append(dict(game=g, rom=TABLE[g]['rom'], decisions=decisions, ret=round(float(ret[i]), 4),
                        cells=round(p['progress'] * t['progress_unit']), deaths=int(ends['death'][i]),
                        game_overs=int(ends['game_over'][i]), stalls=int(ends['stall'][i]),
                        parts={k: round(v, 4) for k, v in p.items() if v}))
    return out


def _w(args):
    try:
        return sweep(*args)
    except Exception as e:  # noqa: BLE001 - one bad ROM: retry its batch one game at a time
        if len(args[0]) == 1:
            return [dict(game=args[0][0], error=repr(e)[:300])]
        return sum((_w(([g], args[1])) for g in args[0]), [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--decisions', type=int, default=400)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--only', nargs='*')
    a = ap.parse_args()
    table = json.loads(Path(os.environ['RL_TABLE_GAMES']).read_text())
    done = {json.loads(l)['game'] for l in open(a.out)} if Path(a.out).exists() else set()
    games = [g for g in table if g not in done and (not a.only or g in a.only)]
    jobs = [(games[i:i + BATCH], a.decisions) for i in range(0, len(games), BATCH)]
    with mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, 'a') as fh:
        for rows in pool.imap_unordered(_w, jobs):
            for r in rows:
                fh.write(json.dumps(r) + '\n')
            fh.flush()


if __name__ == '__main__':
    main()
