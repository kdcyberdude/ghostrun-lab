"""Check that the reward10 reward works on NeSLE the way it does on fceumm, without needing identical RAM.

Same game code means the same RAM addresses; what can go wrong is the game running incorrectly. So each game is
played the same way on both emulators and the reward signals are compared as behaviour:
  audit   sticky-random play from the fresh start (same seeds): deaths vs lives lost, game over, damage, score
  search  reward-guided savestate search (same workers and seconds): best level, event kinds, faults
Gameplay differs between the two (random timing), so the numbers are compared as patterns, not frame by frame.

usage: python reward_check.py GAME.. --out DIR [--episodes 12] [--steps 3000] [--workers 12] [--seconds 240]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'reward10'))
from nesram import run_isolated  # noqa: E402


def _use(emu):
    import audit
    import search
    if emu == 'nesle':
        from neslenes import NesleNes
        audit.Nes = search.Nes = NesleNes
    return audit, search


def audit_run(emu, game, seed, steps, out, tag):
    audit, _ = _use(emu)
    try:
        r = audit.run(game, seed, steps, out, tag)
    except Exception as e:  # an emulator fault is a result, not a crash of the whole check
        return dict(tag=tag, error=repr(e))
    rows = [json.loads(line) for line in (Path(out) / f'{tag}.jsonl').read_text().splitlines()]
    lives = [row['lives'] for row in rows if row.get('lives') is not None]
    health = [row['health'] for row in rows if row.get('health') is not None]
    score = [row['score'] for row in rows if row.get('score') is not None]
    kinds = {}
    for _, kind, *_ in r['events']:
        kinds[kind] = kinds.get(kind, 0) + 1
    return dict(tag=tag, steps=r['steps'], ret=r['ret'], deaths=r['deaths'],
                lives_lost=sum(max(0, a - b) for a, b in zip(lives, lives[1:])),
                health_drops=sum(1 for a, b in zip(health, health[1:]) if b < a),
                score_gain=(max(score) - score[0]) if score else None,
                max_level=max((row['level'] for row in rows if row.get('level') is not None), default=None),
                event_kinds=kinds, game_over=any(k == 'game_over' for _, k, *_ in r['events']))


def search_run(emu, game, seconds, seed, rollout, out):
    _, search = _use(emu)
    try:
        r = search.worker(game, seconds, seed, rollout, out)
    except Exception as e:
        return dict(seed=seed, error=repr(e))
    res = json.loads((Path(out) / f'seed{seed}.json').read_text())
    kinds = {}
    for _, kind, *_ in res['path_events']:
        kinds[kind] = kinds.get(kind, 0) + 1
    return dict(seed=seed, best_level=r['best_level'], best_progress=r['best_progress'], deaths=r['deaths'],
                ret=round(r['ret'], 3), frames=r['frames'], n_events=r['n_events'], path_event_kinds=kinds,
                level_events=len(res['level_events']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('games', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--seconds', type=float, default=240)
    ap.add_argument('--rollout', type=int, default=40)
    ap.add_argument('--skip-search', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    for game in a.games:
        row = dict(game=game, started=time.time())
        jobs = [(emu, game, 1000 + s, a.steps, str(out / game / emu / 'audit'), f'random{s}')
                for emu in ('fceumm', 'nesle') for s in range(a.episodes)]
        res = run_isolated(audit_run, jobs, a.workers * 2)
        row['audit'] = {emu: [r for (e, *_), r in zip(jobs, res) if e == emu] for emu in ('fceumm', 'nesle')}
        if not a.skip_search:
            jobs = [(emu, game, a.seconds, s, a.rollout, str(out / game / emu / 'search'))
                    for emu in ('fceumm', 'nesle') for s in range(a.workers)]
            res = run_isolated(search_run, jobs, a.workers * 2)
            row['search'] = {emu: [r for (e, *_), r in zip(jobs, res) if e == emu] for emu in ('fceumm', 'nesle')}
        (out / game).mkdir(parents=True, exist_ok=True)
        (out / game / 'check.json').write_text(json.dumps(row, indent=1, default=str) + '\n')
        print(game, 'done', flush=True)


if __name__ == '__main__':
    main()
