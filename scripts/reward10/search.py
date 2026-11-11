"""Reward-guided savestate search (Go-Explore style) used to validate a reward spec.

If following only the reward reaches level ends, the reward points the right way.
Every result keeps the exact action sequence from power-on, so audit.py can replay
it deterministically and render the events. Diagnostic only, never training data.

usage: python search.py <game> --seconds 300 --workers 24 --out DIR
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nesram import Nes, run_isolated  # noqa: E402
from reward import RewardTracker  # noqa: E402

FRAMES_PER_DECISION = 8
BUCKETS_PER_LEVEL = 60


def load(game):
    return importlib.import_module(f'games.{game}')


def cell_of(tr, g):
    lvl = tr.best_level if tr.best_level is not None else -1
    play = bool(tr.prev.in_play) if tr.prev is not None else False
    prog = tr.best_progress or 0.0
    # in_play is part of the cell so end-of-level transitions are archived and continued;
    # an optional module hook search_key(signals) adds game phases (e.g. post-fight screens)
    extra = g.search_key(tr.prev) if hasattr(g, 'search_key') and tr.prev is not None else None
    # optional module SEARCH_BUCKETS: cells per PROGRESS_UNIT, so search resolution is not tied to reward scale
    buckets = getattr(g, 'SEARCH_BUCKETS', BUCKETS_PER_LEVEL)
    return (lvl, int(prog / g.PROGRESS_UNIT * buckets), play, extra)


def apply_action(n, act):
    """An action is a button tuple held for one decision, or a macro: list of (buttons, frames)."""
    if isinstance(act, list):
        for buttons, frames in act:
            n.step(buttons, frames)
        return sum(f for _, f in act)
    n.step(act, FRAMES_PER_DECISION)
    return FRAMES_PER_DECISION


def score_of(tr, g):
    """(level, progress in levels): a later level always ranks above more progress in an earlier one."""
    lvl = tr.best_level if tr.best_level is not None else -1
    return (lvl, (tr.best_progress or 0.0) / g.PROGRESS_UNIT)


def worker(game, seconds, seed, rollout, out, prefix=()):
    g = load(game)
    rng = random.Random(seed)
    # optional module hooks (search only): SEARCH_ACTIONS = action indices the search samples (e.g. macros),
    # SEARCH_ALLOW_DEATHS = keep searching through deaths that do not end the game (knockdowns, opponent scores)
    pool = list(getattr(g, 'SEARCH_ACTIONS', range(len(g.ACTIONS))))
    allow_deaths = getattr(g, 'SEARCH_ALLOW_DEATHS', False)
    n = Nes(g.ROM)
    g.boot(n)
    tr = RewardTracker(g)
    tr.update(n.ram(), 0)
    ret0 = 0.0
    for i, a in enumerate(prefix, 1):   # continue from a saved run (actions from power-on)
        f = apply_action(n, g.ACTIONS[a])
        ret0 += tr.update(n.ram(), i, frames=f)[0]
    archive = {cell_of(tr, g): dict(snap=n.save(), tr=copy.deepcopy(tr), acts=list(prefix), ret=ret0, visits=0)}
    t0, frames, events, best = time.time(), 0, [], None
    while time.time() - t0 < seconds:
        keys = list(archive)
        if rng.random() < 0.5:   # push the frontier
            keys.sort(key=lambda k: (k[0], k[1]), reverse=True)
            keys = keys[:5]
        w = [1 / math.sqrt(archive[k]['visits'] + 1) for k in keys]
        k = rng.choices(keys, w)[0]
        e = archive[k]
        e['visits'] += 1
        n.load(e['snap'])
        tr = copy.deepcopy(e['tr'])
        acts, ret, a = list(e['acts']), e['ret'], pool[rng.randrange(len(pool))]
        # long rollouts sometimes, so multi-second cutscenes/tallies can complete
        for _ in range(rollout * rng.choice((1, 1, 2, 4, 8))):
            if rng.random() > 0.8:
                a = pool[rng.randrange(len(pool))]
            f = apply_action(n, g.ACTIONS[a])
            frames += f
            acts.append(a)
            r, done, ev = tr.update(n.ram(), len(acts), frames=f)
            ret += r
            for kind, val in ev:
                events.append(dict(kind=kind, value=val, step=len(acts), level=tr.best_level))
            if any(kind in ('game_over', 'skip') or (kind == 'death' and not allow_deaths) for kind, _ in ev):
                break
            c = cell_of(tr, g)
            old = archive.get(c)
            if old is None or (tr.deaths, len(acts)) < (old['tr'].deaths, len(old['acts'])):
                archive[c] = dict(snap=n.save(), tr=copy.deepcopy(tr), acts=list(acts), ret=ret,
                                  visits=old['visits'] if old else 0)
            if done:
                break
    top = max(archive.values(), key=lambda e: (*score_of(e['tr'], g), -len(e['acts'])))
    res = dict(seed=seed, frames=frames, fps=frames / seconds, cells=len(archive),
               best_level=top['tr'].best_level, best_progress=top['tr'].best_progress,
               score=score_of(top['tr'], g), deaths=top['tr'].deaths, ret=top['ret'],
               won=top['tr'].done and not any(e['kind'] == 'game_over' for e in events[-1:]),
               acts=top['acts'], path_events=top['tr'].events, n_events=len(events),
               level_events=[e for e in events if e['kind'] == 'level'][:50])
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f'seed{seed}.json').write_text(json.dumps(res))
    return {k: v for k, v in res.items() if k not in ('acts', 'level_events', 'path_events')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game')
    ap.add_argument('--seconds', type=float, default=300)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--rollout', type=int, default=40)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', help='result JSON whose acts (from power-on) are replayed before searching')
    a = ap.parse_args()
    prefix = json.loads(Path(a.prefix).read_text())['acts'] if a.prefix else ()
    res = run_isolated(worker, [(a.game, a.seconds, s, a.rollout, a.out, prefix) for s in range(a.workers)],
                       a.workers)
    res.sort(key=lambda r: tuple(r['score']), reverse=True)
    for r in res[:5]:
        print(json.dumps(r))
    print('levels reached by seed:', sorted(r['best_level'] for r in res))


if __name__ == '__main__':
    main()
