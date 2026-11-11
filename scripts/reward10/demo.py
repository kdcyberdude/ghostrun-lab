"""Build one long fresh-start run per game by chaining reward-only searches (CPU only).

Each round runs search.worker from near the end of the best run so far (--backoff decisions before it),
so every worker pushes the same frontier. A round that does not improve (level, progress) restarts from the
start of the current level; two rounds in a row without improvement end the run. OUT/best.json keeps
the actions from power-on (replay with audit.py replay), the path events and the per-round history.
The result is demonstration/exploration data (imitation pretraining, checkpoint starts), never
evaluation: evaluation stays a fresh start played by the model itself.

usage: python demo.py <game> --workers 5 --round-seconds 600 --max-rounds 40 --out DIR  (rerun to resume)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nesram import run_isolated  # noqa: E402
from search import worker  # noqa: E402


def start_point(best, backoff, stalled):
    """Actions to replay before a round: after a stall the first step of the current level, otherwise
    `backoff` decisions before the end of the best run, because its furthest state is usually an
    unavoidable death (SMB 60/60, Kage 57/60 random rollouts die from it), never before the level start."""
    starts = [e[0] for e in best['path_events'] if e[1] == 'level']
    lvl = starts[-1] if starts else 0
    return best['acts'][:lvl if stalled else max(lvl, len(best['acts']) - backoff)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game')
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--round-seconds', type=float, default=600)
    ap.add_argument('--max-rounds', type=int, default=40)
    ap.add_argument('--rollout', type=int, default=40)
    ap.add_argument('--backoff', type=int, default=150)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best, prefix, stalls, hist = None, [], 0, []
    if (out / 'best.json').exists():   # resume an interrupted run from its best result
        best = json.loads((out / 'best.json').read_text())
        hist = best.pop('history')
        best.pop('game')
        prefix = start_point(best, a.backoff, False)
    for k in range(len(hist), a.max_rounds):
        t0, rd = time.time(), out / f'round{k}'
        for p in rd.glob('seed*.json'):   # partial results of an interrupted round
            p.unlink()
        run_isolated(worker, [(a.game, a.round_seconds, 1000 * k + s, a.rollout, str(rd), prefix)
                              for s in range(a.workers)], a.workers)
        res = [json.loads(p.read_text()) for p in rd.glob('seed*.json')]
        top = max(res, key=lambda r: (*r['score'], -len(r['acts'])))
        improved = best is None or tuple(top['score']) > tuple(best['score'])
        if improved:
            best, stalls = top, 0
        else:
            stalls += 1
        hist.append(dict(round=k, start_step=len(prefix), round_score=top['score'], best_score=best['score'],
                         best_level=best['best_level'], steps=len(best['acts']), won=best['won'],
                         minutes=round((time.time() - t0) / 60, 1)))
        print(json.dumps(hist[-1]), flush=True)
        (out / 'best.json').write_text(json.dumps(dict(game=a.game, **best, history=hist)))
        if best['won'] or stalls >= 2:
            break
        prefix = start_point(best, a.backoff, not improved)


if __name__ == '__main__':
    main()
