"""Summarize reward_check.py results (<out>/<game>/check.json) per emulator into one JSON file.

usage: python summarize.py OUT_DIR SUMMARY.json GAME.. [--command CMD] [--note NOTE]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def emu_summary(row, emu):
    audit = [r for r in row['audit'][emu] if 'error' not in r]
    search = [r for r in row.get('search', {}).get(emu, []) if 'error' not in r]
    total = lambda k: sum(r.get(k) or 0 for r in audit)  # noqa: E731
    return dict(audit_errors=len(row['audit'][emu]) - len(audit), game_overs=sum(r['game_over'] for r in audit),
                deaths=total('deaths'), lives_lost=total('lives_lost'), health_drops=total('health_drops'),
                score_gain=total('score_gain'), runs_with_score=sum(1 for r in audit if r.get('score_gain')),
                search_errors=len(row.get('search', {}).get(emu, [])) - len(search),
                search_levels_by_seed=sorted(r['best_level'] for r in search),
                search_frames_per_worker=round(sum(r['frames'] for r in search) / max(1, len(search))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('summary')
    ap.add_argument('games', nargs='+')
    ap.add_argument('--command', default='')
    ap.add_argument('--note', default='')
    a = ap.parse_args()
    rows = {g: json.loads((Path(a.out) / g / 'check.json').read_text()) for g in a.games}
    res = dict(what='reward10 reward on NeSLE (CPU build of its GPU code) vs fceumm: 12 sticky-random games from the '
                    'fresh start (seeds 1000-1011, 3000 decisions max) and a 12-worker x 240 s reward-guided search '
                    'per emulator', command=a.command, note=a.note,
               games={g: {emu: emu_summary(row, emu) for emu in ('fceumm', 'nesle')} for g, row in rows.items()})
    Path(a.summary).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
