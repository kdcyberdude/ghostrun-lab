#!/usr/bin/env python3
"""Small result tables from one rl/train.py run directory (reads only metrics.jsonl and eval/it*/summary.json).

usage: python scripts/summarize_run.py RUN_DIR OUT_DIR [--name NAME]

Writes OUT_DIR/NAME_eval_history.csv  one row per test play (iteration, game, mood, level reached, end, return)
       OUT_DIR/NAME_training.csv      every 10th metrics row: iteration, decisions, speed, entropy, KL, explained var
       OUT_DIR/NAME_level_clear.csv   last logged per-level clear rate and visits (runs with level starts only)
"""
import argparse
import csv
import glob
import json
import os


def level_name(game, level):
    return f'{level // 4 + 1}-{level % 4 + 1}' if game == 'smb' else f'L{level}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--name', default=None)
    a = ap.parse_args()
    name = a.name or os.path.basename(a.run.rstrip('/'))
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.out, f'{name}_eval_history.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'game', 'mood', 'level_reached', 'level_name', 'end', 'deaths', 'decisions', 'return'])
        for path in sorted(glob.glob(os.path.join(a.run, 'eval/it*/summary.json'))):
            it = int(os.path.basename(os.path.dirname(path))[2:])
            for r in json.load(open(path)):
                w.writerow([it, r['game'], r.get('mood', ''), r['level'], level_name(r['game'], r['level']), r['end'],
                            r['deaths'], r['decisions'], round(r['ret'], 4)])

    last = None
    with open(os.path.join(a.run, 'metrics.jsonl')) as src, \
            open(os.path.join(a.out, f'{name}_training.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        cols = ['it', 'decisions', 'sps', 'mfu', 'ent', 'kl', 'explained_var', 'reward_mean']
        w.writerow(cols)
        for i, line in enumerate(src):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            last = r
            if i % 10 == 0:
                w.writerow([r.get(c, '') if not isinstance(r.get(c), float) else round(r[c], 5) for c in cols])

    plan = (last or {}).get('level_plan', {})
    if plan:
        with open(os.path.join(a.out, f'{name}_level_clear.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iteration', 'game', 'level', 'level_name', 'clear_rate', 'visits'])
            for game, p in plan.items():
                for lv, d in sorted(p['levels'].items(), key=lambda x: int(x[0])):
                    w.writerow([last['it'], game, lv, level_name(game, int(lv)), round(d['clear'], 3),
                                round(d['visits'], 1)])


if __name__ == '__main__':
    main()
