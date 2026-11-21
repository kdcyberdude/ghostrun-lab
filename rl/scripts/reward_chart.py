"""Training reward chart of a coverage run (RUN/metrics.jsonl) -> RUN/notes/charts/reward.png
  1. reward per game (all levels): the average return of the games that ended in each iteration
  2. reward per 100 decisions in a level (learners' moves; metrics 'level_reward', logged from 2026-11-20) for the
     hard levels: how fast Mario still finds new places there
  3. places ever found in a level (8x8 blocks any game touched; 'blocks_level'): flat = nothing new is being found
usage: python rl/scripts/reward_chart.py RUN"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

run = Path(sys.argv[1])
rows = {}
for line in open(run / 'metrics.jsonl'):
    d = json.loads(line)
    rows[d['it']] = d                     # a restart repeats a few iterations: keep the last
its = sorted(rows)
t0 = rows[its[0]]['time']
hrs = np.array([(rows[i]['time'] - t0) / 3600 for i in its])
name = lambda lv: f'{lv // 4 + 1}-{lv % 4 + 1}'  # noqa: E731
HARD = [(22, '#8a5cf6'), (24, '#2a9d8f'), (25, '#d9822b'), (27, '#264653'), (28, '#e76f51'), (31, '#d62828')]


def smooth(y, k=15):
    y = np.asarray(y, float)
    ok = ~np.isnan(y)
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        out[i] = w.mean() if len(w) else np.nan
    return np.where(ok | ~np.isnan(out), out, np.nan)


fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
ep = [rows[i].get('coverage', {}).get('ep_mean', np.nan) for i in its]
ax[0].plot(hrs, ep, color='#2a6fdb', alpha=.25)
ax[0].plot(hrs, smooth(ep), color='#2a6fdb', lw=2)
ax[0].set_ylabel('reward per game'); ax[0].grid(alpha=.3)
ax[0].set_title('1. Reward per training game (all levels; line = average of the last 15 iterations)')
lr_its = [k for k, i in enumerate(its) if 'level_reward' in rows[i]]
for lv, c in HARD:
    y = [np.nan] * len(its)
    for k in lr_its:
        r = rows[its[k]]['level_reward'].get(str(lv)) or rows[its[k]]['level_reward'].get(lv)
        if r and r[1] > 0:
            y[k] = 100 * r[0] / r[1]
    ax[1].plot(hrs, smooth(y), lw=2, color=c, label=name(lv))
ax[1].set_ylabel('reward per 100 decisions'); ax[1].grid(alpha=.3); ax[1].legend(ncol=6, loc='upper left')
ax[1].set_title('2. Reward per 100 decisions spent in a level (hard levels; logged from it 5448)')
for lv, c in HARD:
    y = [rows[i].get('coverage', {}).get('blocks_level', [np.nan] * 32)[lv] for i in its]
    ax[2].plot(hrs, y, lw=2, color=c, label=name(lv))
ax[2].set_ylabel('places found ever (8x8 blocks)'); ax[2].grid(alpha=.3); ax[2].legend(ncol=6, loc='upper left')
ax[2].set_title('3. Places ever found in the hard levels (flat = nothing new found there)')
ax[2].set_xlabel(f'hours since it {its[0]}')
fig.tight_layout()
out = run / 'notes' / 'charts' / 'reward.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110)
print(out)
