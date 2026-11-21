"""Chart of RUN/validation/history.jsonl (rl/validate.py): one-life pass rate per level and full-game reach over time.
usage: python rl/scripts/validation_chart.py RUN   -> RUN/validation/chart.png"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

run = Path(sys.argv[1])
h = [json.loads(x) for x in (run / 'validation' / 'history.jsonl').read_text().splitlines() if x.strip()]
h.sort(key=lambda r: r['it'])
its = [r['it'] for r in h]
name = lambda lv: f'{lv // 4 + 1}-{lv % 4 + 1}'
fig, ax = plt.subplots(3, 1, figsize=(10, 12))
ax[0].plot(its, [r['level_pass_mean'] * 100 for r in h], 'o-', color='#2a6fdb', lw=2, label='average, 31 levels (%)')
ax[0].set_ylim(0, 100); ax[0].set_ylabel('% passed on ONE life'); ax[0].grid(alpha=.3)
a2 = ax[0].twinx(); a2.plot(its, [r['levels_at_70'] for r in h], 's-', color='#d9822b', lw=2, label='levels passing 70%+')
a2.set_ylim(0, 31); a2.set_ylabel('levels at 70%+ (of 31)')
ax[0].legend(loc='upper left'); a2.legend(loc='lower right')
ax[0].set_title('1. Validation: each level from its start, one life, model\'s own moves only (~32 games each)')
ax[1].plot(its, [r['full_best_mean'] + 1 for r in h], 'o-', color='#2a6fdb', lw=2, label='average')
ax[1].plot(its, [r['full_best_median'] + 1 for r in h], 'o--', color='#2a9d8f', lw=2, label='median')
ax[1].plot(its, [r['full_best_max'] + 1 for r in h], '^-', color='#d62828', lw=1.5, label='best game')
ax[1].set_yticks(range(1, 33, 2)); ax[1].set_yticklabels([name(v - 1) for v in range(1, 33, 2)]); ax[1].set_ylim(0.5, 32.5)
ax[1].grid(alpha=.3); ax[1].legend(loc='upper left')
ax[1].set_title(f"2. Full games from 1-1, 3 lives ({h[-1]['full_games']} games): furthest level reached; won: "
                + ', '.join(str(r['full_won']) for r in h[-6:]))
last = h[-1]['level_pass']
lv = sorted(last, key=lambda k: (int(k.split('-')[0]), int(k.split('-')[1])))
ax[2].bar(lv, [last[k] * 100 for k in lv], color=['#d62828' if last[k] < .4 else '#d9822b' if last[k] < .7 else '#2a9d8f' for k in lv])
ax[2].set_ylim(0, 100); ax[2].set_ylabel('% passed on one life'); ax[2].grid(alpha=.3, axis='y')
ax[2].tick_params(axis='x', rotation=90)
ax[2].set_title(f"3. Per level at it {h[-1]['it']} (red < 40%, orange < 70%, green 70%+)")
ax[1].set_xlabel('training iteration'); ax[0].set_xlabel('training iteration')
fig.tight_layout()
fig.savefig(run / 'validation' / 'chart.png', dpi=110)
print(run / 'validation' / 'chart.png')
