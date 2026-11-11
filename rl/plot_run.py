"""One picture of a training run: is it improving?   usage: python rl/plot_run.py runs/rl/NAME [out.png]

Reads NAME/metrics.jsonl (one row per iteration) and NAME/eval/it*/summary.json (test plays) and writes
NAME/progress.png (6 panels, 8 with moods, x axis = millions of decisions). Training-episode numbers are averaged over a window of
iterations, weighted by the number of games that finished, because single iterations finish only a few games.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def smooth(x, y, w, k=15):
    """Weighted moving average of y (weights w) over k points."""
    x, y, w = map(np.asarray, (x, y, w))
    out = [np.sum((y * w)[max(0, i - k + 1):i + 1]) / max(np.sum(w[max(0, i - k + 1):i + 1]), 1e-9)
           for i in range(len(y))]
    return x, np.array(out)


def main(run, out=None):
    run = Path(run)
    rows = [json.loads(l) for l in open(run / 'metrics.jsonl')]
    dec = np.array([r['decisions'] for r in rows]) / 1e6
    games = sorted({g for r in rows for g in r['games']})
    moods = any('div' in r for r in rows)
    fig, ax = plt.subplots(2, 4 if moods else 3, figsize=(21 if moods else 16, 8.5))
    fig.suptitle(f'{run.name}: {dec[-1]:.1f}M decisions, iteration {rows[-1]["it"]}', fontsize=13)

    a = ax[0, 0]   # 1. MAIN: average score of finished training games
    for g in games:
        idx = [i for i, r in enumerate(rows) if g in r['games']]
        if not idx:
            continue
        x = dec[idx]
        y = [rows[i]['games'][g]['ret'] for i in idx]
        w = [rows[i]['games'][g]['n'] for i in idx]
        a.plot(*smooth(x, y, w), label=g)
    a.axhline(0, color='grey', lw=0.5)
    a.set_title('1. Score per finished game (training)  <- main graph, should go up')
    a.set_xlabel('M decisions'); a.legend(fontsize=7)

    a = ax[0, 1]   # 2. test plays (greedy, fixed start)
    ev = []
    for f in sorted(glob.glob(str(run / 'eval' / 'it*' / 'summary.json'))):
        it = int(Path(f).parent.name[2:8])
        r = next((r for r in rows if r['it'] == it), None)
        if r is not None:
            ev.append((r['decisions'] / 1e6, json.load(open(f))))
    for g in games:
        pts = [(d, [s['ret'] for s in summ if s['game'] == g]) for d, summ in ev if any(s['game'] == g for s in summ)]
        if pts:
            ln, = a.plot([p[0] for p in pts], [np.mean(p[1]) for p in pts], 'o-', label=g)
            for d, rs in pts:
                if len(rs) > 1:
                    a.scatter([d] * len(rs), rs, s=8, color=ln.get_color(), alpha=0.5)
    a.axhline(0, color='grey', lw=0.5)
    a.set_title(f'2. Test play score every eval ({len(ev)} so far; dots = moods)')
    a.set_xlabel('M decisions'); a.legend(fontsize=7)

    a = ax[0, 2]   # 3. how training games end
    ends = sorted({e for r in rows for g in r['games'].values() for e in g['ends']})
    for e in ends:
        x, y, w = [], [], []
        for i, r in enumerate(rows):
            n = sum(g['n'] for g in r['games'].values())
            if n:
                x.append(dec[i]); w.append(n)
                y.append(sum(g['ends'].get(e, 0) for g in r['games'].values()) / n)
        if x:
            a.plot(*smooth(x, y, w), label=e)
    a.set_ylim(-0.02, 1.02)
    a.set_title('3. How games end (share): game_over / stall / won')
    a.set_xlabel('M decisions'); a.legend(fontsize=8)

    a = ax[1, 0]   # 4. randomness of the moves
    a.plot(dec, [r['ent'] for r in rows], lw=0.8)
    a.set_title('4. Randomness of moves (entropy): slowly down = deciding')
    a.set_xlabel('M decisions')

    a = ax[1, 1]   # 5. game length + furthest level
    for g in games:
        idx = [i for i, r in enumerate(rows) if g in r['games']]
        if idx:
            a.plot(*smooth(dec[idx], [rows[i]['games'][g]['len'] for i in idx],
                           [rows[i]['games'][g]['n'] for i in idx]), label=f'{g} length')
    a.set_title('5. Decisions per finished game')
    a.set_xlabel('M decisions'); a.legend(fontsize=7)
    b = a.twinx()
    for g in games:
        idx = [i for i, r in enumerate(rows) if g in r['games']]
        if idx:
            b.plot(dec[idx], np.maximum.accumulate([rows[i]['games'][g]['level_max'] for i in idx]), '--',
                   color='k', lw=0.8)
    for g in games:   # level starts saved so far (any training game reached them; rl/csrc/rlenv_core.cuh)
        idx = [i for i, r in enumerate(rows) if r.get('saved_levels', {}).get(g)]
        if idx:
            b.plot(dec[idx], [max(rows[i]['saved_levels'][g]) for i in idx], ':', color='tab:green', lw=1.2)
    b.set_ylabel('best level: power-on games (dashed), any game (green dotted)')

    a = ax[1, 2]   # 6. machine
    a.plot(dec, [r['sps'] for r in rows], lw=0.8, label='decisions/s')
    a.set_xlabel('M decisions'); a.set_title('6. Speed and power'); a.legend(loc='lower left', fontsize=8)
    b = a.twinx()
    b.plot(dec, [r['power_w'] for r in rows], color='tab:red', lw=0.6)
    b.set_ylabel('W (red)')

    if moods:
        drows = [r for r in rows if 'div' in r]
        dx = np.array([r['decisions'] for r in drows]) / 1e6
        a = ax[0, 3]   # 7. variety
        K = len(next(iter(drows[0]['div'].values()))['mood_ret'])
        for g in sorted(drows[-1]['div']):
            a.plot(dx, [r['div'].get(g, {}).get('acc', np.nan) for r in drows], lw=0.8, label=g)
        a.axhline(1 / K, color='grey', ls=':', lw=1)
        a.set_ylim(0, 1.02)
        a.set_title(f'7. Variety: judge tells the {K} moods apart (dotted = no variety)')
        a.set_xlabel('M decisions'); a.legend(fontsize=7)
        a = ax[1, 3]   # 8. each mood's running return (quality gate input)
        for g in sorted(drows[-1]['div']):
            mr = np.array([r['div'][g]['mood_ret'] for r in drows if g in r['div']])
            for k in range(mr.shape[1]):
                a.plot(dx[:len(mr)], mr[:, k], lw=0.7)
        a.set_title('8. Training score of each mood (all should stay close)')
        a.set_xlabel('M decisions')

    fig.tight_layout()
    out = Path(out) if out else run / 'progress.png'
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == '__main__':
    main(*sys.argv[1:])
