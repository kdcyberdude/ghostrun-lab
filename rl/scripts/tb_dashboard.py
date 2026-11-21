"""Live TensorBoard dashboard of a run, kept to 4 charts (user request 2026-11-20: simple, readable at a glance).
Reads RUN/metrics.jsonl (training) and RUN/validation/history.jsonl (rl/validate.py), writes RUN/tb_dashboard/,
x = training iteration, checks for new lines every minute:
  1 levels passed on one life (%)       validation: average over the 31 level starts, model's own moves only
  2 full game: how far from 1-1          validation: median furthest level of 32 games, 1 = 1-1 .. 32 = 8-4
  3 reward per training game             training: places newly reached per game
  4 maze castles: places found           training: 8x8 places ever found in 7-4 and 8-4 (flat = stuck)
 5-9 stuck -> search -> hand-off (train.py --stuck-search, 2026-11-21), one line per level:
  5 stuck: M learner decisions without a new block (a search starts at the trigger line)
  6 search: M decisions used (stops early when nothing new is left; at the cap with finds still coming = too small)
  7 search: rare cells found (and in the last 10M decisions)
  8 new spot: level-start learner games that reach it (%) (path design, from 2026-11-21 ~20:00; the goal: >= 25%)
  9 new spot: practice tries that reach it (%), near (<= 8 decisions before it) / mid (<= 64) / far (> 64)
    (8 and 9 before 2026-11-21 ~20:00: the gate design's curriculum state and gate reach %)
usage: python rl/scripts/tb_dashboard.py RUN   (then: tensorboard --logdir RUN/tb_dashboard)"""
import json
import shutil
import sys
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def name(L):
    L = int(L)
    return f'{L // 4 + 1}-{L % 4 + 1}'

run = Path(sys.argv[1])
logdir = run / 'tb_dashboard'
shutil.rmtree(logdir, ignore_errors=True)          # rewritten from the files on every start: no duplicate points
w = SummaryWriter(str(logdir), flush_secs=30)
pos = {'m': 0, 'v': 0}
seen = set()


def train_row(d):
    it = d['it']
    if it in seen:                                  # a restart repeats a few iterations
        return
    seen.add(it)
    c = d.get('coverage') or {}
    if 'ep_mean' in c:
        w.add_scalar('3 reward per training game', c['ep_mean'], it)
    b = c.get('blocks_level') or []
    if len(b) == 32:
        w.add_scalar('4 maze castles: places found (7-4)', b[27], it)
        w.add_scalar('4 maze castles: places found (8-4)', b[31], it)
    st = d.get('stuck')
    if st:
        if st['dry_M']:
            w.add_scalars('5 stuck: M learner decisions without a new block (search at the trigger line)',
                          {**{name(L): v for L, v in st['dry_M'].items()}, 'trigger': min(st['trigger_M'].values())}, it)
        j = st.get('job')
        if j:
            w.add_scalars('6 search: M decisions used', {name(j['level']): j.get('decisions', 0) / 1e6}, it)
            w.add_scalars('7 search: rare cells found (all / last 10M decisions)',
                          {name(j['level']): j.get('rare_cells', 0), name(j['level']) + ' recent': j.get('rare_recent', 0)}, it)
        h = st.get('handoff') or {}
        if h and 'start_games' in next(iter(h.values())):
            w.add_scalars('8 new spot: level-start learner games that reach it (%)',
                          {name(L): 100.0 * v['start_reached'] / v['start_games'] for L, v in h.items()
                           if v['start_games']}, it)
            w.add_scalars('9 new spot: practice tries that reach it (%) near / mid / far',
                          {f'{name(L)} {k}': 100.0 * v[k][1] / v[k][0] for L, v in h.items()
                           for k in ('near', 'mid', 'far') if v[k][0]}, it)
        elif h and 'stage' in next(iter(h.values())):
            w.add_scalars('8 gates: curriculum state (0 = the state next to the gate)', {name(L): v['stage'] for L, v in h.items()}, it)
            if 'tries' in next(iter(h.values())):
                w.add_scalars('9 gates: practice tries that reach the gate (%)',
                              {name(L): 100.0 * v['reached'] / v['tries'] for L, v in h.items() if v['tries']}, it)


def val_row(r):
    it = r['it']
    w.add_scalar('1 levels passed on one life (%)', 100 * r['level_pass_mean'], it)
    w.add_scalar('2 full game from 1-1: median furthest level (1 = 1-1, 32 = 8-4)', r['full_best_median'] + 1, it)


def read_new(path, key, fn):
    if not path.exists():
        return
    with open(path) as f:
        f.seek(pos[key])
        while True:
            line = f.readline()
            if not line or not line.endswith('\n'):
                break                                 # a half-written line: read it next time
            pos[key] = f.tell()
            if line.strip():
                fn(json.loads(line))


while True:
    read_new(run / 'metrics.jsonl', 'm', train_row)
    read_new(run / 'validation' / 'history.jsonl', 'v', val_row)
    w.flush()
    time.sleep(60)
