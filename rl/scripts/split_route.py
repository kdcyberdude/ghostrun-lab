"""Split a whole-game route (rl/explore.py --whole-game, any game) into the per-level routes the trainer's backward
curriculum reads (train.py --routes DIR --entrances DIR/entrances.npy, rl/routes.py).

Level 0's entrance is the route's power-on state; level L's entrance is the first route state at level L (just
after the level event). The route states after an entrance, at least --spacing decisions apart, become that level's
route L<L>.npz. Only levels the route starts are written; the route's last level has no clear, but its states still
give the policy starts deep in the level.

usage: python rl/scripts/split_route.py GAME runs/routes/GAME-game/route.npz OUTDIR [--spacing 24]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv  # noqa: E402
import slots as slotlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('game')
ap.add_argument('route')
ap.add_argument('out')
ap.add_argument('--spacing', type=int, default=24)
a = ap.parse_args()
r = np.load(a.route)
meta = json.loads(str(r['meta']))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
levels, steps = r['levels'], r['steps']
env = RlEnv({a.game: 1}, device='cpu', seed=0, sticky=False, idle_max=0, dropout=False, level_cap=64, level_slots=64)
lay = env.level_layout(0)
assert all(meta['layout'][k] == lay[k] for k in ('stride', 'key_cap', 'st')), 'route saved with another env build'
entr = {0: (-1, 0)}                                  # level -> (route index of its entrance, -1 = power-on), step
for i, lv in enumerate(levels.tolist()):
    if lv not in entr:
        entr[lv] = (i, int(steps[i]))
slotlib.write(env, 0, [0], {k: r[f'boot_{k}'] for k in slotlib.FIELDS})
for lv, (i, _) in entr.items():
    if i >= 0:
        slotlib.write(env, 0, [lv], {k: r[f'state_{k}'][i:i + 1] for k in slotlib.FIELDS})
np.save(out / 'entrances.npy', slotlib.tail(env, 0, env.level_slots))
for lv, (i0, s0) in sorted(entr.items()):
    idx, last = [], s0
    for i in np.flatnonzero(levels == lv):
        if i == i0 or steps[i] - last < a.spacing:
            continue
        idx.append(i)
        last = int(steps[i])
    rec = {k: r[f'state_{k}'][idx] for k in slotlib.FIELDS}
    np.savez_compressed(out / f'L{lv}.npz', steps=steps[idx] - s0, **rec,
                        meta=json.dumps(dict(game=a.game, layout=lay, from_route=a.route, entrance_step=s0)))
    print(f'level {lv}: entrance at decision {s0}, {len(idx)} route states')
