"""Replay a whole-game route (rl/explore.py --whole-game) from power-on and record it as an MP4.

The route's buttons are pressed one decision (4 frames) at a time on the CPU build, with the search's settings (no
sticky buttons, no dropout, no idle frames, the same restores), so the replay is the exact play the search found.
The replay is an ordinary episode: it does not stop at a death but shows it and counts it (routes found before exact
restores could hide a death in a restore's first frames). Prints the levels reached and the deaths.

usage: python rl/scripts/route_video.py GAME runs/routes/GAME-game/route.npz OUT.mp4 [--speed 1]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import Mp4Writer, PAL  # noqa: E402
from rlenv import RlEnv, END_NAMES  # noqa: E402
import slots as slotlib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('game')
ap.add_argument('route')
ap.add_argument('out')
ap.add_argument('--speed', type=int, default=1, help='keep every k-th frame (1 = real time at 60 fps)')
a = ap.parse_args()
r = np.load(a.route)
meta = json.loads(str(r['meta']))
buttons = r['buttons']
env = RlEnv({a.game: 1}, device='cpu', seed=0, sticky=False, idle_max=0, dropout=False, level_cap=64, level_slots=64)
if meta.get('exact_restore'):
    env.set_restore_exact(True)
if meta.get('search') == 'ramexplore':
    # that search plays on through deaths and game overs (it may continue): replay the same way
    env.set_attempts(1 << 30)
    env.set_attempt_through(True)
    env.set_only_kill(True)
slotlib.write(env, 0, [0], {k: r[f'boot_{k}'] for k in slotlib.FIELDS})
w = np.zeros(64)
w[0] = 1.0
env.set_level_plan(0, 1.0, w)
env.start()
video = Mp4Writer(Path(a.out), fps=60 // a.speed)
levels, deaths, end, f = [], [], None, 0
for t, b in enumerate(buttons):
    env.actions[0] = int(b)
    env.step()
    done = bool(env.done[0])
    for k in range(4):
        if not done:
            if f % a.speed == 0:
                o, e = env.obs[0, k].numpy().astype(np.int32), env.emph[0, k].numpy().astype(np.int32)
                video.write(PAL[o | (e[:, None] << 6)])
            f += 1
    if int(env.events[0]) & 4:                                  # reward10 kEvDeath
        deaths.append((int(env.level_now[0]), t + 1))
    lv = int(env.level_now[0])
    if not levels or levels[-1][0] != lv:
        levels.append((lv, t))
    if done:
        end = (END_NAMES.get(int(env.info[0, 2]), '?'), int(env.info[0, 4]), t + 1)
        break
video.close()
print('levels reached (level, decision):', levels)
print('deaths (level, decision):', deaths or 'none')
print('ended:', end or f'still playing after {len(buttons)} decisions', '| search said:',
      {k: meta[k] for k in ('level', 'won', 'goal_reached', 'route_decisions')})
print(f'video {a.out}: {f // a.speed} frames, {f / 60:.0f} s of play')
