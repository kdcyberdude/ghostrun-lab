"""Dropout as the environment actually applies it (CPU build, bit-identical to the GPU env), all 12 games, sticky on,
random legal moves. Checks, per decision:
  1. masks == the legal parts for the input the timeline says is off now (column 0);
  2. the timeline is the truth: column j at decision t equals column 0 at decision t + j (same episode);
  3. the pad bytes sent to the console never use the dropped input (direction bit never set; A/B hold dropped -> the
     button is never held on two frames in a row; A/B turbo dropped -> the button switches at most once in the
     decision's 4 frames, as per-frame sticky repeats allow, never the on/off/on of turbo);
  4. schedule statistics: share of decisions with an input off, run lengths (8..22), gaps (mean ~35), per-input share.
usage: python rl/tests/test_dropout_env.py [DECISIONS] [ENVS_PER_GAME]"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, GAMES, MANIFEST  # noqa: E402
from actions import random_legal  # noqa: E402

NAMES = ['Up', 'Down', 'Left', 'Right', 'A-hold', 'B-hold', 'A-turbo', 'B-turbo']
DIR_BIT = {0: 16, 1: 32, 2: 64, 3: 128}
DPAD = [0, 16, 16 | 128, 128, 32 | 128, 32, 32 | 64, 64, 16 | 64]


def legal_masks(dropped, start_ok, select_ok):
    dp = [1] * 9
    if dropped in DIR_BIT:
        dp = [0 if DPAD[d] & DIR_BIT[dropped] else 1 for d in range(9)]
    a = [1, 0 if dropped == 4 else 1, 0 if dropped == 6 else 1]
    b = [1, 0 if dropped == 5 else 1, 0 if dropped == 7 else 1]
    return dp + a + b + [1, 1 if start_ok else 0] + [1, 1 if select_ok else 0]


def main(steps=3000, per_game=4):
    env = RlEnv({g: per_game for g in GAMES}, device='cpu', seed=77, sticky=True, idle_max=30, dropout=True)
    env.start()
    gen = torch.Generator().manual_seed(77)
    N = env.n
    hist_tl = [[] for _ in range(N)]     # per env, current episode: timeline per decision
    bad = Counter()
    ok_ss = {}                            # learn start/select legality from the masks of the first decision
    off_counts, runs, gaps = Counter(), [], []
    cur_run = [None] * N                  # (input, length)
    cur_gap = [0] * N
    decisions = 0
    for t in range(steps):
        tl = env.timeline.numpy().reshape(N, 8, 45).copy()
        masks = env.masks.numpy().copy()
        act = random_legal(env.masks, gen)
        env.actions.copy_(act)
        env.step()
        pads = env.pads.numpy()
        done = env.done.numpy().astype(bool)
        for e in range(N):
            col = tl[e, :, 0]
            dropped = int(np.argmax(col)) if col.any() else -1
            if col.sum() > 1:
                bad['two inputs off at once'] += 1
            # 1. masks
            if e not in ok_ss:
                ok_ss[e] = (masks[e, 16] == 1, masks[e, 18] == 1)
            if list(masks[e]) != legal_masks(dropped, *ok_ss[e]):
                bad['masks != legal(dropped)'] += 1
            # 3. pads (the 4 frames of this decision)
            p = pads[e]
            if dropped in DIR_BIT and any(x & DIR_BIT[dropped] for x in p):
                bad['dropped direction pressed'] += 1
            for btn, hold, turbo in ((1, 4, 6), (2, 5, 7)):
                on = [bool(x & btn) for x in p]
                if dropped == hold and any(on[i] and on[i + 1] for i in range(3)):
                    bad['dropped hold button held'] += 1
                # sticky repeats are per frame, so a decision is the old move for k frames then the new one: a
                # held button switches at most once; turbo (on even frames) would switch on/off every frame
                if dropped == turbo and sum(on[i] != on[i + 1] for i in range(3)) > 1:
                    bad['dropped turbo button pulsed'] += 1
            # 2. timeline truth: compare with earlier predictions of this decision
            h = hist_tl[e]
            for j in range(1, min(45, len(h) + 1)):
                if not np.array_equal(h[-j][:, j], col):
                    bad['timeline prediction wrong'] += 1
                    break
            h.append(tl[e])
            if len(h) > 45:
                h.pop(0)
            # 4. statistics
            decisions += 1
            if dropped >= 0:
                off_counts[dropped] += 1
                if cur_run[e] and cur_run[e][0] == dropped:
                    cur_run[e][1] += 1
                else:
                    if cur_gap[e]:
                        gaps.append(cur_gap[e])
                    cur_gap[e] = 0
                    cur_run[e] = [dropped, 1]
            else:
                if cur_run[e]:
                    runs.append(cur_run[e][1])
                    cur_run[e] = None
                cur_gap[e] += 1
            if done[e]:                    # new episode: new schedule, the next decision has no history
                hist_tl[e] = []
                cur_run[e] = None
                cur_gap[e] = 0
    off = sum(off_counts.values())
    print(f'{decisions} decisions ({N} envs x {steps}), episodes ended: counted per env')
    print(f'an input off in {100 * off / decisions:.1f}% of decisions (design ~30%)')
    print('per input: ' + ', '.join(f'{NAMES[i]} {100 * off_counts[i] / decisions:.2f}%' for i in range(8)) + ' (design ~3.75% each)')
    if runs:
        print(f'off-runs: {len(runs)}, length min {min(runs)} max {max(runs)} mean {np.mean(runs):.1f} (design 8..22, mean 15)')
    if gaps:
        print(f'healthy gaps: {len(gaps)}, min {min(gaps)} mean {np.mean(gaps):.1f} (design >=1, mean ~35)')
    print('problems:', dict(bad) if bad else 'none')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main(*(int(x) for x in sys.argv[1:])))
