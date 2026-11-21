import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import situation  # noqa: E402
from places import VAR_MIN_TRIES, VAR_SHARE, Places  # noqa: E402
from situation import REAL_MIN  # noqa: E402
from situation import COUNT_ROLLOUTS, Situations, slot  # noqa: E402


class Env:
    def __init__(self):
        self.saved, self.copied = [], []

    def place_save(self, gi, env_idx, slot_idx):
        self.saved += [(int(e), int(s)) for e, s in zip(env_idx, slot_idx) if s >= 0]

    def place_copy(self, gi, src, dst):
        self.copied += list(zip(src, dst))


def key_of(screen, x=0, y=20):
    return (screen << 11) | (x << 6) | y


def play(sit, r, keys, done=None, learner=None):
    """Drive a Situations through one stored rollout r [T+1, N, 2048] with place keys [T, N] (after each decision)."""
    T, n = keys.shape
    done = torch.zeros(T, n, dtype=torch.bool) if done is None else done
    out = [sit.step(t, r[t + 1], keys[t], keys[t] & 63, slice(None), done[t]) for t in range(T)]
    sit.end_rollout(r, done, torch.ones(n, dtype=torch.bool) if learner is None else learner)
    return out


def test_progress_bytes_are_place_bound_flags_not_clocks_or_positions():
    """byte 0: a clock (changes every 5 decisions wherever the player is), byte 1: a flag that flips only on screen 3,
    byte 2: a position (many values)."""
    n, T = 64, 400
    g = torch.Generator().manual_seed(0)
    r = torch.zeros(T + 1, n, 2048, dtype=torch.uint8)
    screen = torch.randint(0, 200, (T + 1, n), generator=g)
    r[:, :, 0] = (torch.arange(T + 1)[:, None] // 5 % 4).to(torch.uint8)
    flag = torch.zeros(n, dtype=torch.uint8)
    for t in range(1, T + 1):
        flag = torch.where((screen[t - 1] == 3) & (torch.rand(n, generator=g) < 0.5), 1 - flag, flag)
        r[t, :, 1] = flag
    r[:, :, 2] = torch.randint(0, 256, (T + 1, n), generator=g, dtype=torch.uint8)
    sit = Situations(n, T, 'cpu', warmup=n * (T - 1) - 1)
    sit.prev = screen[0] << 11
    out = play(sit, r, screen[1:] << 11)
    assert sit.chosen and sit.idx.tolist() == [1], sit.idx.tolist()
    assert all((k == -1).all() for k, _ in out)              # nothing is reported before the counts have run


def ready(n=4, T=8):
    sit = Situations(n, T, 'cpu', warmup=0)
    sit.changes[1], sit.steps = 100, torch.tensor(100000)             # byte 1: a rare flag with the values 0, 2, 3
    sit.vals[1, [0, 2, 3]] = True
    sit.screens[1, 0] = True
    sit.choose()
    assert sit.idx.tolist() == [1]
    sit.rollouts = COUNT_ROLLOUTS
    return sit


def test_a_rare_value_on_a_common_screen_is_a_situation_a_common_value_is_not():
    sit = ready()
    S = 7
    sit.scr[slot(torch.tensor(S))] = 5000
    sit.occ[slot(torch.tensor((S * 2048 + 1) * 256 + 0))] = 4900       # the flag is nearly always 0 on this screen
    sit.occ[slot(torch.tensor((S * 2048 + 1) * 256 + 2))] = 100        # 2% of the decisions: 2
    sit.scr[slot(torch.tensor(9))] = 50                                # screen 9 was hardly ever seen
    ram = torch.zeros(4, 2048, dtype=torch.uint8)
    ram[1, 1], ram[2, 1], ram[3, 1] = 2, 3, 2
    key = torch.tensor([key_of(S, 1, 5), key_of(S, 2, 9), key_of(S, 3, 30), key_of(9)])
    fresh = torch.zeros(4, dtype=torch.bool)
    for t in range(situation.DWELL):
        assert (sit.step(t, ram, key, key & 63, slice(None), fresh)[0] == -1).all()   # not held long enough (flicker)
        key = key + (1 << 6)                                           # walking on: another block, the same height ...
        key[:3] += torch.tensor([t % 2, 0, 0])                         # ... (console 0 changes height: no standing)
    vk, vn = sit.step(situation.DWELL, ram, key, key & 63, slice(None), fresh)
    assert vk[0] == -1 and vk[3] == -1                                 # the usual value; an unknown screen
    assert vk[1] >= (1 << 62) and vn[1] == 100                         # the rare value, seen 100 times
    assert vk[2] == -1                                                 # a value never seen here: rare but not real
    assert sit.dbg.tolist() == [-1, 1 * 256 + 2, -1, -1]
    sit.occ[slot(torch.tensor((S * 2048 + 1) * 256 + 3))] = REAL_MIN
    key[1] += 5 << 6                                                   # elsewhere on the screen: the same situation
    vk2, vn2 = sit.step(0, ram, key, key & 63, slice(None), fresh)
    assert vk2[1] == vk[1] and vk2[2] >= (1 << 62) and vk2[2] != vk2[1] and vn2[2] == REAL_MIN
    sit.occ[slot(torch.tensor((S * 2048 + 1) * 256 + 0))] = 3000       # no rule on this screen any more: the usual
    assert (sit.step(0, ram, key, key & 63, slice(None), fresh)[0][1:3] == -1).all()   # value: 60% of the decisions
    sit.occ[slot(torch.tensor((S * 2048 + 1) * 256 + 0))] = 4900
    fresh[1] = True
    assert sit.step(1, ram, key, key & 63, slice(None), fresh)[0][1] == -1   # a game that just started: none


def test_standing_where_the_others_fly_past_is_a_situation():
    sit = ready()
    air, ground = key_of(5, 10, 14), key_of(5, 11, 22)
    sit.blk[slot(torch.tensor(air))], sit.rst[slot(torch.tensor(air))] = 3000, 2          # crossed, hardly rested in
    sit.blk[slot(torch.tensor(ground))], sit.rst[slot(torch.tensor(ground))] = 3000, 2500  # ordinary ground
    ram = torch.zeros(4, 2048, dtype=torch.uint8)
    fresh = torch.zeros(4, dtype=torch.bool)
    key = torch.tensor([air, ground, air, -1])
    h = torch.tensor([112, 176, 109, -1])                 # console 2 is in the air: its exact height changes
    for t in range(2):
        assert (sit.step(t, ram, key, h, slice(None), fresh)[0] == -1).all()  # not yet three decisions at one height
        h[2] += 1
    vk, vn = sit.step(2, ram, key, h, slice(None), fresh)
    assert vk[0] >= (1 << 62) and vn[0] == 2              # resting where nearly everyone flies past
    assert vk[1] == -1 and vk[2] == -1 and vk[3] == -1    # ordinary ground; in the air in the same block; no place


def test_counts_come_from_the_learners_and_resume_exactly():
    n, T = 3, 6
    sit = ready(n, T)
    sit.rollouts = 0
    r = torch.zeros(T + 1, n, 2048, dtype=torch.uint8)
    r[:, 1, 1] = 2
    keys = torch.tensor([key_of(4, 1, 5), key_of(4, 2, 5), key_of(4, 3, 5)]).repeat(T, 1)
    play(sit, r, keys, learner=torch.tensor([True, True, False]))
    assert sit.rollouts == 1
    assert int(sit.scr[slot(torch.tensor(4))]) == 2 * T                       # the explorer (console 2) is not counted
    assert int(sit.occ[slot(torch.tensor((4 * 2048 + 1) * 256 + 2))]) == T
    assert int(sit.rst[slot(keys[0, 0])]) == T - 2                            # resting from the third decision on
    sit2 = Situations(n, T, 'cpu')
    sit2.load_state_dict(sit.state_dict())
    assert sit2.chosen and sit2.rollouts == 1 and torch.equal(sit2.occ, sit.occ) and torch.equal(sit2.y1, sit.y1)
    assert torch.equal(sit2.dwell, sit.dwell) and torch.equal(sit2.vlist, sit.vlist) and int(sit.dwell[1, 0]) == T - 1


def places(**kw):
    return Places(1, 4, 64, 8, 1, 4, 1, 8, 'cpu', seed=1, **kw)


def test_without_situation_slots_nothing_changes():
    assert places().cap == places(var=0).cap and places(var=8).cap == places().cap + 4 * 1 * 2 + 8


def test_rare_situations_are_saved_least_tried_first_from_learners_and_explorers():
    pl = places(var=8, var_k=2, var_per_level=4)
    X, X2, X3 = (1 << 62) + (5 << 30), (1 << 62) + (6 << 30), (1 << 62) + (7 << 30)
    lv, z = torch.full((4,), 27), torch.zeros(4)
    fresh = torch.zeros(4, dtype=torch.bool)
    key = torch.tensor([500, 500, 600, 500])
    vkey = torch.tensor([X, -1, X3, X2])
    vn = torch.tensor([3., 0., 5000., 0.])
    counted = torch.tensor([True, True, True, False])                      # console 3 is an explorer
    pl.vtried[0, Places._vhash(torch.tensor(X3))] = 10 ** 6                # X3 was tried a lot and given up before
    env = Env()
    pl.step(env, 0, 0, 0, 0, slice(0, 4), None, None, lv, z, fresh, key, counted, vkey=(vkey, vn))
    assert sorted(e for e, s in env.saved if s >= pl.vstage0) == [0, 3]    # two per step: the ones not tried yet
    pl.end_rollout(env)
    assert sorted(pl.vslot_key[0][pl.vslot_level[0] == 27].tolist()) == [X, X2]
    assert sorted(d for _, d in env.copied if d >= pl.v0) == [pl.v0, pl.v0 + 1]
    assert pl.level_of(0, np.array([pl.v0, 27, -1])).tolist() == [27, 27, -1]
    w = np.zeros(64)
    w[27] = 1.0
    full = pl.weights(0, w, np.full(64, 0.2))
    assert abs(full[pl.v0:pl.v0 + 8].sum() - 0.8 * VAR_SHARE) < 1e-9 and abs(full.sum() - 1) < 1e-9
    assert abs(full[27] + full[pl.L:pl.L + 8].sum() - (1 - 0.8 * VAR_SHARE)) < 1e-9   # start + ordinary places
    env = Env()
    pl.step(env, 0, 0, 0, 0, slice(0, 4), None, None, lv, z, fresh, key, counted, vkey=(vkey, vn))
    assert sorted(e for e, s in env.saved if s >= pl.vstage0) == [2]       # X and X2 are held already
    pl.add_starts(0, [pl.v0, pl.v0, 3, pl.v0 + 1], [True, False, True, False])
    pl2 = places(var=8, var_k=2, var_per_level=4)
    pl2.load_state_dict(pl.state_dict())
    assert (pl2.vslot_key == pl.vslot_key).all() and (pl2.vslot_seen == pl.vslot_seen).all() and pl2.varch[0].numel() == 2
    assert pl2.vslot_tries[0, :2].tolist() == [2, 1] and pl2.vslot_wins[0, :2].tolist() == [1, 0]
    assert torch.equal(pl2.vtried, pl.vtried)


def stage(pl, key):
    pl.vst_key[0, 0], pl.vst_level[0, 0], pl.vst_ok[0, 0], pl.vst_seen[0, 0] = key, 27, True, 3.
    pl.end_rollout(Env())
    return sorted(pl.vslot_key[0][pl.vslot_level[0] == 27].tolist())


def test_the_notebook_decides_which_situations_stay():
    """A situation that was tried and never led to a clear, or always does, gives its slot to a new guess; one that
    sometimes leads to a clear stays and gets the most practice."""
    pl = places(var=8, var_k=2, var_per_level=2)
    A, B, C, D = ((1 << 62) + (i << 30) for i in range(1, 5))
    pl.vslot_key[0, :2], pl.vslot_level[0, :2] = [A, B], 27
    w = np.zeros(64)
    w[27] = 1.0
    share = lambda: pl.weights(0, w, np.zeros(64))[pl.v0:pl.v0 + 8]  # noqa: E731
    assert abs(share().sum() - VAR_SHARE) < 1e-9                           # not tried yet: the full share
    assert stage(pl, C) == [A, B]                                          # a full level, nothing spent: C waits
    pl.vslot_tries[0, :2] = VAR_MIN_TRIES - 1
    assert stage(pl, C) == [A, B]                                          # not tried enough yet
    pl.vslot_tries[0, :2], pl.vslot_wins[0, :2] = VAR_MIN_TRIES, [0, 2]
    assert share()[1] > 2 * share()[0]                                     # B sometimes wins: most of the practice
    assert stage(pl, C) == [B, C]                                          # A never led to a clear: spent
    assert int(pl.vtried[0, Places._vhash(torch.tensor(A))]) == VAR_MIN_TRIES   # ... and remembered
    c = int(np.flatnonzero(pl.vslot_key[0] == C)[0])
    assert pl.vslot_tries[0, c] == 0 and pl.vslot_wins[0, c] == 0 and abs(share().sum() - VAR_SHARE) < 1e-9
    pl.vslot_tries[0, 1], pl.vslot_wins[0, 1] = 200, 3
    assert stage(pl, D) == [C, D]                                          # B: fewer than one win in 40 tries now
    pl.vslot_tries[0, :2], pl.vslot_wins[0, :2] = 20, [19, 5]
    k1 = int(pl.vslot_key[0, 1])
    assert stage(pl, A) == sorted([A, k1])                                 # the learned one (19 of 20) gives way
    pl.vslot_tries[0, :2], pl.vslot_wins[0, :2] = 400, 0
    assert share().sum() < 0.2 * VAR_SHARE                                 # everything spent: the share fades
    assert situation.REST == 1
