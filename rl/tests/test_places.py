"""Places inside levels (rl/places.py, rlenv_place_save), CPU reference build unless --gpu:

1. exact replay: a trained policy plays; at decision `at` the env is saved into a place slot (state A kept); the env
   idles until its episode ends and the next one starts from the place; a second env loaded from A takes one neutral
   decision (= a restart's 4 neutral frames) and then the same moves: pictures, rewards, reward parts, done, events
   and masks must match for K decisions; the finished place episode reports the place's slot and level;
2. an old state (64 level slots, no places) loads into an env with place slots: same saved levels, and a level-start
   episode plays exactly as in the old env;
3. Places bookkeeping on a short real rollout: keys stable for one picture, new places staged and kept, weights sum
   to 1 and give places only failure share.

usage: python rl/tests/test_places.py MODEL.pt          (an eval/itN/model.pt that reaches 1-2)
       python rl/tests/test_places.py MODEL.pt --gpu    GPU build == CPU build with training settings and place
                                                        saves/starts, every output and every saved slot
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Policy, pack  # noqa: E402
from rlenv import RlEnv  # noqa: E402
from places import Places  # noqa: E402

KW = dict(device='cpu', seed=5, sticky=False, idle_max=0, dropout=False)
L, P = 64, 32


def policy(path):
    s = torch.load(path, map_location='cpu', weights_only=False)
    m = Policy(core=s['cfg']['core'], moods=s['cfg'].get('moods', 0))
    m.load_state_dict(s['model'])
    return m.eval()


class Driver:
    def __init__(self, m, n=1):
        self.m, self.h, self.prev = m, torch.zeros(n, m.core), [torch.zeros(n, dtype=torch.long) for _ in range(5)]
        self.mood = torch.zeros(n, dtype=torch.long) if m.moods else None

    @torch.no_grad()
    def move(self, env, greedy=True, gen=None):
        z = self.m.encode(env.obs, env.emph, env.timeline, self.prev, self.mood)
        self.h = self.m.gru(z.float(), self.h)
        parts, _ = self.m.act(self.h, env.masks, greedy=greedy, generator=gen)
        self.prev = parts
        return pack(parts)


def same(a, b, what, keys=('obs', 'emph', 'reward', 'parts', 'done', 'events', 'masks')):
    for k in keys:
        if not torch.equal(getattr(a, k), getattr(b, k)):
            raise AssertionError(f'{what}: {k} differs')


def one_hot_plan(env, slot):
    w = np.zeros(env.level_cap)
    w[slot] = 1.0
    env.set_level_plan(0, 1.0, w)


def exact_replay(m, at=400, K=500):
    env = RlEnv({'smb': 1}, level_cap=L + P, level_slots=L, **KW)
    env.start()
    d = Driver(m)
    for _ in range(at):
        env.actions.copy_(d.move(env))
        env.step()
        assert not env.done.any(), 'episode ended before the save point; pick a smaller --at'
    slot = L + 3
    env.place_save(0, torch.tensor([0], dtype=torch.int32), torch.tensor([slot], dtype=torch.int32))
    A = env.state_dict()
    level = int(env.level_now[0])
    assert env.saved_levels()['smb'] == {} or all(k < L for k in env.saved_levels()['smb'])
    one_hot_plan(env, slot)
    for t in range(6000):
        env.actions.zero_()
        env.step()
        if env.done.any():
            break
    assert env.done.all(), 'idle episode never ended'
    ref = RlEnv({'smb': 1}, level_cap=L + P, level_slots=L, **KW)
    ref.load_state_dict(A)
    ref.actions.zero_()
    ref.step()
    for k in ('obs', 'emph', 'masks', 'timeline'):
        assert torch.equal(getattr(env, k), getattr(ref, k)), f'first observation: {k} differs'
    d = Driver(m)
    for t in range(K):
        mv = d.move(env)
        env.actions.copy_(mv); ref.actions.copy_(mv)
        env.step(); ref.step()
        same(env, ref, f'place restart vs continued, decision {t}')
        if env.done.any():
            assert int(env.start_level[0]) == slot, int(env.start_level[0])
            assert int(env.info[0, 4]) >= level, (env.info[0], level)
            break
    print(f'ok: a place saved at decision {at} (level {level}) replays exactly like the game continued ({t + 1} decisions)')


def grow_old_state(m):
    old = RlEnv({'smb': 1}, level_cap=L, **KW)
    old.start()
    d = Driver(m)
    for t in range(5000):
        old.actions.copy_(d.move(old))
        old.step()
        if old.done.any():
            d = Driver(m)
        if old.saved_levels()['smb']:
            break
    saved = old.saved_levels()['smb']
    assert saved, 'policy never reached a new level'
    sd = old.state_dict()
    del sd['level_cap']          # as written before places existed
    new = RlEnv({'smb': 1}, level_cap=L + P, level_slots=L, **KW)
    new.load_state_dict(sd)
    assert new.saved_levels()['smb'] == saved, (new.saved_levels(), saved)
    w = np.zeros(L)
    w[list(saved)] = 1.0 / len(saved)
    old.set_level_plan(0, 1.0, w)
    new.set_level_plan(0, 1.0, np.concatenate([w, np.zeros(P)]))
    for t in range(6000):
        old.actions.zero_(); new.actions.zero_()
        old.step(); new.step()
        same(old, new, f'idle {t}')
        if old.done.any():
            break
    d1, d2 = Driver(m), Driver(m)
    for t in range(300):
        old.actions.copy_(d1.move(old)); new.actions.copy_(d2.move(new))
        old.step(); new.step()
        same(old, new, f'level start after growing, decision {t}')
    print(f'ok: an old state (64 slots) loads with {P} empty place slots; level starts play identically')


def bookkeeping(m, n=8, T=16, rounds=6):
    env = RlEnv({'smb': n}, level_cap=L + P + T * 2, level_slots=L, device='cpu', seed=3, sticky=True, idle_max=30,
                dropout=True)
    env.start()
    pl = Places(1, n, L, P, 2, T, 1, 4, 'cpu', seed=1)
    z, pr = torch.zeros(n, dtype=torch.long), torch.full((n,), 0.30)
    k1 = pl.keys(env.obs[:, 3], env.emph[:, 3], z, pr)
    k2 = pl.keys(env.obs[:, 3].clone(), env.emph[:, 3].clone(), z, pr + 0.01)
    assert torch.equal(k1, k2), 'nearby progress (same 1/16 of a level) keeps the key'
    k3 = pl.keys(env.obs[:, 3], env.emph[:, 3], torch.ones(n, dtype=torch.long), pr)
    assert not torch.equal(k1, k3), 'level must change the key'
    k4 = pl.keys(env.obs[:, 3], env.emph[:, 3], z, pr + 0.25)
    assert (k1 != k4).all(), 'progress must change the key (maze loop: same picture, other position)'
    k5 = pl.keys(env.obs[:, 3], env.emph[:, 3], z, torch.full((n,), float('nan')))
    assert (k1 != k5).all() and torch.equal(k5, pl.keys(env.obs[:, 3], env.emph[:, 3], z, torch.full((n,), float('nan'))))
    d = Driver(m, n)
    gen = torch.Generator().manual_seed(1)
    for r in range(rounds):
        for t in range(T):
            env.actions.copy_(d.move(env, greedy=False, gen=gen))
            env.step()
            done = env.done.bool()
            d.h[done] = 0
            pl.step(env, t, 0, 0, 0, slice(0, n), env.obs[:, 3], env.emph[:, 3], env.level_now, env.progress_now, done)
        pl.end_rollout(env)
        held = pl.report(0)
        assert all(v <= 4 for v in held.values()), held
        print(f'round {r}: kept {pl.last["kept"]} known {pl.last["known"]} held per level {held}')
    assert pl.last['known'] > 0 and sum(pl.report(0).values()) > 0
    rate = np.full(L, 0.5)
    lw = np.zeros(L)
    lv = sorted(pl.report(0))
    lw[lv] = 1.0 / len(lv)
    w = pl.weights(0, lw, rate)
    assert abs(w.sum() - 1) < 1e-9 and (w[L + P:] == 0).all() and (w[L:L + P] > 0).sum() == sum(pl.report(0).values())
    rate[:] = 1.0
    w1 = pl.weights(0, lw, rate)
    assert (w1[L:] == 0).all(), 'a level cleared every time must not use its places'
    print(f'ok: place keys, staging, keeping and weights ({sum(pl.report(0).values())} places held)')


def gpu_check(m, n=16, steps=3000):
    torch.set_num_threads(16)
    kw = dict(seed=11, sticky=True, idle_max=30, dropout=True, level_cap=L + P, level_slots=L)
    g, c = RlEnv({'smb': n}, device='cuda', **kw), RlEnv({'smb': n}, device='cpu', **kw)
    for e in (g, c):
        e.start()
    h, prev = torch.zeros(n, m.core), [torch.zeros(n, dtype=torch.long) for _ in range(5)]
    mood = torch.zeros(n, dtype=torch.long) if m.moods else None
    gen = torch.Generator().manual_seed(3)
    filled = valid = 0
    for t in range(steps):
        with torch.no_grad():
            z = m.encode(c.obs, c.emph, c.timeline, prev, mood)
            h = m.gru(z.float(), h)
            parts, _ = m.act(h, c.masks, generator=gen)
        mv = pack(parts)
        c.actions.copy_(mv); g.actions.copy_(mv.cuda())
        c.step(); g.step()
        for k in ('obs', 'emph', 'reward', 'parts', 'done', 'events', 'masks', 'timeline', 'pads'):
            if not torch.equal(getattr(g, k).cpu(), getattr(c, k)):
                raise AssertionError(f'step {t}: {k} differs')
        if not torch.allclose(g.progress_now.cpu(), c.progress_now, rtol=0, atol=0, equal_nan=True):
            raise AssertionError(f'step {t}: progress_now differs')
        valid += int((~c.progress_now.isnan()).sum())
        d = c.done.bool()
        if d.any():
            assert torch.equal(g.info.cpu()[d], c.info[d]) and torch.equal(g.start_level.cpu()[d], c.start_level[d])
            h[d] = 0
        prev = [p * (~d).long() for p in parts]
        if t % 40 == 20 and filled < P:      # save two envs into place slots, then start from places too
            ei = torch.tensor([t % n, (t // 3) % n], dtype=torch.int32)
            si = torch.tensor([L + filled, L + filled + 1], dtype=torch.int32)
            c.place_save(0, ei, si); g.place_save(0, ei.cuda(), si.cuda())
            filled += 2
            w = np.zeros(L + P)
            w[L:L + filled] = 1.0
            for e in (g, c):
                e.set_level_plan(0, 0.8, w / w.sum())
        if t % 500 == 0:
            print(f'step {t}: places {filled}', flush=True)
    a, b = g.state_dict()['env'], c.state_dict()['env']
    assert (a == b).all(), 'saved env states differ'
    assert valid > steps * n // 2, valid
    print(f'ok: GPU == CPU for {steps} decisions x {n} envs with place saves and place starts ({filled} places), '
          f'progress valid {valid / (steps * n):.0%}')


def test_equal_rarity_places_refresh_without_duplicate_destinations():
    class Env:
        def place_copy(self, gi, src, dst):
            self.src, self.dst = src, dst

    pl = Places(1, 1, 64, 2, 2, 1, 1, 2, 'cpu', seed=1)
    pl.known = torch.tensor([10, 20, 30, 40])
    pl.count = torch.ones(4, dtype=torch.float64)
    pl.slot_key[0] = [10, 20]
    pl.slot_level[0] = 27
    pl.st_key[0] = torch.tensor([30, 40])
    pl.st_level[0] = 27
    pl.st_ok[0] = True
    env = Env()
    pl.end_rollout(env)
    assert set(pl.slot_key[0]) == {30, 40}
    assert len(env.dst) == len(set(env.dst)) == 2
    assert env.src == [pl.stage0, pl.stage0 + 1]
    assert pl.report(0) == {27: 2}


def test_more_frequent_places_do_not_displace_rare_places():
    class Env:
        def place_copy(self, gi, src, dst):
            assert src == dst == []

    pl = Places(1, 1, 64, 2, 2, 1, 1, 2, 'cpu', seed=1)
    pl.known = torch.tensor([10, 20, 30, 40])
    pl.count = torch.tensor([1., 1., 2., 2.], dtype=torch.float64)
    pl.slot_key[0] = [10, 20]
    pl.slot_level[0] = 27
    pl.st_key[0] = torch.tensor([30, 40])
    pl.st_level[0] = 27
    pl.st_ok[0] = True
    pl.end_rollout(Env())
    assert pl.slot_key[0].tolist() == [10, 20]



def test_doors_come_from_explorers_new_jumps_and_close_on_learners_own_jumps():
    """An explorer's try followed by a jump on in the level that no learner has made -> a door place (the try's start,
    its move); walking, a death's restart, a jump back and a learner's jump while replaying a move do not count; the
    learners' own jump closes it."""
    from places import DOOR_SHARE, DOOR_WARMUP

    class Env:
        def __init__(self):
            self.saves, self.copies = [], []

        def place_save(self, gi, env_idx, slot_idx):
            self.saves.append(slot_idx.tolist())

        def place_copy(self, gi, src, dst):
            self.copies.append((list(src), list(dst)))

    def key(x, area=101, lvl=31, y=176):
        return (((lvl * 128 + area) << 12 | x // 8) << 6) | y // 8

    pl = Places(1, 3, 64, 8, 1, 4, 1, 4, 'cpu', seed=1, readmit=True, explorers=np.array([False, True, False]),
                game_rows=[(0, 3)])
    assert pl.tslot.tolist() == [-1, pl.stage0 + 4, -1] and pl.cap == pl.stage0 + 4 + 1
    pl.rollouts = DOOR_WARMUP - 1      # one more rollout and the learners' jumps have been collected long enough
    env, sl, lv = Env(), slice(0, 3), torch.full((3,), 31)
    no, learners = torch.zeros(3, dtype=torch.bool), torch.tensor([True, False, True])

    def step(t, xs, death=no, own=learners):
        k = torch.tensor([key(x) for x in xs])
        pl.step(env, t, 0, 0, 0, sl, None, None, lv, torch.tensor(xs) / 3000.0, no, k, learners, death, own)

    step(0, [2400, 2400, 2400])
    pl.save_tries(env, 0, sl, 0, torch.tensor([False, True, False]), torch.tensor([key(2400)] * 3), lv,
                  torch.tensor([[5, 5, 5], [0, 0, 0], [0, 0, 0]]), torch.full((3,), 8))
    assert env.saves[-1] == [-1, pl.stage0 + 4, -1]
    step(1, [2600, 3128, 2400], death=torch.tensor([False, False, True]))   # walk / the explorer's jump / a death
    step(2, [2700, 3200, 40])                                               # walk / walk / the death's restart
    step(3, [2800, 3300, 60])
    pl.end_rollout(env)
    assert (pl.slot_door[0] >= 0).sum() == 0             # within the warm-up no door is kept
    pl.rollouts = DOOR_WARMUP
    pl.save_tries(env, 0, sl, 0, torch.tensor([False, True, False]), torch.tensor([key(2400)] * 3), lv,
                  torch.tensor([[7, 7, 7], [0, 0, 0], [0, 0, 0]]), torch.full((3,), 8))
    step(0, [2900, 3300, 80])
    step(1, [3000, 1300, 90])                                               # the explorer's jump back
    pl.end_rollout(env)
    assert (pl.slot_door[0] >= 0).sum() == 0
    pl.t_key.fill_(-1)                                                      # that try is over
    step(0, [2400, 2400, 2400])
    pl.save_tries(env, 0, sl, 0, torch.tensor([False, True, False]), torch.tensor([key(2400)] * 3), lv,
                  torch.tensor([[5, 5, 5], [0, 0, 0], [0, 0, 0]]), torch.full((3,), 8))
    step(1, [2600, 3128, 2400], death=torch.tensor([False, False, True]))
    step(2, [2700, 3200, 40])
    pl.end_rollout(env)
    d = np.flatnonzero(pl.slot_door[0] >= 0)
    assert len(d) == 1 and pl.t_key[1] == -1              # the try opened its door: it is used up and pl.slot_level[0, d[0]] == 31 and pl.slot_key[0, d[0]] == key(2400)
    assert env.copies[-1] == ([pl.stage0 + 4], [64 + int(d[0])])
    assert not bool((pl.ltrans == int(pl.slot_door[0, d[0]])).any()) and pl.last['door_added'] == 1
    mv, ln = pl.door_of(torch.tensor([0, 0]), torch.tensor([64 + int(d[0]), 3]))
    assert mv[0].tolist() == [5, 0, 0] and ln.tolist() == [8, 0]
    w = pl.weights(0, np.eye(64)[31], np.zeros(64))
    assert abs(w[64 + d[0]] - DOOR_SHARE) < 1e-9 and abs(w.sum() - 1) < 1e-9
    # a learner makes the same jump while replaying the door's move: still a door; with its own moves: an ordinary place
    step(0, [2400, 2400, 2400])
    step(1, [3128, 2400, 2400], own=torch.tensor([False, False, True]))
    pl.end_rollout(env)
    assert (pl.slot_door[0] >= 0).sum() == 1
    step(0, [2400, 2400, 2400])
    step(1, [3128, 2400, 2400])
    pl.end_rollout(env)
    assert (pl.slot_door[0] >= 0).sum() == 0 and pl.last['door_opened'] == 1 and pl.slot_key[0, d[0]] == key(2400)



def test_a_try_ends_with_its_game():
    """A jump after the explorer's game ended (e.g. a new game started at a door place replaying its move) is not the
    old try's doing."""
    from places import DOOR_WARMUP

    class Env:
        def place_save(self, gi, env_idx, slot_idx):
            pass

        def place_copy(self, gi, src, dst):
            pass

    def key(x):
        return (((31 * 128 + 101) << 12 | x // 8) << 6) | 22

    pl = Places(1, 2, 64, 8, 1, 4, 1, 4, 'cpu', seed=1, readmit=True, explorers=np.array([False, True]),
                game_rows=[(0, 2)])
    pl.rollouts = DOOR_WARMUP
    env, sl, lv, learners = Env(), slice(0, 2), torch.full((2,), 31), torch.tensor([True, False])

    def step(t, xs, fresh):
        pl.step(env, t, 0, 0, 0, sl, None, None, lv, torch.tensor(xs) / 3000.0, torch.tensor(fresh),
                torch.tensor([key(x) for x in xs]), learners, torch.zeros(2, dtype=torch.bool), learners)

    step(0, [100, 4800], [False, False])
    pl.save_tries(env, 0, sl, 0, torch.tensor([False, True]), torch.tensor([key(4800)] * 2), lv,
                  torch.tensor([[2, 2], [1, 1], [0, 0]]), torch.full((2,), 8))
    step(1, [110, 2400], [False, True])          # the explorer's game ends; the next starts near the pipe
    step(2, [120, 3128], [False, False])         # ... and goes through it
    pl.end_rollout(env)
    assert (pl.slot_door[0] >= 0).sum() == 0


if __name__ == '__main__':
    torch.set_num_threads(8)
    model = policy(sys.argv[1])
    if '--gpu' in sys.argv:
        gpu_check(model)
    else:
        exact_replay(model)
        grow_old_state(model)
        bookkeeping(model)


def test_full_archive_shares_slots_with_a_level_that_has_none():
    """Every slot held by level 15 (trained first); a new place of level 27 takes level 15's most seen place."""
    class Env:
        def place_copy(self, gi, src, dst):
            self.last = (list(src), list(dst))

    pl = Places(1, 2, 64, 8, 1, 4, 1, 8, 'cpu', seed=1, share=True)
    pl.slot_level[0, :] = 15
    pl.slot_key[0, :] = np.arange(8) + 1000
    pl.known = torch.tensor(np.arange(8) + 1000, dtype=torch.int64)
    pl.count = torch.tensor([5., 9., 1., 1., 1., 1., 1., 1.], dtype=torch.float64)
    pl.st_key[0, 0], pl.st_level[0, 0], pl.st_ok[0, 0] = 77, 27, True
    env = Env()
    pl.end_rollout(env)
    assert pl.slot_level[0].tolist().count(27) == 1 and pl.slot_key[0, 1] == 77      # the most seen (count 9) gave way
    assert env.last[1] == [pl.L + 1]


def test_level_start_keeps_half_the_failure_share():
    """A failing level with many fresh places is still practised from its start: start = cleared + half the failure."""
    from places import START_SHARE
    pl = Places(1, 2, 64, 8, 1, 4, 1, 8, 'cpu', seed=1, share=True)
    pl.slot_level[0, :] = 27
    pl.slot_key[0, :] = np.arange(8) + 1000
    w = np.zeros(64)
    w[27] = 1.0
    rate = np.full(64, 0.2)
    full = pl.weights(0, w, rate)
    assert abs(full[27] - (0.2 + 0.8 * START_SHARE)) < 1e-9
    assert abs(full[pl.L:pl.L + 8].sum() - 0.8 * (1 - START_SHARE)) < 1e-9 and abs(full.sum() - 1) < 1e-9
