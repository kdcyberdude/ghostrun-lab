"""Stuck -> search -> hand-off (rl/stuck.py, rl/stucksearch.py), CPU reference build (fast logic, no training):

1. cell keys from block keys: level, screen, 32-pixel x and y; a cell's 16 blocks in the trainer's mask order;
2. the new spot on a way (first rare block the way then stays rare after) and the hand-back distances / weights;
3. trainer side on fake consoles (path design, 2026-11-21 ~19:00): a game from a hand slot is a practice try; a
   counted block of the new spot's cell is reached (another block of the cell is not); a try ends at a death or at its
   slot's length and is counted (learners only) and killed; a slot game that reward10 ended is killed with its end
   reason; level-start learner games log their buttons and count toward the new spot (learned at DONE_RATE of
   DONE_GAMES), a spot is given up after GIVEUP_ITS; start weights follow the slots' weights; a search waits for
   MIN_START_GAMES level-start games; capture keeps a seed's buttons; a hand-off result is applied; checkpoints;
4. coverage counts skip practice games and bcount0 counts only level-start learner games;
5. a real search from 1-1's start (CPU env) and the hand-back from it; a learner-like game from the level start with
   random idle frames is found again by the idle copies and its way (learner's buttons + the search's) replays.

usage: python rl/tests/test_stuck.py
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import slots as slotlib  # noqa: E402
import stuck as stk  # noqa: E402
import stucksearch as ss  # noqa: E402

stk.TEACH = 0.0     # the older tests: every practice try is the model's own (test_teacher turns teacher tries on)


def test_cells():
    lvl, area, bx, by = 27, 3 * 32 + 5, 1234, 21
    bk = torch.tensor([(((lvl * 128 + area) << 12 | bx) << 6) | by, -1])
    c = ss.cell_of(bk)
    assert int(c[1]) == -1
    k = int(c[0])
    assert k >> 21 == lvl and (k >> 14) & 127 == area and (k >> 4) & 1023 == bx // 4 and k & 15 == by // 4
    assert k >> 7 == ((lvl * 128 + area) << 7) | (bx * 8 // 256)       # screen = (level, area, x // 256)
    blocks = ss.cell_blocks(k)
    assert (ss.cell_of(torch.from_numpy(blocks)).numpy() == k).all() and len(set(blocks.tolist())) == 16
    b = torch.from_numpy(blocks)
    assert ((((b >> 6) & 3) << 2) | (b & 3)).tolist() == list(range(16)), 'mask bit order'
    print('cells ok')


def test_tag_cells():
    """GSearch's tagged cells (position | (TAG | byte << 8 | value) << TAG_SHIFT): two tags at one position are two
    cells, and masking gives the position back (targets hand back cell_of keys)."""
    pos = ss.cell_of(torch.tensor([(((27 * 128 + 5) << 12 | 1234) << 6) | 21] * 2))
    tag = torch.tensor([ss.TAG | (0x123 << 8) | 1, ss.TAG | (0x123 << 8) | 2])
    c = pos | (tag << ss.TAG_SHIFT)
    assert int(c[0]) != int(c[1]) and ((c & ((1 << ss.TAG_SHIFT) - 1)) == pos).all() and int(c.max()) < (1 << 63)
    print('tag cells ok')


def test_spot_positions():
    F, T = False, True
    rare = np.array([F, T] + [F] * 10 + [T] * 40)             # a lone rare block (a jump), then rare ground for good
    assert ss.spot_of(rare, False) == 13
    assert ss.spot_of(np.zeros(10, bool), True) == 11 and ss.spot_of(np.zeros(10, bool), False) is None
    mostly = np.array([T] * 3 + ([T] * 9 + [F]) * 5)           # >= 80% rare from the start
    assert ss.spot_of(mostly, False) == 1
    d, w = ss.positions(500, False)
    assert d.tolist() == list(range(501)) and len(w) == 501      # every decision back to the way's start
    d, w = ss.positions(5, True)
    assert d.tolist() == [1, 2, 3, 4, 5]                       # a clear: no state at the spot itself
    print('spot / positions ok')


class FakeEnv:
    def __init__(self, n):
        self.info = torch.zeros(n, 8)
        self.kill = torch.zeros(n, dtype=torch.uint8)
        self.events = torch.zeros(n, dtype=torch.int32)
        self.cur_start = torch.full((n,), -1, dtype=torch.int32)
        self.actions = torch.zeros(n, dtype=torch.int16)
        self.saved = []

    def place_save(self, gi, idx, slots):
        self.saved.append((idx.clone(), slots.clone()))


def test_window():
    """Sliding window: WINDOW of the way's states practised, nearest first; a mastered state (>= MASTER_WINS reached,
    >= MASTER_RATE of its tries) gives its slot to the next earlier state; open tries from it stop counting."""
    N, slot0, LV, L, D = 8, 100, 32, 31, 150
    st = stk.StuckSearch('/tmp/unused', N, torch.device('cpu'), slot0, LV, torch.ones(N, dtype=torch.bool), seed=1)
    cell = ((L * 128 + 101) << 14) | (84 << 4) | 9
    dist = np.arange(D)
    rec = {k: np.repeat(dist[:, None], 4, 1).astype(np.uint8) for k in slotlib.FIELDS}   # each state marked by its distance
    res = dict(cell=cell, mask=1, target=cell, screens=1, cells=3, spot_at=D - 1, length=D, learner=0, matched=False,
               rare=5, buttons=np.zeros(D - 1, np.int16), pos=D - 1 - dist, dist=dist, wt=np.ones(D), lim=2 * dist + 64,
               states=rec)
    d = Path(tempfile.mkdtemp())
    ss.save_result(d, dict(stop='used up', decisions=1, seconds=1), [res])
    written = []
    write0, slotlib.write = slotlib.write, lambda env, gi, sl, r: written.append((list(sl), r['mem'][:, 0].tolist()))
    try:
        st.job = dict(level=L, started=0.0)
        st._apply(None, d, it=0)
        t = st.targets[0]
        assert len(t['slots']) == stk.WINDOW and t['dist'].tolist() == list(range(stk.WINDOW)) and t['next'] == stk.WINDOW
        assert written[0][1] == list(range(stk.WINDOW))
        w = st.weights(np.ones(LV), None, np.zeros(LV), np.arange(LV))[slot0:slot0 + stk.WINDOW]
        assert w[-1] > 50 * w[0]                                   # the earliest states of the window most
        st.c_slot[0], st.c_open[0] = 3, True                       # a try playing from slot 3
        st.acc[0, :5] = torch.tensor([30, 20, 50, 25, 10])         # tries
        st.acc[1, :5] = torch.tensor([30, 19, 20, 25, 10])         # reached: slots 0 and 3 mastered (2: 40%, 4: 10 wins)
        st._outcomes(None, it=1)
        assert written[-1] == ([slot0 + 0, slot0 + 3], [100, 101]), written[-1]
        assert t['dist'][[0, 3]].tolist() == [100, 101] and t['next'] == 102 and t['mastered'] == 2
        assert st.s_lim[[0, 3]].tolist() == [264, 266] and t['tries'][[0, 3]].tolist() == [0, 0]
        assert not bool(st.c_open[0]) and t['quiet'] == 0
        # the way's start reached: no more states, mastered ones stay
        t['next'] = D
        st.acc[0, :1], st.acc[1, :1] = 40, 40
        n = len(written)
        st._outcomes(None, it=2)
        assert len(written) == n and t['quiet'] == 1
    finally:
        slotlib.write = write0
    print('window ok')


def test_teacher():
    """Teacher tries: from a window state the way's buttons from there on, one per decision; the try ends with the
    last one, is counted as a teacher try (not as a try of the window), and the console is killed."""
    N, slot0, LV, L, r = 4, 100, 32, 31, 40
    st = stk.StuckSearch('/tmp/unused', N, torch.device('cpu'), slot0, LV, torch.tensor([True, True, True, False]), seed=1)
    cell = ((L * 128 + 101) << 14) | (84 << 4) | 9
    in_spot = int(ss.cell_blocks(cell)[0])
    rec = {k: np.zeros((3, 4), np.uint8) for k in slotlib.FIELDS}
    res = dict(cell=cell, mask=1, target=cell, screens=1, cells=3, spot_at=r, length=50, learner=0, matched=False,
               rare=5, buttons=np.arange(100, 100 + r).astype(np.int16), pos=np.array([40, 39, 37]),
               dist=np.array([0, 1, 3]), wt=np.ones(3), lim=np.array([64, 66, 70]), states=rec)
    d = Path(tempfile.mkdtemp())
    ss.save_result(d, dict(stop='used up', decisions=1, seconds=1), [res])
    write0, slotlib.write = slotlib.write, lambda env, gi, sl, rr: None
    teach0, stk.TEACH = stk.TEACH, 1.0
    try:
        st.job = dict(level=L, started=0.0)
        st._apply(None, d, it=0)
        assert int(st.t_n[0]) == r and st.s_dist[:3].tolist() == [0, 1, 3] and st.t_btn[0, :3].tolist() == [100, 101, 102]
        env = FakeEnv(N)
        # consoles 0 and 3 (an explorer) start at slot 2 (3 decisions before the spot), 1 at slot 1, 2 a normal game
        env.cur_start[:] = torch.tensor([slot0 + 2, slot0 + 1, 5, slot0 + 2], dtype=torch.int32)
        st.on_restart(slice(None), torch.ones(N, dtype=torch.bool), env.cur_start)
        assert st.c_teach.tolist() == [True, True, False, False]        # learners' practice tries only
        nd, lv = torch.zeros(N, dtype=torch.bool), torch.full((N,), L, dtype=torch.int32)
        bk = torch.full((N,), -1, dtype=torch.int64)
        pressed = []
        for step in range(3):
            on, btn = st.teacher(slice(None))
            pressed.append([int(b) if o else None for o, b in zip(on.tolist(), btn.tolist())])
            if step == 2:
                bk[0] = in_spot                                          # the way's last button reaches the spot
            st.after_step(env, slice(None), bk, lv, nd, nd)
            if step == 0:
                assert env.kill.tolist() == [0, 1, 0, 0]                 # console 1's single button is pressed
                env.kill.zero_()
        assert pressed == [[137, 139, None, None], [138, None, None, None], [139, None, None, None]], pressed
        assert env.kill.tolist() == [1, 1, 0, 0]                         # (console 1 was not restarted here: still killed)
        assert st.acc_t[:, 0].tolist() == [2, 1] and int(st.acc.sum()) == 0   # teacher tries; none of the window's
    finally:
        slotlib.write, stk.TEACH = write0, teach0
    print('teacher ok')


def test_exact_replay():
    """A try that starts EXACTLY on a saved state (set_restore_exact) and presses the way's remaining buttons ends
    in the way's own state; with the 4 neutral restart frames (not exact) it does not."""
    rng = np.random.default_rng(3)
    RUN, JUMP = 3 + 9 * 3, 3 + 9 * 4
    way = np.where(rng.random(60) < 0.3, JUMP, RUN)

    def end_ram(exact, cut):
        env = ss.make_env(1, ss.L0 + 2, 5, exact=exact)
        slotlib.save_boot_entrance(env, 0, 0)
        w = np.zeros(env.level_cap)
        w[0] = 1.0
        env.set_level_plan(0, 1.0, w)
        env.start()
        _, ends, _, _ = ss.replay(env, 0, way, save={cut: ss.L0})
        assert ends[0] == len(way), 'the way died'
        whole = env.ram.cpu().numpy().copy()
        _, ends, _, _ = ss.replay(env, ss.L0, way[cut:])
        return whole, env.ram.cpu().numpy().copy()

    whole, part = end_ram(True, 25)
    assert (whole == part).all(), 'exact restart: the rest of the way must end in the same state'
    whole, part = end_ram(False, 25)
    assert (whole != part).any(), 'with restart frames the try is off the way (if equal, exact restores are not needed)'
    print('exact replay ok')


def test_trainer_side():
    N, slot0, LV, L = 8, 100, 32, 27
    learner = torch.tensor([True] * 7 + [False])
    st = stk.StuckSearch('/tmp/unused', N, torch.device('cpu'), slot0, LV, learner, seed=1, idle=4)
    cell = ((L * 128 + 101) << 14) | (84 << 4) | 9
    blocks = ss.cell_blocks(cell)
    in_spot, other = int(blocks[0]), int(blocks[1])             # mask bit 0 counts, bit 1 does not
    # a hand-off result (as rl/stucksearch.py writes it) applied to the env
    rec = {k: np.zeros((3, 4), np.uint8) for k in slotlib.FIELDS}
    res = dict(cell=cell, mask=1, target=cell, screens=1, cells=3, spot_at=40, length=50, learner=30, matched=True,
               rare=5, buttons=np.zeros(40, np.int16), pos=np.array([40, 39, 0]), dist=np.array([0, 1, 40]),
               wt=np.array([0.5, 0.3, 0.2]), lim=np.array([64, 66, 144]), states=rec)
    d = Path(tempfile.mkdtemp())
    ss.save_result(d, dict(stop='used up', decisions=1, seconds=1), [res])
    written = []
    write0, slotlib.write = slotlib.write, lambda env, gi, sl, r: written.append(list(sl))
    try:
        st.job = dict(level=L, started=0.0)
        st._apply(None, d, it=0)
    finally:
        slotlib.write = write0
    t = st.targets[0]
    assert written == [[slot0, slot0 + 1, slot0 + 2]] and t['cell'] == cell and int(st.lv_row[L]) == 0
    assert st.s_lim[:3].tolist() == [64, 66, 144] and int(st.t_mask[0]) == 1 and st.job is None
    env = FakeEnv(N)
    # consoles 0-4 and 7 (an explorer) start practice tries at hand slot 0; 5 a level-start game of L; 6 power-on
    env.cur_start[:] = torch.tensor([slot0] * 5 + [L, -1, slot0], dtype=torch.int32)
    done = torch.ones(N, dtype=torch.bool)
    st.on_restart(slice(None), done, env.cur_start)
    assert st.c_slot.tolist() == [0] * 5 + [-1, -1, 0] and st.c_open.tolist() == [True] * 5 + [False, False, True]
    assert st.practising(slice(None)).tolist() == [True] * 5 + [False, False, True]
    assert st.from_start(slice(None)).tolist() == [False] * 5 + [True, False, False] and int(st.g_row[5]) == 0
    bk = torch.full((N,), -1, dtype=torch.int64)
    bk[0], bk[2], bk[7], bk[5] = in_spot, other, in_spot, in_spot
    death = torch.zeros(N, dtype=torch.bool)
    death[1] = True
    env.events[3] = 16                                         # game over in a practice game
    env.events[6] = 16                                         # ... and in the power-on game (its kill is harmless)
    env.actions[:] = torch.arange(N, dtype=torch.int16) + 10
    lv = torch.full((N,), L, dtype=torch.int32)
    nd = torch.zeros(N, dtype=torch.bool)
    st.after_step(env, slice(None), bk, lv, nd, death)
    assert st.c_hit.tolist() == [True, False, False, False, False, False, False, True]
    assert bool(st.g_hit[5]) and int(st.h_len[5]) == 1 and int(st.h_buf[5, 0]) == 15 and int(st.h_len[0]) == 0
    assert st.acc[:, 0].tolist() == [1, 0], st.acc[:, 0]       # console 1 died: counted, not reached
    assert env.kill.tolist() == [0, 1, 0, 1, 0, 0, 1, 0]        # its try and the ended games
    assert int(st.c_why[3]) == 1 and int(st.c_why[1]) == 0
    env.events.zero_()
    # the killed games end at the next decision: end reason restored, kills cleared
    done = torch.zeros(N, dtype=torch.bool)
    done[1] = done[3] = done[6] = True
    env.info[:, 2] = 8                                         # the env reports a limit for a killed attempt
    env.info[6, 2] = 1                                         # ... and its own end for the power-on game
    st.fix_end(env, slice(None), done)
    assert int(env.info[3, 2]) == 1 and int(env.info[1, 2]) == 8 and int(env.info[6, 2]) == 1
    assert env.kill.tolist() == [0] * N
    env.cur_start[1] = env.cur_start[3] = 7                    # both start normal slot games
    st.on_restart(slice(None), done, env.cur_start)
    assert int(st.c_slot[1]) == -1 and st.acc[:, 0].tolist() == [2, 0], st.acc[:, 0]
    # capture: the level-start game of L is saved with its buttons; practice and other games are not
    st.capture = dict(level=L, it=0)
    bcount0 = torch.zeros(1 << ss.BLOCK_COUNT_BITS, dtype=torch.int32)
    st.after_step(env, slice(None), bk, lv, nd, nd)
    st.capture_step(env, 0, 0, 0, slice(None), bk, lv, nd, bcount0)
    idx, slots = env.saved[-1]
    k = int((slots >= 0).sum())
    assert k == 1 and int(idx[slots >= 0][0]) == 5
    j = int(torch.nonzero(st.cap_ok)[0])
    assert int(st.cap_hlen[j]) == 2 and st.cap_hist[j, :2].tolist() == [15, 15]
    read0, slotlib.read = slotlib.read, lambda env, gi, sl: {kk: np.zeros((len(sl), 4), np.uint8) for kk in slotlib.FIELDS}
    try:
        seeds, ncap, _, hists = st._seeds(L, env)
    finally:
        slotlib.read = read0
    assert seeds == [L, st.seed0 + j] and ncap == 1 and hists[0].size == 0 and hists[1].tolist() == [15, 15]
    # the tries from slot 0 run to its length (64): counted (learners), console 0 reached, then killed
    for _ in range(64 - 1):
        st.after_step(env, slice(None), torch.full((N,), -1, dtype=torch.int64), lv, nd, nd)
    assert st.acc[:, 0].tolist() == [5, 1], st.acc[:, 0]        # 0, 2, 4 now (+ 2 before); the explorer is no try
    assert env.kill.tolist() == [1, 0, 1, 0, 1, 0, 0, 1], env.kill.tolist()
    # weights: the new spot's slots share `share` equally; the level's normal starts are boosted (unsolved)
    full = np.zeros(slot0)
    full[L], full[5] = 0.4, 0.6
    rate = np.zeros(LV)
    rate[5] = 0.9
    w = st.weights(full, None, rate, np.arange(slot0))
    B, g = stk.LEVEL_BOOST, st.share
    assert len(w) == slot0 + stk.HAND_SLOTS + st.stage_n and abs(w.sum() - 1) < 1e-9
    assert np.allclose(w[slot0:slot0 + 3], g * np.array([1, 2, 41]) / 44)   # share ~ distance + 1 (dist 0, 1, 40)
    assert abs(w[L] - (1 - g) * 0.4 * B / (0.4 * B + 0.6)) < 1e-9
    # places off for a stuck level: its places' start weight goes to its level start
    st.no_places[L] = True
    sl_lv = np.arange(slot0)
    sl_lv[40] = L                                            # a saved place of level L
    full2 = np.zeros(slot0)
    full2[L], full2[40], full2[5] = 0.2, 0.2, 0.6
    w2 = st.weights(full2, None, rate, sl_lv)
    assert w2[40] == 0 and abs(w2[L] - (1 - g) * 0.4 * B / (0.4 * B + 0.6)) < 1e-9 and abs(w2.sum() - 1) < 1e-9
    st.no_places[L] = False
    # the level-start game ends: counted for the spot (reached) and for the level
    done = torch.zeros(N, dtype=torch.bool)
    done[5] = True
    env.cur_start[5] = -1
    st.on_restart(slice(None), done, env.cur_start)
    assert st.acc0[:, 0].tolist() == [1, 1] and int(st.n0[L]) == 1 and not bool(st.from_start(slice(None))[5])
    # learned: DONE_RATE of DONE_GAMES level-start games reach it -> slots freed, the level free for a new search
    st._outcomes(None, it=1)
    assert t['sg'] == 1 and t['sr'] == 1 and st.targets[0] is t
    st.acc0[0, 0], st.acc0[1, 0] = stk.DONE_GAMES, stk.DONE_GAMES // 4
    st._outcomes(None, it=2)
    assert st.targets[0] is None and (st.slot_row[:3] == -1).all() and int(st.s_row[:3].max()) == -1
    assert int(st.lv_row[L]) == -1 and int(st.t_cell[0]) == -1 and int(st.c_slot[0]) == -1
    ev = [e for e in st.report(rate)['events'] if e['kind'] == 'target']
    assert ev and ev[0]['end'] == 'learned' and st.fails[L] == 0
    # given up after GIVEUP_ITS iterations without that and without the window moving back
    st.targets[1] = dict(t, its=stk.GIVEUP_ITS - 1, quiet=stk.GIVEUP_ITS - 1, sg=0, sr=0)
    st.lv_row[L] = 1
    st._outcomes(None, it=3)
    assert st.targets[1] is None and st.fails[L] == 1 and st.report(rate)['events'][0]['end'] == 'given up'
    assert st.level_of([slot0 + 1, L], [L, L]).tolist() == [-1, L]
    # stuck counts: learners' normal decisions only; a search waits for MIN_START_GAMES level-start games
    st.fails[:] = 0
    st.capture = None
    b = SimpleNamespace(b_lnow=torch.full((4, N), L, dtype=torch.int32), b_prac=torch.zeros(4, N, dtype=torch.bool))
    b.b_prac[:, 0] = True
    bl = np.zeros(32, np.int64)
    st.end_rollout(None, b, bl, rate, set(), None, 5)
    assert st.dry[L] == 4 * 6                       # 7 learners, one practising
    bl[L] = 3
    st.end_rollout(None, b, bl, rate, set(), None, 6)
    assert st.dry[L] == 4 * 6                       # a new block: restarted, then this rollout's decisions
    st.dry[L] = 2 * stk.STUCK_DECISIONS
    st.end_rollout(None, b, bl, rate, {L}, None, 7)
    assert st.capture is None and st.no_places[L]   # stuck: places off at once; too few level-start games yet
    st.n0[L] = stk.MIN_START_GAMES
    st.end_rollout(None, b, bl, rate, {L}, None, 8)
    assert st.capture == dict(level=L, it=8)
    # a checkpoint round trip keeps the tables; one of an earlier design drops its targets
    st.targets[2] = dict(t)
    st2 = stk.StuckSearch('/tmp/unused', N, torch.device('cpu'), slot0, LV, learner, seed=1)
    st2.load_state_dict(st.state_dict())
    assert st2.share == st.share and st2.targets[2]['cell'] == cell and int(st2.n0[L]) == stk.MIN_START_GAMES
    assert st2.no_places[L] and (st2.h_buf == st.h_buf).all() and (st2.g_lvl == st.g_lvl).all()
    old = dict(st.state_dict(), version=2)
    st3 = stk.StuckSearch('/tmp/unused', N, torch.device('cpu'), slot0, LV, learner, seed=1)
    st3.load_state_dict(old)
    assert all(x is None for x in st3.targets) and st3.capture is None
    print('trainer side ok')


def test_coverage_counts():
    from coverage import Coverage, smb_block_keys
    c = Coverage(3, torch.device('cpu'), rarity=1.0, blocks=1.0)
    ram = torch.zeros(3, 2048, dtype=torch.uint8)
    ram[:, 0x770], ram[:, 0x772], ram[:, 0x0E] = 1, 3, 8      # in normal play
    ram[:, 0x6D], ram[:, 0x86], ram[:, 0xCE], ram[:, 0xB5] = 1, 40, 100, 1
    c.begin(ram)
    ram2 = ram.clone()
    ram2[:, 0x86] = 140
    r = c.step(ram2, torch.zeros(3, dtype=torch.bool), torch.tensor([False, True, False]),
               torch.tensor([True, False, False]))
    bk = smb_block_keys(ram2)
    assert int(bk[0]) >= 0
    assert float(r[0]) == float(r[1]) and float(c.last_bpay[0]) == float(c.last_bpay[1]) > 0
    s = ss.block_slot(bk)[0]
    assert int(c.bcount[s]) == 2, 'the practice game does not count'
    assert int(c.bcount0[s]) == 1, 'bcount0: only the level-start learner game'
    print('coverage counts ok')


def test_search():
    env = ss.make_env(4, ss.L0 + 4, 3)
    slotlib.save_boot_entrance(env, 0, 0)            # 1-1's start -> level slot 0
    start = slotlib.read(env, 0, [0])
    bcount = np.zeros(1 << ss.BLOCK_COUNT_BITS, np.int32)       # nothing known: every block is rare
    p = dict(ss.DEFAULTS, envs=32, archive=600, min=6000, quiet=6000, max=12000, seconds=300, seed=4)
    senv = ss.make_env(p['envs'], ss.L0 + 1 + p['archive'], p['seed'])
    slotlib.write(senv, 0, [ss.L0], start)
    srch = ss.Search(senv, 1, bcount, 0, 0, p, log=lambda s: None)
    stop = srch.run()
    assert stop in ('decision cap', 'used up', 'archive full'), stop
    assert len(srch.ckey) > 10 and srch.rare_found == len(srch.ckey)
    tg = [x for x in srch.targets() if x[0] != ss.CLEAR]
    assert tg, 'no targets'
    one = max(tg, key=lambda x: len(x[1]))
    assert len(one[1]) > 40, len(one[1])
    # from the level start seed, everything rare: the new spot is the first button, states at it and before it
    out = ss.hand_back([one], start, [np.zeros(0, np.int64)], 9, bcount, 0, 4, log=print)
    assert len(out) == 1 and out[0]['spot_at'] == 1 and out[0]['pos'].tolist() == [1, 0] and not out[0]['matched']
    assert out[0]['states']['mem'].shape[0] == 2 and out[0]['mask'] == (1 << 16) - 1
    # a learner-like game from the level start (random idle frames, idle 0..4) saved after H buttons = a seed, with
    # its buttons logged; a "search" way from that seed (restart, then buttons): the idle copies find the learner's
    # game again and the whole way replays from the level start
    def started(e, slot):
        w = np.zeros(e.level_cap)
        w[slot] = 1.0
        e.set_level_plan(0, 1.0, w)
        e.start()
        return e

    lenv = ss.make_env(1, ss.L0 + 2, 77, idle=4)
    slotlib.write(lenv, 0, [0], start)
    started(lenv, 0)
    rng = np.random.default_rng(5)
    H, RUN, JUMP = 40, 3 + 9 * 3, 3 + 9 * 4                     # right + B held, right + A + B held
    for _ in range(20):
        hist = rng.integers(0, 81, H)
        keys, ends, _, _ = ss.replay(lenv, 0, hist, save={H: ss.L0})
        if ends[0] == H:
            break
    assert ends[0] == H, 'the learner-like game died'
    seed_rec = slotlib.read(lenv, 0, [ss.L0])
    seeds = {k: np.concatenate([start[k], seed_rec[k]]) for k in slotlib.FIELDS}
    venv = ss.make_env(1, ss.L0 + 2, 3)
    slotlib.write(venv, 0, [ss.L0 + 1], seed_rec)
    started(venv, ss.L0 + 1)
    for _ in range(20):
        tail = np.where(rng.random(60) < 0.15, JUMP, RUN)
        vkeys, vends, _, _ = ss.replay(venv, ss.L0 + 1, tail)
        if vends[0] >= 40:
            break
    n = int(vends[0])
    assert n >= 40, 'the search way died'
    key = int(ss.cell_of(torch.tensor([vkeys[0, n - 1]]))[0])
    seq = np.concatenate([[0], tail[:n]]).astype(np.uint8)
    # the learner's buttons, one no-button decision (the search's restart at the seed), the search's buttons: the
    # idle copy that matches the seed then walks the search's way block for block
    cenv = ss.make_env(ss.IDLE_COPIES, ss.L0 + 2, 9, idle=4)
    slotlib.write(cenv, 0, [0], start)
    started(cenv, 0)
    ck, ce, _, same = ss.replay(cenv, 0, np.concatenate([hist, seq]),
                                match=(H, seed_rec['mem'][0, ss.RAM_OFFSET:ss.RAM_OFFSET + 2048]))
    assert same.any(), 'no idle copy matches the learner game'
    who = int(np.argmax(same))
    assert (ck[who, H + 1:H + 1 + n] == vkeys[0, :n]).all(), 'a restart is not one no-button decision'
    bc = np.zeros_like(bcount)
    first = keys[0][keys[0] >= 0]                               # the learner's ground is known
    bc[ss.block_slot(torch.from_numpy(first)).numpy()] = 100
    out = ss.hand_back([(key, seq, 1, 1, 3)], seeds, [np.zeros(0, np.int64), hist], 9, bc, 20, 4, log=print)
    assert len(out) == 1, 'the way from the level start did not replay'
    t = out[0]
    assert t['matched'] and t['learner'] == H and t['spot_at'] > H and t['states']['mem'].shape[0] == len(t['pos'])
    assert t['pos'][-1] == 0 and (np.diff(t['pos']) < 0).all()         # back to the level start
    print('search ok:', stop, len(srch.ckey), 'cells; learner way: new spot at button', t['spot_at'], 'of',
          t['length'], ',', len(t['pos']), 'states')


if __name__ == '__main__':
    test_cells()
    test_tag_cells()
    test_spot_positions()
    test_window()
    test_teacher()
    test_exact_replay()
    test_trainer_side()
    test_coverage_counts()
    test_search()
    print('all ok')
