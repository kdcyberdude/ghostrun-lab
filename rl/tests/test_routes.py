"""Backward curriculum (rl/routes.py): start weights and pointer moves. No emulator.

usage: python -m pytest rl/tests/test_routes.py
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import routes as R  # noqa: E402
from routes import Routes  # noqa: E402

L = 8


def make():
    return Routes(L, L + 32, {2: 10, 5: 3})   # level 2: slots 8..17, level 5: slots 18..20


def play(r, lv, rates, n=100, rounds=40):
    """Episodes from every position of level lv with the given clear rates (position -> rate), until nothing moves."""
    for _ in range(rounds):
        before = r.p[lv]
        start = np.concatenate([np.full(n, r.slot(lv, q)) for q in rates])
        clear = np.concatenate([np.arange(n) < x * n for x in rates.values()])
        r.add(start, clear)
        if r.p[lv] == before and _ > 12:
            break


def test_layout():
    r = make()
    assert r.slot0[2] == 8 and r.slot0[5] == 18
    assert r.level_of[8] == 2 and r.level_of[17] == 2 and r.level_of[18] == 5 and r.level_of[21] == -1
    assert r.slot(2, 0) == 2 and r.slot(2, 1) == 8 and r.slot(2, 10) == 17
    assert r.p[2] == 11 and r.p[5] == 4 and r.p[0] == 1       # nothing solved yet


def test_weights_start_at_the_end_of_the_route():
    r = make()
    w = r.weights([0, 2, 5])
    assert abs(w.sum() - 1) < 1e-12 and (w[[1, 3, 4, 6, 7]] == 0).all() and (w[21:] == 0).all()
    lv = w[0]                                   # all three levels are equally unknown: equal shares
    assert abs(w[2] + w[8:18].sum() - lv) < 1e-12 and abs(w[5] + w[18:21].sum() - lv) < 1e-12
    assert abs(w[2] - lv * R.ENTRANCE) < 1e-12 and (w[8:15] == 0).all()
    assert np.allclose(w[15:18] / lv, np.array(R.LOOK[::-1]) * (1 - R.ENTRANCE))   # the last three states, last most


def test_shares_around_the_pointer():
    r = make()
    r.p[2] = 6                                  # solved from position 6: frontier 5, 4, 3; hold 6; back 7..10
    w = r.shares(2)
    assert abs(sum(w.values()) - 1) < 1e-12 and set(w) == {0, 3, 4, 5, 6, 7, 8, 9, 10}
    assert abs(w[0] - R.ENTRANCE) < 1e-12 and abs(w[6] - R.HOLD) < 1e-12 and abs(w[9] - R.BACK / 4) < 1e-12
    assert np.allclose([w[5], w[4], w[3]], np.array(R.LOOK) * R.FRONT)
    r.p[2] = 1                                  # only the entrance is left before the solved stretch
    w = r.shares(2)
    assert abs(w[0] - (R.ENTRANCE + R.FRONT)) < 1e-12 and abs(sum(w.values()) - 1) < 1e-12
    r.p[2] = 0                                  # solved from the entrance: entrance plus the whole route
    w = r.shares(2)
    assert abs(w[0] - R.AT_ENTRANCE) < 1e-12 and np.allclose([w[q] for q in range(1, 11)], (1 - R.AT_ENTRANCE) / 10)
    assert r.shares(0) == {0: 1.0} or abs(sum(r.shares(0).values()) - 1) < 1e-12   # no route: entrance only


def test_pointer_moves_only_to_a_state_that_clears():
    r = make()
    play(r, 2, {10: 0.0, 9: 0.0, 8: 0.0})
    assert r.p[2] == 11
    play(r, 2, {10: 0.75, 9: 0.0, 8: 0.0})
    assert r.p[2] == 10 and r.p[5] == 4


def test_easy_states_do_not_carry_the_pointer_past_an_unsolved_one():
    r = make()                                  # 8-4, 2026-11-14: 0 / 0 / 4 / 86 / 100 % from the last five states
    play(r, 2, {10: 1.0, 9: 0.86, 8: 0.04, 7: 0.0, 6: 0.0})
    assert r.p[2] == 9


def test_a_bad_state_is_stepped_over():
    r = make()                                  # 2-3: 88 / 80 / 8 / 28 / 100 %: states 8 and 9 are bad to start from
    play(r, 2, {10: 1.0, 9: 0.28, 8: 0.08, 7: 0.80, 6: 0.88})
    assert r.p[2] <= 6


def test_few_episodes_do_not_move_the_pointer():
    r = make()
    r.add(np.full(8, 17), np.ones(8, bool))
    assert r.p[2] == 11


def test_pointer_steps_back_when_its_state_stops_clearing():
    r = make()
    r.p[2] = 7
    play(r, 2, {7: 0.1, 8: 0.2, 9: 0.9, 6: 0.0, 5: 0.0, 4: 0.0})
    assert r.p[2] == 9
    play(r, 2, {9: 0.45, 8: 0.2, 7: 0.1, 6: 0.0})     # between the two thresholds: stays
    assert r.p[2] == 9


def test_entrance_rate_sets_the_level_share_and_the_last_step():
    r = make()
    for _ in range(30):
        r.add(np.r_[np.full(50, 0), np.full(50, 2)], np.r_[np.ones(50, bool), np.zeros(50, bool)])
    er = r.entrance_rate()
    assert er[0] > 0.95 and er[2] < 0.05 and r.p[0] == 0
    w = r.weights([0, 2])
    assert w[0] < 0.06 and w[2] + w[8:18].sum() > 0.94      # the failing level gets nearly every episode
    assert w[0] > 0                                         # the solved one is still measured
    r.p[2] = 1
    play(r, 2, {0: 0.8})
    assert r.p[2] == 0                                      # the entrance itself is the last state to solve
    play(r, 2, {0: 0.1})
    assert r.p[2] == 1


def test_power_on_episodes_are_ignored_and_state_round_trips():
    r = make()
    r.add(np.array([-1, -1, 17]), np.array([True, False, True]))
    assert r.seen[0].sum() == 1 and r.seen[0, 17] == 1
    r.p[2] = 6
    q = make()
    q.load_state_dict(r.state_dict())
    assert q.p[2] == 6 and q.seen[0, 17] == 1
    assert q.report()[2]['rates'][-1] == [1.0, 100]         # per route state: episodes, clear %
    other = Routes(L, L + 32, {2: 12, 5: 3})                # level 2's route was replaced: its curriculum restarts
    other.load_state_dict(r.state_dict())
    assert other.p[2] == 13 and other.seen[0, 17] == 0 and other.p[5] == 4


def test_first_version_checkpoint_loads():
    r = make()
    old = dict(K=r.K.copy(), p=np.where(r.K > 0, r.K - 2, 0), win=np.zeros((2, L)), ent=np.ones((2, L)),
               still=np.zeros(L, np.int64), slot=np.full((2, L + 32), 3.0))
    r.load_state_dict(old)
    assert r.p[2] == 8 and r.seen[0, 17] == 3 and r.seen[0, 2] == 1


def test_weights_are_shares_of_playing_time():
    """Two levels that fail equally: the one whose attempts are 10x shorter gets 10x the episodes, so both get the
    same number of decisions."""
    r = make()
    for _ in range(30):
        r.add(np.array([2] * 50 + [5] * 50), np.zeros(100, bool), np.array([20] * 50 + [200] * 50))
    w = r.weights([2, 5])
    assert abs(w[2] / w[5] - 10.0) < 1e-6
    without = Routes(L, L + 32, {2: 10, 5: 3})      # no lengths recorded: plain episode shares
    for _ in range(30):
        without.add(np.array([2] * 50 + [5] * 50), np.zeros(100, bool))
    w = without.weights([2, 5])
    assert abs(w[2] / w[5] - 1.0) < 1e-6
