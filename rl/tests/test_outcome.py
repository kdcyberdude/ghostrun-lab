import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from outcome import E, EVENTS, HORIZONS, Outcome  # noqa: E402


def test_labels_count_events_in_window_and_stop_at_game_end():
    o = Outcome(8, 'cpu', 0.1)
    T, N = 20, 3
    ev = torch.zeros(T, N, dtype=torch.uint8)
    rew = torch.zeros(T, N)
    done = torch.zeros(T, N, dtype=torch.bool)
    J = EVENTS.index('jump')
    ev[10, 0] = 1 << J                 # env 0: a jump at step 10
    done[5, 1] = True                  # env 1: game ends at 5, a jump at 7 belongs to the next game
    ev[7, 1] = 1 << J
    rew[18, 2] = 1.0                   # env 2: new ground at 18
    y, k = o.labels(ev, rew, done)
    h4, h16 = 0, E                     # horizon-major columns
    # env 0: jump within 4 for t = 7..10, within 16 for t = 0..10 (t <= 10 < t + 16)
    assert [int(y[t, 0, h4 + J]) for t in range(T)] == [1 if 7 <= t <= 10 else 0 for t in range(T)]
    assert all(int(y[t, 0, h16 + J]) == (t <= 10) for t in range(T))
    # unknown: no event and the 16-window runs past the rollout (t + 16 > 20) with no end inside
    assert not bool(k[11, 0, h16 + J]) and bool(k[4, 0, h16 + J]) and bool(k[10, 0, h16 + J])
    # env 1: steps 0..5 never see the next game's jump, and their window is known (the game ended)
    assert y[:6, 1, h16 + J].sum() == 0 and bool(k[:6, 1, h16 + J].all())
    assert int(y[6, 1, h4 + J]) == 1 and int(y[7, 1, h4 + J]) == 1 and int(y[8, 1, h4 + J]) == 0
    # env 2: 'new' from reward
    N_ = EVENTS.index('new')
    assert int(y[18, 2, h4 + N_]) == 1 and int(y[15, 2, h4 + N_]) == 1 and int(y[14, 2, h4 + N_]) == 0
    assert y.shape == (T, N, E * len(HORIZONS))


def test_events_and_promise_shapes():
    parts = torch.zeros(4, 7)
    parts[0, 2] = -1      # death
    parts[1, 3] = -1      # damage
    parts[2, 4] = 0.5     # score up
    parts[3, 1] = 1       # new level
    ev = Outcome.events(parts, torch.tensor([False, False, False, True]))
    assert ev.tolist() == [1, 2, 4, 8 | 16]
    o = Outcome(8, 'cpu', 0.1, chunk=3)
    p = o.promise(torch.randn(5, 8))
    assert p.shape == (5, 36)
    # a fresh head: every event starts near p = sigmoid(-3); promise = 1 - (1 - p)^2
    q = torch.sigmoid(torch.tensor(-3.0))
    assert torch.allclose(p, (1 - (1 - q) ** 2).expand(5, 36), atol=1e-5)
    # its loss is finite and trains toward the labels
    h = torch.randn(64, 8)
    parts5 = [torch.zeros(64, dtype=torch.long)] * 5
    y = torch.zeros(64, 2 * E)
    y[:, E + EVENTS.index('jump')] = 1
    k = torch.ones(64, 2 * E, dtype=torch.bool)
    l0 = o.loss(h, parts5, y, k)
    for _ in range(50):
        o.opt.zero_grad()
        o.loss(h, parts5, y, k).backward()
        o.opt.step()
    assert o.loss(h, parts5, y, k) < l0
