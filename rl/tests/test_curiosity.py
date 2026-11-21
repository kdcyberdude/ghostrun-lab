"""Curiosity about actions (rl/curiosity.py): the heads agree about moves they were trained on and disagree about a
move they never saw, a random outcome pays less than an unseen move, and the bonus is 0 at an episode end."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curiosity import Curiosity, OUT  # noqa: E402
from model import SIZES  # noqa: E402

CORE = 32


def _parts(n, dpad):
    return [torch.full((n,), dpad, dtype=torch.long)] + [torch.zeros(n, dtype=torch.long) for _ in SIZES[1:]]


def test_unseen_move_disagrees_more_than_seen_and_random_outcomes():
    """A stream of fresh situations (as in training): move 3 ("right") leads to a fixed function of the situation,
    move 5 ("down") to pure noise, move 7 ("left") is never played."""
    torch.manual_seed(0)
    n = 512
    cur = Curiosity(2 * n, CORE, 'cpu', seed=1, coef=1.0, lr=1e-3)
    w = torch.randn(CORE, OUT)
    right, down, left = _parts(n, 3), _parts(n, 5), _parts(n, 7)
    parts = [torch.cat([a, b]) for a, b in zip(right, down)]
    e, keep = torch.arange(2 * n), torch.ones(2 * n, dtype=torch.bool)
    for _ in range(2000):
        h = torch.randn(n, CORE)
        loss = cur.loss(torch.cat([h, h]), parts, torch.cat([torch.tanh(h @ w), torch.randn(n, OUT)]), keep, e, 1)
        cur.opt.zero_grad()
        loss.backward()
        cur.opt.step()
    h = torch.randn(n, CORE)
    seen = float(cur.disagreement(h, right).mean())
    noisy = float(cur.disagreement(h, down).mean())
    unseen = float(cur.disagreement(h, left).mean())
    assert unseen > 2 * seen, (seen, unseen)
    assert unseen > 5 * noisy, (noisy, unseen)


def test_bonus_scale_and_episode_end():
    cur = Curiosity(4, CORE, 'cpu', seed=1, coef=0.5, clip=10.0)
    T, N = 3, 4
    fp = torch.randn(T, N, OUT)
    dis = torch.tensor([[1.0, 1.0, 1.0, 100.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    done = torch.zeros(T, N, dtype=torch.bool)
    done[1, 2] = True
    target, bonus = cur.prepare(fp, dis, done)
    assert bonus[1, 2] == 0                                   # the move into an episode end pays nothing
    assert torch.isclose(bonus[0, 3], torch.tensor(0.5 * 10.0))   # capped at clip x the average
    assert target.shape == (T, N, OUT)
    assert torch.isfinite(target).all()
