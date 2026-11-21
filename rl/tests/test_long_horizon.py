"""Analytic return/boundary checks for longer recurrent PPO rollouts (no neural updates)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import Trainer


def gae(reward, value, done, gamma=.999, lam=.99):
    T, N = reward.shape
    owner = SimpleNamespace(T=T, N=N, dev=reward.device, cfg=SimpleNamespace(gamma=gamma, lam=lam))
    return Trainer.gae(owner, SimpleNamespace(b_val=value, b_done=done), reward)


def test_delayed_reward_can_outweigh_early_penalty_inside_longer_window():
    reward = torch.zeros(128, 1)
    reward[0] = -.02
    reward[120] = 1
    adv, _ = gae(reward, torch.zeros(129, 1), torch.zeros(128, 1, dtype=torch.bool))
    assert abs(float(adv[0]) - (-.02 + (.999 * .99)**120)) < 1e-6
    assert adv[0] > 0
    old, _ = gae(reward[:64], torch.zeros(65, 1), torch.zeros(64, 1, dtype=torch.bool), .995, .98)
    assert old[0] < 0  # zero critic here intentionally isolates actual observed reward exposure


def test_lambda_one_matches_discounted_returns_with_terminal_and_rollout_boundaries():
    gen = torch.Generator().manual_seed(17)
    reward = torch.randn(128, 3, generator=gen)
    value = torch.randn(129, 3, generator=gen)
    done = torch.zeros(128, 3, dtype=torch.bool)
    done[31, 1] = True
    done[70, 1] = True
    done[127, 2] = True
    _, actual = gae(reward, value, done, lam=1.)
    expected = torch.zeros_like(reward)
    for t in range(128):
        for n in range(3):
            discount = 1.
            for k in range(t, 128):
                expected[t, n] += discount * reward[k, n]
                discount *= .999
                if done[k, n]:
                    break
            else:
                expected[t, n] += discount * value[128, n]
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_bootstrap_survives_rollout_cutoff_but_never_crosses_done():
    reward = torch.zeros(128, 2)
    value = torch.zeros(129, 2)
    value[-1] = 2
    done = torch.zeros(128, 2, dtype=torch.bool)
    done[-1, 1] = True
    adv, _ = gae(reward, value, done)
    assert abs(float(adv[-1, 0]) - 1.998) < 1e-6
    # 128 float32 recurrence steps accumulate rounding relative to the double-precision closed form.
    assert abs(float(adv[0, 0]) - 1.998 * (.999 * .99)**127) < 5e-6
    assert torch.equal(adv[:, 1], torch.zeros(128))


def test_held_moves_cut_the_trace_of_the_policy_move_before_them():
    """A policy move followed by a held (random) move bootstraps from the value there: it gets neither credit nor
    blame for what the held move did. The held moves' own trace runs on into the policy's moves after them."""
    T = 6
    reward = torch.zeros(T, 1)
    reward[3] = -1.0                                   # a held move at t=2..3 walks into a death
    reward[5] = 2.0                                    # the policy, back in control, then finds something
    value = torch.full((T + 1, 1), 0.5)
    done = torch.zeros(T, 1, dtype=torch.bool)
    forced = torch.tensor([[False], [False], [True], [True], [False], [False]])
    owner = SimpleNamespace(T=T, N=1, dev='cpu', cfg=SimpleNamespace(gamma=1.0, lam=1.0, explore_eps=0.01))
    adv, ret = Trainer.gae(owner, SimpleNamespace(b_val=value, b_done=done, b_forced=forced), reward)
    assert float(adv[1]) == 0.0                        # r 0 + V(s2) 0.5 - V(s1) 0.5: the held try's death is not its fault
    assert float(adv[0]) == 0.0                        # and nothing leaks back through it
    assert abs(float(adv[2]) - (-1.0 + 2.0 + 0.5 - 0.5)) < 1e-6   # the try's own advantage: everything after it
    plain, _ = Trainer.gae(SimpleNamespace(T=T, N=1, dev='cpu', cfg=SimpleNamespace(gamma=1.0, lam=1.0)),
                           SimpleNamespace(b_val=value, b_done=done), reward)
    assert abs(float(plain[1]) - 1.0) < 1e-6           # without held moves nothing is cut


def test_advantages_are_normalised_by_the_policys_own_moves():
    """Held (random) moves are mostly worse than the policy; they must not shift the baseline of its own moves."""
    torch.manual_seed(0)
    T, N = 48, 64
    own = torch.rand(T, N) < 0.5
    adv = torch.where(own, torch.randn(T, N), torch.randn(T, N) - 1.0)    # held moves 1 std worse
    owner = SimpleNamespace(game_of_env=torch.zeros(N, dtype=torch.long), ddp=False, dev='cpu')
    out = Trainer.per_game_norm(owner, adv, own)
    assert abs(float(out[own].mean())) < 1e-5 and abs(float(out[own].std(unbiased=False)) - 1) < 1e-4
    assert float(Trainer.per_game_norm(owner, adv)[own].mean()) > 0.3     # the old statistics: own moves biased up
