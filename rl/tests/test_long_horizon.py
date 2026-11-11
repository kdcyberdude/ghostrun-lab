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
