import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ramnovelty import CAP, RamNovelty  # noqa: E402


def frames(n, T, seed=0):
    """byte 0 = frame counter (changes every decision), byte 1 = a switch that flips once, the rest fixed."""
    g = torch.Generator().manual_seed(seed)
    r = torch.randint(0, 256, (1, 1, 2048), generator=g, dtype=torch.uint8).repeat(T + 1, n, 1)
    r[:, :, 0] = torch.arange(T + 1, dtype=torch.uint8)[:, None]
    return r


def test_timer_is_skipped_and_rare_switch_pays_once_per_place():
    n, T = 4, 50
    rn = RamNovelty(1.0, 'cpu', warmup=0)
    r = frames(n, T)
    place = lambda ram: torch.full((ram.shape[0],), 7, dtype=torch.int64)  # noqa: E731
    done = torch.zeros(T, n, dtype=torch.bool)
    learner = torch.ones(n, dtype=torch.bool)
    rn.apply(r, done, place, learner)          # learn the rates: byte 0 changes every decision
    w = rn.weights()
    assert w[0] == 0 and w[1] == 1
    r2 = frames(n, T)
    r2[30:, 0, 1] ^= 0xFF                      # console 0 flips the switch at decision 29
    out = rn.apply(r2, done, place, learner)
    assert out[29, 0] > 0.99 and out[:, 1:].abs().sum() == 0 and out[:29].abs().sum() == 0
    out = rn.apply(r2, done, place, learner)   # the same change again: novelty 1/sqrt(2)
    assert abs(float(out[29, 0]) - 2 ** -0.5) < 1e-5


def test_episode_ends_explorers_and_no_place_never_count():
    n, T = 3, 10
    rn = RamNovelty(1.0, 'cpu', warmup=0)
    r = frames(n, T)
    r[5:, :, 2] = 9                            # everyone: byte 2 changes at decision 4
    done = torch.zeros(T, n, dtype=torch.bool)
    done[4, 0] = True                          # console 0: that decision ended its game
    learner = torch.tensor([True, False, True])
    place = lambda ram: torch.tensor([1, 1, -1])  # noqa: E731   console 2 is outside normal play
    out = rn.apply(r, done, place, learner)
    assert out.abs().sum() == 0
    assert rn.count.sum() == 0


def test_cap_and_warmup():
    n, T = 2, 3
    rn = RamNovelty(0.5, 'cpu', warmup=10 ** 9)
    r = frames(n, T)
    r[2:, :, 100:110] += 1                     # ten rare bytes change together
    place = lambda ram: torch.zeros(ram.shape[0], dtype=torch.int64)  # noqa: E731
    done = torch.zeros(T, n, dtype=torch.bool)
    learner = torch.ones(n, dtype=torch.bool)
    assert rn.apply(r, done, place, learner).abs().sum() == 0      # warmup: counted, not paid
    rn.warmup = 0
    rn.count.zero_()
    rn.changes.zero_()
    rn.steps.fill_(10000)                      # a long history in which these bytes never changed
    out = rn.apply(r, done, place, learner)
    assert abs(float(out[1, 0]) - 0.5 * CAP) < 1e-6


def test_position_like_byte_pays_per_bucket_flag_per_value():
    n, T = 1, 40
    rn = RamNovelty(1.0, 'cpu', warmup=0)
    rn.steps.fill_(10 ** 6)                    # a long history: bytes 5 and 6 rarely change
    r = frames(n, T)
    r[:, :, 0] = 0                             # no frame counter here
    r[:, 0, 5] = torch.arange(0, T + 1, dtype=torch.uint8)   # byte 5 walks like a position: 1, 2, 3 ... 40
    out = rn.apply(r, torch.zeros(T, n, dtype=torch.bool), lambda ram: torch.zeros(ram.shape[0], dtype=torch.int64),
                   torch.ones(n, dtype=torch.bool))
    assert rn.vals[5].sum() > 8                # many values: a position, not counted any more
    assert int((rn.count > 0).sum()) == 8      # only its first 8 values (while it still looked discrete)


def test_state_dict_round_trip():
    rn = RamNovelty(1.0, 'cpu', warmup=0)
    r = frames(2, 5)
    rn.apply(r, torch.zeros(5, 2, dtype=torch.bool), lambda ram: torch.zeros(2, dtype=torch.int64),
             torch.ones(2, dtype=torch.bool))
    s = {k: v.clone() for k, v in rn.state_dict().items()}
    rn2 = RamNovelty(1.0, 'cpu', warmup=0)
    rn2.load_state_dict(s)
    assert torch.equal(rn2.vals, rn.vals) and torch.equal(rn2.count, rn.count) and torch.equal(rn2.changes, rn.changes) and rn2.steps == rn.steps
