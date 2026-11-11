"""Packed move ids <-> parts, and random legal moves (tests, baselines)."""
import torch

PARTS = [(0, 9), (9, 12), (12, 15), (15, 17), (17, 19)]  # mask byte ranges: d-pad, A, B, Start, Select
MULT = [1, 9, 27, 81, 162]


def pack(parts):
    """parts: list of 5 long tensors -> int16 move ids."""
    return sum(p * m for p, m in zip(parts, MULT)).to(torch.int16)


def unpack(ids):
    ids = ids.long()
    return [ids % 9, (ids // 9) % 3, (ids // 27) % 3, (ids // 81) % 2, (ids // 162) % 2]


def random_legal(masks, gen=None):
    parts = [torch.multinomial(masks[:, a:b].float(), 1, generator=gen).squeeze(1) for a, b in PARTS]
    return pack(parts)
