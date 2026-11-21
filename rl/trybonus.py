"""Try bonus (user design 2026-11-20): reward the player for pressing a button combination it has rarely pressed at
the place where it stands, not only for reaching new places. Same for every game: it never looks at what a button
does, only counts which combination was pressed where.

  combination  d-pad (9: none + 8 directions) x A (off / hold / turbo) x B (off / hold / turbo) = 81. Start and Select
               are left out: in nearly every NES game Start pauses, so "new" presses of it would pay for pausing.
  place        the --places key of the moment before the move (SMB: Mario's 8x8 block; other games: rl/places.py's
               picture key). No key (title, dying, pipe animation) = no bonus, no count.
  bonus        coef / sqrt(1 + n), n = how often the player's games (all consoles, all training; explorers' not
               counted, they do not learn) pressed this combination at this place before. Counts start at zero
               when this is switched on, so every combination is fresh everywhere at first; on common routes they
               reach thousands within minutes and the bonus there fades to ~0.
What a try leads to is paid by the existing reward (a new place, a new area); this bonus only gets the try made.
Counts live in a hashed table (rows = places, 81 columns), like Trainer.x_tried.
"""
from __future__ import annotations

import torch

BITS = 20
COMBOS = 81


def combo(parts):
    """The 5 move parts (d-pad, A, B, Start, Select) -> combination index 0..80 (Start / Select ignored)."""
    return parts[0] * 9 + parts[1] * 3 + parts[2]


class TryBonus:
    def __init__(self, coef, dev):
        self.coef = coef
        self.dev = dev
        self.count = torch.zeros(1 << BITS, COMBOS, dtype=torch.int32, device=dev)
        self.paid = torch.zeros((), device=dev)          # bonus paid since the last stats() (for the log)
        self.fresh = torch.zeros((), dtype=torch.long, device=dev)   # presses of a never-pressed (place, combination)

    @staticmethod
    def row(key):
        return ((key * 0x5851F42D4C957F2D) >> (64 - BITS)) & ((1 << BITS) - 1)

    def step(self, key, parts, learner):
        """Before a move: key [B] place keys (-1 = none), parts (5 x [B]) the moves, learner [B] bool. Returns the
        bonus [B] for these moves and counts the learners' presses."""
        ok = (key >= 0) & learner
        r, c = self.row(key.clamp_min(0)), combo(parts)
        n = self.count[r, c].float()
        bonus = torch.where(ok, self.coef * (1.0 + n).rsqrt(), torch.zeros_like(n))
        self.count.index_put_((r[ok], c[ok]), torch.ones_like(r[ok], dtype=torch.int32), accumulate=True)
        self.paid += bonus.sum()
        self.fresh += (ok & (n == 0)).sum()
        return bonus

    def stats(self):
        out = dict(paid=float(self.paid), fresh=int(self.fresh), pairs=int((self.count > 0).sum()))
        self.paid.zero_()
        self.fresh.zero_()
        return out

    def state_dict(self):
        return dict(count=self.count)

    def load_state_dict(self, s):
        self.count.copy_(s['count'].to(self.dev))
