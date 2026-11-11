"""Which saved level a new training episode starts from (user request 2026-11-11), the same for every game.

Uses only what every game reports, at the moment it happens: a level was cleared (reward10 'level' event), or a game
ended on a level (a win clears it; skipped-level endings are not counted). Counting at the moment matters: a long game
finishing now started over an hour ago, so counting levels at game end reports the old policy (measured 2026-11-11: 1-1
recorded 99.7% clear while the current policy cleared it 0 of 16 times).

Per game and level: clear rate r = (clears + 0.5) / (visits + 1) over recent episodes (counts decay per rollout, so a
level with little recent evidence counts as unknown, r = 0.5, and gets tried again). Failure rate f = 1 - r.
A new episode starts on a level with probability f(level) / sum of f over the levels in play (user request 2026-11-11:
"normalized version of percentage of times it is failing"). Power-on is simply the level before the first saved one.
An episode does not stop at the end of its start level: it plays on, carrying power-ups and lives forward.
"""
from __future__ import annotations

import numpy as np

DECAY = 0.98        # per 16-decision rollout: half-life ~34 rollouts (~8 minutes); train.py scales it to --steps


class LevelPlan:
    def __init__(self, games, cap, decay=DECAY):
        self.games, self.cap, self.decay = list(games), cap, decay
        self.visits = np.zeros((len(self.games), cap))
        self.clears = np.zeros((len(self.games), cap))

    def add(self, visits, clears):
        """One rollout's outcomes, [games, cap]: visits = levels cleared + games ended on the level; clears."""
        self.visits = self.visits * self.decay + visits
        self.clears = self.clears * self.decay + clears

    def rate(self, gi):
        return (self.clears[gi] + 0.5) / (self.visits[gi] + 1.0)

    def plan(self, gi, saved, max_share):
        """saved: the game's saved levels. Returns (weights [cap] over the saved levels, summing to 1, or None if
        nothing is saved; share of episodes starting from a saved level, the rest start at power-on)."""
        if not saved:
            return None, max_share
        f = 1.0 - self.rate(gi)
        base = min(saved) - 1          # power-on: the level before the first saved one
        w = np.zeros(self.cap)
        for lv in (base, *saved):
            w[lv] = f[lv]
        w /= w.sum()
        share = (1.0 - w[base]) * max_share
        w[base] = 0.0
        return w / w.sum(), share

    def state_dict(self):
        return dict(games=self.games, visits=self.visits, clears=self.clears)

    def load_state_dict(self, s):
        assert s['games'] == self.games and s['visits'].shape == self.visits.shape
        self.visits, self.clears = s['visits'].copy(), s['clears'].copy()
