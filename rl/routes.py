"""Backward curriculum over routes (rl/explore.py): where a level's training episodes start. The same for every game.

A route is a level's saved states from its entrance to its clear, a second or two apart: positions 1..K, position 0 is
the entrance. Each level has a pointer p: from position p on the level counts as solved (at first p = K + 1, nothing
solved; p = 0, solved from the entrance). Half of the level's episodes start at the three states before the solved
stretch, mostly the nearest. When one of them clears the level often enough, the pointer moves to it (Salimans & Chen
2018, arXiv 1812.03381; the second phase of Go-Explore). So the policy only ever has to learn a short new stretch that
ends in play it already knows, and a maze's right turn is learned where it pays seconds later.

Every state is judged by its own clear rate. (An average over a window of states let the pointer walk past a stretch
nobody could play, carried by the easy states behind it: 8-4, 2026-11-14.) Looking three states back steps over a
saved state that is bad to start from (mid-jump) while its neighbours are fine. If the first solved state stops
clearing, the pointer steps back to the next one.

Some episodes always start at the entrance: their clear rate is the level's real score, and it sets the level's share
of all episodes (failure rate, as rl/level_plan.py does). Some start in the solved stretch, so it stays solved. A level
without a route is trained from its entrance only.

Episodes from saved slots are single attempts (rlenv_core.cuh attempt_limit): they end at the clear, the first death
or the decision limit. Only saved console states are involved; nothing enters the actor.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import slots as slotlib

MOVE_AT, RETREAT_AT = 0.5, 0.35          # clear rate that moves the pointer to a state / below which it steps back
MOVE_N = 24.0                            # (decayed) episodes from a state needed to judge it
DECAY = 0.9                              # of the per-state counts, per rollout
LOOK = (0.6, 0.25, 0.15)                 # shares of the 1st, 2nd, 3rd state before the solved stretch
ENTRANCE, FRONT, HOLD, BACK = 0.1, 0.5, 0.15, 0.25   # a level's episodes: entrance, those three, state p, after p
AT_ENTRANCE = 0.7                        # entrance share once solved from there (the rest: the whole route)
FLOOR = 0.02                             # least failure rate a level is weighted by, so it is still measured


def files(directory):
    """level -> route file."""
    return {int(p.stem[1:]): p for p in sorted(Path(directory).glob('L*.npz'), key=lambda p: int(p.stem[1:]))}


class Routes:
    def __init__(self, level_slots, cap, counts):
        """counts: level -> number of route states; their slots follow the level slots, level after level."""
        self.L, self.cap = level_slots, cap
        self.K = np.zeros(level_slots, np.int64)
        self.slot0 = np.zeros(level_slots, np.int64)
        self.level_of = np.full(cap, -1, np.int64)    # slot -> level
        self.level_of[:level_slots] = np.arange(level_slots)
        s = level_slots
        for lv, k in sorted(counts.items()):
            assert s + k <= cap, 'more route states than route slots (--route-slots)'
            self.K[lv], self.slot0[lv] = k, s
            self.level_of[s:s + k] = lv
            s += k
        self.p = self.K + 1                           # solved from position p on
        self.seen = np.zeros((3, cap))                # decayed episodes / clears / decisions per start slot

    @classmethod
    def from_dir(cls, env, gi, directory, route_slots):
        """Write every route of `directory` into game gi's place slots."""
        recs = {lv: np.load(p) for lv, p in files(directory).items()}
        recs = {lv: r for lv, r in recs.items() if len(r['steps'])}
        lay = env.level_layout(gi)
        for lv, r in recs.items():
            saved = json.loads(str(r['meta']))['layout']
            assert all(saved[k] == lay[k] for k in ('stride', 'key_cap', 'st')), f'route L{lv}: other env build'
        self = cls(env.level_slots, env.level_slots + route_slots, {lv: len(r['steps']) for lv, r in recs.items()})
        if recs:
            order = sorted(recs)
            ids = np.concatenate([self.slot0[lv] + np.arange(self.K[lv]) for lv in order])
            slotlib.write(env, gi, ids, {k: np.concatenate([recs[lv][k] for lv in order]) for k in slotlib.FIELDS})
        return self

    def slot(self, lv, pos):
        return lv if pos == 0 else self.slot0[lv] + pos - 1

    def entrance_rate(self):
        return (self.seen[1, :self.L] + 0.5) / (self.seen[0, :self.L] + 1.0)

    def shares(self, lv):
        """position -> share of level lv's episodes (sum 1)."""
        K, p = int(self.K[lv]), int(self.p[lv])
        if p == 0:
            w = {0: AT_ENTRANCE if K else 1.0}
            w.update({q: (1 - AT_ENTRANCE) / K for q in range(1, K + 1)})
            return w
        front = [q for q in (p - 1, p - 2, p - 3) if q >= 0]
        hold, back = p <= K, range(p + 1, K + 1)
        w = {0: ENTRANCE}
        look = LOOK[:len(front)]
        for q, a in zip(front, look):                 # shares with nowhere to go also train the frontier
            w[q] = w.get(q, 0.0) + (FRONT + (0 if hold else HOLD) + (0 if back else BACK)) * a / sum(look)
        if hold:
            w[p] = HOLD
        w.update({q: BACK / len(back) for q in back})
        return w

    def weights(self, levels):
        """Start weights over all slots (sum 1) for the levels in play (those with a saved entrance).

        The shares above are shares of playing time: each slot's weight is divided by the mean length of the episodes
        started there, so a start whose attempts die in seconds gets as many decisions as one whose attempts last.
        (As episode shares, 8-4's frontier got 11% of the episodes but a small part of the decisions, 2026-11-15.)"""
        full = np.zeros(self.cap)
        fail = np.maximum(1.0 - self.entrance_rate(), FLOOR)
        for lv in levels:
            for q, x in self.shares(lv).items():
                full[self.slot(lv, q)] += fail[lv] * x
        n, d = self.seen[0], self.seen[2]
        known = n >= 1.0
        mean = np.where(known, d / np.maximum(n, 1e-9), np.nan)
        fill = np.nanmedian(mean[known & (full > 0)]) if (known & (full > 0)).any() else 1.0
        full = full / np.maximum(np.where(np.isnan(mean), fill, mean), 1.0)
        return full / full.sum()

    def add(self, start, cleared, length=None):
        """One rollout's finished episodes: start slot (int array, -1 power-on), whether the level was cleared and
        (optional) the episode's length in decisions. Counts them and moves the pointers."""
        start, cleared = np.asarray(start, np.int64), np.asarray(cleared, bool)
        ok = start >= 0
        self.seen *= DECAY
        np.add.at(self.seen[0], start[ok], 1.0)
        np.add.at(self.seen[1], start[ok], cleared[ok].astype(float))
        if length is not None:
            np.add.at(self.seen[2], start[ok], np.asarray(length, float)[ok])
        n, c = self.seen[0], self.seen[1]
        for lv in range(self.L):
            p, rate = int(self.p[lv]), lambda q: c[self.slot(lv, q)] / n[self.slot(lv, q)]  # noqa: E731
            judged = lambda q: n[self.slot(lv, q)] >= MOVE_N  # noqa: E731
            solved = [q for q in (p - 3, p - 2, p - 1) if q >= 0 and judged(q) and rate(q) >= MOVE_AT]
            if solved:
                self.p[lv] = solved[0]                # the earliest of the three that clears often enough
            elif p <= self.K[lv] and judged(p) and rate(p) < RETREAT_AT:
                self.p[lv] = p + 1                    # the solved stretch's first state no longer clears

    def report(self):
        """Per level with a route or a measured entrance: pointer, route length, entrance clear rate, and per
        route state [recent episodes started there, their clear rate in percent]."""
        er, out = self.entrance_rate(), {}
        for lv in np.flatnonzero((self.K > 0) | (self.seen[0, :self.L] > 0)):
            s = self.seen[:2, self.slot0[lv]:self.slot0[lv] + self.K[lv]]
            out[int(lv)] = dict(pointer=int(self.p[lv]), states=int(self.K[lv]), entrance=float(er[lv]),
                                entrance_n=float(self.seen[0, lv]),
                                rates=[[round(float(n), 1), round(float(100 * c / max(n, 1e-9)))] for n, c in s.T])
        return out

    def state_dict(self):
        return dict(K=self.K, p=self.p, seen=self.seen)

    def load_state_dict(self, s):
        """A level whose route changed since the save starts its curriculum again; the others continue."""
        same = s['K'] == self.K
        self.p = np.where(same, s['p'], self.K + 1)
        if 'seen' in s:
            seen = s['seen'] if s['seen'].shape[0] == 3 else np.concatenate([s['seen'], np.zeros((1, self.cap))])
            self.seen = seen * same[np.maximum(self.level_of, 0)]
        else:   # saved by the first version (window average): its window start becomes the pointer
            self.seen = s['slot'] * same[np.maximum(self.level_of, 0)]
            self.seen[:, :self.L] = s['ent']
