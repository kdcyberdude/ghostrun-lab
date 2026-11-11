"""Places inside levels: Go-Explore (Ecoffet et al. 2021, arXiv 1901.10995) on top of the level starts, the same
for every game (user request 2026-11-11: solve a maze without knowing it is a maze).

A place is (level, the newest picture shrunk to 11 x 8 grey cells, each cut to 8 brightness steps, the current
within-level progress in 1/16ths of a level), as in Go-Explore's domain-agnostic cell plus its position; the progress
is the game's generic reward10 progress (rlenv.progress_now), so a maze section the game sends Mario back through
looks the same on screen but is a different place from the same picture past the correct branch. During each rollout, every env's place is looked up in the table of places seen in earlier
rollouts; envs standing on a place nobody has seen yet are saved whole (console, reward tracker; rlenv_place_save)
into staging slots, at most `k` per game per step. After the rollout the table counts every place seen, and each
staged new place is kept in the game's place slots (per level at most `per_level`; when full, the most often seen
place of that level gives way to a rarer one).

New episodes then start from places too (rl/level_plan.py chooses the level by its failure rate, as before): within a
level, the share (1 - failure) stays with the level start and the failure share is split over the level start and
its places by 1 / sqrt(1 + times seen), so rarely reached places (the frontier: the far end of a maze branch nobody
gets through) are replayed most, and solved levels hardly use places. Going round a loop shows only places seen many
times, so it neither creates nor favours anything; a new branch does.

Only pictures, the level number and reward10's within-level progress are involved: no game-specific code; nothing enters the
actor (a place start looks like any moment of the game).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fceux_palette import RGB

GW, GH, STEPS = 11, 8, 8
PROG_STEPS = 16                    # progress buckets per level (PROGRESS_UNIT)
PROG_MIX = 0x5851F42D4C957F2D      # odd 64-bit multiplier (fits int64)
LEVEL_MIX = -7046029254386353131   # 0x9E3779B97F4A7C15 as int64


class Places:
    def __init__(self, n_games, N, level_slots, perm, k, T, units, per_level, dev, seed):
        self.G, self.N, self.L, self.P, self.K, self.T, self.U = n_games, N, level_slots, perm, k, T, units
        self.per_level, self.dev = per_level, dev
        self.stage0 = level_slots + perm
        self.cap = self.stage0 + T * units * k
        rgb = torch.tensor(RGB, dtype=torch.float32)
        self.lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).to(dev)
        g = torch.Generator().manual_seed(seed * 977 + 3)
        self.mult = (torch.randint(1, 2 ** 62, (GH * GW,), generator=g, dtype=torch.int64) | 1).to(dev)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed * 991 + 5)
        self.known = torch.empty(0, dtype=torch.int64, device=dev)    # sorted place keys seen in earlier rollouts
        self.count = torch.empty(0, dtype=torch.float64, device=dev)  # times seen (decisions), same order
        self.slot_key = np.zeros((n_games, perm), np.int64)
        self.slot_level = np.full((n_games, perm), -1, np.int64)      # -1 = empty
        S = T * units * k
        self.st_key = torch.zeros(n_games, S, dtype=torch.int64, device=dev)
        self.st_level = torch.full((n_games, S), -1, dtype=torch.int64, device=dev)
        self.st_ok = torch.zeros(n_games, S, dtype=torch.bool, device=dev)
        self.b_key = torch.zeros(T, N, dtype=torch.int64, device=dev)
        self.b_ok = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.arange_k = torch.arange(k, device=dev)
        self.last = {}

    def keys(self, obs_last, emph_last, level, prog):
        """obs_last [B, 224, 240], emph_last [B, 224], level [B], prog [B] float (NaN = unknown) -> place key [B] int64."""
        idx = obs_last.long() | (emph_last.long() << 6).unsqueeze(-1)
        y = F.adaptive_avg_pool2d(self.lum[idx].unsqueeze(1), (GH, GW)).flatten(1)
        q = (y * (STEPS / 256.0)).long().clamp(0, STEPS - 1)
        b = torch.nan_to_num(prog.float() * PROG_STEPS, nan=-1.0).floor().clamp(-1e6, 1e6).long() + 2
        return (q * self.mult).sum(1) ^ ((level.long() + 1) * LEVEL_MIX) ^ (b * PROG_MIX)

    @torch.no_grad()
    def step(self, env, t, gi, unit, a, sl, obs_last, emph_last, level, prog, fresh):
        """After env step t of envs sl (game gi, its envs a.. of unit `unit`): key every env, stage up to k new places.
        fresh: envs whose episode just ended or started (their state is a restart, not a place)."""
        lv = level.long()
        ok = (lv >= 0) & (lv < self.L) & ~fresh
        key = self.keys(obs_last, emph_last, lv, prog)
        self.b_key[t, sl] = key
        self.b_ok[t, sl] = ok
        if self.known.numel():
            pos = torch.searchsorted(self.known, key).clamp(max=self.known.numel() - 1)
            new = ok & (self.known[pos] != key)
        else:
            new = ok
        n = key.shape[0]
        k = min(self.K, n)
        score = new.float() * (torch.rand(n, device=self.dev, generator=self.gen) + 1e-3)
        v, i = score.topk(k)
        good = v > 0
        j = (t * self.U + unit) * self.K + self.arange_k[:k]
        slots = torch.where(good, self.stage0 + j, -1).to(torch.int32)
        env.place_save(gi, (i + a).to(torch.int32), slots)
        self.st_key[gi, j] = key[i]
        self.st_level[gi, j] = lv[i]
        self.st_ok[gi, j] = good

    def _counts(self, keys):
        """Times seen of keys (torch int64 on dev) -> float64 numpy (0 = never)."""
        if self.known.numel() == 0 or keys.numel() == 0:
            return np.zeros(keys.numel())
        pos = torch.searchsorted(self.known, keys).clamp(max=self.known.numel() - 1)
        return torch.where(self.known[pos] == keys, self.count[pos], torch.zeros_like(self.count[pos])).cpu().numpy()

    @torch.no_grad()
    def end_rollout(self, env):
        """Count this rollout's places, keep the staged new ones. One host sync."""
        uk, uc = torch.unique(self.b_key[self.b_ok], return_counts=True)
        allk = torch.cat([self.known, uk])
        allc = torch.cat([self.count, uc.double()])
        self.known, inv = torch.unique(allk, return_inverse=True)
        self.count = torch.zeros(self.known.numel(), dtype=torch.float64, device=self.dev).index_add_(0, inv, allc)
        sk, sl_, so = self.st_key.cpu().numpy(), self.st_level.cpu().numpy(), self.st_ok.cpu().numpy()
        kept = 0
        for gi in range(self.G):
            idx = np.flatnonzero(so[gi])
            if len(idx) == 0:
                continue
            _, first = np.unique(sk[gi, idx], return_index=True)
            idx = idx[np.sort(first)]
            have = set(self.slot_key[gi][self.slot_level[gi] >= 0].tolist())
            idx = np.array([i for i in idx if sk[gi, i] not in have], np.int64)
            if len(idx) == 0:
                continue
            cand_n = self._counts(torch.from_numpy(sk[gi, idx]).to(self.dev))
            used = self.slot_level[gi] >= 0
            place_n = self._counts(torch.from_numpy(self.slot_key[gi]).to(self.dev))
            free = list(np.flatnonzero(~used)[::-1])
            src, dst = [], []
            for i, cn in sorted(zip(idx, cand_n), key=lambda x: x[1]):
                lvl = sl_[gi, i]
                mine = np.flatnonzero(self.slot_level[gi] == lvl)
                if len(mine) < self.per_level and free:
                    d = free.pop()
                elif len(mine):
                    d = mine[np.argmax(place_n[mine])]
                    if place_n[d] <= cn:
                        continue
                else:
                    continue
                src.append(self.stage0 + int(i))
                dst.append(self.L + int(d))
                self.slot_key[gi, d], self.slot_level[gi, d], place_n[d] = sk[gi, i], lvl, cn
            env.place_copy(gi, src, dst)
            kept += len(src)
        self.st_ok.zero_()
        self.b_ok.zero_()
        self.last = dict(kept=kept, known=int(self.known.numel()))

    def level_of(self, gi, slot):
        """Level of a start slot (numpy int array; -1 power-on stays -1)."""
        slot = np.asarray(slot)
        out = slot.copy()
        p = (slot >= self.L) & (slot < self.stage0)
        out[p] = self.slot_level[gi, slot[p] - self.L]
        return out

    def weights(self, gi, level_w, rate):
        """level_w [L] (rl/level_plan.py, sums to 1 over saved levels), rate [L] clear rates -> weights over all cap
        slots: each level's failure share spread over its start and its places by 1 / sqrt(1 + times seen)."""
        full = np.zeros(self.cap)
        full[:self.L] = level_w
        n = self._counts(torch.from_numpy(self.slot_key[gi]).to(self.dev))
        for lvl in np.unique(self.slot_level[gi][self.slot_level[gi] >= 0]):
            w = level_w[lvl]
            if w <= 0:
                continue
            mine = np.flatnonzero(self.slot_level[gi] == lvl)
            s = 1.0 / np.sqrt(1.0 + n[mine])
            s0 = 1.0 / np.sqrt(2.0)
            f = 1.0 - rate[lvl]
            tot = s0 + s.sum()
            full[lvl] = w * (1 - f) + w * f * s0 / tot
            full[self.L + mine] = w * f * s / tot
        return full / full.sum()

    def report(self, gi):
        lv, c = np.unique(self.slot_level[gi][self.slot_level[gi] >= 0], return_counts=True)
        return {int(a): int(b) for a, b in zip(lv, c)}

    def state_dict(self):
        return dict(known=self.known, count=self.count, slot_key=self.slot_key, slot_level=self.slot_level,
                    gen=self.gen.get_state(), shape=(self.L, self.P, self.K, self.T, self.U))

    def load_state_dict(self, s):
        assert tuple(s['shape'])[:2] == (self.L, self.P), 'place slots changed'
        self.known, self.count = s['known'].to(self.dev), s['count'].to(self.dev)
        self.slot_key, self.slot_level = s['slot_key'].copy(), s['slot_level'].copy()
        self.gen.set_state(s['gen'].cpu())
