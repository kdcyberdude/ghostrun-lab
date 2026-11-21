"""Places inside levels: Go-Explore (Ecoffet et al. 2021, arXiv 1901.10995) on top of the level starts, the same
for every game (user request 2026-11-11: solve a maze without knowing it is a maze).

A place is (level, the newest picture shrunk to 11 x 8 grey cells, each cut to 8 brightness steps, the current
within-level progress in 1/16ths of a level), as in Go-Explore's domain-agnostic cell plus its position; the progress
is the game's generic reward10 progress (rlenv.progress_now), so a maze section the game sends Mario back through
looks the same on screen but is a different place from the same picture past the correct branch. During each rollout, every env's place is looked up in the table of places seen in earlier
rollouts; envs standing on a place nobody has seen yet are saved whole (console, reward tracker; rlenv_place_save)
into staging slots, at most `k` per game per step. After the rollout the table counts every place seen, and each
staged new place is kept in the game's place slots (per level at most `per_level`; when full, the most often seen
place of that level gives way to an equally rare or rarer new one; when every slot is taken, a level below
`per_level` takes the most seen place of the level holding the most, so levels trained later get places too).

New episodes then start from places too (rl/level_plan.py chooses the level by its failure rate, as before): within a
level, the share (1 - failure) stays with the level start and the failure share is split over the level start and
its places by 1 / sqrt(1 + times seen), so rarely reached places (the frontier: the far end of a maze branch nobody
gets through) are replayed most, and solved levels hardly use places. Going round a loop shows only places seen many
times, so it neither creates nor favours anything; a new branch does.

Only pictures, the level number and reward10's within-level progress are involved: no game-specific code; nothing enters the
actor (a place start looks like any moment of the game).

Doors (2026-11-19, with explorers: train.py Config.explore_count). Only learners' spots become places (a learner
started at spots only explorers reached plays where the policy never goes, and 4-4 decayed: notes/least-tried
Control 10). An explorer's find reaches the learners only when it opened a door: a move the explorer tried (its state
at the try's start is kept in the explorer's own slot) was followed, within the try plus DOOR_GRACE decisions, by a
jump the learners have never made: a change of level/area or of more than one 256-pixel column at once that leads on
in the level's progress (rlenv.progress_now), not within DEATH_GAP decisions of a death (a pipe, a vine, a warp;
walking moves one column at a time; loops and death restarts lead back). That try's start becomes a door place; a start there replays the explorer's
move first (DOOR_REPLAY of the time; train.py x_restart), else the policy plays at once. Door places get DOOR_SHARE of
their level's failure share and stay until learners make that jump with their own moves (no held move for DOOR_GRACE
decisions before it) in a game that did not start at a door place; then they are ordinary places.

Situations (2026-11-20, `var` slots, rl/situation.py): the same spot in an unusual state of the world (a rare value of
a progress byte, or standing where the others only fly past) is a different situation. rl/situation.py says which
consoles are in a situation that is rare at a common place and how often it was seen (step's `vkey`); those are
saved into their own `var` slots after all other slots (at most `var_per_level` per level), from learners and
explorers alike. Each keeps its NOTEBOOK: tries = the learners' games started from it, wins = those that cleared its
level (add_starts). A rare situation is only a guess: offline tests 2026-11-20 showed that rarity alone saves junk
(map tiles, sprite flags), and that neither the critic's value nor the return of the next ~40 decisions tells a right
maze step from a wrong one (the pay comes too late and too seldom). So the notebook decides: a situation that was
tried VAR_MIN_TRIES times is SPENT when it wins less than once in VAR_KEEP tries (it leads nowhere) or at least
VAR_DONE of them (it is learned); a spent one gives its slot to a new guess, and its tries are remembered per
situation (`vtried`), so the guesses not tried yet come first: the level's rare situations are swept one after the
other and the ones that lead to a clear stay. Their level gives them VAR_SHARE of its failure share, split by
p (1 - p) with p = (1 + wins) / (2 + tries): most where the outcome is open, little where it always fails or always
wins; the share fades when every one is spent (min(1, s_max / VAR_FRESH)). The ordinary places, their slots and
their weights are untouched.
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
DOOR_GRACE = 32       # decisions after a try's end in which a jump still counts as the try's doing
DEATH_GAP = 128       # decisions after a death in which a jump is the restart, not a door
DOOR_SHARE = 0.5      # of a level's failure share, split over its door places
DOOR_REPLAY = 0.5     # chance that a start at a door place replays the move first (else the policy plays at once)
DOORS_PER_LEVEL = 16  # door places per level (the oldest gives way)
DOOR_COPIES = 4       # door places per door (different explorers' tries through the same door)
START_SHARE = 0.5     # of a level's (non-door) failure share that stays with its level start (weights())
DOOR_WARMUP = 20      # rollouts of learners' jumps collected before the first door (else every jump looks new)
VAR_SHARE = 0.25      # of a level's failure share, for its situation places (while fresh)
VAR_FRESH = 0.1       # p (1 - p) of a place that failed ~8 of 8 tries: below this the situations' share fades
VAR_MIN_TRIES = 12    # games started from a situation place before it can be spent
VAR_KEEP = 40         # ... spent: fewer than one win in this many tries
VAR_DONE = 0.9        # ... or at least this share of wins (learned)
VAR_BITS = 20         # hashed table of the tries each situation already had (4 MB per game)
DOOR_NOW = ('t_key', 't_level', 't_move', 't_len', 't_age', 't_found', 't_code', 'prev_cell', 'prev_prog', 'since_death')


class Places:
    def __init__(self, n_games, N, level_slots, perm, k, T, units, per_level, dev, seed, readmit=False,
                 explorers=None, game_rows=None, share=False, var=0, var_k=2, var_per_level=16):
        self.G, self.N, self.L, self.P, self.K, self.T, self.U = n_games, N, level_slots, perm, k, T, units
        self.per_level, self.dev = per_level, dev
        # readmit (2026-11-19): any key not in the archive can be staged, rarest first (never seen, then seen least);
        # without it only never-seen keys can, so a rare spot that was once replaced (e.g. above 8-4's floating pipe)
        # can never come back. end_rollout's rule (replace only a place seen at least as often) is unchanged. With
        # readmit the visit counts are the learners' only (step's `counted`); counts from a checkpoint without that
        # meaning are dropped on load (they made a once-busy start spot, e.g. 8-4's pipe top, look common forever).
        self.readmit = readmit
        self.share = share    # 2026-11-20 --place-share: slots move to levels without places; level start keeps START_SHARE
        self.arch = [torch.empty(0, dtype=torch.int64, device=dev) for _ in range(n_games)]  # sorted archive keys
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
        # doors (module notes): explorers [N] bool (numpy) gives each explorer its own try slot after the staging slots
        self.doors = explorers is not None
        if self.doors:
            ex = np.asarray(explorers, bool)
            tslot = np.full(N, -1, np.int64)
            for a, z in game_rows:
                tslot[a:z] = np.where(ex[a:z], self.cap + np.cumsum(ex[a:z]) - 1, -1)
            self.cap += int(max(ex[a:z].sum() for a, z in game_rows))
            self.tslot = torch.from_numpy(tslot).to(dev)
            self.row_end = np.array([z for a, z in game_rows])
            i64 = dict(dtype=torch.int64, device=dev)
            self.t_key = torch.full((N,), -1, **i64)          # the explorer's current try: place key at its start,
            self.t_level = torch.full((N,), -1, **i64)        # level, move (d-pad, A, B), length, decisions since
            self.t_move = torch.zeros(N, 3, **i64)
            self.t_len = torch.zeros(N, **i64)
            self.t_age = torch.full((N,), 1 << 30, **i64)
            self.t_found = torch.zeros(N, dtype=torch.bool, device=dev)   # opened a door this rollout (slot kept)
            self.t_code = torch.full((N,), -1, **i64)
            self.prev_cell = torch.full((N,), -1, **i64)      # last valid (level, area, column) of every env
            self.prev_prog = torch.full((N,), float('-inf'), device=dev)   # and its level progress (rlenv.progress_now)
            self.since_death = torch.full((N,), 1 << 30, **i64)
            self.jump_now = torch.zeros(N, dtype=torch.bool, device=dev)   # a forward jump at the last step (rl/outcome.py)
            self.ltrans = torch.empty(0, **i64)               # sorted jump codes the learners have made
            self.b_trans = torch.full((T, N), -1, **i64)
            self.slot_door = np.full((n_games, perm), -1, np.int64)   # door code of a place (-1 = ordinary)
            self.slot_move = np.zeros((n_games, perm, 3), np.int64)
            self.slot_len = np.zeros((n_games, perm), np.int64)
            self.slot_stamp = np.zeros((n_games, perm), np.int64)
            self.rollouts = 0
        # situation places (module notes): their staging slots and slots come after every other slot
        self.V, self.KV, self.var_per_level = var, var_k, var_per_level
        if var:
            Sv = T * units * var_k
            self.vstage0 = self.cap
            self.v0 = self.vstage0 + Sv
            self.cap = self.v0 + var
            self.vslot_key = np.zeros((n_games, var), np.int64)
            self.vslot_level = np.full((n_games, var), -1, np.int64)
            self.vslot_tries = np.zeros((n_games, var), np.int64)     # games started from each situation place
            self.vslot_seen = np.zeros((n_games, var), np.float64)    # times the situation was seen when it was saved
            self.vslot_wins = np.zeros((n_games, var), np.int64)      # ... of the tries, the games that cleared the level
            self.vtried = torch.zeros(n_games, 1 << VAR_BITS, dtype=torch.int32, device=dev)   # tries of spent ones
            self.vst_past = torch.zeros(n_games, Sv, dtype=torch.int32, device=dev)
            self.vst_key = torch.zeros(n_games, Sv, dtype=torch.int64, device=dev)
            self.vst_level = torch.full((n_games, Sv), -1, dtype=torch.int64, device=dev)
            self.vst_ok = torch.zeros(n_games, Sv, dtype=torch.bool, device=dev)
            self.vst_seen = torch.zeros(n_games, Sv, dtype=torch.float64, device=dev)
            self.arange_kv = torch.arange(var_k, device=dev)
            self.varch = [torch.empty(0, dtype=torch.int64, device=dev) for _ in range(n_games)]
        if self.doors:
            self._sync_doors()

    def keys(self, obs_last, emph_last, level, prog):
        """obs_last [B, 224, 240], emph_last [B, 224], level [B], prog [B] float (NaN = unknown) -> place key [B] int64."""
        idx = obs_last.long() | (emph_last.long() << 6).unsqueeze(-1)
        y = F.adaptive_avg_pool2d(self.lum[idx].unsqueeze(1), (GH, GW)).flatten(1)
        q = (y * (STEPS / 256.0)).long().clamp(0, STEPS - 1)
        b = torch.nan_to_num(prog.float() * PROG_STEPS, nan=-1.0).floor().clamp(-1e6, 1e6).long() + 2
        return (q * self.mult).sum(1) ^ ((level.long() + 1) * LEVEL_MIX) ^ (b * PROG_MIX)

    @torch.no_grad()
    def step(self, env, t, gi, unit, a, sl, obs_last, emph_last, level, prog, fresh, key=None, counted=None,
             death=None, own=None, vkey=None):
        """After env step t of envs sl (game gi, its envs a.. of unit `unit`): key every env, stage up to k new places.
        fresh: envs whose episode just ended or started (their state is a restart, not a place).
        key: [B] int64 place keys to use instead of the picture key (-1 = not a place), e.g. the coverage block.
        counted: [B] bool, the envs whose visits count and whose spots can be staged (None = all): the learners
        (train.py Config.explore_count); explorers reach the places only through doors (module notes).
        death: [B] bool, a death this step; own: [B] bool, learners playing their own moves (no held move for
        DOOR_GRACE decisions): only their jumps count as known (doors).
        vkey: (keys [B] int64, times seen [B]) of the consoles in a rare situation (rl/situation.py; -1 = not in
        one), with `var` slots."""
        lv = level.long()
        ok = (lv >= 0) & (lv < self.L) & ~fresh
        if key is None:
            key = self.keys(obs_last, emph_last, lv, prog)
        else:
            ok &= key >= 0
        self.b_key[t, sl] = key
        base = ok
        if counted is not None:
            ok = ok & counted
        self.b_ok[t, sl] = ok
        if self.doors:
            self._door_step(t, sl, key, prog, fresh, counted, death, own)
        if self.known.numel():
            pos = torch.searchsorted(self.known, key).clamp(max=self.known.numel() - 1)
            seen = self.known[pos] == key
            new = ok & ~seen
        else:
            seen = torch.zeros_like(ok)
            new = ok
        n = key.shape[0]
        k = min(self.K, n)
        score = new.float() * (torch.rand(n, device=self.dev, generator=self.gen) + 1e-3)
        if self.readmit:
            arch = self.arch[gi]
            if arch.numel():
                ap = torch.searchsorted(arch, key).clamp(max=arch.numel() - 1)
                cand = ok & (arch[ap] != key)
            else:
                cand = ok
            cnt = torch.where(seen, self.count[pos], torch.zeros_like(key, dtype=torch.float64)) if self.known.numel() \
                else torch.zeros(n, dtype=torch.float64, device=self.dev)
            score = cand.float() * (torch.rand(n, device=self.dev, generator=self.gen) + 1e-3) / (1.0 + cnt.float())
        v, i = score.topk(k)
        good = v > 0
        j = (t * self.U + unit) * self.K + self.arange_k[:k]
        slots = torch.where(good, self.stage0 + j, -1).to(torch.int32)
        env.place_save(gi, (i + a).to(torch.int32), slots)
        self.st_key[gi, j] = key[i]
        self.st_level[gi, j] = lv[i]
        self.st_ok[gi, j] = good
        if self.V and vkey is not None:
            self._var_step(env, t, gi, unit, a, lv, vkey[0], vkey[1], base)

    def _var_step(self, env, t, gi, unit, a, lv, vkey, vn, base):
        """Consoles in a rare situation (vkey >= 0, seen vn times) are saved into the situation staging slots,
        the situations tried least so far first, at most KV per step (module notes)."""
        cand = base & (vkey >= 0)
        arch = self.varch[gi]
        if arch.numel():
            ap = torch.searchsorted(arch, vkey).clamp(max=arch.numel() - 1)
            cand &= arch[ap] != vkey
        n = vkey.shape[0]
        kv = min(self.KV, n)
        past = self.vtried[gi][self._vhash(vkey)]
        score = cand.float() * (torch.rand(n, device=self.dev, generator=self.gen) + 1e-3) / (1.0 + past.float())
        v, i = score.topk(kv)
        good = v > 0
        j = (t * self.U + unit) * self.KV + self.arange_kv[:kv]
        env.place_save(gi, (i + a).to(torch.int32), torch.where(good, self.vstage0 + j, -1).to(torch.int32))
        self.vst_key[gi, j] = vkey[i]
        self.vst_level[gi, j] = lv[i]
        self.vst_seen[gi, j] = vn[i].double()
        self.vst_past[gi, j] = past[i]
        self.vst_ok[gi, j] = good

    @staticmethod
    def _vhash(key):
        return (key >> 30) & ((1 << VAR_BITS) - 1)

    def _var_spent(self, gi):
        """Tried enough and leading nowhere, or learned (module notes)."""
        t, w = self.vslot_tries[gi], self.vslot_wins[gi]
        return (t >= VAR_MIN_TRIES) & ((w * VAR_KEEP < t) | (w >= VAR_DONE * t))

    def _door_step(self, t, sl, key, prog, fresh, counted, death, own):
        """Jumps of envs sl this step: the learners' are recorded, an explorer's new one opens its try's door. Block
        keys only: (((level*128 + area) << 12 | x // 8) << 6) | y // 8, so key >> 11 = (level, area, 256-px column)."""
        if counted is None:
            counted = torch.ones_like(fresh)
        if death is None:
            death = torch.zeros_like(fresh)
        if own is None:
            own = counted
        # x wrapped below 0 (block x clamped to 4095) is no column; outside normal play (pipe or death animation)
        # there is none either, and a try's clock stands still (a pipe's animation is longer than DOOR_GRACE)
        cell = torch.where((key >= 0) & (((key >> 6) & 4095) != 4095), key >> 11, -1)
        sd = torch.where(death, 0, self.since_death[sl] + 1)
        prev = self.prev_cell[sl]
        jump = (prev >= 0) & (cell >= 0) & ~fresh & (sd > DEATH_GAP) & \
            (((prev >> 7) != (cell >> 7)) | ((prev - cell).abs() >= 2))
        code = (prev << 21) | cell
        pr = torch.nan_to_num(prog.float(), nan=float('-inf'))
        ahead = pr > self.prev_prog[sl]     # a door leads on in the level (loops and restarts lead back)
        self.b_trans[t, sl] = torch.where(jump & own, code, -1)
        self.jump_now[sl] = jump & ahead
        if self.ltrans.numel():
            pos = torch.searchsorted(self.ltrans, code).clamp(max=self.ltrans.numel() - 1)
            seen = self.ltrans[pos] == code
        else:
            seen = torch.zeros_like(jump)
        new = jump & ahead & ~counted & ~seen & ~self.t_found[sl] & (self.t_key[sl] >= 0) & \
            (self.t_age[sl] <= self.t_len[sl] + DOOR_GRACE)
        self.t_found[sl] |= new
        self.t_code[sl] = torch.where(new, code, self.t_code[sl])
        keep = fresh | (cell >= 0)
        self.prev_cell[sl] = torch.where(keep, cell, prev)
        self.prev_prog[sl] = torch.where(keep & (cell >= 0), pr, torch.where(keep, float('-inf'), self.prev_prog[sl]))
        self.since_death[sl] = sd
        self.t_age[sl] += (cell >= 0).long()
        self.t_key[sl] = torch.where(fresh, -1, self.t_key[sl])     # a try ends with its game

    def save_tries(self, env, gi, sl, a, began, key, level, move, length):
        """Before the step: explorers sl (game gi, its envs a..) that begin a try keep their state in their own slot,
        with the try (place key, level, move [3, B], length); not one that opened a door this rollout (kept)."""
        if not self.doors:
            return
        ok = began & ~self.t_found[sl] & (self.tslot[sl] >= 0)
        n = ok.shape[0]
        env.place_save(gi, torch.arange(a, a + n, dtype=torch.int32, device=self.dev),
                       torch.where(ok, self.tslot[sl], -1).to(torch.int32))
        self.t_key[sl] = torch.where(ok, key, self.t_key[sl])
        self.t_level[sl] = torch.where(ok, level.long(), self.t_level[sl])
        self.t_move[sl] = torch.where(ok[:, None], move.T, self.t_move[sl])
        self.t_len[sl] = torch.where(ok, length, self.t_len[sl])
        self.t_age[sl] = torch.where(ok, 0, self.t_age[sl])

    def door_of(self, game, slot):
        """game [B], slot [B] (an episode's start slot, -1 = power-on) -> the door move [B, 3] and length [B] (0 = not a
        door place)."""
        s = slot.long().clamp(0, self.cap - 1)
        return self.door_move[game, s], torch.where((slot >= 0) & (slot < self.cap), self.door_len[game, s], 0)

    def _sync_doors(self):
        self.door_move = torch.zeros(self.G, self.cap, 3, dtype=torch.int64, device=self.dev)
        self.door_len = torch.zeros(self.G, self.cap, dtype=torch.int64, device=self.dev)
        for gi in range(self.G):
            d = np.flatnonzero(self.slot_door[gi] >= 0)
            if len(d):
                idx = torch.from_numpy(self.L + d).to(self.dev)
                self.door_move[gi, idx] = torch.from_numpy(self.slot_move[gi, d]).to(self.dev)
                self.door_len[gi, idx] = torch.from_numpy(self.slot_len[gi, d]).to(self.dev)

    def _end_doors(self, env):
        """After the rollout's places: learners' jumps join ltrans; door places they went through become ordinary;
        explorers' new doors become door places (their try slot -> an archive slot). One host copy."""
        nt = self.b_trans[self.b_trans >= 0]
        if nt.numel():
            self.ltrans = torch.unique(torch.cat([self.ltrans, nt]))
        self.b_trans.fill_(-1)
        self.rollouts += 1
        lt = self.ltrans.cpu().numpy()
        opened = 0
        for gi in range(self.G):
            d = np.flatnonzero(self.slot_door[gi] >= 0)
            gone = d[np.isin(self.slot_door[gi][d], lt)]
            self.slot_door[gi][gone] = -1
            opened += len(gone)
        e = torch.nonzero(self.t_found).flatten().cpu().numpy()
        added = 0
        found = [] if not len(e) else [(int(lv), self.jump_name(int(c))) for lv, c in
                                       zip(self.t_level.cpu().numpy()[e][:12], self.t_code.cpu().numpy()[e][:12])]
        if len(e) and self.rollouts > DOOR_WARMUP:
            code, key = self.t_code.cpu().numpy()[e], self.t_key.cpu().numpy()[e]
            lvl, mv = self.t_level.cpu().numpy()[e], self.t_move.cpu().numpy()[e]
            ln, ts = self.t_len.cpu().numpy()[e], self.tslot.cpu().numpy()[e]
            rows = self.game_rows_of(e)
            for i in range(len(e)):
                gi, c, lv = rows[i], code[i], lvl[i]
                if np.isin(c, lt) or not (0 <= lv < self.L):
                    continue
                if (self.slot_door[gi] == c).sum() >= DOOR_COPIES:
                    continue
                mine = np.flatnonzero(self.slot_level[gi] == lv)
                doors = mine[self.slot_door[gi][mine] >= 0]
                free = np.flatnonzero(self.slot_level[gi] < 0)
                if len(doors) < DOORS_PER_LEVEL and len(free):
                    d = free[0]
                elif len(doors) >= DOORS_PER_LEVEL:
                    d = doors[np.argmin(self.slot_stamp[gi][doors])]
                else:
                    plain = mine[self.slot_door[gi][mine] < 0]
                    if len(plain) == 0:
                        continue
                    d = plain[np.argmax(self._counts(torch.from_numpy(self.slot_key[gi][plain]).to(self.dev)))]
                env.place_copy(gi, [int(ts[i])], [self.L + int(d)])
                self.slot_key[gi, d], self.slot_level[gi, d], self.slot_door[gi, d] = key[i], lv, c
                self.slot_move[gi, d], self.slot_len[gi, d], self.slot_stamp[gi, d] = mv[i], ln[i], self.rollouts
                added += 1
        self.t_key[self.t_found] = -1       # a try opens one door
        self.t_found.zero_()
        self.t_code.fill_(-1)
        self._sync_doors()
        self.last.update(doors=int(sum((self.slot_door[gi] >= 0).sum() for gi in range(self.G))), door_found=len(e),
                         door_added=added, door_opened=opened, jumps_known=len(lt), door_jumps=found)

    @staticmethod
    def jump_name(code):
        a, b = code >> 21, code & ((1 << 21) - 1)
        return f"a{(a >> 7) % 128}:x{(a & 127) * 256}->a{(b >> 7) % 128}:x{(b & 127) * 256}"

    def game_rows_of(self, rows):
        return [int(np.searchsorted(self.row_end, r, side='right')) for r in rows]

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
            available = np.ones(self.P, dtype=bool)
            src, dst = [], []
            for i, cn in sorted(zip(idx, cand_n), key=lambda x: x[1]):
                lvl = sl_[gi, i]
                mine = np.flatnonzero((self.slot_level[gi] == lvl) & (not self.doors or self.slot_door[gi] < 0))
                ordinary = (self.slot_level[gi] >= 0) & (not self.doors or self.slot_door[gi] < 0)
                per = np.bincount(self.slot_level[gi][ordinary], minlength=max(int(lvl) + 1, 1))
                rich = int(np.argmax(per))
                if len(mine) < self.per_level and free:
                    d = free.pop()
                elif self.share and len(mine) < self.per_level and per[rich] > len(mine) + 1:
                    # all slots taken (e.g. by the levels trained first): the level holding the most places gives up
                    # its most seen one, so every level that still finds new places gets its share (2026-11-20)
                    choices = np.flatnonzero(ordinary & (self.slot_level[gi] == rich) & available)
                    if len(choices) == 0:
                        continue
                    d = choices[np.argmax(place_n[choices])]
                elif len(mine):
                    choices = mine[available[mine]]
                    if len(choices) == 0:
                        continue
                    d = choices[np.argmax(place_n[choices])]
                    if place_n[d] < cn:
                        continue
                else:
                    continue
                src.append(self.stage0 + int(i))
                dst.append(self.L + int(d))
                available[d] = False  # one source per destination in the batched copy
                self.slot_key[gi, d], self.slot_level[gi, d], place_n[d] = sk[gi, i], lvl, cn
            env.place_copy(gi, src, dst)
            kept += len(src)
        self.st_ok.zero_()
        self.b_ok.zero_()
        self.last = dict(kept=kept, known=int(self.known.numel()))
        if self.doors:
            self._end_doors(env)
        if self.V:
            self._end_var(env)
        self._sync_arch()

    def _end_var(self, env):
        """Keep the staged rare situations, the ones tried least so far first: a level holds at most var_per_level; in
        a full level a spent one (module notes) gives way; with every slot taken, the level holding the most gives
        up a spent one. The tries of a situation that gives way are remembered."""
        vk, vl, vo = self.vst_key.cpu().numpy(), self.vst_level.cpu().numpy(), self.vst_ok.cpu().numpy()
        vs, vp = self.vst_seen.cpu().numpy(), self.vst_past.cpu().numpy()
        kept = 0
        for gi in range(self.G):
            idx = np.flatnonzero(vo[gi])
            if len(idx) == 0:
                continue
            _, first = np.unique(vk[gi, idx], return_index=True)
            idx = idx[np.sort(first)]
            have = set(self.vslot_key[gi][self.vslot_level[gi] >= 0].tolist())
            idx = np.array([i for i in idx if vk[gi, i] not in have], np.int64)
            if len(idx) == 0:
                continue
            spent = self._var_spent(gi)
            available = np.ones(self.V, dtype=bool)
            src, dst, old, old_n = [], [], [], []
            for i in idx[np.argsort(vp[gi, idx], kind='stable')]:
                cn = vs[gi, i]
                lvl = vl[gi, i]
                used = self.vslot_level[gi] >= 0
                mine = np.flatnonzero(self.vslot_level[gi] == lvl)
                free = np.flatnonzero(~used & available)
                if len(mine) < self.var_per_level and len(free):
                    d = free[0]
                elif len(mine) >= self.var_per_level:
                    choices = mine[available[mine] & spent[mine]]
                    if len(choices) == 0:
                        continue
                    d = choices[np.argmax(self.vslot_tries[gi][choices])]
                else:
                    per = np.bincount(self.vslot_level[gi][used], minlength=int(lvl) + 1)
                    rich = int(np.argmax(per))
                    choices = np.flatnonzero((self.vslot_level[gi] == rich) & available & spent)
                    if per[rich] <= len(mine) + 1 or len(choices) == 0:
                        continue
                    d = choices[np.argmax(self.vslot_tries[gi][choices])]
                if used[d]:
                    old.append(int(self.vslot_key[gi, d]))
                    old_n.append(int(self.vslot_tries[gi, d]))
                src.append(self.vstage0 + int(i))
                dst.append(self.v0 + int(d))
                available[d] = False
                self.vslot_key[gi, d], self.vslot_level[gi, d] = vk[gi, i], lvl
                self.vslot_tries[gi, d], self.vslot_wins[gi, d], self.vslot_seen[gi, d] = 0, 0, cn
            env.place_copy(gi, src, dst)
            kept += len(src)
            if old:
                self.vtried[gi].index_add_(0, self._vhash(torch.tensor(old, dtype=torch.int64, device=self.dev)),
                                           torch.tensor(old_n, dtype=torch.int32, device=self.dev))
        self.vst_ok.zero_()
        self.last.update(sit_kept=kept, sit=int(sum((self.vslot_level[gi] >= 0).sum() for gi in range(self.G))),
                         sit_tries=int(self.vslot_tries.sum()), sit_wins=int(self.vslot_wins.sum()),
                         sit_proven=' '.join(f'{a}:{b}' for a, b in zip(*np.unique(
                             self.vslot_level[(self.vslot_wins > 0) & (self.vslot_level >= 0)], return_counts=True))))

    def add_starts(self, gi, slots, won):
        """Game gi's learner episodes that ended: their start slots and whether they cleared the start's level -
        the notebook of the situation places (tries, wins)."""
        if not self.V:
            return
        s, won = np.asarray(slots, np.int64), np.asarray(won, bool)
        m = (s >= self.v0) & (s < self.v0 + self.V)
        np.add.at(self.vslot_tries[gi], s[m] - self.v0, 1)
        np.add.at(self.vslot_wins[gi], s[m] - self.v0, won[m].astype(np.int64))

    def _sync_arch(self):
        for gi in range(self.G):
            k = self.slot_key[gi][self.slot_level[gi] >= 0]
            self.arch[gi] = torch.from_numpy(np.sort(k)).to(self.dev)
            if self.V:
                k = self.vslot_key[gi][self.vslot_level[gi] >= 0]
                self.varch[gi] = torch.from_numpy(np.sort(k)).to(self.dev)

    def level_of(self, gi, slot):
        """Level of a start slot (numpy int array; -1 power-on stays -1)."""
        slot = np.asarray(slot)
        out = slot.copy()
        p = (slot >= self.L) & (slot < self.stage0)
        out[p] = self.slot_level[gi, slot[p] - self.L]
        if self.V:
            v = (slot >= self.v0) & (slot < self.v0 + self.V)
            out[v] = self.vslot_level[gi, slot[v] - self.v0]
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
            f = 1.0 - rate[lvl]
            if self.doors and (self.slot_door[gi][mine] >= 0).any():
                door = self.slot_door[gi][mine] >= 0
                full[self.L + mine[door]] = w * f * DOOR_SHARE / door.sum()
                f, mine = f * (1 - DOOR_SHARE), mine[~door]
            s = 1.0 / np.sqrt(1.0 + n[mine])
            # the level start keeps the cleared share and half the failure share; the places split the other half by
            # rarity. (Until 2026-11-20 the start weighed like one place seen once: with ~30 fresh places a failing
            # level was practised almost only from its middle, its start-to-end play decayed, it failed more and its
            # places took still more: castles 1-4/3-4/4-4/5-4 fell 75-95% -> 14-34% in validation within an hour.)
            if self.share:
                full[lvl] = w * rate[lvl] + w * f * START_SHARE
                full[self.L + mine] = w * f * (1 - START_SHARE) * s / s.sum()
            else:
                s0 = 1.0 / np.sqrt(2.0)
                tot = s0 + s.sum()
                full[lvl] = w * rate[lvl] + w * f * s0 / tot     # the cleared share stays with the level start
                full[self.L + mine] = w * f * s / tot
        if self.V:
            p = (1.0 + self.vslot_wins[gi]) / (2.0 + self.vslot_tries[gi])
            for lvl in np.unique(self.vslot_level[gi][self.vslot_level[gi] >= 0]):
                w = level_w[lvl]
                if w <= 0:
                    continue
                mine = np.flatnonzero(self.vslot_level[gi] == lvl)
                s = p[mine] * (1.0 - p[mine])
                take = (1.0 - rate[lvl]) * VAR_SHARE * min(1.0, s.max() / VAR_FRESH)
                full[lvl] *= 1.0 - take          # the level's start, places and doors give up `take` of their weight
                full[self.L + np.flatnonzero(self.slot_level[gi] == lvl)] *= 1.0 - take
                full[self.v0 + mine] = w * take * s / s.sum()
        return full / full.sum()

    def report(self, gi):
        lv, c = np.unique(self.slot_level[gi][self.slot_level[gi] >= 0], return_counts=True)
        return {int(a): int(b) for a, b in zip(lv, c)}

    def report_var(self, gi):
        if not self.V:
            return {}
        lv, c = np.unique(self.vslot_level[gi][self.vslot_level[gi] >= 0], return_counts=True)
        return {int(a): int(b) for a, b in zip(lv, c)}

    def state_dict(self):
        return dict(known=self.known, count=self.count, slot_key=self.slot_key, slot_level=self.slot_level,
                    var=None if not self.V else dict(vslot_key=self.vslot_key, vslot_level=self.vslot_level,
                                                      vslot_tries=self.vslot_tries, vslot_seen=self.vslot_seen,
                                                      vslot_wins=self.vslot_wins, vtried=self.vtried,
                                                      shape=(self.V, self.KV)),
                    gen=self.gen.get_state(), shape=(self.L, self.P, self.K, self.T, self.U),
                    counts='learners' if self.readmit else 'all',
                    doors=None if not self.doors else dict(ltrans=self.ltrans, slot_door=self.slot_door,
                                                            slot_move=self.slot_move, slot_len=self.slot_len,
                                                            slot_stamp=self.slot_stamp, rollouts=self.rollouts,
                                                            now={k: getattr(self, k) for k in DOOR_NOW}))

    def load_state_dict(self, s):
        assert tuple(s['shape'])[:2] == (self.L, self.P), 'place slots changed'
        self.known, self.count = s['known'].to(self.dev), s['count'].to(self.dev)
        self.slot_key, self.slot_level = s['slot_key'].copy(), s['slot_level'].copy()
        self.gen.set_state(s['gen'].cpu())
        if self.readmit and s.get('counts', 'all') != 'learners':
            self.known = torch.empty(0, dtype=torch.int64, device=self.dev)
            self.count = torch.empty(0, dtype=torch.float64, device=self.dev)
            print('[places] visit counts reset: from now on they count the learners only', flush=True)
        if self.doors and s.get('doors') is not None:
            d = s['doors']
            self.ltrans = d['ltrans'].to(self.dev)
            self.slot_door, self.slot_move = d['slot_door'].copy(), d['slot_move'].copy()
            self.slot_len, self.slot_stamp, self.rollouts = d['slot_len'].copy(), d['slot_stamp'].copy(), d['rollouts']
            for k in DOOR_NOW:
                getattr(self, k).copy_(d['now'][k])
        if self.V and s.get('var') is not None and tuple(s['var']['shape']) == (self.V, self.KV):
            v = s['var']
            self.vslot_key, self.vslot_level = v['vslot_key'].copy(), v['vslot_level'].copy()
            self.vslot_tries, self.vslot_seen = v['vslot_tries'].copy(), v['vslot_seen'].copy()
            self.vslot_wins = v['vslot_wins'].copy()
            self.vtried.copy_(v['vtried'])
        if self.doors:
            self._sync_doors()
        self._sync_arch()
