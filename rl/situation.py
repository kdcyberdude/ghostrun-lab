"""Situations (user design 2026-11-20): a situation is the place PLUS the state of the world. A player can have been
at every place of a level and still not have been in every situation: the same spot after a switch was flipped, after
two right turns in a maze, or standing where everyone else only flies past, is a different situation. This module
says, for every console and decision, whether it is in a situation that is RARE AT A COMMON PLACE; rl/places.py
(`var` slots) saves the game there and starts practice games from it, so a chain of right steps is learned one step
at a time. Nothing here knows what a byte means.

  progress bytes   the bytes of the game's 2 KB RAM in which "the world remembers something that happened at a
                   particular place": a byte that (1) ever took at most FEW values (a flag, a counter), (2) changes
                   in at most RATE_MAX of decisions, (3) changed at least MIN_CHANGES times, and (4) changes only on
                   a few screens (at most PLACE_SHARE of the screens where anything changes, at least PLACE_FLOOR).
                   A clock digit fails (4): it changes wherever the player happens to be. Chosen once, from the
                   learners' play of the first WARMUP decisions, then frozen.
  world state      for every (screen, progress byte, value) the decisions the learners spent with that value on that
                   screen are counted. A console is in a rare world state when one progress byte is an EXCEPTION TO
                   THE RULE of its screen: on that screen (seen at least SCREEN_MIN decisions) the byte has one value
                   in at least DOM of the decisions, the console's value is held in at most RATIO of them but was
                   seen at least REAL_MIN decisions (rare but real), and the console has held it for at least DWELL
                   decisions (no flicker). The least seen such (byte, value) names the situation, one per (screen,
                   byte, value): one byte at a time, never the combination.
                   This is only a GUESS. Offline tests 2026-11-20 (notes/situations_bytes.py, situations_offline.py,
                   value_probe.py): 137 bytes are chosen, most of them level layout and enemy-slot flags; a key over
                   all of them made nearly every moment "new"; "any byte with a rare value" flagged 15% of all
                   decisions; with DOM and DWELL still 12%, the saved situations map tiles and sprite flags. In 7-4
                   there are 77 such (screen, byte, value) and 3 of them are the maze's right steps; neither the
                   critic's value nor the trainer's return (next ~40 decisions) of a situation tells those from the
                   rest (the second maze is passed too seldom). So rl/places.py tries every guess and keeps the
                   ones whose games clear the level (its notebook).
  standing         the player's exact height was the same for the last three decisions (standing or walking on
                   something; in the air it changes - the 8-pixel row of the place key does not at the top of a jump:
                   offline test 2026-11-20, every saved 8-4 "stand" was mid-air), in a block the learners crossed at
                   least BLOCK_MIN decisions but rested in at most RATIO of them: standing where the others only fly
                   past.
  situation key    hash(screen, byte, value) or hash(place key, REST); always >= 2^62, never a place key.
  times seen       the count behind the decision (decisions with that value on the screen / rests in the block).
Screens are the player's: place key >> 11 (level, area, x // 256), as rl/places.py's doors.
Nothing is reported until the bytes are chosen and the counts have run for COUNT_ROLLOUTS rollouts.
Cost: per decision one gather of the progress bytes and one table lookup per byte; per rollout one counting pass over
the stored RAM, CHUNK decisions at a time (observe() only until the bytes are chosen). Memory: three hashed int32
tables (48 MB).
"""
from __future__ import annotations

import torch

FEW = 8
RATE_MAX = 0.01
MIN_CHANGES = 50
PLACE_SHARE = 0.05
PLACE_FLOOR = 8
SCREEN_SLOTS = 1024        # hashed screens per byte (choosing the bytes)
WARMUP = 5_000_000         # learner decisions observed before the progress bytes are chosen
COUNT_ROLLOUTS = 20        # rollouts counted (after the bytes are chosen) before a situation is reported
RATIO = 0.1                # rare: at most this share of the screen's / block's decisions
DOM = 0.8                  # the rule: the byte's usual value holds at least this share of the screen's decisions
DWELL = 4                  # decisions a console has held the value
REAL_MIN = 30              # rare but real: the learners spent at least this many decisions in the world state
SCREEN_MIN = 2000          # a common screen: the learners spent at least this many decisions there
BLOCK_MIN = 200            # a common block
CHUNK = 4
BITS = 22
MIX = 0x5851F42D4C957F2D
REST = 1


def slot(key):
    return ((key * MIX) >> (64 - BITS)) & ((1 << BITS) - 1)


def mix(key, cond):
    return (((key * -7046029254386353131) ^ (cond * MIX)) & ((1 << 62) - 1)) | (1 << 62)


class Situations:
    def __init__(self, n, T, dev, warmup=None, seed=0):
        self.n, self.T, self.dev = n, T, dev
        self.warmup = WARMUP if warmup is None else warmup
        i64 = dict(dtype=torch.int64, device=dev)
        self.idx = torch.empty(0, **i64)                              # the progress bytes
        self.vlist = torch.empty(0, FEW, **i64)                       # the values each of them took (padded)
        self.last = torch.zeros(n, 0, dtype=torch.uint8, device=dev)  # each console's value of each progress byte
        self.dwell = torch.zeros(n, 0, dtype=torch.uint8, device=dev)  # ... and for how many decisions it held it
        self.dbg = torch.full((n,), -1, **i64)                        # byte * 256 + value of the last reported state
        self.y1 = torch.full((n,), -1, **i64)                         # the player's height one / two decisions ago
        self.y2 = torch.full((n,), -1, **i64)
        self.prev = torch.full((n,), -1, **i64)                       # place key after the last rollout's last step
        self.b_key = torch.full((T, n), -1, **i64)                    # this rollout: place key after each step
        self.b_rest = torch.zeros(T, n, dtype=torch.bool, device=dev)
        self.chosen = False
        self.rollouts = 0                                             # counted since the bytes were chosen
        self.found = torch.zeros(2, **i64)                            # rare world states / rare stands reported
        # choosing the bytes
        self.changes = torch.zeros(2048, **i64)
        self.steps = torch.zeros((), **i64)
        self.vals = torch.zeros(2048, 256, dtype=torch.bool, device=dev)
        self.screens = torch.zeros(2048, SCREEN_SLOTS, dtype=torch.bool, device=dev)
        # counts (learners' decisions)
        self.occ = torch.zeros(1 << BITS, dtype=torch.int32, device=dev)    # (screen, progress byte, value)
        self.scr = torch.zeros(1 << BITS, dtype=torch.int32, device=dev)    # screen
        self.blk = torch.zeros(1 << BITS, dtype=torch.int32, device=dev)    # block
        self.rst = torch.zeros(1 << BITS, dtype=torch.int32, device=dev)    # resting in the block

    # --- per decision ---
    @torch.no_grad()
    def step(self, t, ram, key, height, sl, fresh):
        """After decision t: ram [B, 2048] uint8, key [B] place keys (-1 = none; coverage.smb_block_keys layout),
        height [B] the player's exact height (-1 = none), sl = these consoles (slice or index) in [0, n), fresh [B]
        bool (the episode just ended / started). Returns the situation keys [B] (-1 = not in a rare situation) and
        how often each was seen [B] float."""
        key = torch.where(fresh, -1, key)
        y = torch.where(key >= 0, height, -1)
        rest = (y >= 0) & (y == self.y1[sl]) & (y == self.y2[sl])
        self.y2[sl] = torch.where(fresh, -1, self.y1[sl])
        self.y1[sl] = y
        self.b_key[t, sl] = key
        self.b_rest[t, sl] = rest
        none = torch.full_like(key, -1), torch.zeros(key.shape[0], device=self.dev)
        if self.chosen and self.idx.numel():
            v8 = ram[:, self.idx]
            self.dwell[sl] = torch.where((v8 == self.last[sl]) & ~fresh[:, None], (self.dwell[sl] + 1).clamp(max=200), 0)
            self.last[sl] = v8
        if not self.chosen or self.rollouts < COUNT_ROLLOUTS:
            return none
        ok = key >= 0
        kc = key.clamp_min(0)
        bs = slot(kc)
        bc, rc = self.blk[bs].float(), self.rst[bs].float()
        r_ok = ok & rest & (bc >= BLOCK_MIN) & (rc <= RATIO * bc)
        vk, vn = torch.where(r_ok, mix(kc, REST), -1), rc
        if self.idx.numel():
            screen = kc >> 11
            base = (screen[:, None] * 2048 + self.idx) * 256
            v = v8.long()
            o = self.occ[slot(base + v)].float()                                   # [B, K] decisions with this value
            sc = self.scr[slot(screen)].float()
            rare = (ok & (sc >= SCREEN_MIN))[:, None] & (o <= RATIO * sc[:, None]) & (o >= REAL_MIN) & \
                (self.dwell[sl] >= DWELL)
            c, k = rare.nonzero(as_tuple=True)                                     # few (console, byte) pairs
            good = torch.zeros_like(rare)
            if c.numel():
                top = self.occ[slot(base[c, k][:, None] + self.vlist[k])].float().amax(1)   # the usual value's count
                good[c, k] = top >= DOM * sc[c]
            omin, j = torch.where(good, o, torch.full_like(o, float('inf'))).min(1)
            w_ok = omin < float('inf')
            bv = self.idx[j] * 256 + v.gather(1, j[:, None]).squeeze(1)
            vk, vn = torch.where(w_ok, mix(screen, bv * 2 + 2), vk), torch.where(w_ok, omin, vn)
            self.dbg[sl] = torch.where(w_ok, bv, -1)
            self.found[0] += w_ok.sum()
        self.found[1] += (vk >= 0).sum()
        return vk, vn

    # --- per rollout ---
    @torch.no_grad()
    def end_rollout(self, b_ram, b_done, learner):
        """b_ram [T+1, N, 2048] (b_ram[t + 1] = RAM after decision t), b_done [T, N], learner [N] bool: until the bytes
        are chosen gather their statistics, afterwards count the learners' world states, blocks and rests."""
        T = b_done.shape[0]
        if not self.chosen:
            self._observe(b_ram, b_done, learner)
        else:
            for t in range(0, T, CHUNK):
                e = min(T, t + CHUNK)
                kb = self.b_key[t:e]
                ok = learner[None] & (kb >= 0) & ~b_done[t:e].bool()
                kc = kb.clamp_min(0)
                one = torch.ones(int(ok.sum()), dtype=torch.int32, device=self.dev)
                self.blk.index_add_(0, slot(kc[ok]), one)
                self.scr.index_add_(0, slot(kc[ok] >> 11), one)
                r = ok & self.b_rest[t:e]
                self.rst.index_add_(0, slot(kc[r]), torch.ones(int(r.sum()), dtype=torch.int32, device=self.dev))
                if self.idx.numel():
                    v = b_ram[t + 1:e + 1][ok][:, self.idx].long()            # [M, K]
                    k = ((kc[ok] >> 11)[:, None] * 2048 + self.idx) * 256 + v
                    self.occ.index_add_(0, slot(k).flatten(),
                                        torch.ones(k.numel(), dtype=torch.int32, device=self.dev))
            self.rollouts += 1
        self.prev = self.b_key[T - 1].clone()

    def _observe(self, b_ram, b_done, learner):
        T = b_done.shape[0]
        before = torch.cat([self.prev[None], self.b_key[:T - 1]])            # place key before each decision
        for t in range(0, T, CHUNK):
            e = min(T, t + CHUNK)
            ok = learner[None] & ~b_done[t:e].bool() & (self.b_key[t:e] >= 0) & (before[t:e] >= 0)
            ch = (b_ram[t:e] != b_ram[t + 1:e + 1]) & ok[..., None]
            self.changes += ch.sum((0, 1))
            self.steps += ok.sum()
            c, i, b = ch.nonzero(as_tuple=True)
            if b.numel():
                self.vals[b, b_ram[t + 1:e + 1][c, i, b].long()] = True
                h = (((before[t:e][c, i] >> 11) * MIX) >> 40) & (SCREEN_SLOTS - 1)
                self.screens[b, h] = True
        if int(self.steps) >= self.warmup:
            self.choose()

    def choose(self):
        rate = self.changes.float() / self.steps.clamp_min(1).float()
        nscreens = self.screens.sum(1)
        limit = max(PLACE_FLOOR, int(PLACE_SHARE * int(self.screens.any(0).sum())))
        ok = (self.changes >= MIN_CHANGES) & (self.vals.sum(1) <= FEW) & (rate <= RATE_MAX) & (nscreens <= limit)
        self.idx = ok.nonzero().flatten()
        K = len(self.idx)
        self.vlist = torch.zeros(K, FEW, dtype=torch.int64, device=self.dev)
        for a, b in enumerate(self.idx.tolist()):
            vv = self.vals[b].nonzero().flatten()
            self.vlist[a] = vv[torch.arange(FEW, device=self.dev) % len(vv)]
        self.last = torch.zeros(self.n, K, dtype=torch.uint8, device=self.dev)
        self.dwell = torch.zeros(self.n, K, dtype=torch.uint8, device=self.dev)
        self.chosen = True
        self.info = dict(bytes=[f'{int(a):#05x}' for a in self.idx], screens=nscreens[self.idx].tolist(),
                         changes=self.changes[self.idx].tolist(), screen_limit=limit)
        print(f"[situations] progress bytes chosen after {int(self.steps)} decisions: {len(self.idx)} "
              f"(changes on at most {limit} screens): {self.info['bytes']}", flush=True)

    def stats(self):
        out = dict(bytes=len(self.idx) if self.chosen else None, counted=self.rollouts,
                   rare_world=int(self.found[0]), rare=int(self.found[1]))
        self.found.zero_()
        return out

    def state_dict(self):
        return dict(idx=self.idx, vlist=self.vlist, last=self.last, dwell=self.dwell,
                    y1=self.y1, y2=self.y2, prev=self.prev, chosen=self.chosen, rollouts=self.rollouts,
                    changes=self.changes, steps=self.steps, vals=self.vals, screens=self.screens,
                    occ=self.occ, scr=self.scr, blk=self.blk, rst=self.rst, info=getattr(self, 'info', None))

    def load_state_dict(self, s):
        self.idx, self.vlist = s['idx'].to(self.dev), s['vlist'].to(self.dev)
        self.last, self.dwell = s['last'].to(self.dev), s['dwell'].to(self.dev)
        self.chosen, self.rollouts = bool(s['chosen']), int(s['rollouts'])
        for k in ('y1', 'y2', 'prev', 'changes', 'steps', 'vals', 'screens', 'occ', 'scr', 'blk', 'rst'):
            getattr(self, k).copy_(s[k])
        if s.get('info') is not None:
            self.info = s['info']
