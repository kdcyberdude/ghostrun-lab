"""RAM-change novelty (user design 2026-11-20): reward the player when the game's memory changes in a way never seen
at this place. Every game keeps everything that matters in its 2 KB of RAM (switches, keys, maze checkpoints, power,
which room); nothing here knows what any byte means.

  rate      how often each of the 2048 bytes changes per decision, counted over the learners' play (all training).
            A frame counter or timer changes nearly every decision, a switch or checkpoint almost never.
  weight    min(1, P0 / rate): rarely changing bytes count fully, constantly changing ones fade. Bytes changing in more
            than SKIP_RATE of decisions are skipped outright (their weight would be < P0 / SKIP_RATE = 0.1) - this keeps
            the work sparse.
  discrete  only bytes that ever took at most FEW different values count (flags, counters, modes, sound-effect ids);
            positions, sprites, tiles and buffers (many values) are left to the place reward. Offline tests 2026-11-20:
            v1 (every byte, exact value) paid enemy / sprite positions on every visit, as much as the camera reward;
            v2 (many-valued bytes coarsened to value // 16) still paid 93% of decisions (sound / screen buffers).
  novelty   1 / sqrt(1 + n), n = how often the learners saw this (place, byte, value) before (all training).
  reward    coef * min(CAP, sum over the bytes changed by this decision of weight * novelty).
  place     the coarse "where" the change happened: the run's camera key at one-screen size (SMB: level, area,
            camera x // 256; -1 outside normal play, then nothing counts). For another game: its picture key.
Steps that end an episode (the RAM after them is the next game's start) never count. Explorers' games are not
counted and not paid (they do not teach). Until WARMUP learner decisions were seen only rates and counts are
gathered (no pay): the rates must be known before they can weigh anything.
Cost: computed once per rollout from the stored RAM (Buffers.b_ram), CHUNK decisions at a time on the GPU; nothing
runs inside the game loop. Within a chunk every console reads the counts from before the chunk. Memory: a fixed
2^24-entry int32 count table (64 MB), 2048 rates, a 2048 x 256 table of values seen.
"""
from __future__ import annotations

import torch

BITS = 24
FEW = 8              # a byte that took at most this many values keeps them exactly
CHUNK = 4            # decisions per GPU pass
P0 = 0.01            # a byte changing in at most 1% of decisions counts fully
SKIP_RATE = 0.1      # bytes changing in more than 10% of decisions are skipped
CAP = 1.0            # at most coef * CAP per decision
WARMUP = 20_000_000  # learner decisions observed before anything is paid (~40 min at 12k consoles)


class RamNovelty:
    def __init__(self, coef, dev, warmup=WARMUP):
        self.coef, self.dev, self.warmup = coef, dev, warmup
        self.count = torch.zeros(1 << BITS, dtype=torch.int32, device=dev)
        self.changes = torch.zeros(2048, dtype=torch.int64, device=dev)    # decisions in which each byte changed
        self.steps = torch.zeros((), dtype=torch.int64, device=dev)        # counted decisions
        self.vals = torch.zeros(2048, 256, dtype=torch.bool, device=dev)   # values each byte changed to
        self.paid = torch.zeros(2048, device=dev)                          # pay per byte since the last stats()
        self.fresh = torch.zeros((), dtype=torch.int64, device=dev)       # never-seen (place, byte, value) since stats()

    def weights(self):
        """[2048] weight of a change of each byte (0 = skipped)."""
        rate = self.changes.float() / self.steps.clamp_min(1).float()
        w = (P0 / rate.clamp_min(1e-12)).clamp(max=1.0)
        return torch.where(rate > SKIP_RATE, torch.zeros_like(w), w)

    @staticmethod
    def slot(place, byte, vkey):
        k = (place * 2048 + byte) * 512 + vkey
        return ((k * 0x5851F42D4C957F2D) >> (64 - BITS)) & ((1 << BITS) - 1)

    def step(self, ram0, ram1, place, ok):
        """C decisions: ram0 / ram1 [C, N, 2048] uint8 before / after, place [C, N] int64 (-1 = none), ok [C, N] bool
        (a learner, not an episode end). Returns the reward [C, N]; counts these decisions."""
        ok = ok & (place >= 0)
        pay_on = bool(self.steps >= self.warmup)
        ch = (ram0 != ram1) & ok[..., None]
        self.changes += ch.sum((0, 1))
        self.steps += ok.sum()
        w = self.weights()
        c, i, b = (ch & (w > 0)).nonzero(as_tuple=True)
        out = torch.zeros(ram0.shape[:2], device=self.dev)
        if b.numel() == 0:
            return out
        v = ram1[c, i, b].long()
        self.vals[b, v] = True
        few = (self.vals.sum(1) <= FEW)[b]
        c, i, b, v = c[few], i[few], b[few], v[few]
        if b.numel() == 0:
            return out
        s = self.slot(place[c, i], b, v)
        n = self.count[s]
        self.fresh += (n == 0).sum()
        self.count.index_add_(0, s, torch.ones_like(s, dtype=torch.int32))
        if pay_on:
            pay = w[b] * (1.0 + n.float()).rsqrt()
            out.index_put_((c, i), pay, accumulate=True)
            out = self.coef * out.clamp(max=CAP)
            self.paid.index_add_(0, b, pay * self.coef)    # per byte, before the cap (for the log only)
        return out

    def apply(self, b_ram, b_done, place_fn, learner):
        """A whole rollout: b_ram [T+1, N, 2048], b_done [T, N], place_fn(ram [N, 2048]) -> [N] place keys,
        learner [N] bool -> rewards [T, N]."""
        T = b_done.shape[0]
        out = torch.zeros(T, b_ram.shape[1], device=self.dev)
        for t in range(0, T, CHUNK):
            e = min(T, t + CHUNK)
            p = torch.stack([place_fn(b_ram[k]) for k in range(t, e + 1)])
            ok = learner[None] & ~b_done[t:e].bool() & (p[1:] >= 0)
            out[t:e] = self.step(b_ram[t:e], b_ram[t + 1:e + 1], p[:-1], ok)
        return out

    def stats(self):
        top = torch.topk(self.paid, 5)
        out = dict(paid=float(self.paid.sum()), fresh=int(self.fresh), active=bool(self.steps >= self.warmup),
                   bytes_counted=int((self.weights() > 0).sum()),
                   table_used=float((self.count > 0).float().mean()),     # near 1 = collisions hide new changes
                   top=[(f'{int(a):#05x}', round(float(v), 1)) for v, a in zip(top.values, top.indices) if v > 0])
        self.paid.zero_()
        self.fresh.zero_()
        return out

    def state_dict(self):
        return dict(count=self.count, changes=self.changes, steps=self.steps, vals=self.vals)

    def load_state_dict(self, s):
        self.count.copy_(s['count'].to(self.dev))
        self.changes.copy_(s['changes'].to(self.dev))
        self.steps.copy_(s['steps'].to(self.dev))
        self.vals.copy_(s['vals'].to(self.dev))
