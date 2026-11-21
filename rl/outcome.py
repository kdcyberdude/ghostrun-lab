"""What will this move cause? (user objective 2026-11-19: the model should understand what every thing in the game
does to Mario, in its own latent terms, and use that to try rare-but-real possibilities.)

A small head on the policy's own state (the GRU output h) plus a move predicts the CHANCE of each of 6 events within
the next 4 and the next 16 decisions:
  death       the game's reward10 'death' part (a life lost)
  damage      the 'damage' part (hurt, e.g. Mario shrinks)
  score       the 'score' part went up (an enemy beaten, a coin, an item eaten)
  level       the 'level' part (a new level reached)
  jump        Mario is suddenly somewhere else and further on in the level (another area, or more than one 256-px
              column at once, not near a death; rl/places.py doors): a pipe, a vine, a warp
  new         the step paid reward (coverage: a camera position / 8x8 block new in this game)
Same events for every game (rlenv.parts are reward10's generic parts), no game rules.

Its loss also trains the policy's network (the "understanding" lives in h, shared with the policy). It is trained on
every console's moves, the explorers' tries included: those are the rare presses.

Explorers (Config.explore_count) use it to choose their tries: per legal move (d-pad x A x B, 36)
  promise = chance of a jump or a new level within 16 decisions
  score   = promise + 1 / sqrt(1 + tries of that move at this place)
and play the highest score. Untried moves still come first; once every move was tried a few times, the ones the
head thinks MAY lead somewhere new keep being tried (a pipe: Down is usually nothing, but pipes sometimes lead on).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SIZES

EVENTS = ('death', 'damage', 'score', 'level', 'jump', 'new')
HORIZONS = (4, 16)
E = len(EVENTS)
GOOD = (EVENTS.index('level'), EVENTS.index('jump'))    # the promise of a try (16-decision horizon)
FAR = 1 << 20


def move_parts(dev):
    """The 36 tried moves (d-pad x A x B, as Trainer.explore numbers them: j = dpad*4 + A*2 + B) as 5 part tensors."""
    j = torch.arange(36, device=dev)
    z = torch.zeros_like(j)
    return [j // 4, (j // 2) % 2, j % 2, z, z]


class Outcome:
    def __init__(self, core, dev, coef, lr=2.5e-4, hidden=256, chunk=1024):
        self.dev, self.coef, self.chunk = dev, coef, chunk
        self.head = nn.Sequential(nn.Linear(core + sum(SIZES), hidden), nn.ReLU(),
                                  nn.Linear(hidden, E * len(HORIZONS))).to(dev)
        nn.init.zeros_(self.head[2].weight)
        nn.init.constant_(self.head[2].bias, -3.0)       # events are rare: start near p = 0.05
        self.opt = torch.optim.Adam(self.head.parameters(), lr=lr)
        self.moves = move_parts(dev)

    def logits(self, h, parts):
        x = torch.cat([h.float()] + [F.one_hot(parts[k], n).float() for k, n in enumerate(SIZES)], 1)
        return self.head(x)                                  # B, len(HORIZONS) * E (horizon-major)

    @staticmethod
    def events(parts, jump):
        """After a step: env.parts [B, 7] and the places jump flag [B] -> event bits [B] uint8 (all but 'new')."""
        ev = (parts[:, 2] != 0).to(torch.uint8)
        ev |= (parts[:, 3] != 0).to(torch.uint8) << 1
        ev |= (parts[:, 4] > 0).to(torch.uint8) << 2
        ev |= (parts[:, 1] > 0).to(torch.uint8) << 3
        ev |= jump.to(torch.uint8) << 4
        return ev

    @torch.no_grad()
    def labels(self, b_ev, b_rew, b_done):
        """b_ev [T, N] event bits of each step's result, b_rew [T, N], b_done [T, N] -> y [T, N, 2E] (did the event
        happen within h decisions, counting this step, before the game ended) and known [T, N, 2E] (it happened, or
        the whole window lies inside this rollout or the game ended inside it)."""
        T, N = b_ev.shape
        bits = torch.arange(E - 1, device=self.dev)
        ev = torch.cat([((b_ev.long().unsqueeze(-1) >> bits) & 1).bool(), (b_rew > 0).unsqueeze(-1)], -1)  # T, N, E
        nxt = torch.full((T + 1, N, E), FAR, dtype=torch.long, device=self.dev)   # first event step >= t, same game
        end = torch.full((T + 1, N), FAR, dtype=torch.long, device=self.dev)      # first end step >= t
        for t in range(T - 1, -1, -1):
            tt = torch.full_like(end[t], t)
            d = b_done[t]
            end[t] = torch.where(d, tt, end[t + 1])
            later = torch.where(d.unsqueeze(-1), FAR, nxt[t + 1])
            nxt[t] = torch.where(ev[t], t, later)
        ts = torch.arange(T, device=self.dev).view(T, 1, 1)
        ys, ks = [], []
        for h in HORIZONS:
            y = nxt[:T] < ts + h
            ys.append(y)
            ks.append(y | (end[:T].unsqueeze(-1) < ts + h) | (ts + h <= T))
        return torch.cat(ys, -1).float(), torch.cat(ks, -1)

    def loss(self, hs, parts, y, known):
        """Minibatch loss: hs [B, core] (gradient flows into the policy's network), parts, y / known [B, 2E]."""
        lg = self.logits(hs, parts)
        bce = F.binary_cross_entropy_with_logits(lg.float(), y, reduction='none')
        k = known.float()
        return (bce * k).sum() / k.sum().clamp_min(1) * (2 * E)    # per (step, all outputs): ~ nats per step

    @torch.no_grad()
    def promise(self, h):
        """h [n, core] -> [n, 36] chance of a jump or a new level within 16 decisions, per tried move."""
        out = []
        H, j = len(HORIZONS), (len(HORIZONS) - 1) * E
        for i in range(0, h.shape[0], self.chunk):
            hc = h[i:i + self.chunk].float()
            n = hc.shape[0]
            x = hc.unsqueeze(1).expand(n, 36, hc.shape[1]).reshape(n * 36, -1)
            p = torch.sigmoid(self.logits(x, [m.repeat(n) for m in self.moves])).view(n, 36, H * E)
            none = torch.ones(n, 36, device=self.dev)
            for g in GOOD:
                none = none * (1 - p[..., j + g])
            out.append(1 - none)
        return torch.cat(out)

    def state_dict(self):
        return dict(head=self.head.state_dict(), opt=self.opt.state_dict())

    def load_state_dict(self, s):
        self.head.load_state_dict(s['head'])
        self.opt.load_state_dict(s['opt'])
