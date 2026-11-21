"""Curiosity about actions, from the screen only (user design 2026-11-18): "do I already know what this button does
here?". Same code for every game, no RAM, no game knowledge.

Pieces:
  fingerprint  a small network with fixed random weights (never trained) turns the newest picture (grey, 56 x 60,
               as rl/novelty.py) into 64 numbers. Fixed, so it cannot learn to make every picture look the same (which
               would make every guess right), and its numbers do not drift during training.
  heads        K small predictors on the policy's own state (the GRU output h) plus the move taken: they guess the
               fingerprint of the picture after the move (normalised per number by running mean / std). Each head
               learns from its own fixed random half of the consoles and starts from its own random weights, so they
               agree where moves like this one were seen often and disagree where they were not. Their loss also
               trains the policy's network (it replaces the RAM-change prediction as the "understand the game" task).
               The last layer starts at zero: at first every head says 0 and nothing changes for the policy.
  bonus        after each move, how much the K guesses for that move disagree (variance over heads, mean over the
               64 numbers), divided by its long-run average (so `coef` is the average bonus per decision) and capped
               at `clip` times that average. A truly random outcome (flicker, a random enemy) is learned as an average
               by every head, so the heads agree and it pays nothing; plain prediction error would pay for it forever.
               The move into an episode's end pays nothing (the next picture is another episode's start).
The bonus is computed during the rollout from the acting policy's h and move; the targets (fingerprints) too; the
heads learn in the PPO minibatch loop, together with the policy (Trainer.mb_loss).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fceux_palette import RGB
from model import SIZES

H, W, OUT = 56, 60, 64


def _fingerprint_net():
    return nn.Sequential(nn.Conv2d(1, 32, 5, stride=2), nn.LeakyReLU(), nn.Conv2d(32, 64, 3, stride=2), nn.LeakyReLU(),
                         nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(), nn.Flatten(), nn.Linear(64 * 10 * 11, 256),
                         nn.LeakyReLU(), nn.Linear(256, OUT))


class Heads(nn.Module):
    """K independent 2-layer predictors, batched: x [B, d_in] -> [K, B, OUT]."""

    def __init__(self, k, d_in, hidden=256):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(k, d_in, hidden) * (2.0 / d_in) ** 0.5)
        self.b1 = nn.Parameter(torch.zeros(k, hidden))
        self.w2 = nn.Parameter(torch.zeros(k, hidden, OUT))
        self.b2 = nn.Parameter(torch.zeros(k, OUT))

    def forward(self, x):
        y = torch.relu(torch.einsum('bi,kih->kbh', x, self.w1) + self.b1[:, None])
        return torch.einsum('kbh,kho->kbo', y, self.w2) + self.b2[:, None]


class Curiosity:
    def __init__(self, n_envs, core, dev, seed, coef, heads=4, clip=10.0, lr=2.5e-4, ema=0.99, chunk=2048):
        self.dev, self.coef, self.clip, self.ema, self.chunk, self.K = dev, coef, clip, ema, chunk, heads
        g = torch.Generator().manual_seed(seed * 131 + 17)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=g)))
            self.target = _fingerprint_net().to(dev)
            self.heads = Heads(heads, core + sum(SIZES)).to(dev)
        self.target.requires_grad_(False)
        # each head learns from a fixed random half of the consoles (its own "pages of the book")
        self.mask = (torch.rand(heads, n_envs, generator=g) < 0.5).to(dev)
        self.opt = torch.optim.Adam(self.heads.parameters(), lr=lr)
        rgb = torch.tensor(RGB, dtype=torch.float32)
        self.lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).to(dev) / 255.0  # 512 colours
        # running per-number mean / second moment of the fingerprints (count-weighted merge, float64)
        self.n = torch.zeros((), dtype=torch.float64, device=dev)
        self.mean = torch.zeros(OUT, dtype=torch.float64, device=dev)
        self.m2 = torch.zeros(OUT, dtype=torch.float64, device=dev)
        self.avg = torch.zeros((), dtype=torch.float64, device=dev)   # long-run average disagreement (0 = none yet)

    def inputs(self, h, parts):
        return torch.cat([h.float()] + [F.one_hot(parts[k], n).float() for k, n in enumerate(SIZES)], 1)

    @torch.no_grad()
    def fingerprint(self, obs_last, emph_last):
        """obs_last [B, 224, 240] uint8 colour numbers, emph_last [B, 224] -> [B, 64] raw fingerprints."""
        out = []
        for o, e in zip(obs_last.split(self.chunk), emph_last.split(self.chunk)):
            y = self.lum[o.long() | (e.long() << 6).unsqueeze(-1)]          # B, 224, 240 in 0..1
            y = F.avg_pool2d(y.unsqueeze(1), 4) - 0.5                        # B, 1, 56, 60
            out.append(self.target(y))
        return torch.cat(out)

    @torch.no_grad()
    def disagreement(self, h, parts):
        """Policy state h [B, core] and the move about to be played -> [B] raw disagreement of the heads."""
        out = []
        for i in range(0, h.shape[0], self.chunk):
            p = self.heads(self.inputs(h[i:i + self.chunk], [q[i:i + self.chunk] for q in parts]))
            out.append(p.var(0, unbiased=False).mean(-1))
        return torch.cat(out)

    @torch.no_grad()
    def prepare(self, b_fp, b_dis, b_done):
        """After a rollout: update the fingerprint statistics, return the normalised targets [T, N, 64] and the
        bonus [T, N], and update the long-run average disagreement."""
        keep = ~b_done
        x = b_fp[keep].double()
        bn = torch.tensor(float(x.shape[0]), dtype=torch.float64, device=self.dev)
        if x.shape[0]:
            bmean = x.mean(0)
            bm2 = ((x - bmean) ** 2).sum(0)
            tot = self.n + bn
            d = bmean - self.mean
            self.mean = self.mean + d * bn / tot
            self.m2 = self.m2 + bm2 + d * d * self.n * bn / tot
            self.n = tot
        std = (self.m2 / self.n.clamp_min(1)).sqrt().clamp_min(1e-6)
        target = ((b_fp.double() - self.mean) / std).float()
        dis_mean = b_dis[keep].double().mean() if bool(keep.any()) else self.avg
        self.avg = torch.where(self.avg > 0, self.ema * self.avg + (1 - self.ema) * dis_mean, dis_mean)
        bonus = self.coef * (b_dis.double() / self.avg.clamp_min(1e-12)).clamp(max=self.clip).float() * keep.float()
        return target, bonus

    def loss(self, hs, parts, target, keep, e, T):
        """Heads' loss for a PPO minibatch: hs [T*mb, core] (gradient flows into the policy's network), parts,
        target [T*mb, 64], keep [T*mb] bool, e [mb] env indices -> mean squared error over each head's own half."""
        pred = self.heads(self.inputs(hs, parts))                                   # K, B, 64
        m = self.mask[:, e].unsqueeze(1).expand(self.K, T, e.shape[0]).reshape(self.K, -1) & keep.unsqueeze(0)
        err = (pred - target.unsqueeze(0)).pow(2).mean(-1)                          # K, B
        mf = m.float()
        return ((err * mf).sum(1) / mf.sum(1).clamp_min(1)).mean()

    def state_dict(self):
        return dict(target=self.target.state_dict(), heads=self.heads.state_dict(), opt=self.opt.state_dict(),
                    mask=self.mask, n=self.n, mean=self.mean, m2=self.m2, avg=self.avg)

    def load_state_dict(self, s):
        self.target.load_state_dict(s['target'])
        self.heads.load_state_dict(s['heads'])
        self.opt.load_state_dict(s['opt'])
        for k in ('mask', 'n', 'mean', 'm2', 'avg'):
            setattr(self, k, s[k].to(self.dev))
