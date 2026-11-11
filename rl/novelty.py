"""Curiosity bonus for every game, from the screen only: Random Network Distillation (Burda et al. 2018,
arXiv 1810.12894).

A fixed, randomly initialised network (`target`) maps a small grey version of the newest picture to 256 numbers; a
second network (`predictor`) is trained to reproduce them on the pictures the agent has seen. Where the agent has been
often the predictor is accurate; on a screen it has rarely or never seen (a power-up taken, a new room, a new weapon)
the error is large. Only screens more novel than is typical right now are rewarded (an always-positive bonus would just
pay for staying alive): per game, over the current rollout,
    reward += int_coef * max(0, (error - mean(error)) / std(error))
No game knowledge, no RAM: the same code for all 12 games. The picture fed here is the policy's own observation (the
last of its 4 frames), so nothing new enters the actor.

Input: the last frame's NES colour numbers -> FCEUX luminance (with emphasis), 4x4 average pooled to 56 x 60, then
normalised per pixel by running mean / std (all games together) and clipped to [-5, 5] as in the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fceux_palette import RGB

H, W = 56, 60


def _net(out=256, extra=False):
    layers = [nn.Conv2d(1, 32, 5, stride=2), nn.LeakyReLU(), nn.Conv2d(32, 64, 3, stride=2), nn.LeakyReLU(),
              nn.Conv2d(64, 64, 3, stride=1), nn.LeakyReLU(), nn.Flatten(), nn.Linear(64 * 10 * 11, 512)]
    if extra:  # the predictor is a little deeper than the target (as in the paper)
        layers += [nn.ReLU(), nn.Linear(512, 512), nn.ReLU()]
    layers += [nn.Linear(512, out)] if extra else [nn.ReLU(), nn.Linear(512, out)]
    return nn.Sequential(*layers)


class RND:
    def __init__(self, dev, seed, coef=0.5, lr=1e-4, train_frac=0.25, mb=8192):
        self.dev, self.coef, self.train_frac, self.mb = dev, coef, train_frac, mb
        g = torch.Generator().manual_seed(seed * 101 + 9)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            self.target = _net().to(dev)
            self.predictor = _net(extra=True).to(dev)
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        rgb = torch.tensor(RGB, dtype=torch.float32)
        self.lum = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).to(dev)  # 512 colours (with emphasis)
        # running statistics: per-pixel input mean/var, error mean/var (count-weighted, Chan et al. merge)
        self.in_n = torch.zeros((), dtype=torch.float64, device=dev)
        self.in_mean = torch.zeros(H, W, dtype=torch.float64, device=dev)
        self.in_m2 = torch.zeros(H, W, dtype=torch.float64, device=dev)
        self.err_n = torch.zeros((), dtype=torch.float64, device=dev)
        self.err_mean = torch.zeros((), dtype=torch.float64, device=dev)
        self.err_m2 = torch.zeros((), dtype=torch.float64, device=dev)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed * 103 + 11)

    @torch.no_grad()
    def small(self, obs_last, emph_last):
        """obs_last [B, 224, 240] uint8 colour numbers, emph_last [B, 224] -> [B, 56, 60] uint8 luminance."""
        idx = obs_last.long() | (emph_last.long() << 6).unsqueeze(-1)
        y = self.lum[idx]                                        # B, 224, 240 in 0..255
        y = F.avg_pool2d(y.unsqueeze(1), 4).squeeze(1)           # B, 56, 60
        return y.round().clamp(0, 255).to(torch.uint8)

    def _norm(self, x):
        std = (self.in_m2 / self.in_n.clamp_min(1)).sqrt().clamp_min(1e-3)
        return ((x.float() - self.in_mean.float()) / std.float()).clamp(-5, 5).unsqueeze(1)

    def _error(self, x):
        z = self._norm(x)
        return (self.predictor(z) - self.target(z)).pow(2).mean(1)

    @staticmethod
    def _merge(n, mean, m2, bn, bmean, bm2):
        tot = n + bn
        d = bmean - mean
        mean2 = mean + d * bn / tot.clamp_min(1)
        m22 = m2 + bm2 + d * d * n * bn / tot.clamp_min(1)
        return tot, mean2, m22

    @torch.no_grad()
    def update_input_stats(self, x):
        """x [..., 56, 60] uint8 pictures of a rollout (before learning on it)."""
        for c in x.reshape(-1, H, W).split(1 << 17):   # in parts: a whole long rollout in float64 would not fit
            xf = c.double()
            bn = torch.tensor(float(xf.shape[0]), dtype=torch.float64, device=self.dev)
            bmean = xf.mean(0)
            bm2 = ((xf - bmean) ** 2).sum(0)
            self.in_n, self.in_mean, self.in_m2 = self._merge(self.in_n, self.in_mean, self.in_m2, bn, bmean, bm2)

    @torch.no_grad()
    def bonus(self, x, game):
        """x [T, N, 56, 60] uint8, game [N] -> curiosity bonus [T, N] (coef * max(0, error standardised per game over
        this rollout)) and the raw error."""
        T, N = x.shape[:2]
        err = torch.cat([self._error(c) for c in x.reshape(T * N, H, W).split(self.mb)]).view(T, N)
        z = torch.zeros_like(err)
        for gi in game.unique().tolist():
            m = (game == gi).unsqueeze(0).expand_as(err)
            e = err[m]
            z[m] = (e - e.mean()) / e.std().clamp_min(1e-8)
        return self.coef * z.clamp_min(0), err

    def train(self, x):
        """A random `train_frac` of the rollout's pictures, in minibatches of `mb`, one pass."""
        flat = x.reshape(-1, H, W)
        k = int(flat.shape[0] * self.train_frac)
        idx = torch.randperm(flat.shape[0], device=self.dev, generator=self.gen)[:k]
        loss = None
        for i in idx.split(self.mb):
            loss = self._error(flat[i]).mean()
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return float(loss.detach()) if loss is not None else 0.0

    def state_dict(self):
        return dict(target=self.target.state_dict(), predictor=self.predictor.state_dict(), opt=self.opt.state_dict(),
                    gen=self.gen.get_state(), **{k: getattr(self, k) for k in
                                                 ('in_n', 'in_mean', 'in_m2', 'err_n', 'err_mean', 'err_m2')})

    def load_state_dict(self, s):
        self.target.load_state_dict(s['target'])
        self.predictor.load_state_dict(s['predictor'])
        self.opt.load_state_dict(s['opt'])
        self.gen.set_state(s['gen'].cpu())
        for k in ('in_n', 'in_mean', 'in_m2', 'err_n', 'err_mean', 'err_m2'):
            setattr(self, k, s[k].to(self.dev))
