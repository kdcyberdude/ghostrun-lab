"""Shared 12-game policy (DECISIONS.md 2026-11-09/10), ~6M parameters.

  4 frames (t-3..t), NES colour numbers + per-line emphasis -> RGB through FCEUX's colour table (12 x 224 x 240)
  -> learned 2x2 stride-2 conv (16 ch)
  -> Impoola CNN: 3 ConvSequences 32/64/64 (conv3x3, maxpool 3/2, 2 residual blocks; code from
     github.com/raphajaner/impoola impoola/train/nn.py, MIT), ReLU, global average pooling -> 64
  -> + dropout timeline (8x45 -> 64) + previous move (5 one-hot parts, 19) + mood (one-hot, `moods` wide; 0 = none):
     a random number drawn per episode, held for the whole episode; training rewards each mood for playing in a way
     a judge can tell apart (rl/diversity.py), so different moods = different ways of playing
  -> Linear 512 -> 2 SimBa residual blocks (LayerNorm, MLP 512-2048-512) -> LayerNorm
  -> GRU 512 (memory carried through the episode, reset at episode start)
  -> ordered action heads: d-pad (9) -> A (3) -> B (3) -> Start (2) -> Select (2), each conditioned on the earlier
     choices and masked by the legal parts the env reports
  -> value: HL-Gauss categorical head (101 bins) over per-game normalised returns.
No reward, game id or RAM is an input to the actor (the mood is a random number, not game data).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fceux_palette import RGB

PARTS = [(0, 9), (9, 12), (12, 15), (15, 17), (17, 19)]
SIZES = [9, 3, 3, 2, 2]
MULT = [1, 9, 27, 81, 162]
NEG = -1e9


# ---- Impoola / IMPALA blocks (copied from impoola/train/nn.py, init comments removed) ----
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same')
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block0 = ResidualBlock(out_channels)
        self.res_block1 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        return x


class SimbaBlock(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * mult)
        self.fc2 = nn.Linear(d * mult, d)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(self.ln(x))))


class HLGauss:
    """Histogram loss with Gaussian targets (Imani & White 2018; Farebrother et al. 2024, arXiv 2403.03950)."""

    def __init__(self, vmin=-10.0, vmax=10.0, bins=101, sigma_ratio=0.75, device='cpu'):
        self.edges = torch.linspace(vmin, vmax, bins + 1, device=device)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.sigma = sigma_ratio * (vmax - vmin) / bins
        self.vmin, self.vmax = vmin, vmax

    def target(self, y):
        y = y.clamp(self.vmin, self.vmax).unsqueeze(-1)
        cdf = torch.special.erf((self.edges - y) / (np.sqrt(2.0) * self.sigma))
        p = cdf[..., 1:] - cdf[..., :-1]
        return p / (cdf[..., -1:] - cdf[..., :1])

    def mean(self, logits):
        return (logits.float().softmax(-1) * self.centers).sum(-1)


class Policy(nn.Module):
    def __init__(self, core=512, mlp_mult=4, cnn=(32, 64, 64), stem=16, bins=101, moods=0):
        super().__init__()
        self.register_buffer('palette', torch.tensor(RGB, dtype=torch.float32) / 255.0, persistent=False)
        # (R, G, B, 0) per colour: 4 frames x 4 = 16 channels, the width the tensor-core convolutions want without
        # cuDNN's padding pass; the stem's weight gets a matching zero column, so the result is the same 12-channel conv
        self.register_buffer('palette4', F.pad(self.palette, (0, 1)), persistent=False)
        self.stem = nn.Conv2d(12, stem, kernel_size=2, stride=2)
        self.fused_stem = True   # CUDA + autocast: colour lookup and stem in one kernel (rl/fused.py)
        seqs, c = [], stem
        for o in cnn:
            seqs.append(ConvSequence(c, o))
            c = o
        self.cnn = nn.Sequential(*seqs)
        self.cnn_out = c
        self.timeline = nn.Linear(8 * 45, 64)
        self.moods = moods
        self.inp = nn.Linear(c + 64 + sum(SIZES) + moods, core)
        self.blocks = nn.Sequential(SimbaBlock(core, mlp_mult), SimbaBlock(core, mlp_mult))
        self.ln = nn.LayerNorm(core)
        self.gru = nn.GRUCell(core, core)
        self.core = core
        # ordered heads: part k sees the core and embeddings of parts 0..k-1
        self.heads = nn.ModuleList(nn.Linear(core, n) for n in SIZES)
        self.cond = nn.ModuleList(nn.ModuleList(nn.Embedding(SIZES[j], SIZES[k]) for j in range(k)) for k in range(5))
        self.value = nn.Linear(core, bins)
        nn.init.zeros_(self.value.weight)
        nn.init.zeros_(self.value.bias)
        for h in self.heads:
            nn.init.normal_(h.weight, std=0.01)
            nn.init.zeros_(h.bias)

    # ---- pieces ----
    def rgb(self, obs, emph, dtype=None, pad=False):
        """obs [B,4,224,240] uint8 colour numbers, emph [B,4,224] -> [B,12,224,240] in [0,1] (FCEUX colours), built
        once in `dtype` and already in channels-last memory (the layout the convolutions use). pad: [B,16,...] with a
        zero 4th channel per frame."""
        if pad and obs.is_cuda and dtype is not None:
            from fused import rgb16
            return rgb16(obs.contiguous(), emph.contiguous(), self.palette4.to(dtype)).permute(0, 3, 1, 2)
        idx = obs.long() | (emph.long() << 6).unsqueeze(-1)
        pal = self.palette4 if pad else self.palette
        pal = pal if dtype is None else pal.to(dtype)
        B, F_, H, W = obs.shape
        x = pal[idx.permute(0, 2, 3, 1)]            # B,H,W,4,3 (or 4)
        return x.reshape(B, H, W, -1).permute(0, 3, 1, 2)   # NCHW view, channels-last strides

    def stem_conv(self, obs, emph, dtype):
        """The 2x2 stride-2 stem on the 16-channel padded picture (same values as on the 12-channel one)."""
        w = self.stem.weight                         # 16, 12, 2, 2 (frame-major RGB)
        w4 = F.pad(w.view(w.shape[0], 4, 3, 2, 2), (0, 0, 0, 0, 0, 1)).view(w.shape[0], 16, 2, 2)
        if dtype is not None:
            w4 = w4.to(dtype)
        x = self.rgb(obs, emph, dtype, pad=True)
        return F.conv2d(x, w4.contiguous(memory_format=torch.channels_last),
                        None if self.stem.bias is None else self.stem.bias.to(w4.dtype), stride=2)

    def encode(self, obs, emph, timeline, prev_parts, mood=None):
        """Everything before the GRU, for any batch of frames (no recurrence): -> [B, core]. mood [B] long (moods > 0)."""
        dt = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else None
        if obs.is_cuda and dt == torch.bfloat16 and self.fused_stem:
            from fused import fused_stem
            x = fused_stem(obs, emph, self.palette, self.stem.weight, self.stem.bias, dt)
        else:
            x = self.stem_conv(obs, emph, dt)
        x = self.cnn(x)
        x = F.relu(x).mean((2, 3))
        t = F.relu(self.timeline(timeline.float().flatten(1)))
        p = torch.cat([F.one_hot(prev_parts[k], n) for k, n in enumerate(SIZES)], 1).to(x.dtype)
        extra = [F.one_hot(mood, self.moods).to(x.dtype)] if self.moods else []
        z = self.inp(torch.cat([x, t.to(x.dtype), p] + extra, 1))
        return self.ln(self.blocks(z))

    def recur(self, z, h, starts):
        """z [T,B,core], h [B,core] at the first step, starts [T,B] bool (episode starts at this step: memory reset).
        Returns the GRU outputs [T,B,core] and the last state."""
        out = []
        for t in range(z.shape[0]):
            h = h * (~starts[t]).unsqueeze(1).to(h.dtype)
            h = self.gru(z[t].float(), h)
            out.append(h)
        return torch.stack(out), h

    def head_logits(self, h, k, chosen):
        lg = self.heads[k](h)
        for j in range(k):
            lg = lg + self.cond[k][j](chosen[j])
        return lg

    def act(self, h, masks, greedy=False, generator=None):
        """Sample (or take the most likely) move part by part. h [B,core]; masks [B,19] uint8.
        Returns parts (list of 5 long tensors), total log-prob [B]."""
        chosen, logp = [], torch.zeros(h.shape[0], device=h.device)
        for k, (a, b) in enumerate(PARTS):
            lg = self.head_logits(h, k, chosen).float().masked_fill(masks[:, a:b] == 0, NEG)
            lp = lg.log_softmax(-1)
            c = lg.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1, generator=generator).squeeze(1)
            logp = logp + lp.gather(1, c[:, None]).squeeze(1)
            chosen.append(c)
        return chosen, logp

    def evaluate(self, h, masks, parts):
        """Log-prob and entropy (sum over parts, each conditioned on the taken earlier parts) of given moves."""
        logp = torch.zeros(h.shape[0], device=h.device)
        ent = torch.zeros(h.shape[0], device=h.device)
        for k, (a, b) in enumerate(PARTS):
            lg = self.head_logits(h, k, parts).float().masked_fill(masks[:, a:b] == 0, NEG)
            lp = lg.log_softmax(-1)
            logp = logp + lp.gather(1, parts[k][:, None]).squeeze(1)
            ent = ent - (lp.exp() * lp).sum(-1)
        return logp, ent


def pack(parts):
    return sum(p * m for p, m in zip(parts, MULT)).to(torch.int16)


def unpack(ids):
    ids = ids.long()
    return [ids % 9, (ids // 9) % 3, (ids // 27) % 3, (ids // 81) % 2, (ids // 162) % 2]


def count_params(m):
    return sum(p.numel() for p in m.parameters())
