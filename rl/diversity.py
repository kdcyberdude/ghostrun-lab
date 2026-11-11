"""Different ways of playing (user request 2026-11-10): a random mood per episode + a judge + a quality gate.

Every episode draws a random mood z (one of K), held for the whole episode and given to the policy as an input.
A small judge network sees only how the episode's reward has built up so far - the reward split into its parts
(progress, level, death, damage, score, win, stall; rlenv.parts, all 12 games), summed since the episode start, plus
the time in the episode and the game - and guesses z. The policy gets a small bonus per step,
    div_coef * (log q(z | features) - log(1/K)),   clipped below at -log K,
which is large when the judge can tell the moods apart: the moods must differ in what the reward sees (a different
pace, more or fewer kills / score, other deaths or damage, levels at other times), not in invisible button wiggles.
(DIAYN, Eysenbach et al. 2018, with the judge restricted to reward-signal features in the spirit of SIPO, Fu et al.
2023, which measures diversity in state rather than action space.)

Quality gate (SMERL, Kumar et al. 2020; DOMiNO, Zahavy et al. 2023): per game and mood a running (exponentially
decayed, per iteration) mean, variance and count of the finished episodes' task return (the env's reward10 return,
bonus excluded). A mood gets the bonus unless it is clearly worse than the best mood:
    mean + 2 * standard error  <  best mean - margin * max(|best mean|, 1).
The standard error keeps a few unlucky episodes from closing the gate (moods start out playing identically); a mood
that does fall behind learns from the task reward alone until it catches up. Deaths are allowed; only the episode's
overall return counts.

The judge and its inputs are never actor inputs; RAM stays out of the policy (the parts come from the reward).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlenv import GAMES, NUM_PARTS

N_FEAT = NUM_PARTS + 1 + len(GAMES)


def features(cum, ep_t, game):
    """cum [..., NUM_PARTS] reward parts summed since the episode start, ep_t [...] decisions so far, game [...]."""
    return torch.cat([torch.asinh(cum * 4.0), torch.asinh(ep_t.float() / 300.0).unsqueeze(-1),
                      F.one_hot(game, len(GAMES)).float()], -1)


def mood_of(seed: int, env_idx, ep_count, K: int):
    """Deterministic random mood per (run seed, env, episode number): no RNG state, exact on resume."""
    m = (1 << 62) - 1
    x = (env_idx * 0x9E3779B1 + ep_count * 0x85EBCA77 + seed * 0xC2B2AE3D + 0x27D4EB2F) & m
    x = (x ^ (x >> 29)) * 0xBF58476D & m
    x = (x ^ (x >> 31)) * 0x94D049BB & m
    x = x ^ (x >> 27)
    return (x % K).long()


class Judge(nn.Module):
    def __init__(self, K, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K))

    def forward(self, f):
        return self.net(f)


class Diversity:
    """Per-env mood / running reward parts, the judge, the bonus and the gate."""

    def __init__(self, K, N, game_of_env, seed, dev, coef=1e-4, margin=0.1, lr=1e-3, gate_beta=0.98,
                 judge_steps=4, judge_batch=32768):
        self.K, self.N, self.dev, self.seed = K, N, dev, seed
        self.coef, self.margin, self.beta = coef, margin, gate_beta
        self.judge_steps, self.judge_batch = judge_steps, judge_batch
        self.game = game_of_env
        self.env_idx = torch.arange(N, device=dev)
        self.ep_count = torch.zeros(N, dtype=torch.long, device=dev)
        self.mood = mood_of(seed, self.env_idx, self.ep_count, K)
        self.cum = torch.zeros(N, NUM_PARTS, device=dev)
        self.ep_t = torch.zeros(N, dtype=torch.long, device=dev)
        G = len(GAMES)
        # decayed sums of finished episodes per game and mood: count, return, return^2 (task return only)
        self.En = torch.zeros(G, K, dtype=torch.float64, device=dev)
        self.Es = torch.zeros(G, K, dtype=torch.float64, device=dev)
        self.Eq = torch.zeros(G, K, dtype=torch.float64, device=dev)
        self.judge = Judge(K).to(dev)
        self.opt = torch.optim.Adam(self.judge.parameters(), lr=lr)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed * 31 + 7)

    # ---- per step (rollout) ----
    def after_step(self, sl, parts, done, feat_out):
        """Env slice `sl` just stepped: add its reward parts, write the features after this step into feat_out
        ([n, N_FEAT]); a finished episode's next one gets a fresh mood and empty sums."""
        self.cum[sl] += parts
        self.ep_t[sl] += 1
        feat_out.copy_(features(self.cum[sl], self.ep_t[sl], self.game[sl]))
        d = done.bool()
        keep = (~d).unsqueeze(1).float()
        self.cum[sl] *= keep
        self.ep_t[sl] *= (~d).long()
        self.ep_count[sl] += d.long()
        self.mood[sl] = torch.where(d, mood_of(self.seed, self.env_idx[sl], self.ep_count[sl], self.K), self.mood[sl])

    # ---- per rollout (learner) ----
    @property
    def R(self):
        """[G, K] running mean task return per game and mood (0 where none finished yet)."""
        return self.Es / self.En.clamp_min(1e-9)

    def gate(self):
        """[G, K] 1.0 where the mood may earn the bonus."""
        seen = self.En > 0.5
        mean = self.R
        var = (self.Eq / self.En.clamp_min(1e-9) - mean ** 2).clamp_min(0)
        se = (var / self.En.clamp_min(1.0)).sqrt()
        best = torch.where(seen, mean, torch.full_like(mean, -1e30)).max(1, keepdim=True).values
        tol = self.margin * best.abs().clamp_min(1.0)
        return (~seen | (mean + 2 * se >= best - tol)).float()

    @torch.no_grad()
    def bonus(self, feat, mood):
        """feat [T, N, N_FEAT], mood [T, N] -> (bonus [T, N] incl. gate, stats)."""
        logits = self.judge(feat)
        logq = logits.log_softmax(-1).gather(-1, mood.unsqueeze(-1)).squeeze(-1)
        raw = (logq + math.log(self.K)).clamp_min(-math.log(self.K))
        g = self.gate()[self.game.unsqueeze(0).expand_as(mood), mood].float()
        acc = (logits.argmax(-1) == mood).float()
        return self.coef * raw * g, dict(acc=acc, raw=raw, gate=g)

    def train_judge(self, feat, mood):
        f, z = feat.reshape(-1, feat.shape[-1]), mood.reshape(-1)
        loss = None
        for _ in range(self.judge_steps):
            i = torch.randint(0, f.shape[0], (min(self.judge_batch, f.shape[0]),), device=self.dev, generator=self.gen)
            loss = F.cross_entropy(self.judge(f[i]), z[i])
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
        return float(loss.detach()) if loss is not None else 0.0

    def update_gate(self, done, mood, ret):
        """done [T, N] bool, mood [T, N], ret [T, N] (finished episodes' task return where done)."""
        G = len(GAMES)
        idx = (self.game.unsqueeze(0).expand_as(mood) * self.K + mood)[done]
        r = ret[done].double()
        n = torch.zeros(G * self.K, dtype=torch.float64, device=self.dev).index_add_(0, idx, torch.ones_like(r))
        s = torch.zeros(G * self.K, dtype=torch.float64, device=self.dev).index_add_(0, idx, r)
        q = torch.zeros(G * self.K, dtype=torch.float64, device=self.dev).index_add_(0, idx, r * r)
        b = self.beta
        self.En = b * self.En + n.view(G, self.K)
        self.Es = b * self.Es + s.view(G, self.K)
        self.Eq = b * self.Eq + q.view(G, self.K)

    # ---- checkpoint ----
    def state_dict(self):
        return dict(ep_count=self.ep_count, mood=self.mood, cum=self.cum, ep_t=self.ep_t, En=self.En, Es=self.Es, Eq=self.Eq,
                    judge=self.judge.state_dict(), opt=self.opt.state_dict(), gen=self.gen.get_state())

    def load_state_dict(self, s):
        for k in ('ep_count', 'mood', 'cum', 'ep_t', 'En', 'Es', 'Eq'):
            getattr(self, k).copy_(s[k])
        self.judge.load_state_dict(s['judge'])
        self.opt.load_state_dict(s['opt'])
        self.gen.set_state(s['gen'].cpu())
