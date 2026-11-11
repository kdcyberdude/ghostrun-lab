"""PPO for the shared 12-game NES policy (rl/model.py) on the GPU environment (rl/rlenv.py).

Recurrent PPO with stored GRU state: each iteration collects T decisions from every env; the learner replays whole
T-step sequences of a subset of envs from the GRU state stored at the rollout start (memory reset at episode
starts). GAE with discount 0.995 and lambda 0.95; every reward10 end is terminal. Clipped policy loss and per-minibatch
advantage normalisation from github.com/raphajaner/impoola (impoola/train/ppo_criterion.py); the value is an HL-Gauss
categorical head over per-game normalised returns (running mean/std per game, all ranks combined).

Single GPU:   python rl/train.py --run runs/rl/NAME
Two GPUs:     torchrun --nproc_per_node 2 rl/train.py --run runs/rl/NAME     (data parallel, envs split per rank)
Resume:       the same command; the newest checkpoint in --run is loaded (exact: envs, RNG, optimizer, counters).
Stop:         create --run/STOP, send SIGTERM/SIGINT, or reach --time-limit: a checkpoint is written first.

--lag 1 (user-approved 2026-11-10): the next rollout is collected while the learner trains on the previous one, so
every update learns from data of the policy one update older (importance ratio and clip are against that behaviour
policy, as in APPO). Before a checkpoint the pending rollout is learned alone, so checkpoints never hold unlearned
data and resume stays exact. --lag 0 is plain sequential PPO.
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Policy, HLGauss, pack, unpack, count_params, SIZES  # noqa: E402
from rlenv import RlEnv, GAMES, END_NAMES  # noqa: E402
from diversity import Diversity, N_FEAT  # noqa: E402
from novelty import RND, H as RND_H, W as RND_W  # noqa: E402
from level_plan import LevelPlan  # noqa: E402
from places import Places  # noqa: E402


@dataclasses.dataclass
class Config:
    run: str = 'runs/rl/dev'
    envs_per_game: int = 21           # per rank
    games: str = ','.join(GAMES)
    steps: int = 128                  # T, decisions per env per iteration (= training sequence length)
    epochs: int = 2
    minibatch_envs: int = 16          # envs (x T steps) per learner minibatch
    lr: float = 2.5e-4
    adam_eps: float = 1e-5
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    ret_beta: float = 0.995           # EMA of per-game return mean / second moment
    moods: int = 0                    # K > 0: random mood per episode (policy input) + judge bonus + quality gate
    div_coef: float = 1e-4            # diversity bonus per step: div_coef * (log q(mood | reward parts) + log K)
    div_margin: float = 0.1           # a mood earns the bonus while its average return >= best - margin*max(|best|,1)
    int_coef: float = 0.0             # > 0: curiosity bonus (rl/novelty.py, RND on the newest picture): reward +=
                                      # int_coef * max(0, prediction error standardised per game over the rollout)
    mood_kick: float = 0.0            # > 0: once, add N(0, mood_kick) to the policy's mood input weights (moods start
                                      # out playing alike after zero-widening; the judge can only reward differences)
    level_starts: float = 0.0         # > 0: new training episodes start by difficulty over power-on and the saved
                                      # level starts in proportion to each level's failure rate (rlenv_core.cuh,
                                      # rl/level_plan.py; every game's reward10 'level' event saves one per level);
                                      # 1 = failure rate alone, less scales the saved share down.
                                      # Test plays: power-on.
    level_cap: int = 64               # levels 0..level_cap-1 of each game can be saved (always collected)
    places: int = 0                   # > 0: Go-Explore places inside levels (rl/places.py): this many place slots per game
    only_levels: str = ''             # e.g. '15,27,30': every episode starts at one of these saved levels (by failure)
    episode_level_end: int = -1       # optional native level-clear episode boundary; no extra reward
    place_k: int = 4                  # new places staged per game per step
    place_per_level: int = 128        # place slots one level may hold
    adv_norm: str = 'game'            # 'game': advantages scaled to mean 0 / std 1 per game over the rollout before the
                                      # minibatch normalisation (games' return scales differ ~60x); 'batch': minibatch only
    core: int = 512
    seed: int = 1
    sticky: int = 1
    idle_max: int = 30
    dropout: int = 1
    precision: str = 'auto'           # auto: bf16 where supported, else fp16 with loss scaling
    time_limit: float = 0.0           # seconds; 0 = none
    ckpt_minutes: float = 10.0
    ckpt_iters: int = 0               # also checkpoint every K iterations (0 = off; for resume tests)
    eval_every_minutes: float = 30.0
    eval_steps: int = 100000          # safety cap: an evaluation plays until every episode ends (game over/won/stall)
    eval_threads: int = 16            # CPU threads of the background evaluation process
    total_iters: int = 0              # 0 = until stopped
    deterministic: int = 0            # 1: deterministic cuDNN (resume test)
    log_every: int = 1
    device: str = 'cuda'             # 'cpu' only for small logic tests
    compile: int = 1                  # torch.compile the encoder (CUDA)
    compile_mode: str = 'default'     # torch.compile mode for the learner step, e.g. max-autotune-no-cudagraphs
    obs_host: int = 1                 # keep the rollout pictures in pinned host RAM (CUDA; frees VRAM for many envs)
    infer_chunk: int = 2048           # rollout encoder batch (bounds activation memory)
    obs_slots: int = 2                # device picture slots with --obs-host (1: half the VRAM, the env waits for each copy-out)
    obs_pack6: int = 0                # lossless 6-bit palette packing in pinned rollout storage; actor inputs unchanged
    lag: int = 0                      # 1: collect the next rollout during learning (single GPU)
    per_game_rollout: int = 1         # CUDA: each game plays its rollout on its own stream at its own pace
    rollout_split: int = 1            # ... and each game's envs in this many parts with their own streams, so one
                                      # part's emulation overlaps another part's inference (env results per move
                                      # are identical; the sampled moves use the shared generator in unit order)


def parse() -> Config:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(Config):
        ap.add_argument('--' + f.name.replace('_', '-'), type=type(f.default), default=f.default)
    return Config(**vars(ap.parse_args()))


class Stop:
    flag = False

    @staticmethod
    def handler(*_):
        Stop.flag = True


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


FRAME_BYTES = 4 * 224 * 240
SLOW_GAMES = ('punchout',)  # longest emulation per decision on the GPU (measured 2026-11-10)


class DeviceObs:
    """Rollout pictures on the device, one region per rollout buffer: region r slot t = the pictures the decision at
    step t saw (slot T: the next ones)."""

    def __init__(self, T, N, dev, regions):
        self.T = T
        self.buf = [torch.zeros(T + 1, N, 4, 224, 240, dtype=torch.uint8, device=dev) for _ in range(regions)]

    def slot(self, r, t):
        return self.buf[r][t]

    def before_write(self, t):
        pass

    def saved(self, r, t):
        pass

    def handoff(self, src, dst):
        self.buf[dst][0].copy_(self.buf[src][self.T])

    def before_write_game(self, gi, t):
        pass

    def saved_game(self, r, t, gi, rows):
        pass

    def epoch(self, r, perm, mb):
        self._r, self._perm, self._mb = r, perm, mb

    def batch(self, i):
        e = self._perm[i * self._mb:(i + 1) * self._mb]
        return self.buf[self._r][:self.T, e].flatten(0, 1)

    def done_with(self, i):
        pass


class HostObs:
    """Rollout pictures in pinned host RAM (per region: T+1 slots of N envs). The env draws into one of two device
    slots (step t in slot t % 2); each is copied out asynchronously on a side stream. The learner's minibatches (whole
    T-step sequences of `mb` envs) are gathered back over PCIe by a kernel on another stream, one minibatch ahead."""

    def __init__(self, lib, T, N, dev, mb, regions, nslots=2, pack6=False):
        assert nslots in (1, 2) and T % nslots == 0, 'steps must be even (the next rollout starts in slot 0)'
        self.nslots = nslots
        self.lib, self.T, self.N, self.dev = lib, T, N, dev
        self.pack6 = bool(pack6)
        self.row_bytes = FRAME_BYTES * 3 // 4 if self.pack6 else FRAME_BYTES
        self.region_bytes = (T + 1) * N * self.row_bytes
        self.ptr = lib.rl_host_alloc(self.region_bytes * regions)
        if not self.ptr:
            raise RuntimeError(f'pinned host allocation of {self.region_bytes * regions / 2**30:.1f} GiB failed')
        self.slots = [torch.zeros(N, 4, 224, 240, dtype=torch.uint8, device=dev) for _ in range(nslots)]
        self.packed = torch.empty(N, self.row_bytes, dtype=torch.uint8, device=dev) if self.pack6 else None
        self.d2h = torch.cuda.Stream(dev)
        self.gather = torch.cuda.Stream(dev)
        self.copied = [torch.cuda.Event() for _ in range(2)]
        self.mbuf = [torch.empty(T * mb, 4, 224, 240, dtype=torch.uint8, device=dev) for _ in range(2)]
        self.pmbuf = [torch.empty(T * mb, self.row_bytes, dtype=torch.uint8, device=dev)
                      for _ in range(2)] if self.pack6 else self.mbuf
        self.fetched = [torch.cuda.Event() for _ in range(2)]
        self.used = [torch.cuda.Event() for _ in range(2)]
        self.d2h_g, self.copied_g = [], []

    def __del__(self):
        if getattr(self, 'ptr', None):
            torch.cuda.synchronize(self.dev)
            self.lib.rl_host_free(self.ptr)
            self.ptr = None

    def slot(self, r, t):
        return self.slots[t % self.nslots]

    def before_write(self, t):
        """The env is about to draw step t+1 into its slot: that slot's previous copy-out must have finished."""
        torch.cuda.current_stream(self.dev).wait_event(self.copied[(t + 1) % self.nslots])

    def saved(self, r, t):
        """Slot t % nslots holds step t; keep every frame, optionally packed without loss."""
        current = torch.cuda.current_stream(self.dev)
        src = self.slots[t % self.nslots]
        if self.pack6:
            current.wait_event(self.copied[(t - 1) % 2])  # one packed staging row: do not overwrite a live copy
            if self.lib.rl_pack6(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(self.packed.data_ptr()),
                                 self.N * FRAME_BYTES, ctypes.c_void_p(current.cuda_stream)) != 0:
                raise RuntimeError(self.lib.rlenv_error().decode())
            src = self.packed
        self.d2h.wait_stream(current)
        dst = self.ptr + r * self.region_bytes + t * self.N * self.row_bytes
        if self.lib.rl_memcpy_async(ctypes.c_void_p(dst), ctypes.c_void_p(src.data_ptr()), self.N * self.row_bytes,
                                    ctypes.c_void_p(self.d2h.cuda_stream)) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())
        self.copied[t % self.nslots].record(self.d2h)

    def handoff(self, src, dst):
        pass  # step T is in device slot T % 2 == 0, where the next rollout starts (host row 0 is copied then)

    # per-game rollout: each game's rows are copied out on that game's own copy stream
    def _game_streams(self, gi):
        while len(self.d2h_g) <= gi:
            self.d2h_g.append(torch.cuda.Stream(self.dev))
            self.copied_g.append([torch.cuda.Event(), torch.cuda.Event()])
        return self.d2h_g[gi], self.copied_g[gi]

    def before_write_game(self, gi, t):
        torch.cuda.current_stream(self.dev).wait_event(self._game_streams(gi)[1][(t + 1) % self.nslots])

    def saved_game(self, r, t, gi, rows):
        st, ev = self._game_streams(gi)
        current = torch.cuda.current_stream(self.dev)
        src = self.slots[t % self.nslots][rows]
        if self.pack6:
            current.wait_event(ev[(t - 1) % 2])
            packed = self.packed[rows]
            if self.lib.rl_pack6(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(packed.data_ptr()),
                                 src.numel(), ctypes.c_void_p(current.cuda_stream)) != 0:
                raise RuntimeError(self.lib.rlenv_error().decode())
            src = packed
        st.wait_stream(current)
        dst = self.ptr + r * self.region_bytes + (t * self.N + rows.start) * self.row_bytes
        if self.lib.rl_memcpy_async(ctypes.c_void_p(dst), ctypes.c_void_p(src.data_ptr()),
                                    (rows.stop - rows.start) * self.row_bytes, ctypes.c_void_p(st.cuda_stream)) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())
        ev[t % self.nslots].record(st)

    def epoch(self, r, perm, mb):
        """This epoch's minibatches: whole T-step sequences of mb envs from region r; start fetching the first.
        (All copy-outs of region r finished: the trainer synchronises the device between rollout and learning.)
        The gather uses the copy engines (one 2-D copy per env), so it runs beside the learner's kernels."""
        self.base = self.ptr + r * self.region_bytes
        self.perm_host = np.ascontiguousarray(perm.cpu().numpy().astype(np.int64))
        self.mb = mb
        self.nmb = len(self.perm_host) // mb
        self.gather.wait_stream(torch.cuda.current_stream(self.dev))
        self._fetch(0)

    def _fetch(self, i):
        k = i % 2
        self.gather.wait_event(self.used[k])
        envs = self.perm_host[i * self.mb:(i + 1) * self.mb]
        if self.lib.rl_gather_envs(ctypes.c_void_p(self.pmbuf[k].data_ptr()), ctypes.c_void_p(self.base),
                                   envs.ctypes.data_as(ctypes.c_void_p), self.mb, self.T, self.N, self.row_bytes,
                                   ctypes.c_void_p(self.gather.cuda_stream)) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())
        if self.pack6 and self.lib.rl_unpack6(ctypes.c_void_p(self.pmbuf[k].data_ptr()),
                                             ctypes.c_void_p(self.mbuf[k].data_ptr()), self.T * self.mb * FRAME_BYTES,
                                             ctypes.c_void_p(self.gather.cuda_stream)) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())
        self.fetched[k].record(self.gather)

    def batch(self, i):
        if i + 1 < self.nmb:
            self._fetch(i + 1)
        torch.cuda.current_stream(self.dev).wait_event(self.fetched[i % 2])
        return self.mbuf[i % 2]

    def done_with(self, i):
        self.used[i % 2].record(torch.cuda.current_stream(self.dev))


class Buffers:
    """Everything one rollout stores except the pictures (those live in region `r` of the picture store)."""

    def __init__(self, r, T, N, core, dev, moods=0, rnd=False):
        self.r = r
        self.b_small = torch.zeros(T, N, RND_H, RND_W, dtype=torch.uint8, device=dev) if rnd else None  # curiosity input
        self.b_mood = torch.zeros(T, N, dtype=torch.long, device=dev) if moods else None   # mood used at each step
        self.b_feat = torch.zeros(T, N, N_FEAT, device=dev) if moods else None           # judge input after each step
        self.b_emph = torch.zeros(T + 1, N, 4, 224, dtype=torch.uint8, device=dev)
        self.b_masks = torch.zeros(T + 1, N, 19, dtype=torch.uint8, device=dev)
        self.b_tl = torch.zeros(T + 1, N, 8, 45, dtype=torch.uint8, device=dev)
        self.b_prev = torch.zeros(T + 1, 5, N, dtype=torch.long, device=dev)  # previous move parts
        self.b_start = torch.zeros(T + 1, N, dtype=torch.bool, device=dev)    # episode starts at this step
        self.b_parts = torch.zeros(T, 5, N, dtype=torch.long, device=dev)
        self.b_logp = torch.zeros(T, N, device=dev)
        self.b_val = torch.zeros(T + 1, N, device=dev)
        self.b_rew = torch.zeros(T, N, device=dev)
        self.b_done = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.b_info = torch.zeros(T, N, 8, device=dev)
        self.b_slvl = torch.zeros(T, N, dtype=torch.int32, device=dev)  # finished episode's start level (-1 power-on)
        self.b_lnow = torch.zeros(T, N, dtype=torch.int32, device=dev)  # level after each decision (-1 unknown)
        self.b_lev = torch.zeros(T, N, dtype=torch.bool, device=dev)    # a new level was reached at this decision
        self.b_rollback = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.h0 = torch.zeros(N, core, device=dev)

    FIRST = ('b_emph', 'b_masks', 'b_tl', 'b_prev', 'b_start')  # step-0 entries carried from the previous rollout

    def handoff_from(self, other, T):
        for k in self.FIRST:
            getattr(self, k)[0].copy_(getattr(other, k)[T])


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ddp = int(os.environ.get('WORLD_SIZE', '1')) > 1
        if self.ddp:
            dist.init_process_group('nccl')
            self.rank, self.world = dist.get_rank(), dist.get_world_size()
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        else:
            self.rank, self.world = 0, 1
        assert not (cfg.lag and self.ddp), '--lag 1 is single-GPU'
        self.cuda = cfg.device == 'cuda'
        self.dev = torch.device('cuda', torch.cuda.current_device()) if self.cuda else torch.device('cpu')
        torch.backends.cudnn.benchmark = not cfg.deterministic
        torch.backends.cudnn.deterministic = bool(cfg.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed * 1000 + self.rank)
        bf16 = self.cuda and torch.cuda.get_device_capability(self.dev)[0] >= 8
        self.amp_dtype = {'auto': (torch.bfloat16 if bf16 else torch.float16) if self.cuda else torch.float32, 'bf16': torch.bfloat16,
                          'fp16': torch.float16, 'fp32': torch.float32}[cfg.precision]
        self.run = Path(cfg.run)
        self.run.mkdir(parents=True, exist_ok=True)
        games = cfg.games.split(',')
        units = max(1, cfg.rollout_split)
        cap = cfg.level_cap + (cfg.places + cfg.steps * units * cfg.place_k if cfg.places > 0 else 0)
        self.env = RlEnv({g: cfg.envs_per_game for g in games}, device=self.dev,
                         seed=cfg.seed * 7919 + self.rank, sticky=bool(cfg.sticky), idle_max=cfg.idle_max,
                         dropout=bool(cfg.dropout), level_cap=cap, level_slots=cfg.level_cap)
        self.level_plan = LevelPlan(self.env.games, cfg.level_cap, 0.98 ** (cfg.steps / 16)) if cfg.level_cap > 0 else None
        self.N, self.T = self.env.n, cfg.steps
        if cfg.episode_level_end >= 0:
            self.env.set_episode_level_end(cfg.episode_level_end)
        self.places = Places(len(self.env.games), self.N, cfg.level_cap, cfg.places, cfg.place_k, cfg.steps, units,
                             cfg.place_per_level, self.dev, cfg.seed * 1000 + self.rank) if cfg.places > 0 else None
        self.local_game = torch.tensor(sum(([gi] * c for gi, c in enumerate(self.env.counts)), []))
        self.game_of_env = self.env.game_of_env
        compile_ = cfg.compile and self.cuda
        self.model = Policy(core=cfg.core, moods=cfg.moods).to(self.dev)
        self.div = Diversity(cfg.moods, self.N, self.game_of_env, cfg.seed * 1000 + self.rank, self.dev,
                             coef=cfg.div_coef, margin=cfg.div_margin) if cfg.moods else None
        self.rnd = RND(self.dev, cfg.seed * 1000 + self.rank, coef=cfg.int_coef) if cfg.int_coef > 0 else None
        if self.cuda:
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.ddp:  # same initial weights on every rank
            for p in self.model.parameters():
                dist.broadcast(p.data, 0)
        # the behaviour policy: the model itself, or (lag) a copy refreshed before each rollout
        if cfg.lag:
            self.actor = Policy(core=cfg.core, moods=cfg.moods).to(self.dev)
            if self.cuda:
                self.actor = self.actor.to(memory_format=torch.channels_last)
            self.actor.requires_grad_(False)
        else:
            self.actor = self.model
        self.actor_enc = torch.compile(self.actor.encode, dynamic=False, mode=cfg.compile_mode) if compile_ else self.actor.encode
        self.mb_loss_c = torch.compile(self.mb_loss, dynamic=False, mode=cfg.compile_mode) if compile_ else self.mb_loss
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=cfg.adam_eps, fused=self.cuda)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp_dtype == torch.float16 and self.cuda)
        self.hl = HLGauss(device=self.dev)
        G = len(GAMES)
        self.ret_mean = torch.zeros(G, device=self.dev, dtype=torch.float64)
        self.ret_sq = torch.ones(G, device=self.dev, dtype=torch.float64)
        self.ret_seen = torch.zeros(G, device=self.dev, dtype=torch.float64)
        N, T, d = self.N, self.T, self.dev
        regions = 2 if cfg.lag else 1
        mb = min(cfg.minibatch_envs, N)
        self.obs = HostObs(self.env.lib, T, N, d, mb, regions, cfg.obs_slots, cfg.obs_pack6) if (cfg.obs_host and self.cuda) else DeviceObs(T, N, d, regions)
        self.bufs = [Buffers(r, T, N, cfg.core, d, cfg.moods, cfg.int_coef > 0) for r in range(regions)]
        self.nb = 0                  # buffer the next rollout goes into; its step 0 holds the current env outputs
        self.last = None             # buffer collected last (its step T = the current env outputs)
        self.env.drop_obs_buffers()  # the trainer's own picture slots are bound instead
        self.h = torch.zeros(N, cfg.core, device=d)
        self.it = 0
        self.decisions = 0
        self.episodes = []           # finished-episode records since the last log
        self.t_last_ckpt = time.time()
        self.t_last_eval = time.time()
        self.t_start = time.time()
        # the rollout's own sampling generator (the learner's randperm uses the default one): with --lag they run
        # in different threads, and separate generators keep both sequences reproducible
        self.gen_roll = torch.Generator(device=d)
        self.gen_roll.manual_seed(cfg.seed * 1000 + self.rank + 500009)
        if self.cuda:
            self.roll_stream = torch.cuda.Stream(d, priority=-1)  # env + inference first; the learner fills the rest
        # power from the GPU's energy counter (exact average over each iteration, not spot samples)
        self.nvml = None
        if self.cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device() if self.dev.index is None
                                                              else self.dev.index)
            except Exception:  # noqa: BLE001 (power logging is optional)
                self.nvml = None
        if self.rank == 0:
            name = torch.cuda.get_device_name(self.dev) if self.cuda else 'cpu'
            print(f'[rl] world {self.world} envs/rank {N} games {len(games)} params {count_params(self.model)/1e6:.2f}M '
                  f'amp {self.amp_dtype} device {name} lag {cfg.lag}', flush=True)

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.dev)

    def autocast(self):
        return torch.autocast(self.dev.type, dtype=self.amp_dtype, enabled=self.amp_dtype != torch.float32)

    # ---------------- value normalisation ----------------
    def norm_stats(self):
        var = (self.ret_sq - self.ret_mean ** 2).clamp_min(1e-4)
        return self.ret_mean.float(), var.sqrt().float()

    def update_stats(self, returns):
        g = self.game_of_env.expand_as(returns).reshape(-1)
        r = returns.reshape(-1).double()
        G = len(GAMES)
        s = torch.zeros(3, G, device=self.dev, dtype=torch.float64)
        s[0].index_add_(0, g, torch.ones_like(r))
        s[1].index_add_(0, g, r)
        s[2].index_add_(0, g, r * r)
        if self.ddp:
            dist.all_reduce(s)
        have = s[0] > 0
        m, q = s[1] / s[0].clamp_min(1), s[2] / s[0].clamp_min(1)
        first = have & (self.ret_seen == 0)
        b = self.cfg.ret_beta
        self.ret_mean = torch.where(first, m, torch.where(have, b * self.ret_mean + (1 - b) * m, self.ret_mean))
        self.ret_sq = torch.where(first, q, torch.where(have, b * self.ret_sq + (1 - b) * q, self.ret_sq))
        self.ret_seen += s[0]

    # ---------------- rollout ----------------
    def encode_chunks(self, obs, emph, tl, prev, mood=None):
        C = self.cfg.infer_chunk
        return torch.cat([self.actor_enc(obs[a:a + C], emph[a:a + C], tl[a:a + C], [p[a:a + C] for p in prev],
                                         None if mood is None else mood[a:a + C])
                          for a in range(0, obs.shape[0], C)])

    def start(self):
        b = self.bufs[self.nb]
        self.env.bind_obs(self.obs.slot(b.r, 0), b.b_emph[0])
        self.env.start()
        b.b_masks[0].copy_(self.env.masks)
        b.b_tl[0].copy_(self.env.timeline)
        b.b_start[0].fill_(True)

    def refresh_actor(self):
        """Behaviour policy for the next rollout := the current weights and value statistics."""
        if self.actor is not self.model:
            with torch.no_grad():
                for pa, p in zip(self.actor.parameters(), self.model.parameters()):
                    pa.copy_(p)
        self.actor_stats = tuple(x.clone() for x in self.norm_stats())

    def collect(self):
        """One rollout into buffer self.nb (call refresh_actor first). Returns that buffer."""
        b = self.bufs[self.nb]
        self.apply_level_plan()
        if self.last is not None and self.last is not b:
            b.handoff_from(self.last, self.T)
            self.obs.handoff(self.last.r, b.r)
        elif self.last is b:
            self.obs.handoff(b.r, b.r)
            for k in Buffers.FIRST:
                getattr(b, k)[0].copy_(getattr(b, k)[self.T])
        t0 = time.time()
        if self.cfg.per_game_rollout and self.cuda:
            self.rollout_per_game(b)
        else:
            self.rollout(b)
        self.sync()
        if self.places is not None:
            self.places.end_rollout(self.env)
        if self.cuda and not self.cfg.lag:
            # the per-game streams' cached activation memory would otherwise stay reserved through learning (and the
            # learner's through the next rollout): hand it back between the phases so their peaks do not add up
            torch.cuda.empty_cache()
        self.rollout_s = time.time() - t0
        self.last = b
        if self.cfg.lag:
            self.nb = 1 - self.nb
        return b

    @torch.no_grad()
    def rollout(self, b):
        m, env, T = self.actor, self.env, self.T
        mu, sd = self.actor_stats
        g = self.game_of_env
        b.h0.copy_(self.h)
        h = self.h
        self.obs.saved(b.r, 0)
        dv = self.div
        for t in range(T):
            if dv is not None:
                b.b_mood[t].copy_(dv.mood)
            with self.autocast():
                z = self.encode_chunks(self.obs.slot(b.r, t), b.b_emph[t], b.b_tl[t], list(b.b_prev[t]),
                                       None if dv is None else b.b_mood[t])
            h = h * (~b.b_start[t]).unsqueeze(1).float()
            h = m.gru(z.float(), h)
            parts, logp = m.act(h, b.b_masks[t], generator=self.gen_roll)
            b.b_val[t] = mu[g] + sd[g] * self.hl.mean(m.value(h))
            for k in range(5):
                b.b_parts[t, k] = parts[k]
            b.b_logp[t] = logp
            env.actions.copy_(pack(parts))
            self.obs.before_write(t)
            env.bind_obs(self.obs.slot(b.r, t + 1), b.b_emph[t + 1])
            env.step()
            self.obs.saved(b.r, t + 1)
            b.b_rew[t] = env.reward
            b.b_rollback[t] = (env.events & 64) != 0
            done = env.done.bool()
            b.b_done[t] = done
            if dv is not None:
                dv.after_step(slice(None), env.parts, done, b.b_feat[t])
            if self.rnd is not None:
                b.b_small[t] = self.rnd.small(self.obs.slot(b.r, t + 1)[:, 3], b.b_emph[t + 1, :, 3])
            if self.places is not None:
                for gi, gsl in enumerate(env.game_slices()):
                    self.places.step(env, t, gi, 0, 0, gsl, self.obs.slot(b.r, t + 1)[gsl, 3], b.b_emph[t + 1, gsl, 3],
                                     env.level_now[gsl], env.progress_now[gsl], done[gsl])
            b.b_masks[t + 1].copy_(env.masks)
            b.b_tl[t + 1].copy_(env.timeline)
            b.b_start[t + 1] = done
            b.b_prev[t + 1] = b.b_parts[t] * (~done).long()
            b.b_info[t] = env.info
            b.b_slvl[t] = env.start_level
            if env.level_cap > 0:
                b.b_lnow[t] = env.level_now
                b.b_lev[t] = env.parts[:, 1] > 0
        with self.autocast():
            z = self.encode_chunks(self.obs.slot(b.r, T), b.b_emph[T], b.b_tl[T], list(b.b_prev[T]),
                                   None if dv is None else dv.mood)
        hT = m.gru(z.float(), h * (~b.b_start[T]).unsqueeze(1).float())
        b.b_val[T] = mu[g] + sd[g] * self.hl.mean(m.value(hT))
        self.h = h  # state after the last step (before the reset of step T's start, applied in the next rollout)
        self.decisions += T * self.N * self.world
        self.collect_episodes(b)

    @torch.no_grad()
    def rollout_per_game(self, b):
        """The same rollout as rollout(), but every game runs its own chain (inference -> move -> its env step ->
        copy-out) on its own CUDA stream, with no host synchronisation: fast games finish early and their inference
        overlaps the slow games' emulation instead of everyone waiting for the slowest game at every step."""
        m, env, T = self.actor, self.env, self.T
        mu, sd = self.actor_stats
        g = self.game_of_env
        main = torch.cuda.current_stream(self.dev)
        b.h0.copy_(self.h)
        if not hasattr(self, 'units'):
            # units: (game index, first env within the game, count, global rows); a game with enough envs is split
            # into `rollout_split` parts so their emulation and inference interleave on separate streams
            self.units = []
            for gi, gsl in enumerate(env.game_slices()):
                n = gsl.stop - gsl.start
                k = max(1, min(self.cfg.rollout_split, n // 256))
                for j in range(k):
                    a, z = n * j // k, n * (j + 1) // k
                    self.units.append((gi, a, z - a, slice(gsl.start + a, gsl.start + z)))
            # the slowest console (Punch-Out, ~4x SMB's work per frame) bounds the rollout: its blocks go first
            self.unit_j = [sum(u[0] == self.units[i][0] for u in self.units[:i]) for i in range(len(self.units))]
            self.game_streams = [torch.cuda.Stream(self.dev, priority=-1 if env.games[gi] in SLOW_GAMES else 0)
                                 for gi, *_ in self.units]
        slices = [u[3] for u in self.units]
        hg = []
        for gi, sl in enumerate(slices):
            s = self.game_streams[gi]
            s.wait_stream(main)
            with torch.cuda.stream(s):
                hg.append(self.h[sl].clone())
                self.obs.saved_game(b.r, 0, gi, sl)
        dv = self.div
        for t in range(T):
            for gi, sl in enumerate(slices):
                with torch.cuda.stream(self.game_streams[gi]):
                    if dv is not None:
                        b.b_mood[t, sl] = dv.mood[sl]
                    with self.autocast():
                        z = self.encode_chunks(self.obs.slot(b.r, t)[sl], b.b_emph[t, sl], b.b_tl[t, sl],
                                               [p[sl] for p in b.b_prev[t]], None if dv is None else b.b_mood[t, sl])
                    h = hg[gi] * (~b.b_start[t, sl]).unsqueeze(1).float()
                    h = m.gru(z.float(), h)
                    hg[gi] = h
                    parts, logp = m.act(h, b.b_masks[t, sl], generator=self.gen_roll)
                    b.b_val[t, sl] = mu[g[sl]] + sd[g[sl]] * self.hl.mean(m.value(h))
                    for k in range(5):
                        b.b_parts[t, k, sl] = parts[k]
                    b.b_logp[t, sl] = logp
                    env.actions[sl] = pack(parts)
                    self.obs.before_write_game(gi, t)
                    ug, ua, un, _ = self.units[gi]
                    env.step_range(ug, ua, un, self.obs.slot(b.r, t + 1), b.b_emph[t + 1])
                    self.obs.saved_game(b.r, t + 1, gi, sl)
                    b.b_rew[t, sl] = env.reward[sl]
                    b.b_rollback[t, sl] = (env.events[sl] & 64) != 0
                    done = env.done[sl].bool()
                    b.b_done[t, sl] = done
                    if dv is not None:
                        dv.after_step(sl, env.parts[sl], done, b.b_feat[t, sl])
                    if self.rnd is not None:
                        b.b_small[t, sl] = self.rnd.small(self.obs.slot(b.r, t + 1)[sl, 3], b.b_emph[t + 1, sl, 3])
                    if self.places is not None:
                        self.places.step(env, t, ug, self.unit_j[gi], ua, sl, self.obs.slot(b.r, t + 1)[sl, 3],
                                         b.b_emph[t + 1, sl, 3], env.level_now[sl], env.progress_now[sl], done)
                    b.b_masks[t + 1, sl] = env.masks[sl]
                    b.b_tl[t + 1, sl] = env.timeline[sl]
                    b.b_start[t + 1, sl] = done
                    b.b_prev[t + 1, :, sl] = b.b_parts[t, :, sl] * (~done).long()
                    b.b_info[t, sl] = env.info[sl]
                    b.b_slvl[t, sl] = env.start_level[sl]
                    if env.level_cap > 0:
                        b.b_lnow[t, sl] = env.level_now[sl]
                        b.b_lev[t, sl] = env.parts[sl, 1] > 0
        for gi, sl in enumerate(slices):
            with torch.cuda.stream(self.game_streams[gi]):
                with self.autocast():
                    z = self.encode_chunks(self.obs.slot(b.r, T)[sl], b.b_emph[T, sl], b.b_tl[T, sl],
                                           [p[sl] for p in b.b_prev[T]], None if dv is None else dv.mood[sl])
                hT = m.gru(z.float(), hg[gi] * (~b.b_start[T, sl]).unsqueeze(1).float())
                b.b_val[T, sl] = mu[g[sl]] + sd[g[sl]] * self.hl.mean(m.value(hT))
        for gi, s in enumerate(self.game_streams):
            main.wait_stream(s)
            hg[gi].record_stream(main)  # used below on the main stream; allocated on the game stream
        self.h = torch.cat(hg)  # state after the last step (before the reset of step T's start)
        self.decisions += T * self.N * self.world
        self.collect_episodes(b)

    def apply_level_plan(self):
        """Level-start shares and weights for this rollout, from the clear rates so far (rl/level_plan.py)."""
        if self.level_plan is None or self.cfg.level_starts <= 0:
            return
        saved = self.env.saved_levels()
        self.level_report = {}
        for gi, g in enumerate(self.env.games):
            w, share = self.level_plan.plan(gi, saved[g], self.cfg.level_starts)
            r = self.level_plan.rate(gi)
            if self.cfg.only_levels and w is not None:
                keep = [int(x) for x in self.cfg.only_levels.split(',') if int(x) in saved[g]]
                if keep:
                    f = np.zeros_like(w)
                    f[keep] = np.maximum(1.0 - r[keep], 0.02)
                    w, share = f / f.sum(), 1.0
            self.env.set_level_plan(gi, share, w if w is None or self.places is None else self.places.weights(gi, w, r))
            self.level_report[g] = dict(share=share, levels={
                str(l): dict(clear=float(r[l]), visits=float(self.level_plan.visits[gi, l]), weight=float(w[l]),
                             deaths=saved[g][l]) for l in sorted(saved[g])})

    def collect_episodes(self, b):
        """Finished episodes of this rollout (one host copy per rollout, no per-step sync)."""
        t, e = b.b_done.nonzero(as_tuple=True)
        info = b.b_info[t, e].cpu().numpy()
        start = b.b_slvl[t, e].cpu().numpy()
        if self.places is not None and len(start):
            lg = self.local_game[e.cpu()].numpy()
            start = np.array([self.places.level_of(int(a), [int(s)])[0] for a, s in zip(lg, start)], np.int64)
        games = self.game_of_env[e].cpu().numpy()
        self.episodes += [dict(game=GAMES[g], ret=float(row[0]), len=int(row[1]),
                               end=END_NAMES.get(int(row[2]), str(int(row[2]))), deaths=int(row[3]),
                               level=int(row[4]), overflow=int(row[5]), start=int(s))
                          for g, row, s in zip(games, info, start)]
        if self.level_plan is not None:
            self.count_levels(b)

    def count_levels(self, b):
        """This rollout's level outcomes, as they happened (rl/level_plan.py). One host copy per rollout."""
        cap, G = self.cfg.level_cap, len(self.env.games)
        g = torch.repeat_interleave(torch.arange(G, device=self.dev), torch.tensor(self.env.counts, device=self.dev))
        g = g.unsqueeze(0).expand_as(b.b_lnow)
        lv, end = b.b_lnow.long(), b.b_info[..., 2]
        cleared = b.b_lev & (lv >= 1) & (lv <= cap)            # the level before the one just reached
        ended = b.b_done & (lv >= 0) & (lv < cap) & (end != 3) & (end != 5) & (end != 6)  # skip/fault/level-clear
        won = ended & (end == 2)
        slot = lambda m, off: (g[m] * cap + lv[m] - off)  # noqa: E731
        v = torch.zeros(G * cap, device=self.dev)
        c = torch.zeros(G * cap, device=self.dev)
        for acc, idx in ((v, slot(cleared, 1)), (c, slot(cleared, 1)), (v, slot(ended & ~won, 0)), (v, slot(won, 0)),
                         (c, slot(won, 0))):
            acc.index_add_(0, idx, torch.ones(idx.shape[0], device=self.dev))
        self.level_plan.add(v.view(G, cap).double().cpu().numpy(), c.view(G, cap).double().cpu().numpy())

    # ---------------- learning ----------------
    def gae(self, b, rew):
        T, cfg = self.T, self.cfg
        adv = torch.zeros(T, self.N, device=self.dev)
        last = torch.zeros(self.N, device=self.dev)
        for t in reversed(range(T)):
            nonterm = (~b.b_done[t]).float()
            delta = rew[t] + cfg.gamma * b.b_val[t + 1] * nonterm - b.b_val[t]
            last = delta + cfg.gamma * cfg.lam * nonterm * last
            adv[t] = last
        return adv, adv + b.b_val[:T]

    def per_game_norm(self, adv):
        """Mean 0 / std 1 advantages per game over the whole rollout (all ranks). Without it the per-minibatch
        normalisation mixes games whose returns differ ~60x in scale (life-loss games vs progress-only games), so the
        quiet games' policy gradient shrinks below the entropy bonus and their policy drifts back to uniform."""
        g = self.game_of_env.expand_as(adv).reshape(-1)
        a = adv.reshape(-1).double()
        s = torch.zeros(3, len(GAMES), device=self.dev, dtype=torch.float64)
        s[0].index_add_(0, g, torch.ones_like(a))
        s[1].index_add_(0, g, a)
        s[2].index_add_(0, g, a * a)
        if self.ddp:
            dist.all_reduce(s)
        n = s[0].clamp_min(1)
        mean = s[1] / n
        std = (s[2] / n - mean ** 2).clamp_min(0).sqrt().clamp_min(1e-6)
        return ((a - mean[g]) / std[g]).float().view_as(adv)

    def sync_grads(self):
        if not self.ddp:
            return
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat)
        flat /= self.world
        o = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[o:o + n].view_as(g))
            o += n

    def mb_loss(self, obs, e, b_emph, b_tl, b_prev, h0, b_start, b_masks, b_parts, b_logp, adv, target, b_mood=None):
        """PPO loss of one minibatch (whole T-step sequences of the envs e); compiled as one graph on CUDA."""
        cfg, m, T, mb = self.cfg, self.model, self.T, e.shape[0]
        with self.autocast():
            z = m.encode(obs, b_emph[:T, e].flatten(0, 1), b_tl[:T, e].flatten(0, 1),
                         [p.flatten(0, 1) for p in b_prev[:T, :, e].transpose(0, 1)],
                         None if b_mood is None else b_mood[:, e].flatten(0, 1))
        hs, _ = m.recur(z.view(T, mb, -1), h0[e], b_start[:T, e])
        hs = hs.flatten(0, 1)
        parts = [p.flatten(0, 1) for p in b_parts[:, :, e].transpose(0, 1)]
        logp, ent = m.evaluate(hs, b_masks[:T, e].flatten(0, 1), parts)
        vlog = m.value(hs).float()
        a = adv[:, e].reshape(-1)
        a = (a - a.mean()) / (a.std() + 1e-8)
        logratio = logp - b_logp[:, e].reshape(-1)
        ratio = logratio.exp()
        pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
        v = -(target[:, e].reshape(-1, target.shape[-1]) * vlog.log_softmax(-1)).sum(-1).mean()
        entm = ent.mean()
        loss = pg + cfg.vf_coef * v - cfg.ent_coef * entm
        with torch.no_grad():
            kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
        return loss, torch.stack([pg.detach(), v.detach(), entm.detach(), kl, clipfrac])

    def learn(self, b):
        cfg, m, T = self.cfg, self.model, self.T
        rew, dstats = b.b_rew, None
        if self.div is not None:
            bonus, dstats = self.div.bonus(b.b_feat, b.b_mood)
            rew = b.b_rew + bonus
            dstats['bonus'] = bonus
        istats = None
        if self.rnd is not None:
            self.rnd.update_input_stats(b.b_small)
            ib, ierr = self.rnd.bonus(b.b_small, self.game_of_env)
            ib = ib * (~b.b_done).float()   # the picture after an end is the next episode's start
            rew = rew + ib
            istats = (ib, ierr)
        adv, ret = self.gae(b, rew)
        if cfg.adv_norm == 'game':
            adv = self.per_game_norm(adv)
        self.update_stats(ret)
        mu, sd = self.norm_stats()
        g = self.game_of_env
        normalized_ret = (ret - mu[g]) / sd[g]
        target = self.hl.target(normalized_ret)                        # T, N, bins
        acc = torch.zeros(6, device=self.dev)                            # pg v ent kl clipfrac gnorm (summed)
        kl_first, n = None, 0
        mb = min(cfg.minibatch_envs, self.N)
        for _ in range(cfg.epochs):
            perm = torch.randperm(self.N, device=self.dev)
            self.obs.epoch(b.r, perm, mb)
            for i in range(self.N // mb):
                e = perm[i * mb:(i + 1) * mb]
                loss, st = self.mb_loss_c(self.obs.batch(i), e, b.b_emph, b.b_tl, b.b_prev, b.h0, b.b_start,
                                          b.b_masks, b.b_parts, b.b_logp, adv, target, b.b_mood)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                # average the still-scaled grads first: every rank then sees the same values, so the fp16 scaler
                # skips an overflowing step on all ranks together (unscaling first lets ranks disagree and diverge)
                self.sync_grads()
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
                self.obs.done_with(i)
                if kl_first is None:  # before any update: ~0 with --lag 0 (rollout and learner agree)
                    kl_first = st[3].clone()
                acc[:5] += st
                acc[5] += gn.float()
                n += 1
        acc = (acc / max(n, 1)).tolist()                                 # one host sync per iteration
        out = dict(zip(['pg', 'v', 'ent', 'kl', 'clipfrac', 'gnorm'], acc))
        out['kl_first'] = float(kl_first)
        with torch.no_grad():
            y, pred = ret.reshape(-1), b.b_val[:T].reshape(-1)
            out['explained_var'] = float(1 - (y - pred).var() / (y.var() + 1e-8))
        out['reward_mean'] = float(b.b_rew.mean())
        out['rollback_events'] = int(b.b_rollback.sum())
        out['value_target_clipfrac'] = float((normalized_ret.abs() > self.hl.vmax).float().mean())
        if self.rnd is not None:
            out['rnd_loss'] = self.rnd.train(b.b_small)
            ib, ierr = istats
            g = self.game_of_env.unsqueeze(0).expand_as(ib)
            out['curiosity'] = {GAMES[gi]: dict(bonus=float(ib[g == gi].mean()), error=float(ierr[g == gi].mean()))
                                for gi in self.game_of_env.unique().tolist()}
        if self.div is not None:
            out['judge_loss'] = self.div.train_judge(b.b_feat, b.b_mood)
            self.div.update_gate(b.b_done, b.b_mood, b.b_info[..., 0])
            out['div'] = self.div_summary(dstats)
        return out

    def div_summary(self, st):
        """Per game: how well the judge tells the moods apart (1/K = they all play alike, 1 = always), the mean bonus,
        the share of moods the gate lets earn it, and each mood's running return."""
        g = self.game_of_env.unsqueeze(0).expand_as(st['acc'])
        gate = self.div.gate()
        res = {}
        for gi in self.game_of_env.unique().tolist():
            m = g == gi
            res[GAMES[gi]] = dict(acc=float(st['acc'][m].mean()), bonus=float(st['bonus'][m].mean()),
                                  gate_open=float(gate[gi].mean()), mood_ret=[round(float(x), 4) for x in self.div.R[gi]])
        return res

    # ---------------- checkpoints ----------------
    def state(self):
        """Taken only when no collected rollout is waiting to be learned (see train)."""
        # the current env outputs: step T of the rollout collected last (handed to step 0 only when the next starts)
        b, t = (self.last, self.T) if self.last is not None else (self.bufs[self.nb], 0)
        return dict(cfg=dataclasses.asdict(self.cfg), it=self.it, decisions=self.decisions, model=self.model.state_dict(),
                    opt=self.opt.state_dict(), scaler=self.scaler.state_dict(), ret_mean=self.ret_mean,
                    ret_sq=self.ret_sq, ret_seen=self.ret_seen, h=self.h, env=self.env.state_dict(),
                    obs=self.obs.slot(b.r, t).cpu(), emph=b.b_emph[t].cpu(), masks=b.b_masks[t].cpu(),
                    tl=b.b_tl[t].cpu(), prev=b.b_prev[t].cpu(), start=b.b_start[t].cpu(),
                    rng_cpu=torch.get_rng_state(), rng_roll=self.gen_roll.get_state(), rng_cuda=torch.cuda.get_rng_state(self.dev) if self.cuda else None,
                    elapsed=time.time() - self.t_start + self.elapsed0,
                    div=None if self.div is None else self.div.state_dict(),
                    mood_kicked=getattr(self, 'mood_kicked', False),
                    rnd=None if self.rnd is None else self.rnd.state_dict(),
                    level_plan2=None if self.level_plan is None else self.level_plan.state_dict(),
                    places=None if self.places is None else self.places.state_dict())

    def save(self, tag='latest'):
        self.sync()
        path = self.run / f'ckpt_rank{self.rank}_{tag}.pt'
        atomic_torch_save(self.state(), path)
        if self.rank == 0:
            (self.run / 'ckpt_info.json').write_text(json.dumps(dict(it=self.it, decisions=self.decisions,
                                                                     time=time.time(), world=self.world)))
        self.t_last_ckpt = time.time()

    def load(self):
        path = self.run / f'ckpt_rank{self.rank}_latest.pt'
        self.elapsed0 = 0.0
        if not path.exists():
            return False
        s = torch.load(path, map_location=self.dev, weights_only=False)
        self.load_model_opt(s)
        self.scaler.load_state_dict(s['scaler'])
        self.ret_mean, self.ret_sq, self.ret_seen = s['ret_mean'], s['ret_sq'], s['ret_seen']
        if self.level_plan is not None and s.get('level_plan2') is not None:   # (level_plan: counted at game end, stale)
            self.level_plan.load_state_dict(s['level_plan2'])
        self.h = s['h']
        self.env.load_state_dict(s['env'])
        b = self.bufs[self.nb]
        self.env.bind_obs(self.obs.slot(b.r, 0), b.b_emph[0])
        self.obs.slot(b.r, 0).copy_(s['obs']); b.b_emph[0].copy_(s['emph']); b.b_masks[0].copy_(s['masks'])
        b.b_tl[0].copy_(s['tl']); b.b_prev[0].copy_(s['prev']); b.b_start[0].copy_(s['start'])
        torch.set_rng_state(s['rng_cpu'].cpu())  # (map_location moved it to the device)
        self.gen_roll.set_state(s['rng_roll'].cpu())
        if self.cuda:
            torch.cuda.set_rng_state(s['rng_cuda'].cpu(), self.dev)
        if self.div is not None and s.get('div') is not None:
            self.div.load_state_dict(s['div'])
        if self.rnd is not None and s.get('rnd') is not None:
            self.rnd.load_state_dict(s['rnd'])
        if self.places is not None and s.get('places') is not None:
            self.places.load_state_dict(s['places'])
        self.it, self.decisions, self.elapsed0 = s['it'], s['decisions'], s.get('elapsed', 0.0)
        if self.rank == 0:
            print(f'[rl] resumed from {path} at iteration {self.it}, {self.decisions} decisions', flush=True)
        return True

    def load_model_opt(self, s):
        """Load weights + optimizer. A checkpoint without the mood input (or with fewer moods) is widened with zero
        weights (and zero Adam moments) for the new inputs, so the loaded policy plays exactly as before."""
        sd = dict(s['model'])
        w_old, w_new = sd['inp.weight'], self.model.inp.weight
        grown = w_old.shape[1] < w_new.shape[1]
        if grown:
            sd['inp.weight'] = F.pad(w_old, (0, w_new.shape[1] - w_old.shape[1]))
            print(f"[rl] widened inp.weight {tuple(w_old.shape)} -> {tuple(w_new.shape)} (new mood inputs start at 0)",
                  flush=True)
        self.model.load_state_dict(sd)
        self.mood_kicked = bool(s.get('mood_kicked', False))
        K = self.cfg.moods
        if K and self.cfg.mood_kick > 0 and not self.mood_kicked:
            gen = torch.Generator(device='cpu').manual_seed(self.cfg.seed * 7 + 12345)
            noise = torch.randn(w_new.shape[0], K, generator=gen) * self.cfg.mood_kick
            with torch.no_grad():
                self.model.inp.weight[:, -K:] += noise.to(self.model.inp.weight)
            self.mood_kicked = True
            print(f'[rl] mood inputs kicked with N(0, {self.cfg.mood_kick}) so the moods start out different', flush=True)
        osd = s['opt']
        if grown:
            k = [n for n, _ in self.model.named_parameters()].index('inp.weight')
            st = osd['state'][k]
            for key in ('exp_avg', 'exp_avg_sq'):
                st[key] = F.pad(st[key], (0, w_new.shape[1] - w_old.shape[1]))
        self.opt.load_state_dict(osd)

    # ---------------- logging ----------------
    def log(self, row):
        if self.rank != 0:
            return
        with open(self.run / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps(row) + '\n')

    def episode_summary(self):
        eps = self.episodes
        if self.ddp:
            allx = [None] * self.world
            dist.all_gather_object(allx, eps)
            eps = sum(allx, [])
        self.episodes = []
        by, lv = {}, {}
        for e in eps:
            if e.get('start', -1) >= 0:   # started from a saved level start: kept apart (the main numbers stay power-on)
                d = lv.setdefault(e['game'], dict(n=0, cleared=0, by_start={}))
                cl = int(e['level'] > e['start'])
                d['n'] += 1; d['cleared'] += cl
                bs = d['by_start'].setdefault(str(e['start']), [0, 0])
                bs[0] += 1; bs[1] += cl
                continue
            d = by.setdefault(e['game'], dict(n=0, ret=0.0, len=0, level_max=0, level_mean=0.0, ends={}))
            d['n'] += 1; d['ret'] += e['ret']; d['len'] += e['len']; d['level_mean'] += e['level']
            d['level_max'] = max(d['level_max'], e['level'])
            d['ends'][e['end']] = d['ends'].get(e['end'], 0) + 1
        for d in by.values():
            d['ret'] /= d['n']; d['len'] /= d['n']; d['level_mean'] /= d['n']
        return by, lv

    def stop_requested(self):
        cfg = self.cfg
        stop = Stop.flag or (self.run / 'STOP').exists()
        if cfg.time_limit and time.time() - self.t_start > cfg.time_limit:
            stop = True
        if cfg.total_iters and self.it + 1 >= cfg.total_iters:
            stop = True
        if self.ddp:
            s = torch.tensor([int(stop)], device=self.dev)
            dist.all_reduce(s)
            stop = bool(s.item())
        return stop

    def energy_j(self):
        if self.nvml is None:
            return None
        import pynvml
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.nvml) / 1000.0

    def _collect_thread(self, box):
        try:
            with torch.cuda.stream(self.roll_stream):
                box['buf'] = self.collect()
        except BaseException as ex:  # re-raised in the main thread
            box['error'] = ex

    def start_eval(self):
        """Save the current weights to eval/itN/model.pt and play them in a separate CPU process (rl/evaluate.py:
        every episode to its end, videos to RUN/videos/itN_GAME[_moodK].mp4), so training does not wait for it.
        One evaluation at a time: if the previous one is still playing, this one only keeps its model.pt."""
        import subprocess
        cfg = self.cfg
        d = self.run / 'eval' / f'it{self.it:06d}'
        d.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(dict(model=self.model.state_dict(), it=self.it, decisions=self.decisions,
                               cfg=dataclasses.asdict(cfg)), d / 'model.pt')
        if getattr(self, 'eval_proc', None) is not None and self.eval_proc.poll() is None:
            print(f'[rl] evaluation of {self.eval_tag} still playing; it{self.it:06d} kept as model.pt only', flush=True)
            return
        self.eval_tag = f'it{self.it:06d}'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        self.eval_proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve().parent / 'evaluate.py'), str(d / 'model.pt'), str(d),
             '--device', 'cpu', '--games', cfg.games, '--threads', str(cfg.eval_threads), '--steps', str(cfg.eval_steps),
             '--video-dir', str(self.run / 'videos'), '--tag', self.eval_tag],
            stdout=open(d / 'eval.log', 'w'), stderr=subprocess.STDOUT, env=env, start_new_session=True)

    # ---------------- main loop ----------------
    def train(self):
        import threading
        cfg = self.cfg
        if not self.load():
            self.start()
        signal.signal(signal.SIGTERM, Stop.handler)
        signal.signal(signal.SIGINT, Stop.handler)
        pending = None   # collected rollout not learned yet
        # FLOPs per learned decision: encoder forward in the rollout + forward and backward (x3) per epoch; the GRU and
        # heads are under 1%. MFU is against the RTX 5090's 209.5 TFLOPS dense BF16.
        flops_per_dec = 0.896e9 * (1 + 3 * cfg.epochs)
        energy0 = self.energy_j()
        while True:
            t0 = time.time()
            if pending is None:
                self.refresh_actor()
                pending = self.collect()
            # Learn `pending`; with --lag collect the next rollout meanwhile, unless a checkpoint/stop follows (then
            # nothing is left unlearned) or this is the first update (compilation happens alone).
            stop = self.stop_requested()
            ckpt = stop or time.time() - self.t_last_ckpt > cfg.ckpt_minutes * 60 or \
                bool(cfg.ckpt_iters and (self.it + 1) % cfg.ckpt_iters == 0)
            overlap = bool(cfg.lag) and not ckpt and self.it > 0
            box, th = {}, None
            if overlap:
                self.refresh_actor()
                self.sync()
                th = threading.Thread(target=self._collect_thread, args=(box,))
                th.start()
            tl0 = time.time()
            stats = self.learn(pending)
            self.sync()
            if self.cuda and not cfg.lag:
                torch.cuda.empty_cache()
            learn_s = time.time() - tl0
            if th is not None:
                th.join()
                if 'error' in box:
                    raise box['error']
                self.sync()
            pending = box.get('buf')
            t2 = time.time()
            self.it += 1
            n = self.T * self.N * self.world
            e1 = self.energy_j()
            power = (e1 - energy0) / (t2 - t0) if e1 is not None and energy0 is not None else None
            energy0 = e1
            row = dict(it=self.it, decisions=self.decisions, time=time.time(), sps=n / (t2 - t0),
                       power_w=power, mfu=n / (t2 - t0) * flops_per_dec / 209.5e12,
                       joules_per_1k_dec=None if power is None else power * (t2 - t0) / n * 1000,
                       rollout_s=self.rollout_s, learn_s=learn_s, overlap=overlap,
                       mem_gb=torch.cuda.max_memory_allocated(self.dev) / 1e9 if self.cuda else 0.0, **stats,
                       **dict(zip(('games', 'level_starts'), self.episode_summary())))
            if self.cfg.level_cap > 0 and self.rank == 0:
                row['saved_levels'] = {g: sorted(v) for g, v in self.env.saved_levels().items()}
                if getattr(self, 'level_report', None):
                    row['level_plan'] = self.level_report
                if self.places is not None:
                    row['places'] = dict(self.places.last, per_level={g: self.places.report(gi)
                                                                      for gi, g in enumerate(self.env.games)})
            if self.it % cfg.log_every == 0:
                self.log(row)
                if self.rank == 0:
                    print(f"[rl] it {self.it} dec {self.decisions/1e6:.2f}M sps {row['sps']:.0f} "
                          f"(rollout {self.rollout_s:.1f}s learn {learn_s:.1f}s{' overlapped' if overlap else ''}) "
                          f"pg {stats['pg']:.3f} v {stats['v']:.3f} ent {stats['ent']:.2f} kl {stats['kl']:.4f} "
                          f"kl0 {stats['kl_first']:.1e} ev {stats['explained_var']:.2f} "
                          f"mfu {row['mfu']*100:.1f}% power {row['power_w'] or 0:.0f}W"
                          + (''.join(f" | {g} judge {d['acc']:.2f} gate {d['gate_open']:.2f}" for g, d in stats['div'].items())
                             if 'div' in stats else '')
                          + (''.join(f" | {g} curiosity {d['bonus']:.2e}" for g, d in stats['curiosity'].items())
                             if 'curiosity' in stats else '')
                          + (f" | places kept {row['places']['kept']} known {row['places']['known']} held "
                             f"{sum(sum(v.values()) for v in row['places']['per_level'].values())}" if 'places' in row else '')
                          + (''.join(f" | {g} saved levels {len(v)} (max {max(v) if v else '-'})"
                                     for g, v in row.get('saved_levels', {}).items()))
                          + (''.join(f" | {g} " + ' '.join(f"L{l}:{d['clear']*100:.0f}%/w{d['weight']*100:.0f}"
                                                           for l, d in p['levels'].items())
                                     for g, p in row.get('level_plan', {}).items()) ), flush=True)
            if ckpt:
                assert pending is None
                self.save()
            if self.rank == 0 and cfg.eval_every_minutes > 0 and \
                    time.time() - self.t_last_eval > cfg.eval_every_minutes * 60 and not stop:
                self.start_eval()
                self.t_last_eval = time.time()
            if self.ddp:
                dist.barrier()
            if stop:
                break
        if self.rank == 0:
            print('[rl] stopped cleanly', flush=True)


if __name__ == '__main__':
    Trainer(parse()).train()
