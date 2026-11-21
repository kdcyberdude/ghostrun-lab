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
from model import Policy, HLGauss, pack, unpack, count_params, SIZES, PARTS  # noqa: E402
from rlenv import RlEnv, GAMES, END_NAMES, rom_path  # noqa: E402
from diversity import Diversity, N_FEAT  # noqa: E402
from novelty import RND, H as RND_H, W as RND_W  # noqa: E402
from curiosity import Curiosity, OUT as CUR_OUT  # noqa: E402
from level_plan import LevelPlan  # noqa: E402
from places import DOOR_GRACE, DOOR_REPLAY, Places  # noqa: E402
from outcome import EVENTS as EVENTS_, Outcome  # noqa: E402
from trybonus import TryBonus  # noqa: E402
from ramnovelty import RamNovelty  # noqa: E402
from situation import Situations  # noqa: E402
from stuck import StuckSearch  # noqa: E402

SIT_K = 2      # situations staged per step and unit (rl/places.py var_k)
from grid_reward import GridReward  # noqa: E402
from routes import Routes  # noqa: E402
from coverage import Coverage, RomMap, smb_block_keys, smb_height, smb_keys  # noqa: E402
import slots as slotlib  # noqa: E402


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
    grid_exploration: int = 0         # opt-in bounded world-grid reward (SMB initially); replaces reward10 shaping
    attempt_limit: int = 4000        # grid mode and routes: first death / level clear / this many decisions
    grid_history_scale: float = 1.0  # prior visits per familiarity unit; env count makes decay invariant to parallelism
    grid_regions: int = 0            # experimental 128x32 region novelty plus smaller fine-cell bonuses
    grid_coverage: int = 0           # persistent episodic coverage, bounded rarity boost, ongoing stagnation cost
    grid_cell_size: int = 32         # world-grid pixels per cell; 8 selects the finer coverage grid
    grid_stagnation_quadratic: int = 0  # coverage: 8-decision grace, quadratic ramp to .001 at 16 decisions
    grid_persistent_visits: int = 0   # only cells absent from shared lifetime history earn exploration bonuses
    coverage: int = 0                 # 1 (SMB): the only reward is +1 per camera position new in the episode
                                      # (rl/coverage.py); the env's reward10 reward is ignored, warp zones do not end
                                      # the episode, and the graphs compare the found positions with the ROM's map
    ram_io: int = 0                   # 1: the 2 KB RAM is a policy input and the next decision's RAM changes are
                                      # predicted (auxiliary loss, rl/model.py ram_logits)
    ram_coef: float = 0.5             # weight of that loss (nats per RAM byte, summed over its 8 bits); 0 = RAM stays an
                                      # input but nothing predicts it
    ram_target: str = 'flip'          # 'flip': predict which RAM bits change; 'next': predict the whole next RAM
                                      # (every bit, the unchanged ones too)
    rarity: float = 0.0               # coverage: extra pay rarity / sqrt(1 + episodes that found the cell before)
    blocks: float = 0.0               # coverage: an 8x8 block Mario is in for the first time in the episode pays
                                      # blocks / (1 + games that touched it before / block_half)  (rl/coverage.py)
    block_half: float = 100.0         # games after which a block pays half
    # Held random moves (2026-11-18): independent per-decision sampling cannot produce a long move against the
    # policy's habit (4-4: ~20 decisions of Left without a jump at p ~ 1e-7 each), so no reward that pays only where
    # the play already goes can teach it. In a reward drought a console now and then holds one random move for a
    # while; such steps are stored with the policy's own log-prob and enter the policy loss only when they turned
    # out well. Generic: no game data. Off in test plays.
    explore_eps: float = 0.0          # chance per decision to start a held random move (0 = off)
    explore_after: int = 100          # ... only after this many decisions without a paying step
    explore_pay: float = 0.5          # a step pays if its reward is at least this
    explore_min: int = 8              # hold length in decisions, log-uniform between min and max
    explore_max: int = 64
    explore_adv_min: float = 2.0      # a held step enters the policy loss if its normalised advantage exceeds this
    # Explorers (2026-11-19): the 8-4 floating pipe needs Down at one spot Mario already reaches; nothing paid for an
    # untried move at a known place. With explore_count, a fixed share (explore_share) of each game's consoles are
    # explorers: tries of each move (d-pad x A x B, 36) are counted per place (the --places key, e.g. Mario's 8x8
    # block); an explorer not holding starts a try of the least-tried legal move here with chance 1 / (1 + tries of
    # that move here) (untried = at once), lasting explore_min decisions, and its drought holds also play the
    # least-tried move. Explorers' games never enter learning (policy, value, advantage statistics, level clear
    # rates); what they find reaches the policy only through the saved places (--places, best with
    # --place-readmit) that the other consoles then start from and learn on their own. Mixing that many tries into
    # the learners' games broke the learned 4-4 route in six controlled runs (the policy learns to play for random
    # interruptions; runs/rl/smb-castles/notes/least-tried/PLAN.md). Generic: no game data.
    explore_count: int = 0
    explore_share: float = 0.25       # ... share of each game's consoles that are explorers
    place_readmit: int = 0            # --places: a rare spot not in the archive can be staged again (rl/places.py)
    place_share: int = 0              # 1: place slots move to levels without places and the level start keeps half the
                                      # failure share (rl/places.py; tried 2026-11-20: castles fell ~4 points)
    try_bonus: float = 0.0            # > 0: reward for pressing a button combination (d-pad x A x B, 81) rarely
                                      # pressed at this place: try_bonus / sqrt(1 + presses there before), the
                                      # player's presses counted over all training (rl/trybonus.py); needs
                                      # --explore-count (the place key of every console)
    ram_novelty: float = 0.0          # > 0 (2026-11-20, user design): reward a change of the game's RAM never seen at
                                      # this screen: per changed byte weight min(1, .01 / its change rate) x
                                      # 1 / sqrt(1 + times seen (screen, byte, new value)), summed, capped at 1, times
                                      # this (rl/ramnovelty.py); computed once per rollout from the stored RAM
    situations: int = 0               # > 0 (2026-11-20, user design): this many extra place slots for rare SITUATIONS:
                                      # the same spot in an unusual state of the world (a progress byte of the RAM
                                      # away from its usual value) or standing where the others only fly past
                                      # (rl/situation.py, rl/places.py `var`); needs --places, --blocks, --coverage
    outcome: float = 0.0              # > 0: "what will this move cause" head (rl/outcome.py): predicts the chance of
                                      # death / damage / score / new level / jump / new ground within 4 and 16
                                      # decisions from the policy state + move; its loss (this weight) also trains the
                                      # policy's network; explorers pick tries by its promise + a count bonus
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
    curiosity: float = 0.0            # > 0: curiosity about actions (rl/curiosity.py): K heads guess the next picture's
                                      # fixed random fingerprint from the policy state + move; reward += this x
                                      # (their disagreement / its long-run average), the average bonus per decision
    cur_heads: int = 4                # guessing heads
    cur_loss_coef: float = 0.5        # weight of the heads' loss (it also trains the policy's network)
    cur_clip: float = 10.0            # cap on one move's bonus, in multiples of the average
    mood_kick: float = 0.0            # > 0: once, add N(0, mood_kick) to the policy's mood input weights (moods start
                                      # out playing alike after zero-widening; the judge can only reward differences)
    level_starts: float = 0.0         # > 0: new training episodes start by difficulty over power-on and the saved
                                      # level starts in proportion to each level's failure rate (rlenv_core.cuh,
                                      # rl/level_plan.py; every game's reward10 'level' event saves one per level);
                                      # 1 = failure rate alone, less scales the saved share down.
                                      # Test plays: power-on.
    level_cap: int = 64               # levels 0..level_cap-1 of each game can be saved (always collected)
    places: int = 0                   # > 0: Go-Explore places inside levels (rl/places.py): this many place slots per game
                                      # with --blocks a place is Mario's 8x8 block (the coverage block key), not a picture
    only_levels: str = ''             # e.g. '15,27,30': every episode starts at one of these saved levels (by failure)
    routes: str = ''                  # directory of routes (rl/explore.py): every episode is one attempt at a level,
                                      # started by the backward curriculum over the level's route (rl/routes.py)
    route_slots: int = 2048           # room for route states
    init_from: str = ''               # a new run starts from this checkpoint's weights, optimizer and return statistics
    entrances: str = ''               # ... and from this entrance archive (.npy; rl/explore.py writes one)
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
    ckpt_minutes: float = 15.0
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
    imitate: float = 0.0              # > 0 (with stuck_search): weight of the imitation loss on teacher tries
                                      # (rl/stuck.py TEACHER TRIES: the search's buttons, pressed by the trainer)
    stuck_search: int = 0             # 1: stuck -> search -> hand-off (user design 2026-11-21, rl/stuck.py,
                                      # rl/stucksearch.py): a level with no new block for stuck_decisions learner
                                      # decisions is searched by a CPU process; its finds become hand-off starts
    stuck_decisions: int = 1_000_000
    stuck_sync: int = 0               # 1: wait for each search (tests)
    stuck_params: str = ''            # JSON overrides of the search's budget (rl/stucksearch.py DEFAULTS; tests)
    stuck_gpu: int = 1                # 1: the search runs on the GPU in this process between rollouts (training
                                      # waits); 0: a CPU process beside training
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
TRIED_BITS = 20             # rows of the (place, move) tries table (Config.explore_count): 2^20 x 36 int32 = 144 MB


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

    def __init__(self, r, T, N, core, dev, moods=0, rnd=False, ram=False, cur=False, out=False, tryb=False):
        self.r = r
        self.b_try = torch.zeros(T, N, device=dev) if tryb else None   # try bonus of each move (rl/trybonus.py)
        self.b_ev = torch.zeros(T, N, dtype=torch.uint8, device=dev) if out else None   # event bits after each move (rl/outcome.py)
        self.b_fp = torch.zeros(T, N, CUR_OUT, device=dev) if cur else None   # fingerprint of the picture after each move
        self.b_dis = torch.zeros(T, N, device=dev) if cur else None           # the heads' disagreement about each move
        self.b_ram = torch.zeros(T + 1, N, 2048, dtype=torch.uint8, device=dev) if ram else None  # RAM at each step
        self.b_small = torch.zeros(T, N, RND_H, RND_W, dtype=torch.uint8, device=dev) if rnd else None  # curiosity input
        self.b_mood = torch.zeros(T, N, dtype=torch.long, device=dev) if moods else None   # mood used at each step
        self.b_feat = torch.zeros(T, N, N_FEAT, device=dev) if moods else None           # judge input after each step
        self.b_emph = torch.zeros(T + 1, N, 4, 224, dtype=torch.uint8, device=dev)
        self.b_masks = torch.zeros(T + 1, N, 19, dtype=torch.uint8, device=dev)
        self.b_tl = torch.zeros(T + 1, N, 8, 45, dtype=torch.uint8, device=dev)
        self.b_prev = torch.zeros(T + 1, 5, N, dtype=torch.long, device=dev)  # previous move parts
        self.b_start = torch.zeros(T + 1, N, dtype=torch.bool, device=dev)    # episode starts at this step
        self.b_cell = torch.full((T + 1, N), -1, dtype=torch.int64, device=dev)
        self.b_parts = torch.zeros(T, 5, N, dtype=torch.long, device=dev)
        self.b_logp = torch.zeros(T, N, device=dev)
        self.b_forced = torch.zeros(T, N, dtype=torch.bool, device=dev)   # a held random move (explore_eps)
        self.b_stuck = torch.zeros(T, N, dtype=torch.bool, device=dev)    # ... whose hold began in a reward drought
        self.b_prac = None                      # a decision of a practice game (rl/stuck.py; set by the trainer)
        self.b_start0 = None                    # ... of a learner game from a level start (rl/stuck.py; coverage bcount0)
        self.b_val = torch.zeros(T + 1, N, device=dev)
        self.b_rew = torch.zeros(T, N, device=dev)
        self.b_done = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.b_info = torch.zeros(T, N, 8, device=dev)
        self.b_slvl = torch.zeros(T, N, dtype=torch.int32, device=dev)  # finished episode's start level (-1 power-on)
        self.b_lnow = torch.zeros(T, N, dtype=torch.int32, device=dev)  # level after each decision (-1 unknown)
        self.b_lev = torch.zeros(T, N, dtype=torch.bool, device=dev)    # a new level was reached at this decision
        self.b_rollback = torch.zeros(T, N, dtype=torch.bool, device=dev)
        self.h0 = torch.zeros(N, core, device=dev)

    FIRST = ('b_emph', 'b_masks', 'b_tl', 'b_prev', 'b_start', 'b_cell', 'b_ram')  # step-0 entries carried over

    def handoff_from(self, other, T):
        for k in self.FIRST:
            if getattr(self, k) is not None:
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
        if cfg.grid_coverage and (not cfg.grid_exploration or cfg.grid_regions or cfg.attempt_limit > 5000):
            raise ValueError('grid coverage needs grid-exploration=1, grid-regions=0, attempt-limit<=5000')
        if cfg.grid_exploration:
            if (games != ['smb'] or cfg.gamma != 1 or cfg.lag or self.ddp or cfg.places or cfg.int_coef or
                    cfg.div_coef or cfg.moods <= 0):
                raise ValueError('grid mode needs SMB, gamma=1, moods>0, lag=0, places=0, int/div-coef=0, single rank')
        if cfg.routes and (len(games) != 1 or cfg.places or cfg.grid_exploration or self.ddp):
            raise ValueError('routes need one game, places=0, no grid mode, single rank')
        units = max(1, cfg.rollout_split)
        cap = cfg.level_cap + (cfg.places + cfg.steps * units * cfg.place_k if cfg.places > 0 else 0) + \
            (cfg.route_slots if cfg.routes else 0)
        # explorer consoles (Config.explore_count): a fixed random share, the same on every resume; each has its own
        # slot for its try's start (rl/places.py doors)
        ge = torch.Generator().manual_seed(cfg.seed * 7 + self.rank + 13)
        n_env = cfg.envs_per_game * len(games)
        x_explorer = (torch.rand(n_env, generator=ge) < cfg.explore_share) if cfg.explore_count else \
            torch.zeros(n_env, dtype=torch.bool)
        if cfg.explore_count:
            cap += int(max(x_explorer[i * cfg.envs_per_game:(i + 1) * cfg.envs_per_game].sum() for i in range(len(games))))
        if cfg.situations > 0:
            if cfg.places <= 0 or cfg.blocks <= 0 or not cfg.coverage:
                raise ValueError('--situations needs --places, --blocks and --coverage')
            cap += cfg.steps * units * SIT_K + cfg.situations      # after every other slot (rl/places.py)
        if cfg.stuck_search:
            if games != ['smb'] or cfg.places <= 0 or not cfg.coverage or cfg.blocks <= 0 or cfg.explore_eps <= 0 or \
                    self.ddp:
                raise ValueError('--stuck-search needs SMB, --places, --coverage, --blocks, --explore-eps, one rank')
            cap += StuckSearch.slots_needed(cfg.steps, units)     # hand-off states and seed captures, last
        self.env = RlEnv({g: cfg.envs_per_game for g in games}, device=self.dev,
                         seed=cfg.seed * 7919 + self.rank, sticky=bool(cfg.sticky), idle_max=cfg.idle_max,
                         dropout=bool(cfg.dropout), level_cap=cap, level_slots=cfg.level_cap, guide_discount=cfg.gamma)
        self.level_plan = LevelPlan(self.env.games, cfg.level_cap, 0.98 ** (cfg.steps / 16)) if cfg.level_cap > 0 else None
        self.N, self.T = self.env.n, cfg.steps
        self.grid = GridReward(self.N, self.dev, history_scale=cfg.grid_history_scale,
                               regional=bool(cfg.grid_regions), coverage=bool(cfg.grid_coverage),
                               cell_size=cfg.grid_cell_size,
                               quadratic_stagnation=bool(cfg.grid_stagnation_quadratic),
                               persistent_visits=bool(cfg.grid_persistent_visits)) if cfg.grid_exploration else None
        if self.grid is not None:
            self.env.set_grid_attempts(cfg.attempt_limit, cfg.grid_cell_size)
        if cfg.episode_level_end >= 0:
            self.env.set_episode_level_end(cfg.episode_level_end)
        if cfg.coverage and (games != ['smb'] or cfg.routes or self.grid is not None or self.ddp):
            raise ValueError('coverage needs --games smb, no routes, no grid mode, single rank')
        if cfg.coverage or cfg.ram_io:
            self.env.bind_ram(range(2048))
        self.cov = Coverage(self.N, self.dev, cfg.rarity, cfg.blocks, cfg.block_half) if cfg.coverage else None
        self.cov_ref = RomMap(rom_path('smb'), self.dev) if cfg.coverage else None
        if cfg.coverage:
            self.env.set_allow_skip(True)
        self.cov_stats = None
        self.tb = None
        if cfg.coverage and self.rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.run / 'tb'))
        self.routes = None                # made in train(), once the env's saved slots are loaded
        if cfg.routes:
            self.env.set_attempts(cfg.attempt_limit)
        assert self.N == n_env
        self.places = Places(len(self.env.games), self.N, cfg.level_cap, cfg.places, cfg.place_k, cfg.steps, units,
                             cfg.place_per_level, self.dev, cfg.seed * 1000 + self.rank,
                             readmit=bool(cfg.place_readmit),
                             explorers=x_explorer.numpy() if cfg.explore_count else None,
                             game_rows=[(s.start, s.stop) for s in self.env.game_slices()],
                             share=bool(cfg.place_share), var=cfg.situations, var_k=SIT_K) if cfg.places > 0 else None
        self.sit = Situations(self.N, cfg.steps, self.dev, seed=cfg.seed) if cfg.situations > 0 else None
        assert self.places is None or self.places.cap == cap - (StuckSearch.slots_needed(cfg.steps, units)
                                                                if cfg.stuck_search else 0), (self.places.cap, cap)
        self.stuck = StuckSearch(self.run, self.N, self.dev, self.places.cap, cfg.level_cap, ~x_explorer.to(self.dev),
                                 cfg.seed * 1000 + self.rank, cfg.stuck_decisions, bool(cfg.stuck_sync),
                                 json.loads(cfg.stuck_params) if cfg.stuck_params else None, T=cfg.steps,
                                 units=units, gpu=bool(cfg.stuck_gpu), idle=cfg.idle_max) if cfg.stuck_search else None
        if cfg.imitate > 0 and self.stuck is None:
            raise ValueError('--imitate needs --stuck-search 1 (the teacher tries are its hand-off ways)')
        if self.stuck is not None:
            StuckSearch.setup_env(self.env)   # slot games end only when the trainer kills them (rl/stuck.py ENDS)
        self.local_game = torch.tensor(sum(([gi] * c for gi, c in enumerate(self.env.counts)), []))
        self.game_of_env = self.env.game_of_env
        compile_ = cfg.compile and self.cuda
        self.model = Policy(core=cfg.core, moods=cfg.moods, ram=bool(cfg.ram_io)).to(self.dev)
        self.div = Diversity(cfg.moods, self.N, self.game_of_env, cfg.seed * 1000 + self.rank, self.dev,
                             coef=cfg.div_coef, margin=cfg.div_margin) if cfg.moods else None
        self.rnd = RND(self.dev, cfg.seed * 1000 + self.rank, coef=cfg.int_coef) if cfg.int_coef > 0 else None
        if cfg.curiosity > 0 and self.ddp:
            raise ValueError('--curiosity is single-GPU (its heads have their own optimizer, not synced across ranks)')
        self.cur = Curiosity(self.N, cfg.core, self.dev, cfg.seed * 1000 + self.rank, cfg.curiosity, cfg.cur_heads,
                             cfg.cur_clip, lr=cfg.lr) if cfg.curiosity > 0 else None
        if cfg.outcome > 0 and (self.ddp or cfg.places <= 0):
            raise ValueError('--outcome is single-GPU and needs --places (its jump event is the places door jump)')
        self.out = Outcome(cfg.core, self.dev, cfg.outcome, lr=cfg.lr) if cfg.outcome > 0 else None
        if cfg.try_bonus > 0 and not cfg.explore_count:
            raise ValueError('--try-bonus needs --explore-count (place keys of every console)')
        self.tryb = TryBonus(cfg.try_bonus, self.dev) if cfg.try_bonus > 0 else None
        if cfg.ram_novelty > 0 and not cfg.coverage:
            raise ValueError('--ram-novelty needs --coverage (stored RAM + the screen key)')
        self.ramnov = RamNovelty(cfg.ram_novelty, self.dev) if cfg.ram_novelty > 0 else None
        if self.cuda:
            self.model = self.model.to(memory_format=torch.channels_last)
        if self.ddp:  # same initial weights on every rank
            for p in self.model.parameters():
                dist.broadcast(p.data, 0)
        # the behaviour policy: the model itself, or (lag) a copy refreshed before each rollout
        if cfg.lag:
            self.actor = Policy(core=cfg.core, moods=cfg.moods, ram=bool(cfg.ram_io)).to(self.dev)
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
        self.bufs = [Buffers(r, T, N, cfg.core, d, cfg.moods, cfg.int_coef > 0, bool(cfg.coverage or cfg.ram_io),
                             cfg.curiosity > 0, cfg.outcome > 0, cfg.try_bonus > 0)
                     for r in range(regions)]
        if cfg.stuck_search:
            for bb in self.bufs:
                bb.b_prac = torch.zeros(T, N, dtype=torch.bool, device=d)
                bb.b_start0 = torch.zeros(T, N, dtype=torch.bool, device=d)
                bb.b_teach = torch.zeros(T, N, dtype=torch.bool, device=d)   # a teacher try's decision (rl/stuck.py)
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
        self.x_hold = torch.zeros(self.N, dtype=torch.long, device=d)       # decisions left of a held random move
        self.x_parts = torch.zeros(5, self.N, dtype=torch.long, device=d)   # that move
        self.x_drought = torch.zeros(self.N, dtype=torch.long, device=d)    # decisions since the last paying step
        # the episode started from a saved place (--places): it starts as if already in a drought (the frontier is
        # where random tries are worth most; a place start would otherwise leave the place before a hold can start)
        self.x_place = torch.zeros(self.N, dtype=torch.bool, device=d)
        if cfg.explore_eps > 0 and cfg.places > 0:
            self.env.bind_cur_start()
        self.x_starts = torch.zeros((), dtype=torch.long, device=d)         # holds started in this rollout
        if cfg.explore_count and (cfg.explore_eps <= 0 or cfg.places <= 0):
            raise ValueError('--explore-count needs --explore-eps > 0 and --places (it counts tries per place key)')
        self.x_key = torch.full((self.N,), -1, dtype=torch.int64, device=d)  # current place key (-1 = unknown)
        # tries of each move (d-pad x A x B) per place, hashed rows (Config.explore_count)
        self.x_tried = torch.zeros(1 << TRIED_BITS, 36, dtype=torch.int32, device=d) if cfg.explore_count else None
        self.x_promise = torch.zeros((), device=d)                       # summed promise of explorers' choices
        self.x_promise_n = torch.zeros((), dtype=torch.long, device=d)
        self.x_cstarts = torch.zeros((), dtype=torch.long, device=d)        # of which started by an untried move
        self.x_stuck = torch.zeros(self.N, dtype=torch.bool, device=d)      # the current hold began in a drought
        # explorer consoles (Config.explore_count): a fixed random share, the same on every resume
        self.x_explorer = x_explorer.to(d)
        self.x_began = torch.zeros(self.N, dtype=torch.bool, device=d)      # an explorer's try begins this decision
        self.x_doors = torch.zeros((), dtype=torch.long, device=d)          # door starts that replay the door's move
        self.x_quiet = torch.zeros(self.N, dtype=torch.long, device=d)      # decisions since the last held move
        self.x_atdoor = torch.zeros(self.N, dtype=torch.bool, device=d)     # the episode started at a door place
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
    def encode_chunks(self, obs, emph, tl, prev, mood=None, ram=None):
        C = self.cfg.infer_chunk
        return torch.cat([self.actor_enc(obs[a:a + C], emph[a:a + C], tl[a:a + C], [p[a:a + C] for p in prev],
                                         None if mood is None else mood[a:a + C], None if ram is None else ram[a:a + C])
                          for a in range(0, obs.shape[0], C)])

    def start(self):
        b = self.bufs[self.nb]
        self.env.bind_obs(self.obs.slot(b.r, 0), b.b_emph[0])
        if self.routes is not None:
            self.apply_level_plan()       # the very first episodes start from saved slots too
        self.env.start()
        b.b_masks[0].copy_(self.env.masks)
        b.b_tl[0].copy_(self.env.timeline)
        b.b_start[0].fill_(True)
        b.b_cell[0].copy_(self.env.cells)
        if b.b_ram is not None:
            b.b_ram[0].copy_(self.env.ram)
        if self.cov is not None:
            self.cov.begin(self.env.ram)

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
                if getattr(b, k) is not None:
                    getattr(b, k)[0].copy_(getattr(b, k)[self.T])
        t0 = time.time()
        if self.cfg.per_game_rollout and self.cuda:
            self.rollout_per_game(b)
        else:
            self.rollout(b)
        if self.grid is not None:
            reward, self.grid_stats = self.grid.apply(b.b_cell, b.b_done, b.b_rew, b.b_mood,
                                                     self.cfg.moods, b.b_info)
            b.b_rew.copy_(reward)
        if self.cov is not None:
            b.b_rew.copy_(self.cov.apply(b.b_ram, b.b_done, b.b_prac, b.b_start0))
            self.cov_stats = self.cov.stats(self.cov_ref)
        if self.ramnov is not None:
            rr = self.ramnov.apply(b.b_ram, b.b_done, lambda r: smb_keys(r) >> 4, ~self.x_explorer)
            self.rn_stats = dict(self.ramnov.stats(), share=float(rr.sum() / (b.b_rew.sum() + rr.sum()).clamp_min(1e-8)))
            b.b_rew += rr
        if self.sit is not None:
            self.sit.end_rollout(b.b_ram, b.b_done, ~self.x_explorer)
        if self.tryb is not None:
            self.try_stats = dict(self.tryb.stats(), share=float(b.b_try.sum() / (b.b_rew.sum() + b.b_try.sum()).clamp_min(1e-8)))
            b.b_rew += b.b_try
        if self.cfg.explore_eps > 0:    # decisions since the last paying step (the rollout counted every step up)
            pay = (b.b_rew >= self.cfg.explore_pay) | b.b_done
            last = torch.where(pay, torch.arange(self.T, device=self.dev)[:, None], -1).amax(0)
            since = self.T - 1 - last
            restart = b.b_done.gather(0, last.clamp_min(0)[None])[0] & (last >= 0)   # the last event is a new episode
            since = since + torch.where(restart & self.x_place, self.cfg.explore_after, 0)
            self.x_drought = torch.where(last >= 0, since, self.x_drought)
        self.collect_episodes(b)
        self.sync()
        if self.places is not None:
            self.places.end_rollout(self.env)
        if self.stuck is not None:
            g0 = self.env.games[0]
            self.stuck.end_rollout(self.env, b, (self.cov_stats or {}).get('blocks_level', []),
                                   self.level_plan.rate(0), set(self.env.saved_levels()[g0]), self.cov, self.it)
        if self.cuda and not self.cfg.lag:
            # the per-game streams' cached activation memory would otherwise stay reserved through learning (and the
            # learner's through the next rollout): hand it back between the phases so their peaks do not add up
            torch.cuda.empty_cache()
        self.rollout_s = time.time() - t0
        self.last = b
        if self.cfg.lag:
            self.nb = 1 - self.nb
        return b

    def prac(self, sl):
        """[n] bool: consoles sl are in a practice game (rl/stuck.py); such games teach but count nowhere else."""
        return self.stuck.practising(sl) if self.stuck is not None else torch.zeros_like(self.x_explorer[sl])

    def teach(self, m, h, masks, sl, parts, logp):
        """Teacher tries (rl/stuck.py): their consoles press the search's button instead of the policy's move.
        Returns the parts, their log-prob under the policy (the stored 'old' log-prob must be of the pressed
        buttons) and [n] bool (the decision is the teacher's)."""
        on, btn = self.stuck.teacher(sl)
        parts = [torch.where(on, tp, p) for tp, p in zip(unpack(btn), parts)]
        return parts, torch.where(on, m.evaluate(h, masks, parts)[0], logp), on

    def x_restart(self, sl, done):
        """Held-move state of consoles sl after a step: a finished episode drops its hold; the new one starts with no
        drought, or with a full one (explore_after) when it starts from a saved place."""
        if self.stuck is not None:
            self.stuck.on_restart(sl, done, self.env.cur_start[sl])
        self.x_hold[sl] *= ~done
        if self.cfg.explore_eps > 0 and self.cfg.places > 0:
            self.x_place[sl] = torch.where(done, self.env.cur_start[sl] >= self.cfg.level_cap, self.x_place[sl])
        after = torch.where(self.x_place[sl], self.cfg.explore_after, 0)
        self.x_drought[sl] = torch.where(done, after, self.x_drought[sl])
        if self.places is not None and self.places.doors:
            # a start at a door place (rl/places.py) may first replay the move that opened the door; on a learner it
            # may teach (as a hold begun in a drought)
            mv, ln = self.places.door_of(self.game_of_env[sl], self.env.cur_start[sl])
            self.x_atdoor[sl] = torch.where(done, ln > 0, self.x_atdoor[sl])
            door = done & (ln > 0) & (torch.rand(ln.shape[0], device=self.dev, generator=self.gen_roll) < DOOR_REPLAY)
            self.x_hold[sl] = torch.where(door, ln, self.x_hold[sl])
            self.x_parts[:3, sl] = torch.where(door, mv.T, self.x_parts[:3, sl])
            self.x_parts[3:, sl] = torch.where(door, 0, self.x_parts[3:, sl])
            self.x_stuck[sl] = torch.where(door, ~self.x_explorer[sl], self.x_stuck[sl])
            self.x_doors += door.sum()

    def explore(self, m, h, masks, parts, logp, sl):
        """Held random moves for the consoles sl (see Config.explore_eps). Returns the moves to play, their
        log-prob under the policy, and which of them are held ones."""
        cfg = self.cfg
        hold, d, xp = self.x_hold[sl], self.x_drought[sl], self.x_parts[:, sl]
        n = hold.shape[0]
        u = torch.rand(n, 2, device=self.dev, generator=self.gen_roll)
        start = (hold == 0) & (d >= cfg.explore_after) & (u[:, 0] < cfg.explore_eps)
        length = (cfg.explore_min * (cfg.explore_max / cfg.explore_min) ** u[:, 1]).long()
        dpad = torch.multinomial(masks[:, :9].float(), 1, generator=self.gen_roll).squeeze(1)   # any legal direction
        ab = torch.randint(0, 2, (2, n), device=self.dev, generator=self.gen_roll)            # A, B: off or hold
        if self.x_tried is not None:
            # least-tried legal move at the current place (ties broken at random); untried here -> start at once
            key = self.x_key[sl]
            known = key >= 0
            row = ((key * 0x5851F42D4C957F2D) >> (64 - TRIED_BITS)) & ((1 << TRIED_BITS) - 1)
            tries = self.x_tried[row]                                                         # n, 36
            ok = (masks[:, :9, None, None] * masks[:, 9:11, None][:, None] * masks[:, 12:14][:, None, None]) > 0
            ex = self.x_explorer[sl]
            v = torch.rand(n, device=self.dev, generator=self.gen_roll)
            if self.out is None:
                score = tries.float() + torch.rand(n, 36, device=self.dev, generator=self.gen_roll) * 0.5
                j = score.masked_fill(~ok.reshape(n, 36), float('inf')).argmin(1)
                least = tries.gather(1, j[:, None]).squeeze(1).float()
                chance = 1.0 / (1.0 + least)
            else:
                # rl/outcome.py: promise (chance of a jump / new level within 16 decisions) + count bonus; a move is
                # tried at least as often as its promise says, untried ones at once
                pr = self.out.promise(h)
                score = pr + (1.0 + tries.float()).rsqrt() + torch.rand(n, 36, device=self.dev, generator=self.gen_roll) * 0.01
                j = score.masked_fill(~ok.reshape(n, 36), float('-inf')).argmax(1)
                least = tries.gather(1, j[:, None]).squeeze(1).float()
                pj = pr.gather(1, j[:, None]).squeeze(1)
                chance = torch.maximum(1.0 / (1.0 + least), pj)
                self.x_promise += (pj * (ex & (hold == 0) & known)).sum()
                self.x_promise_n += (ex & (hold == 0) & known).sum()
            fresh = ex & (hold == 0) & known & (v < chance) & ~start
            start = start | fresh
            length = torch.where(fresh, cfg.explore_min, length)     # a try of a move here is short (see Config)
            pick = known & start & ex       # learners' holds keep a uniformly random move
            self.x_began[sl] = pick
            dpad = torch.where(pick, j // 4, dpad)
            ab = torch.where(pick, torch.stack([(j // 2) % 2, j % 2]), ab)
            self.x_tried.index_put_((row[pick], j[pick]), torch.ones_like(j[pick], dtype=torch.int32), accumulate=True)
            self.x_cstarts += fresh.sum()
        zero = torch.zeros_like(dpad)
        new = torch.stack([dpad, ab[0], ab[1], zero, zero])
        xp = torch.where(start, new, xp)
        hold = torch.where(start, length, hold)
        # a hold may teach the policy (when it paid off) only if it began in a drought, on a learner
        self.x_stuck[sl] = torch.where(start, (d >= cfg.explore_after) & ~self.x_explorer[sl], self.x_stuck[sl])
        legal = torch.ones_like(start)
        for k, (a, _) in enumerate(PARTS[:3]):      # dropout can take a held input away: the hold ends
            legal &= masks[:, a:].gather(1, xp[k][:, None]).squeeze(1) > 0
        active = (hold > 0) & legal
        out = [torch.where(active, xp[k], parts[k]) for k in range(5)]
        logp = torch.where(active, m.evaluate(h, masks, out)[0], logp)
        self.x_parts[:, sl] = xp
        self.x_hold[sl] = torch.where(active, hold - 1, torch.zeros_like(hold))
        self.x_quiet[sl] = torch.where(active, 0, self.x_quiet[sl] + 1)
        self.x_drought[sl] = d + 1
        self.x_starts += (start & active).sum()
        return out, logp, active

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
                                       None if dv is None else b.b_mood[t], None if b.b_ram is None else b.b_ram[t])
            h = h * (~b.b_start[t]).unsqueeze(1).float()
            h = m.gru(z.float(), h)
            parts, logp = m.act(h, b.b_masks[t], generator=self.gen_roll)
            if self.cfg.explore_eps > 0:
                parts, logp, b.b_forced[t] = self.explore(m, h, b.b_masks[t], parts, logp, slice(None))
                if self.tryb is not None:
                    b.b_try[t] = self.tryb.step(self.x_key, parts, ~self.x_explorer)
                b.b_stuck[t] = b.b_forced[t] & self.x_stuck
                if self.x_tried is not None:
                    for gi, gsl in enumerate(env.game_slices()):
                        self.places.save_tries(env, gi, gsl, 0, self.x_began[gsl], self.x_key[gsl], env.level_now[gsl],
                                               self.x_parts[:3, gsl], self.x_hold[gsl] + 1)
            if self.cur is not None:
                b.b_dis[t] = self.cur.disagreement(h, parts)
            b.b_val[t] = mu[g] + sd[g] * self.hl.mean(m.value(h))
            if self.cfg.imitate > 0:
                parts, logp, b.b_teach[t] = self.teach(m, h, b.b_masks[t], slice(None), parts, logp)
            for k in range(5):
                b.b_parts[t, k] = parts[k]
            b.b_logp[t] = logp
            if b.b_prac is not None:
                b.b_prac[t] = self.stuck.practising(slice(None))
                b.b_start0[t] = self.stuck.from_start(slice(None))
            env.actions.copy_(pack(parts))
            self.obs.before_write(t)
            env.bind_obs(self.obs.slot(b.r, t + 1), b.b_emph[t + 1])
            env.step()
            self.obs.saved(b.r, t + 1)
            b.b_rew[t] = env.reward
            b.b_cell[t + 1] = env.cells
            if b.b_ram is not None:
                b.b_ram[t + 1] = env.ram
            b.b_rollback[t] = (env.events & 64) != 0
            done = env.done.bool()
            if self.stuck is not None:
                self.stuck.fix_end(env, slice(None), done)
            b.b_done[t] = done
            self.x_restart(slice(None), done)
            if dv is not None:
                dv.after_step(slice(None), env.parts, done, b.b_feat[t])
            if self.rnd is not None:
                b.b_small[t] = self.rnd.small(self.obs.slot(b.r, t + 1)[:, 3], b.b_emph[t + 1, :, 3])
            if self.cur is not None:
                b.b_fp[t] = self.cur.fingerprint(self.obs.slot(b.r, t + 1)[:, 3], b.b_emph[t + 1, :, 3])
            if self.places is not None:
                for gi, gsl in enumerate(env.game_slices()):
                    bk = smb_block_keys(env.ram[gsl]) if self.cfg.blocks > 0 else None
                    self.places.step(env, t, gi, 0, 0, gsl, self.obs.slot(b.r, t + 1)[gsl, 3], b.b_emph[t + 1, gsl, 3],
                                     env.level_now[gsl], env.progress_now[gsl], done[gsl], bk,
                                     (~self.x_explorer[gsl] & ~self.prac(gsl)) if self.cfg.explore_count else None,
                                     env.parts[gsl, 2] != 0,
                                     ~self.x_explorer[gsl] & (self.x_quiet[gsl] > DOOR_GRACE) & ~self.x_atdoor[gsl],
                                     None if self.sit is None else
                                     self.sit.step(t, env.ram[gsl], bk, smb_height(env.ram[gsl]), gsl, done[gsl]))
                    if self.stuck is not None:
                        self.stuck.after_step(env, gsl, bk, env.level_now[gsl], done[gsl], env.parts[gsl, 2] != 0)
                        self.stuck.capture_step(env, t, 0, 0, gsl, bk, env.level_now[gsl], done[gsl], self.cov.bcount0)
                if self.x_tried is not None:
                    self.x_key.copy_(self.places.b_key[t])
            if self.out is not None:
                b.b_ev[t] = Outcome.events(env.parts, self.places.jump_now)
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
                                   None if dv is None else dv.mood, None if b.b_ram is None else b.b_ram[T])
        hT = m.gru(z.float(), h * (~b.b_start[T]).unsqueeze(1).float())
        b.b_val[T] = mu[g] + sd[g] * self.hl.mean(m.value(hT))
        self.h = h  # state after the last step (before the reset of step T's start, applied in the next rollout)
        self.decisions += T * self.N * self.world

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
                                               [p[sl] for p in b.b_prev[t]], None if dv is None else b.b_mood[t, sl],
                                               None if b.b_ram is None else b.b_ram[t, sl])
                    h = hg[gi] * (~b.b_start[t, sl]).unsqueeze(1).float()
                    h = m.gru(z.float(), h)
                    hg[gi] = h
                    parts, logp = m.act(h, b.b_masks[t, sl], generator=self.gen_roll)
                    if self.cfg.explore_eps > 0:
                        parts, logp, b.b_forced[t, sl] = self.explore(m, h, b.b_masks[t, sl], parts, logp, sl)
                        if self.tryb is not None:
                            b.b_try[t, sl] = self.tryb.step(self.x_key[sl], parts, ~self.x_explorer[sl])
                        b.b_stuck[t, sl] = b.b_forced[t, sl] & self.x_stuck[sl]
                        if self.x_tried is not None:
                            ug, ua = self.units[gi][0], self.units[gi][1]
                            self.places.save_tries(env, ug, sl, ua, self.x_began[sl], self.x_key[sl], env.level_now[sl],
                                                   self.x_parts[:3, sl], self.x_hold[sl] + 1)
                    if self.cur is not None:
                        b.b_dis[t, sl] = self.cur.disagreement(h, parts)
                    b.b_val[t, sl] = mu[g[sl]] + sd[g[sl]] * self.hl.mean(m.value(h))
                    if self.cfg.imitate > 0:
                        parts, logp, b.b_teach[t, sl] = self.teach(m, h, b.b_masks[t, sl], sl, parts, logp)
                    for k in range(5):
                        b.b_parts[t, k, sl] = parts[k]
                    b.b_logp[t, sl] = logp
                    if b.b_prac is not None:
                        b.b_prac[t, sl] = self.stuck.practising(sl)
                        b.b_start0[t, sl] = self.stuck.from_start(sl)
                    env.actions[sl] = pack(parts)
                    self.obs.before_write_game(gi, t)
                    ug, ua, un, _ = self.units[gi]
                    env.step_range(ug, ua, un, self.obs.slot(b.r, t + 1), b.b_emph[t + 1])
                    self.obs.saved_game(b.r, t + 1, gi, sl)
                    b.b_rew[t, sl] = env.reward[sl]
                    b.b_cell[t + 1, sl] = env.cells[sl]
                    if b.b_ram is not None:
                        b.b_ram[t + 1, sl] = env.ram[sl]
                    b.b_rollback[t, sl] = (env.events[sl] & 64) != 0
                    done = env.done[sl].bool()
                    if self.stuck is not None:
                        self.stuck.fix_end(env, sl, done)
                    b.b_done[t, sl] = done
                    self.x_restart(sl, done)
                    if dv is not None:
                        dv.after_step(sl, env.parts[sl], done, b.b_feat[t, sl])
                    if self.rnd is not None:
                        b.b_small[t, sl] = self.rnd.small(self.obs.slot(b.r, t + 1)[sl, 3], b.b_emph[t + 1, sl, 3])
                    if self.cur is not None:
                        b.b_fp[t, sl] = self.cur.fingerprint(self.obs.slot(b.r, t + 1)[sl, 3], b.b_emph[t + 1, sl, 3])
                    if self.places is not None:
                        bk = smb_block_keys(env.ram[sl]) if self.cfg.blocks > 0 else None
                        self.places.step(env, t, ug, self.unit_j[gi], ua, sl, self.obs.slot(b.r, t + 1)[sl, 3],
                                         b.b_emph[t + 1, sl, 3], env.level_now[sl], env.progress_now[sl], done, bk,
                                         (~self.x_explorer[sl] & ~self.prac(sl)) if self.cfg.explore_count else None,
                                         env.parts[sl, 2] != 0,
                                         ~self.x_explorer[sl] & (self.x_quiet[sl] > DOOR_GRACE) & ~self.x_atdoor[sl],
                                         None if self.sit is None else
                                         self.sit.step(t, env.ram[sl], bk, smb_height(env.ram[sl]), sl, done))
                        if self.stuck is not None:
                            self.stuck.after_step(env, sl, bk, env.level_now[sl], done, env.parts[sl, 2] != 0)
                            self.stuck.capture_step(env, t, self.unit_j[gi], ua, sl, bk, env.level_now[sl], done,
                                                    self.cov.bcount0)
                        if self.x_tried is not None:
                            self.x_key[sl] = self.places.b_key[t, sl]
                    if self.out is not None:
                        b.b_ev[t, sl] = Outcome.events(env.parts[sl], self.places.jump_now[sl])
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
                                           [p[sl] for p in b.b_prev[T]], None if dv is None else dv.mood[sl],
                                           None if b.b_ram is None else b.b_ram[T, sl])
                hT = m.gru(z.float(), hg[gi] * (~b.b_start[T, sl]).unsqueeze(1).float())
                b.b_val[T, sl] = mu[g[sl]] + sd[g[sl]] * self.hl.mean(m.value(hT))
        for gi, s in enumerate(self.game_streams):
            main.wait_stream(s)
            hg[gi].record_stream(main)  # used below on the main stream; allocated on the game stream
        self.h = torch.cat(hg)  # state after the last step (before the reset of step T's start)
        self.decisions += T * self.N * self.world

    def apply_level_plan(self):
        """Level-start shares and weights for this rollout, from the clear rates so far (rl/level_plan.py)."""
        if self.level_plan is None or self.cfg.level_starts <= 0:
            return
        saved = self.env.saved_levels()
        self.level_report = {}
        if self.routes is not None:
            g = self.env.games[0]
            only = [int(x) for x in self.cfg.only_levels.split(',')] if self.cfg.only_levels else sorted(saved[g])
            self.env.set_level_plan(0, 1.0, self.routes.weights([l for l in only if l in saved[g]]))
            return
        for gi, g in enumerate(self.env.games):
            w, share = self.level_plan.plan(gi, saved[g], self.cfg.level_starts)
            r = self.level_plan.rate(gi)
            if self.cfg.only_levels and w is not None:
                keep = [int(x) for x in self.cfg.only_levels.split(',') if int(x) in saved[g]]
                if keep:
                    f = np.zeros_like(w)
                    f[keep] = np.maximum(1.0 - r[keep], 0.02)
                    w, share = f / f.sum(), 1.0
            ws = w
            if w is not None and self.places is not None:
                ws = self.places.weights(gi, w, r) if self.stuck is None else self.stuck.weights(
                    self.places.weights(gi, w, r), w, r, self.places.level_of(gi, np.arange(self.places.cap)))
            self.env.set_level_plan(gi, share, ws)
            self.level_report[g] = dict(share=share, levels={
                str(l): dict(clear=float(r[l]), visits=float(self.level_plan.visits[gi, l]), weight=float(w[l]),
                             deaths=saved[g][l]) for l in sorted(saved[g])})

    def collect_episodes(self, b):
        """Finished episodes of this rollout (one host copy per rollout, no per-step sync)."""
        t, e = b.b_done.nonzero(as_tuple=True)
        if b.b_prac is not None:          # practice tries (rl/stuck.py) are no episodes of the stats
            keep = ~b.b_prac[t, e]
            t, e = t[keep], e[keep]
        info = b.b_info[t, e].cpu().numpy()
        start = b.b_slvl[t, e].cpu().numpy()
        if self.places is not None and len(start):
            lg = self.local_game[e.cpu()].numpy()
            slot = start
            start = np.array([self.places.level_of(int(a), [int(s)])[0] for a, s in zip(lg, start)], np.int64)
            if self.stuck is not None:
                start = self.stuck.level_of(slot, start)
            won = (info[:, 4] > start) | (info[:, 2] == 2) | (info[:, 2] == 6)     # cleared the start's level
            mine = ~self.x_explorer[e].cpu().numpy()                             # the learners' games only
            for a in np.unique(lg):
                self.places.add_starts(int(a), slot[(lg == a) & mine], won[(lg == a) & mine])
        if self.routes is not None:
            self.routes.add(start, (info[:, 2] == 6) | (info[:, 2] == 2), info[:, 1])
            start = np.where(start >= 0, self.routes.level_of[start], -1)
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
        lr = ~self.x_explorer.unsqueeze(0).expand_as(b.b_done)   # explorers' games are not the policy's (explore_count)
        if b.b_prac is not None:
            lr = lr & ~b.b_prac                                   # nor are practice tries (rl/stuck.py) level visits
        cleared = b.b_lev & (lv >= 1) & (lv <= cap) & lr        # the level before the one just reached
        ended = b.b_done & (lv >= 0) & (lv < cap) & (end != 3) & (end != 5) & (end != 6) & lr  # skip/fault/level-clear
        won = ended & (end == 2)
        native = b.b_done & (end == 6) & ~b.b_lev & (lv >= 0) & (lv < cap) & lr
        slot = lambda m, off: (g[m] * cap + lv[m] - off)  # noqa: E731
        v = torch.zeros(G * cap, device=self.dev)
        c = torch.zeros(G * cap, device=self.dev)
        for acc, idx in ((v, slot(cleared, 1)), (c, slot(cleared, 1)), (v, slot(ended & ~won, 0)), (v, slot(won, 0)),
                         (c, slot(won, 0)), (v, slot(native, 0)), (c, slot(native, 0))):
            acc.index_add_(0, idx, torch.ones(idx.shape[0], device=self.dev))
        self.level_plan.add(v.view(G, cap).double().cpu().numpy(), c.view(G, cap).double().cpu().numpy())

    # ---------------- learning ----------------
    def gae(self, b, rew):
        """GAE. With held moves (explore_eps), a move of the policy's own followed by a held one gets no credit or
        blame for what the held move then did: its trace stops there and it bootstraps from the value of that state
        (as V-trace / Retrace cut traces at off-policy actions). A held move's own trace runs on (its advantage is what
        the try led to, the policy playing on after it)."""
        T, cfg = self.T, self.cfg
        adv = torch.zeros(T, self.N, device=self.dev)
        last = torch.zeros(self.N, device=self.dev)
        cut = torch.zeros(T, self.N, dtype=torch.bool, device=self.dev)
        if getattr(cfg, 'explore_eps', 0) > 0:
            cut[:-1] = b.b_forced[1:] & ~b.b_forced[:-1]
        for t in reversed(range(T)):
            nonterm = (~b.b_done[t]).float()
            delta = rew[t] + cfg.gamma * b.b_val[t + 1] * nonterm - b.b_val[t]
            last = delta + cfg.gamma * cfg.lam * nonterm * (~cut[t]).float() * last
            adv[t] = last
        return adv, adv + b.b_val[:T]

    def per_game_norm(self, adv, own=None):
        """Mean 0 / std 1 advantages per game over the whole rollout (all ranks). Without it the per-minibatch
        normalisation mixes games whose returns differ ~60x in scale (life-loss games vs progress-only games), so the
        quiet games' policy gradient shrinks below the entropy bonus and their policy drifts back to uniform.
        own [T, N] bool: the statistics come from these moves only (the policy's own, not held random moves). Held
        moves are mostly worse than the policy; counted in the mean they pull it down, every own move then looks
        better than average and gets reinforced, which sharpens the policy onto its habits (2026-11-19: with ~50%
        held moves own moves averaged +0.19 std and 4-4 went from 44/64 to 0/64 clears in 25 updates)."""
        g = self.game_of_env.expand_as(adv).reshape(-1)
        a = adv.reshape(-1).double()
        w = torch.ones_like(a) if own is None else own.reshape(-1).double()
        s = torch.zeros(3, len(GAMES), device=self.dev, dtype=torch.float64)
        s[0].index_add_(0, g, w)
        s[1].index_add_(0, g, a * w)
        s[2].index_add_(0, g, a * a * w)
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

    def mb_loss(self, obs, e, b_emph, b_tl, b_prev, h0, b_start, b_masks, b_parts, b_logp, adv, target, pw, vmask,
                b_mood=None, b_ram=None, b_done=None, b_cur=None, emask=None, b_oy=None, b_ok=None, bw=None):
        """PPO loss of one minibatch (whole T-step sequences of the envs e); compiled as one graph on CUDA.
        With b_ram (--ram-io): plus the RAM-change prediction loss (if ram_coef > 0). With b_cur (--curiosity, the
        normalised fingerprint targets): plus the curiosity heads' loss. emask [T, N] (explore_count): the moves
        whose entropy bonus counts (the learners'); None = all."""
        cfg, m, T, mb = self.cfg, self.model, self.T, e.shape[0]
        ram_now = None if b_ram is None else b_ram[:T, e].flatten(0, 1)
        with self.autocast():
            z = m.encode(obs, b_emph[:T, e].flatten(0, 1), b_tl[:T, e].flatten(0, 1),
                         [p.flatten(0, 1) for p in b_prev[:T, :, e].transpose(0, 1)],
                         None if b_mood is None else b_mood[:, e].flatten(0, 1), ram_now)
        hs, _ = m.recur(z.view(T, mb, -1), h0[e], b_start[:T, e])
        hs = hs.flatten(0, 1)
        parts = [p.flatten(0, 1) for p in b_parts[:, :, e].transpose(0, 1)]
        logp, ent = m.evaluate(hs, b_masks[:T, e].flatten(0, 1), parts)
        vlog = m.value(hs).float()
        a = adv[:, e].reshape(-1)
        vw = vmask[:, e].reshape(-1)    # the policy's own moves (0 = held): they set the normalisation and train the value
        am = (a * vw).sum() / vw.sum().clamp_min(1)
        a = (a - am) / ((((a - am) ** 2) * vw).sum() / (vw.sum() - 1).clamp_min(1)).sqrt().add(1e-8)
        logratio = logp - b_logp[:, e].reshape(-1)
        ratio = logratio.exp()
        wt = pw[:, e].reshape(-1)       # 0 for held random moves that did not turn out well (explore_eps)
        pg = (wt * torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip))).sum() / wt.sum().clamp_min(1)
        v = (-(target[:, e].reshape(-1, target.shape[-1]) * vlog.log_softmax(-1)).sum(-1) * vw).sum() / vw.sum().clamp_min(1)
        if emask is None:
            entm = ent.mean()
        else:
            # explorers' games never teach: with no policy loss there, their entropy bonus alone would push the
            # policy toward random moves where explorers crowd (the level start after their many deaths)
            ew = emask[:, e].reshape(-1)
            entm = (ent * ew).sum() / ew.sum().clamp_min(1)
        loss = pg + cfg.vf_coef * v - cfg.ent_coef * entm
        extra = []
        if b_ram is not None and cfg.ram_coef > 0:
            # target: the next decision's RAM bits ('next'), or which of them differ from now ('flip'). After an
            # end the next RAM is another episode's start: those steps are left out
            ram_next = b_ram[1:T + 1, e].flatten(0, 1)
            flip = m.bits(ram_next ^ ram_now).float()
            tgt = m.bits(ram_next).float() if cfg.ram_target == 'next' else flip
            keep = (~b_done[:, e]).reshape(-1).float()
            with self.autocast():
                lg = m.ram_logits(hs, parts)
            bce = F.binary_cross_entropy_with_logits(lg.float(), tgt, reduction='none').sum(-1).mean(-1)
            ram_loss = (bce * keep).sum() / keep.sum().clamp_min(1)
            loss = loss + cfg.ram_coef * ram_loss
            with torch.no_grad():
                right = ((lg > 0).float() == tgt).all(-1).float()          # [B, 2048]: whole byte right
                changed = flip.amax(-1) * keep[:, None]                     # bytes that change at a kept step
                exact = (right * changed).sum() / changed.sum().clamp_min(1)
                all_exact = (right * keep[:, None]).sum() / (keep.sum() * 2048).clamp_min(1)
            extra = [ram_loss.detach(), exact, all_exact]
        if b_cur is not None:
            cur_loss = self.cur.loss(hs, parts, b_cur[:, e].flatten(0, 1), (~b_done[:, e]).reshape(-1), e, T)
            loss = loss + cfg.cur_loss_coef * cur_loss
            extra = extra + [cur_loss.detach()]
        if b_oy is not None:
            # rl/outcome.py: chances of events after each move; also trains the policy's network
            out_loss = self.out.loss(hs, parts, b_oy[:, e].flatten(0, 1), b_ok[:, e].flatten(0, 1))
            loss = loss + cfg.outcome * out_loss
            extra = extra + [out_loss.detach()]
        if bw is not None:
            # imitation (--imitate): the policy's log-probability of the teacher's buttons, summed over the teacher
            # decisions and divided by ALL the minibatch's decisions: weight 1 pulls about as hard as a policy-
            # gradient move with normalised advantage +1
            tw = bw[:, e].reshape(-1)
            bc = -(tw * logp).sum() / tw.numel()
            loss = loss + cfg.imitate * bc
            extra = extra + [bc.detach(), ((tw * logp).sum() / tw.sum().clamp_min(1)).detach()]
        with torch.no_grad():
            kls = (ratio - 1) - logratio
            kl = kls.mean()
            clipfrac = ((ratio - 1).abs() > cfg.clip).float().mean()
            used = (wt > 0).float()          # the moves the policy loss learns from (held moves only when they paid)
            kl_used = (kls * used).sum() / used.sum().clamp_min(1)
        return loss, torch.stack([pg.detach(), v.detach(), entm.detach(), kl, clipfrac] + extra + [kl_used])

    def learn(self, b):
        cfg, m, T = self.cfg, self.model, self.T
        rew, dstats = b.b_rew, None
        if self.div is not None and cfg.div_coef:
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
        cur_target = cstats = None
        if self.cur is not None:
            cur_target, cb = self.cur.prepare(b.b_fp, b.b_dis, b.b_done)
            rew = rew + cb
            cstats = (cb, rew)
        adv, ret = self.gae(b, rew)
        learner = ~self.x_explorer.unsqueeze(0).expand_as(b.b_forced)    # explorers' games never teach
        if cfg.adv_norm == 'game':
            own_m = ~b.b_forced & learner if cfg.explore_eps > 0 else None
            if cfg.imitate > 0:                       # teacher moves are not the policy's own either
                own_m = ~b.b_teach if own_m is None else own_m & ~b.b_teach
            adv = self.per_game_norm(adv, own_m)
        pw, vmask, xstats = torch.ones_like(adv), torch.ones_like(adv), None
        emask = learner.float() if cfg.explore_count else None
        if cfg.explore_eps > 0:
            # only a learner's hold that began while stuck (a drought) can teach the policy, as before 2026-11-19
            hit = b.b_forced & b.b_stuck & learner & (adv > cfg.explore_adv_min)
            pw = ((~b.b_forced | hit) & learner).float()
            vmask = (~b.b_forced & learner).float()
            xstats = dict(forced=float(b.b_forced[learner].float().mean()), starts=int(self.x_starts), hits=int(hit.sum()),
                          forced_ex=float(b.b_forced[~learner].float().mean()) if bool((~learner).any()) else 0.0,
                          hit_adv=float(adv[hit].mean()) if bool(hit.any()) else 0.0, untried=int(self.x_cstarts),
                          door_starts=int(self.x_doors))
            self.x_starts.zero_()
            self.x_cstarts.zero_()
            self.x_doors.zero_()
        bw = None
        if cfg.imitate > 0:
            # teacher tries (rl/stuck.py): the search's buttons, not the policy's: only the imitation loss learns
            # from them (no policy-gradient, value or entropy term)
            own = (~b.b_teach).float()
            pw, vmask, bw = pw * own, vmask * own, b.b_teach.float()
            emask = own if emask is None else emask * own
        self.update_stats(ret)
        mu, sd = self.norm_stats()
        g = self.game_of_env
        normalized_ret = (ret - mu[g]) / sd[g]
        target = self.hl.target(normalized_ret)                        # T, N, bins
        ram = b.b_ram is not None and cfg.ram_io
        ram_pred = ram and cfg.ram_coef > 0
        oy = ok = None
        if self.out is not None:
            oy, ok = self.out.labels(b.b_ev, b.b_rew, b.b_done)
        names = ['pg', 'v', 'ent', 'kl', 'clipfrac', 'gnorm'] + (['ram_loss', 'ram_byte_exact', 'ram_all_exact']
                                                                if ram_pred else []) + (['cur_loss'] if self.cur else []) + (['out_loss'] if self.out else []) + (['bc_loss', 'bc_logp'] if bw is not None else []) + ['kl_used']
        acc = torch.zeros(len(names), device=self.dev)
        kl_first, n = None, 0
        mb = min(cfg.minibatch_envs, self.N)
        for _ in range(cfg.epochs):
            perm = torch.randperm(self.N, device=self.dev)
            self.obs.epoch(b.r, perm, mb)
            for i in range(self.N // mb):
                e = perm[i * mb:(i + 1) * mb]
                loss, st = self.mb_loss_c(self.obs.batch(i), e, b.b_emph, b.b_tl, b.b_prev, b.h0, b.b_start,
                                          b.b_masks, b.b_parts, b.b_logp, adv, target, pw, vmask, b.b_mood,
                                          b.b_ram if ram else None, b.b_done if (ram or self.cur) else None,
                                          cur_target, emask, oy, ok, bw)
                self.opt.zero_grad(set_to_none=True)
                if self.cur is not None:
                    self.cur.opt.zero_grad(set_to_none=True)
                if self.out is not None:
                    self.out.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                # average the still-scaled grads first: every rank then sees the same values, so the fp16 scaler
                # skips an overflowing step on all ranks together (unscaling first lets ranks disagree and diverge)
                self.sync_grads()
                self.scaler.unscale_(self.opt)
                gn = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.opt)
                if self.cur is not None:
                    self.scaler.unscale_(self.cur.opt)
                    torch.nn.utils.clip_grad_norm_(self.cur.heads.parameters(), cfg.max_grad_norm)
                    self.scaler.step(self.cur.opt)
                if self.out is not None:
                    self.scaler.unscale_(self.out.opt)
                    torch.nn.utils.clip_grad_norm_(self.out.head.parameters(), cfg.max_grad_norm)
                    self.scaler.step(self.out.opt)
                self.scaler.update()
                self.obs.done_with(i)
                if kl_first is None:  # before any update: ~0 with --lag 0 (rollout and learner agree)
                    kl_first = st[3].clone()
                acc[:5] += st[:5]
                acc[6:] += st[5:]
                acc[5] += gn.float()
                n += 1
        acc = (acc / max(n, 1)).tolist()                                 # one host sync per iteration
        out = dict(zip(names, acc))
        out['kl_first'] = float(kl_first)
        with torch.no_grad():
            y, pred = ret.reshape(-1), b.b_val[:T].reshape(-1)
            out['explained_var'] = float(1 - (y - pred).var() / (y.var() + 1e-8))
        out['reward_mean'] = float(b.b_rew.mean())
        if self.grid is not None:
            out['grid'] = self.grid_stats
        if self.cov_stats is not None:
            out['coverage'] = self.cov_stats
        out['rollback_events'] = int(b.b_rollback.sum())
        if xstats is not None:
            out['explore'] = xstats
        out['value_target_clipfrac'] = float((normalized_ret.abs() > self.hl.vmax).float().mean())
        if self.rnd is not None:
            out['rnd_loss'] = self.rnd.train(b.b_small)
            ib, ierr = istats
            g = self.game_of_env.unsqueeze(0).expand_as(ib)
            out['curiosity'] = {GAMES[gi]: dict(bonus=float(ib[g == gi].mean()), error=float(ierr[g == gi].mean()))
                                for gi in self.game_of_env.unique().tolist()}
        if cstats is not None:
            cb, total = cstats
            keep = ~b.b_done
            out['cur'] = dict(loss=out.pop('cur_loss'), bonus=float(cb.mean()), share=float(cb.sum() / total.sum().clamp_min(1e-8)),
                              dis=float(b.b_dis[keep].mean()), avg=float(self.cur.avg),
                              top1=float(cb[keep].float().quantile(0.99)) if bool(keep.any()) else 0.0,
                              held=float(cb[b.b_forced & keep].mean()) if bool((b.b_forced & keep).any()) else 0.0)
        if self.tryb is not None:
            out['try'] = self.try_stats
        if self.ramnov is not None:
            out['ramnov'] = self.rn_stats
        if cfg.coverage and b.b_lnow is not None:
            # reward earned per level (learners' moves only), for the per-level reward graph: [reward sum, decisions]
            with torch.no_grad():
                lv = b.b_lnow.long()
                mk = (lv >= 0) & (lv < 64) & learner
                rs = torch.zeros(64, device=self.dev).index_add_(0, lv[mk], b.b_rew[mk].float())
                ns = torch.zeros(64, device=self.dev).index_add_(0, lv[mk], torch.ones_like(b.b_rew[mk]))
                out['level_reward'] = {int(i): [round(float(rs[i]), 2), int(ns[i])] for i in ns.nonzero().flatten().tolist()}
        if self.out is not None:
            with torch.no_grad():
                rate = ((oy * ok).sum((0, 1)) / ok.float().sum((0, 1)).clamp_min(1)).tolist()
            out['outcome'] = dict(loss=out.pop('out_loss'), rate16={k: rate[len(EVENTS_) + i] for i, k in enumerate(EVENTS_)},
                                  promise=float(self.x_promise / self.x_promise_n.clamp_min(1)))
            self.x_promise.zero_()
            self.x_promise_n.zero_()
        if self.div is not None and cfg.div_coef:
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
                    cells=b.b_cell[t].cpu(), grid=None if self.grid is None else self.grid.state_dict(),
                    ram=None if b.b_ram is None else b.b_ram[t].cpu(),
                    coverage=None if self.cov is None else dict(self.cov.state_dict(), seen=self.cov.seen.cpu(),
                                                                found=self.cov.found.cpu(),
                                                                **(dict(bseen=self.cov.bseen.cpu(), bfound=self.cov.bfound.cpu())
                                                                   if self.cov.blocks > 0 else {})),
                    explore=dict(hold=self.x_hold.cpu(), parts=self.x_parts.cpu(), drought=self.x_drought.cpu(),
                                 place=self.x_place.cpu(), key=self.x_key.cpu(), stuck=self.x_stuck.cpu(),
                                 quiet=self.x_quiet.cpu(), atdoor=self.x_atdoor.cpu(),
                                 tried=None if self.x_tried is None else self.x_tried.cpu()),
                    rng_cpu=torch.get_rng_state(), rng_roll=self.gen_roll.get_state(), rng_cuda=torch.cuda.get_rng_state(self.dev) if self.cuda else None,
                    elapsed=time.time() - self.t_start + self.elapsed0,
                    div=None if self.div is None else self.div.state_dict(),
                    mood_kicked=getattr(self, 'mood_kicked', False),
                    rnd=None if self.rnd is None else self.rnd.state_dict(),
                    cur=None if self.cur is None else self.cur.state_dict(),
                    outcome=None if self.out is None else self.out.state_dict(),
                    trybonus=None if self.tryb is None else self.tryb.state_dict(),
                    ramnovelty=None if self.ramnov is None else self.ramnov.state_dict(),
                    situations=None if self.sit is None else self.sit.state_dict(),
                    stuck=None if self.stuck is None else self.stuck.state_dict(),
                    level_plan2=None if self.level_plan is None else self.level_plan.state_dict(),
                    places=None if self.places is None else self.places.state_dict(),
                    routes=None if self.routes is None else self.routes.state_dict())

    def save(self, tag='latest'):
        self.sync()
        path = self.run / f'ckpt_rank{self.rank}_{tag}.pt'
        st = self.state()
        atomic_torch_save(st, path)
        if self.rank == 0 and self.cfg.level_cap > 0:
            # the saved level starts alone (~20 KB each), so test plays can start from them (rl/evaluate.py --starts)
            for gi, g in enumerate(self.env.games):
                rec = slotlib._game_tail(self.env, st['env']['env'], gi)
                arr = np.concatenate([rec[k][:self.cfg.level_cap].reshape(-1) for k in slotlib.FIELDS])
                tmp = self.run / f'level_starts_{g}.tmp.npy'
                np.save(tmp, arr)
                tmp.replace(self.run / f'level_starts_{g}.npy')
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
        if self.grid is not None and s['cfg'].get('grid_cell_size',32) != self.cfg.grid_cell_size:
            raise ValueError('grid size changed: use prepare_grid_run.py --regrid to reset visits and entrance attempts')
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
        b.b_cell[0].copy_(s.get('cells', self.env.cells))
        if b.b_ram is not None and s.get('ram') is not None:
            b.b_ram[0].copy_(s['ram'])
        if self.cov is not None and s.get('coverage') is not None:
            self.cov.load_state_dict(s['coverage'])
            self.cov.seen.copy_(s['coverage']['seen']); self.cov.found.copy_(s['coverage']['found'])
            if self.cov.blocks > 0 and s['coverage'].get('bseen') is not None:
                self.cov.bseen.copy_(s['coverage']['bseen']); self.cov.bfound.copy_(s['coverage']['bfound'])
        if self.grid is not None:
            if s.get('grid') is None:
                raise ValueError('grid mode requires a fresh entrance checkpoint or a grid-mode checkpoint')
            self.grid.load_state_dict(s['grid'])
        if s.get('explore') is not None:
            self.x_hold.copy_(s['explore']['hold']); self.x_parts.copy_(s['explore']['parts'])
            self.x_drought.copy_(s['explore']['drought'])
            if 'place' in s['explore']:
                self.x_place.copy_(s['explore']['place'])
            if s['explore'].get('key') is not None:
                self.x_key.copy_(s['explore']['key'])
            if s['explore'].get('stuck') is not None:
                self.x_stuck.copy_(s['explore']['stuck'])
            if s['explore'].get('quiet') is not None:
                self.x_quiet.copy_(s['explore']['quiet'])
            if s['explore'].get('atdoor') is not None:
                self.x_atdoor.copy_(s['explore']['atdoor'])
            if self.x_tried is not None and s['explore'].get('tried') is not None:
                self.x_tried.copy_(s['explore']['tried'])
        torch.set_rng_state(s['rng_cpu'].cpu())  # (map_location moved it to the device)
        if not s.get('fresh_grid_run'):
            self.gen_roll.set_state(s['rng_roll'].cpu())
        if self.cuda and s['rng_cuda'] is not None:
            torch.cuda.set_rng_state(s['rng_cuda'].cpu(), self.dev)
        if self.div is not None and s.get('div') is not None:
            self.div.load_state_dict(s['div'])
        if self.rnd is not None and s.get('rnd') is not None:
            self.rnd.load_state_dict(s['rnd'])
        if self.cur is not None and s.get('cur') is not None:
            self.cur.load_state_dict(s['cur'])
        if self.places is not None and s.get('places') is not None:
            self.places.load_state_dict(s['places'])
        if self.out is not None and s.get('outcome') is not None:
            self.out.load_state_dict(s['outcome'])
        if self.tryb is not None and s.get('trybonus') is not None:
            self.tryb.load_state_dict(s['trybonus'])
        if self.ramnov is not None and s.get('ramnovelty') is not None:
            self.ramnov.load_state_dict(s['ramnovelty'])
        if self.sit is not None and s.get('situations') is not None:
            self.sit.load_state_dict(s['situations'])
        if self.stuck is not None and s.get('stuck') is not None:
            self.stuck.load_state_dict(s['stuck'])
        self.routes_saved = s.get('routes')
        self.it, self.decisions, self.elapsed0 = s['it'], s['decisions'], s.get('elapsed', 0.0)
        if self.rank == 0:
            print(f'[rl] resumed from {path} at iteration {self.it}, {self.decisions} decisions', flush=True)
        return True

    def init_from(self):
        """A new run from another run's learning: weights, optimizer, return statistics, mood judge, counters. The envs
        start fresh."""
        s = torch.load(self.cfg.init_from, map_location='cpu', weights_only=False)
        self.load_model_opt(s)
        self.scaler.load_state_dict(s['scaler'])
        self.ret_mean, self.ret_sq, self.ret_seen = (s[k].to(self.dev) for k in ('ret_mean', 'ret_sq', 'ret_seen'))
        if self.div is not None and s.get('div') is not None:
            self.div.judge.load_state_dict(s['div']['judge'])
            self.div.opt.load_state_dict(s['div']['opt'])
        self.it, self.decisions = s['it'], s['decisions']
        self.routes_saved = s.get('routes')      # a routes run carries on with its curriculum pointers
        if self.rank == 0:
            print(f'[rl] new run from {self.cfg.init_from} (iteration {self.it}, {self.decisions} decisions)', flush=True)

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
    def tb_log(self, row, stats):
        """Coverage graphs (TensorBoard, RUN/tb), x = decisions played."""
        w, x = self.tb, self.decisions
        c = stats.get('coverage') or {}
        for k in ('ep_mean', 'ep_max', 'ever', 'map_found', 'off_map', 'ever_levels', 'bonus_per_cell', 'blocks_mean', 'block_pay', 'block_share', 'blocks_ever'):
            if c.get(k) is not None:
                w.add_scalar(f'coverage/{k}', c[k], x)
        if c.get('map_frac') is not None:
            w.add_scalar('coverage/map_percent', 100 * c['map_frac'], x)
            for lv, f in enumerate(c['map_level_frac']):
                w.add_scalar(f'coverage_level/{lv // 4 + 1}-{lv % 4 + 1}', 100 * f, x)
        for lv, nb in enumerate(c.get('blocks_level') or []):
            if nb:
                w.add_scalar(f'blocks_level/{lv // 4 + 1}-{lv % 4 + 1}', nb, x)
        for k in ('pg', 'v', 'ent', 'kl', 'kl_used', 'explained_var', 'ram_loss', 'ram_byte_exact', 'ram_all_exact'):
            if k in stats:
                w.add_scalar(f'train/{k}', stats[k], x)
        w.add_scalar('train/decisions_per_s', row['sps'], x)
        for k, v in (stats.get('explore') or {}).items():
            if not isinstance(v, dict):
                w.add_scalar(f'explore/{k}', v, x)
        for k, v in (stats.get('cur') or {}).items():
            w.add_scalar(f'curiosity/{k}', v, x)
        if row.get('stuck'):
            self.tb_stuck(w, row['stuck'], x)
        g = (row.get('games') or {}).get('smb')
        if g:
            w.add_scalar('episode/length', g['len'], x)
            w.add_scalar('episode/level_max', g['level_max'], x)
            w.add_scalar('episode/level_mean', g['level_mean'], x)
        # finished test plays (eval/itN/summary.json, written by the evaluation process)
        for f in sorted((self.run / 'eval').glob('it*/summary.json')):
            if f in getattr(self, 'tb_evals', set()):
                continue
            self.tb_evals = getattr(self, 'tb_evals', set()) | {f}
            for r in json.loads(f.read_text()):
                x_eval = int(f.parent.name[2:]) * self.T * self.N    # decisions at that iteration
                w.add_scalar('eval/cells', r.get('cells', 0), x_eval)
                w.add_scalar('eval/level', r['level'], x_eval)
        w.flush()

    @staticmethod
    def level_name(L):
        return f'{L // 4 + 1}-{L % 4 + 1}'

    def stuck_line(self, st):
        n = self.level_name
        out = ' | stuck (M decisions without a new block / trigger): ' + ' '.join(
            f"{n(int(L))}:{v:.2f}/{st['trigger_M'][L]:.0f}" for L, v in st['dry_M'].items())
        if st['job']:
            j = st['job']
            out += (f" | searching {n(j['level'])} for {j['minutes']:.0f} min: {j.get('decisions', 0) / 1e6:.1f}M decisions "
                    f"({j.get('rate', 0)}/s), {j.get('cells', 0)} cells, {j.get('rare_cells', 0)} rare, "
                    f"{j.get('clears', 0)} clears")
        pc = lambda x: f"{100.0 * x[1] / x[0]:.0f}%" if x[0] else '-'  # noqa: E731
        for L, d in st['handoff'].items():
            out += (f" | new spot {n(int(L))} ({d['spot']}, {d['way']} decisions in, {d['states']} states): from the "
                    f"start {d['start_reached']}/{d['start_games']} reach it; practice {d['reached']}/{d['tries']} "
                    f"(near {pc(d['near'])}, mid {pc(d['mid'])}, far {pc(d['far'])})"
                    + (f"; window {d['window'][0]}-{d['window'][1]} of {d['window'][2]} decisions back, "
                       f"{d['mastered']} mastered" if 'window' in d else '')
                    + (f"; teacher tries {d['teacher'][1]}/{d['teacher'][0]} reach it" if d.get('teacher') else ''))
        if st['handoff']:
            out += f" | practice consoles {st['practice']['consoles']} (share {st['practice']['share']:.3f})"
        if st.get('start_only'):
            out += (' | level start only: ' + ' '.join(n(L) for L in st['start_only']) + ' (level-start games counted: '
                    + ' '.join(f"{n(int(L))} {v}" for L, v in st.get('start_games', {}).items()) + ')')
        return out

    def tb_stuck(self, w, st, x):
        """Stuck -> search -> hand-off (rl/stuck.py), one line per level in each chart."""
        n = self.level_name
        if st['dry_M']:
            w.add_scalars('stuck/1 M learner decisions without a new block (search at trigger)',
                          {**{n(int(L)): v for L, v in st['dry_M'].items()},
                           **{f'{n(int(L))} trigger': v for L, v in st['trigger_M'].items()}}, x)
        j = st['job']
        if j:
            lv = n(j['level'])
            w.add_scalars('stuck/2 search: M decisions used', {lv: j.get('decisions', 0) / 1e6}, x)
            w.add_scalars('stuck/3 search: rare cells found (all, and in the last 3M decisions)',
                          {lv: j.get('rare_cells', 0), f'{lv} recent': j.get('rare_recent', 0)}, x)
        for e in st['events']:
            if e['kind'] == 'search':
                w.add_scalars('stuck/4 search results: targets handed off', {n(e['level']): e.get('handed', 0)}, x)
        if st['handoff']:
            h = st['handoff']
            w.add_scalars('stuck/5 new spot: level-start learner games that reach it (%, recent)',
                          {n(int(L)): 100.0 * d['start_reached'] / d['start_games'] for L, d in h.items()
                           if d['start_games']}, x)
            w.add_scalars('stuck/6 new spot: practice tries that reach it (%; near <= 8, mid <= 64, far > 64 decisions)',
                          {f'{n(int(L))} {k}': 100.0 * d[k][1] / d[k][0] for L, d in h.items()
                           for k in ('near', 'mid', 'far') if d[k][0]}, x)
            w.add_scalars('stuck/7 new spot: practice tries per iteration', {n(int(L)): d['tries'] for L, d in h.items()}, x)
            w.add_scalar('stuck/8 new spot: learner consoles practising', st['practice']['consoles'], x)

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
                d = lv.setdefault(e['game'], dict(n=0, cleared=0, by_start={}, ends={}))
                cl = int(e['level'] > e['start'] or e['end'] in ('won', 'level_clear'))
                d['n'] += 1; d['cleared'] += cl
                d.setdefault('return_sum', 0.0); d['return_sum'] += e.get('ret', 0.0)
                d.setdefault('length_sum', 0); d['length_sum'] += e.get('len', 0)
                d['ends'][e['end']] = d['ends'].get(e['end'], 0) + 1
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
             '--video-dir', str(self.run / 'videos'), '--tag', self.eval_tag] + self.eval_starts(),
            stdout=open(d / 'eval.log', 'w'), stderr=subprocess.STDOUT, env=env, start_new_session=True)

    def eval_starts(self):
        """A run that trains only from some saved levels (--only-levels) is tested from those levels too (the archive
        the last checkpoint wrote); otherwise test plays start at power-on."""
        games = self.cfg.games.split(',')
        arch = self.run / f'level_starts_{games[0]}.npy'
        if not self.cfg.only_levels or len(games) != 1 or not arch.exists():
            return []
        return ['--starts', self.cfg.only_levels, '--archive', str(arch)]

    # ---------------- main loop ----------------
    def train(self):
        import threading
        cfg = self.cfg
        resumed = self.load()
        if not resumed and cfg.init_from:
            self.init_from()
        if not resumed and cfg.entrances:
            slotlib.load_entrances(self.env, 0, cfg.entrances)
        if cfg.routes:
            self.routes = Routes.from_dir(self.env, 0, cfg.routes, cfg.route_slots)
            if (resumed or cfg.init_from) and getattr(self, 'routes_saved', None) is not None:
                self.routes.load_state_dict(self.routes_saved)
            if self.rank == 0:
                print(f'[rl] routes: {int((self.routes.K > 0).sum())} levels, {int(self.routes.K.sum())} states', flush=True)
        if not resumed:
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
                    if self.sit is not None:
                        self.places.last.update(self.sit.stats(), sit_levels={g: self.places.report_var(gi)
                                                                              for gi, g in enumerate(self.env.games)})
                    row['places'] = dict(self.places.last, per_level={g: self.places.report(gi)
                                                                      for gi, g in enumerate(self.env.games)})
                if self.routes is not None:
                    row['routes'] = self.routes.report()
                if self.stuck is not None:
                    row['stuck'] = self.stuck.report(self.level_plan.rate(0))
            if self.it % cfg.log_every == 0:
                self.log(row)
                if self.rank == 0:
                    print(f"[rl] it {self.it} dec {self.decisions/1e6:.2f}M sps {row['sps']:.0f} "
                          f"(rollout {self.rollout_s:.1f}s learn {learn_s:.1f}s{' overlapped' if overlap else ''}) "
                          f"pg {stats['pg']:.3f} v {stats['v']:.3f} ent {stats['ent']:.2f} kl {stats['kl']:.4f} (learned moves {stats['kl_used']:.4f}) "
                          + (f"imitation loss {stats['bc_loss']:.4f} (teacher's buttons: log-prob {stats['bc_logp']:.2f}) "
                             if 'bc_loss' in stats else '') +
                          f"kl0 {stats['kl_first']:.1e} ev {stats['explained_var']:.2f} "
                          f"mfu {row['mfu']*100:.1f}% power {row['power_w'] or 0:.0f}W"
                          + (''.join(f" | {g} judge {d['acc']:.2f} gate {d['gate_open']:.2f}" for g, d in stats['div'].items())
                             if 'div' in stats else '')
                          + (''.join(f" | {g} curiosity {d['bonus']:.2e}" for g, d in stats['curiosity'].items())
                             if 'curiosity' in stats else '')
                          + (f" | places kept {row['places']['kept']} known {row['places']['known']} held "
                             f"{sum(sum(v.values()) for v in row['places']['per_level'].values())}"
                             + (f" | doors: {p['doors']} door places, {p['door_found']} found by explorers ({p['door_added']} "
                                f"kept), {p['door_opened']} opened by learners, {(row.get('explore') or {}).get('door_starts', 0)} "
                                f"learner/explorer starts at doors, {p['jumps_known']} jumps the learners know"
                                if 'doors' in (p := row['places']) else '')
                             + (f" | situations: {p['sit']} saved ({p['sit_kept']} new; {p['sit_tries']} games started from "
                                f"them, {p['sit_wins']} cleared the level; with a clear in levels [{p['sit_proven']}]) "
                                f"in levels {' '.join(f'{k}:{v}' for d in p['sit_levels'].values() for k, v in sorted(d.items()))}; "
                                + (f"{p['bytes']} progress bytes, counts of {p['counted']} rollouts, consoles in a rare "
                                   f"situation this rollout {p['rare']} (rare world state {p['rare_world']})"
                                   if p['bytes'] is not None else 'choosing the progress bytes')
                                if 'sit_levels' in (p := row['places']) else '') if 'places' in row else '')
                          + (self.stuck_line(row['stuck']) if 'stuck' in row else '')
                          + (" | routes (level:solved from state/states, entrance clear%) " + ' '.join(
                              f"{l}:{d['pointer']}/{d['states']} {d['entrance']*100:.0f}%"
                              for l, d in row['routes'].items()) if 'routes' in row else '')
                          + (f" | coverage episodes {c['episodes']} mean {c['ep_mean'] or 0:.0f} max {c['ep_max'] or 0} "
                             f"ever {c['ever']} (of the game's map {c.get('map_found', 0)}/{c.get('map_total', 0)} "
                             f"= {c.get('map_frac', 0)*100:.1f}%, off it {c.get('off_map', 0)}, levels {c['ever_levels']})"
                             if (c := stats.get('coverage')) else '')
                          + (f" | ram loss {stats['ram_loss']:.4f} changed bytes exact {stats['ram_byte_exact']*100:.1f}% "
                             f"all bytes exact {stats['ram_all_exact']*100:.1f}%" if 'ram_loss' in stats else '')
                          + (f" | Mario blocks per episode {c['blocks_mean']:.0f} pay per new block {c['block_pay']:.3f}"
                             f" ({c['block_share'] * 100:.0f}% of the reward)"
                             + (f" blocks ever {c['blocks_ever']} ("
                                + ' '.join(f"{lv // 4 + 1}-{lv % 4 + 1}:{nb}" for lv, nb in enumerate(c['blocks_level']) if nb)
                                + ')' if 'blocks_ever' in c else '')
                             if (c := stats.get('coverage')) and c.get('blocks_mean') is not None else '')
                          + (f" | curiosity about moves: heads' loss {cs['loss']:.3f}, bonus {cs['share']*100:.0f}% of the "
                             f"reward (top 1% of moves {cs['top1']:.3f}, held moves {cs['held']:.3f})" if (cs := stats.get('cur')) else '')
                          + (f" | try bonus: {t['share']*100:.1f}% of the reward, {t['fresh']} first presses of a "
                             f"(place, buttons) pair, {t['pairs']} pairs ever pressed" if (t := stats.get('try')) else '')
                          + (f" | RAM novelty: {'on' if q['active'] else 'warming up'}, {q['share']*100:.1f}% of the reward, "
                             f"{q['fresh']} never-seen changes, {q['bytes_counted']} bytes counted, table {q['table_used']*100:.0f}% used, top {q['top']}"
                             if (q := stats.get('ramnov')) else '')
                          + (f" | what-will-happen head: loss {o['loss']:.3f}, explorers' chosen tries promise {o['promise']*100:.1f}%, "
                             f"real chance in 16 decisions: " + ' '.join(f"{k} {v*100:.1f}%" for k, v in o['rate16'].items())
                             if (o := stats.get('outcome')) else '')
                          + (f" | held random moves {xs['forced']*100:.1f}% of decisions, {xs['starts']} started"
                             + (f" ({xs['untried']} explorer tries of a little-tried move; explorers held "
                                f"{xs['forced_ex']*100:.0f}%)" if self.x_tried is not None else '')
                             + f", {xs['hits']} paid off (advantage {xs['hit_adv']:.1f})" if (xs := stats.get('explore')) else '')
                          + (f" | rarity bonus per new cell {c['bonus_per_cell']:.3f}"
                             if (c := stats.get('coverage')) and 'bonus_per_cell' in c else '')
                          + (''.join(f" | {g} saved levels {len(v)} (max {max(v) if v else '-'})"
                                     for g, v in row.get('saved_levels', {}).items()))
                          + (''.join(f" | {g} " + ' '.join(f"L{l}:{d['clear']*100:.0f}%/w{d['weight']*100:.0f}"
                                                           for l, d in p['levels'].items())
                                     for g, p in row.get('level_plan', {}).items()) ), flush=True)
            if self.tb is not None:
                self.tb_log(row, stats)
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
