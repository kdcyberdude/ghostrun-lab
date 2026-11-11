"""Python side of the 12-game environment (rl/csrc/rlenv_core.cuh): builds the libraries, records each game's boot
pad bytes, allocates the output tensors (torch, on the env's device) and runs decisions.

    env = RlEnv({'smb': 64, 'tetris': 32}, device='cuda', seed=1)   # or device='cpu' (reference build)
    env.start()                       # first observation in env.obs / env.emph, legal parts in env.masks
    env.actions.copy_(moves)          # int16 packed move ids (rl/csrc/controller.cuh), legal under env.masks
    env.step()                        # env.obs, reward, done, masks, timeline, pads, events, info updated in place

Everything stays on the device; nothing is copied to the host per step.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

RL = Path(__file__).resolve().parent
ROOT = RL.parent
NESLE_INC = Path(os.environ.get('NESLE_INC', ROOT / '.cache/nesle-src/cpp/include'))
NESLE_ROM_CPP = Path(os.environ.get('NESLE_ROM_CPP', ROOT / '.cache/nesle-src/cpp/src/rom.cpp'))
BUILD = Path(os.environ.get('RL_BUILD', RL / 'build'))
BOOT_DIR = Path(os.environ.get('RL_BOOT_DIR', ROOT / '.cache/rl/boot'))

MANIFEST = __import__('json').loads((RL / 'games_manifest.json').read_text())
GAMES = list(MANIFEST)  # same order as rl/scripts/gen_reward_consts.py GAMES (rl10 game ids)
SOURCES = ['rlenv_core.cuh', 'rlenv_build.hpp', 'nes_machine.cuh', 'reward10.cuh', 'reward10_consts.h',
           'controller.cuh', 'rlenv_api.h']
OBS_H, OBS_W, FRAMES = 224, 240, 4
MASK_BYTES, TIMELINE, INFO = 19, 8 * 45, 8
PART_NAMES = ('progress', 'level', 'death', 'damage', 'score', 'win', 'stall')  # rlenv.parts columns
NUM_PARTS = len(PART_NAMES)
END_NAMES = {0: 'none', 1: 'game_over', 2: 'won', 3: 'skip', 4: 'stall', 5: 'fault', 6: 'level_clear'}


def _newer(out: Path, srcs) -> bool:
    return not out.exists() or any(s.stat().st_mtime > out.stat().st_mtime for s in srcs)


def build_cpu() -> Path:
    out = BUILD / 'librlenv_cpu.so'
    srcs = [RL / 'csrc' / s for s in SOURCES] + [RL / 'csrc/rlenv_cpu.cpp']
    if _newer(out, srcs):
        BUILD.mkdir(parents=True, exist_ok=True)
        subprocess.run(['c++', '-O2', '-std=c++20', '-fopenmp', '-ffp-contract=off', '-shared', '-fPIC',
                        f'-I{NESLE_INC}', f'-I{RL / "csrc"}', str(RL / 'csrc/rlenv_cpu.cpp'), str(NESLE_ROM_CPP),
                        '-o', str(out)], check=True)
    return out


def build_cuda(archs=None) -> Path:
    """GPU build. archs e.g. ['75', '120']; default: this machine's GPU. -fmad=false keeps the reward's double math
    identical to the host build. RL_CUDA_LIB=path uses a prebuilt library instead (e.g. on Kaggle, no nvcc)."""
    if os.environ.get('RL_CUDA_LIB'):
        return Path(os.environ['RL_CUDA_LIB'])
    if archs is None:
        major, minor = torch.cuda.get_device_capability()
        archs = [f'{major}{minor}']
    out = BUILD / f'librlenv_cuda_sm{"_".join(archs)}.so'
    srcs = [RL / 'csrc' / s for s in SOURCES] + [RL / 'csrc/rlenv.cu']
    if _newer(out, srcs):
        BUILD.mkdir(parents=True, exist_ok=True)
        nvcc = os.environ.get('NVCC', 'nvcc')
        gencode = sum((['-gencode', f'arch=compute_{a},code=sm_{a}'] for a in archs), [])
        subprocess.run([nvcc, '-O3', '-std=c++20', '-fmad=false', '--expt-relaxed-constexpr', '-shared',
                        '-Xcompiler', '-fPIC', '-cudart', 'static', *gencode, f'-I{NESLE_INC}', f'-I{RL / "csrc"}',
                        str(RL / 'csrc/rlenv.cu'), str(NESLE_ROM_CPP), '-o', str(out)], check=True)
    return out


def _load(path: Path):
    lib = ctypes.CDLL(str(path))
    P, I, U64, SZ = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t
    lib.rlenv_create.restype = P
    lib.rlenv_create.argtypes = [I, P, P, P, P, P, P, U64, I, I, I, I]
    lib.rlenv_num_envs.argtypes = [P]
    lib.rlenv_bind.argtypes = [P] * 11
    lib.rlenv_bind_parts.argtypes = [P, P]
    lib.rlenv_bind_obs.argtypes = [P, P, P]
    lib.rlenv_start.argtypes = [P, P]
    lib.rlenv_step.argtypes = [P, P]
    lib.rlenv_state_bytes.restype = SZ
    lib.rlenv_state_bytes.argtypes = [P]
    lib.rlenv_save.argtypes = [P, P, P]
    lib.rlenv_load.argtypes = [P, P, P]
    lib.rlenv_error.restype = ctypes.c_char_p
    lib.rlenv_destroy.argtypes = [P]
    if hasattr(lib, 'rlenv_set_episode_level_end'):
        lib.rlenv_set_episode_level_end.argtypes = [P, I]
    if hasattr(lib, 'rlenv_enable_level_starts'):
        lib.rlenv_enable_level_starts.argtypes = [P, I]
        lib.rlenv_set_level_plan.argtypes = [P, I, ctypes.c_double, P]
        lib.rlenv_bind_start.argtypes = [P, P]
        lib.rlenv_bind_level.argtypes = [P, P]
        if hasattr(lib, 'rlenv_bind_progress'):
            lib.rlenv_bind_progress.argtypes = [P, P]
        lib.rlenv_level_deaths.argtypes = [P, I, P, P]
    if hasattr(lib, 'rlenv_enable_places'):
        lib.rlenv_enable_places.argtypes = [P, I, I]
        lib.rlenv_place_save.argtypes = [P, I, P, P, I, P]
        lib.rlenv_place_copy.argtypes = [P, I, P, P, I, P]
        lib.rlenv_level_layout.argtypes = [P, I, P]
    if hasattr(lib, 'rl_gather_rows'):  # CUDA build only
        lib.rl_host_alloc.restype = P
        lib.rl_host_alloc.argtypes = [SZ]
        lib.rl_host_free.argtypes = [P]
        lib.rl_memcpy_async.argtypes = [P, P, SZ, P]
        lib.rl_gather_rows.argtypes = [P, P, P, I, SZ, P]
        lib.rl_gather_envs.argtypes = [P, P, P, I, I, I, SZ, P]
        if hasattr(lib, 'rl_pack6'):
            lib.rl_pack6.argtypes = [P, P, SZ, P]
            lib.rl_unpack6.argtypes = [P, P, SZ, P]
        lib.rlenv_step_game.argtypes = [P, I, P, P, P]
        if hasattr(lib, "rlenv_step_range"):
            lib.rlenv_step_range.argtypes = [P, I, I, I, P, P, P]
    return lib




def rom_path(slug: str) -> str:
    """ROM of a game: RL_ROM_DIR/<manifest file name> when RL_ROM_DIR is set (e.g. a private Kaggle dataset), else the
    reward10 module's path. The SHA-256 must match the manifest."""
    d = os.environ.get('RL_ROM_DIR')
    p = Path(d) / MANIFEST[slug]['rom'] if d else Path(game_module(slug).ROM)
    if hashlib.sha256(p.read_bytes()).hexdigest() != MANIFEST[slug]['sha256']:
        raise RuntimeError(f'{p}: not the ROM in rl/games_manifest.json')
    return str(p)


def game_module(slug):
    sys.path.insert(0, str(ROOT / 'scripts' / 'reward10'))
    import contract
    return contract.load(slug)


def boot_pads(slug: str, rom_path: str | None = None) -> bytes:
    """Pad byte of every frame of the game's reward10 boot(), recorded on NeSLE (CPU) from power-on. Cached by ROM
    hash; the cache file is all a machine without the CPU harness needs (e.g. Kaggle)."""
    rom_file = rom_path or globals()['rom_path'](slug)
    rom = Path(rom_file).read_bytes()
    key = hashlib.sha256(rom).hexdigest()[:16]
    path = BOOT_DIR / f'{slug}.{key}.pads.bin'
    if path.exists():
        return path.read_bytes()
    sys.path.insert(0, str(ROOT / 'scripts' / 'nesle_check'))
    import neslenes
    bits = {'A': 0, 'B': 1, 'SELECT': 2, 'START': 3, 'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7}

    class Rec(neslenes.NesleNes):
        def __init__(self, r):
            super().__init__(r)
            self.pads = bytearray()

        def step(self, buttons=(), frames=1, record=None):
            self.pads += bytes([sum(1 << bits[b] for b in buttons)]) * frames
            super().step(buttons, frames, record)

    g = game_module(slug)
    n = Rec(rom_file)
    g.boot(n)
    BOOT_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(n.pads))
    return bytes(n.pads)


class RlEnv:
    def __init__(self, counts: dict, device='cuda', seed=0, sticky=True, idle_max=30, dropout=True, roms=None,
                 lib_path=None, level_cap=0, level_slots=None):
        """counts: game slug -> number of envs (any order; envs are laid out in GAMES order).
        roms: optional slug -> ROM path (default: the reward10 module's ROM).
        level_cap > 0: save level starts (levels 0..level_cap-1 of each game; rlenv_core.cuh), used by new episodes
        with the probability set by set_level_prob (default 0). self.start_level: level a finished episode started
        from, -1 = power-on (valid where done).
        level_slots < level_cap: only slots 0..level_slots-1 are level starts; the rest are places (rl/places.py), saved
        by place_save; start_level is then a slot (a place's level: rl/places.py)."""
        self.device = torch.device(device)
        self.games = [g for g in GAMES if counts.get(g, 0) > 0]
        self.counts = [int(counts[g]) for g in self.games]
        self.n = sum(self.counts)
        self.game_of_env = torch.tensor(sum(([GAMES.index(g)] * c for g, c in zip(self.games, self.counts)), []),
                                        dtype=torch.long, device=self.device)
        cuda = self.device.type == 'cuda'
        self.lib = _load(lib_path or (build_cuda() if cuda else build_cpu()))
        roms = roms or {}
        rom_bytes = [Path(roms.get(g) or rom_path(g)).read_bytes() for g in self.games]
        boots = [boot_pads(g, roms.get(g)) for g in self.games]
        self._keep = [rom_bytes, boots]
        arr = lambda T, xs: (T * len(xs))(*xs)  # noqa: E731
        rom_p = arr(ctypes.c_char_p, rom_bytes)
        boot_p = arr(ctypes.c_char_p, boots)
        self._keep += [rom_p, boot_p]
        dev_index = self.device.index if self.device.index is not None else (torch.cuda.current_device() if cuda else 0)
        if cuda:
            torch.cuda.set_device(dev_index)
            torch.zeros(1, device=self.device)  # make sure torch's CUDA context exists first
        self.h = self.lib.rlenv_create(
            len(self.games), arr(ctypes.c_int, [GAMES.index(g) for g in self.games]), rom_p,
            arr(ctypes.c_size_t, [len(r) for r in rom_bytes]), boot_p, arr(ctypes.c_size_t, [len(b) for b in boots]),
            arr(ctypes.c_int, self.counts), seed, int(sticky), int(idle_max), int(dropout), dev_index)
        if not self.h:
            raise RuntimeError(f'rlenv_create: {self.lib.rlenv_error().decode()}')
        N, d = self.n, self.device
        self.actions = torch.zeros(N, dtype=torch.int16, device=d)
        self.obs = torch.zeros(N, FRAMES, OBS_H, OBS_W, dtype=torch.uint8, device=d)
        self.emph = torch.zeros(N, FRAMES, OBS_H, dtype=torch.uint8, device=d)
        self.reward = torch.zeros(N, dtype=torch.float32, device=d)
        self.done = torch.zeros(N, dtype=torch.uint8, device=d)
        self.masks = torch.zeros(N, MASK_BYTES, dtype=torch.uint8, device=d)
        self.timeline = torch.zeros(N, 8, 45, dtype=torch.uint8, device=d)
        self.pads = torch.zeros(N, FRAMES, dtype=torch.uint8, device=d)
        self.events = torch.zeros(N, dtype=torch.int32, device=d)
        self.info = torch.zeros(N, INFO, dtype=torch.float32, device=d)
        p = lambda t: ctypes.c_void_p(t.data_ptr())  # noqa: E731
        self.lib.rlenv_bind(self.h, p(self.actions), p(self.obs), p(self.emph), p(self.reward), p(self.done),
                            p(self.masks), p(self.timeline), p(self.pads), p(self.events), p(self.info))
        # the reward split by source (progress, level, death, damage, score, win, stall); never an actor input
        self.parts = torch.zeros(N, NUM_PARTS, dtype=torch.float32, device=d)
        self.lib.rlenv_bind_parts(self.h, p(self.parts))
        self.level_cap = int(level_cap)
        self.level_slots = int(level_slots if level_slots is not None else level_cap)
        self.start_level = torch.full((N,), -1, dtype=torch.int32, device=d)
        if self.level_cap > 0:
            ok = (self.lib.rlenv_enable_places(self.h, self.level_cap, self.level_slots) if self.level_slots < self.level_cap
                  else self.lib.rlenv_enable_level_starts(self.h, self.level_cap))
            if ok != 0:
                raise RuntimeError(self.lib.rlenv_error().decode())
            self.lib.rlenv_bind_start(self.h, p(self.start_level))
            self.level_now = torch.full((N,), -1, dtype=torch.int32, device=d)  # level after the last decision
            self.lib.rlenv_bind_level(self.h, p(self.level_now))
            # current within-level progress / one level's PROGRESS_UNIT, NaN = not valid now; never an actor input
            self.progress_now = torch.full((N,), float('nan'), dtype=torch.float32, device=d)
            self.lib.rlenv_bind_progress(self.h, p(self.progress_now))

    def set_level_prob(self, prob: float):
        """Chance that a new episode starts from a saved level start (uniform over the game's saved levels)."""
        assert self.level_cap > 0 or prob == 0
        for gi in range(len(self.games)):
            self.set_level_plan(gi, prob)

    def set_level_plan(self, gi, prob, weights=None):
        """Game gi's chance of a level start and its level weights ([level_cap], only saved levels > 0, sum 1; None =
        uniform over the saved levels). Call between decisions."""
        if self.level_cap == 0:
            return
        cdf = None
        if weights is not None:
            cdf = np.minimum(np.cumsum(weights) * 2.0 ** 32, 2.0 ** 32 - 1).astype(np.uint32)
            cdf[np.flatnonzero(weights)[-1]:] = 2 ** 32 - 1
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.lib.rlenv_set_level_plan(self.h, gi, float(prob),
                                      None if cdf is None else cdf.ctypes.data_as(ctypes.c_void_p))

    def saved_levels(self):
        """game slug -> {level: deaths of the saved arrival} (synchronous)."""
        out = {}
        for gi, g in enumerate(self.games):
            buf = np.full(max(self.level_cap, 1), -1, np.int32)
            if self.level_cap > 0:
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                if self.lib.rlenv_level_deaths(self.h, gi, buf.ctypes.data_as(ctypes.c_void_p), self._stream()) != 0:
                    raise RuntimeError(self.lib.rlenv_error().decode())
            out[g] = {int(l): int(d) for l, d in enumerate(buf[:self.level_slots]) if d >= 0}
        return out

    def place_save(self, gi, env_idx, slot_idx):
        """After a step, on the current stream: env env_idx[j] (index within game gi) -> place slot slot_idx[j]
        (skipped where < level_slots). int32 tensors on the env's device."""
        assert env_idx.dtype == torch.int32 and slot_idx.dtype == torch.int32 and env_idx.is_contiguous()
        if self.lib.rlenv_place_save(self.h, gi, ctypes.c_void_p(env_idx.data_ptr()), ctypes.c_void_p(slot_idx.data_ptr()),
                                     int(env_idx.numel()), self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())

    def place_copy(self, gi, src, dst):
        """Slot src[j] -> slot dst[j] of game gi (whole saved state), synchronous."""
        a, b = np.ascontiguousarray(src, np.int32), np.ascontiguousarray(dst, np.int32)
        if len(a) and self.lib.rlenv_place_copy(self.h, gi, a.ctypes.data_as(ctypes.c_void_p),
                                                b.ctypes.data_as(ctypes.c_void_p), len(a), self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())

    def level_layout(self, gi):
        out = np.zeros(5, np.uint32)
        self.lib.rlenv_level_layout(self.h, gi, out.ctypes.data_as(ctypes.c_void_p))
        return dict(stride=int(out[0]), key_cap=int(out[1]), st=int(out[2]), cap=int(out[3]), levels=int(out[4]))

    def _grow_archive(self, buf, old_cap, need):
        """A state saved with old_cap slots per game -> this env's level_cap (new slots empty); None if buf is not
        such a state."""
        lay = [self.level_layout(gi) for gi in range(len(self.games))]
        sizes = lambda cap, l: [cap * l['stride'], cap * l['st'], cap * l['key_cap'] * 4, cap * 8, cap * 4, cap * 4]  # noqa: E731
        old_tail = sum(sum(sizes(old_cap, l)) for l in lay)
        if len(buf) - old_tail + sum(sum(sizes(self.level_cap, l)) for l in lay) != need:
            return None
        prefix, tail = buf[:len(buf) - old_tail], buf[len(buf) - old_tail:]
        out, o = [prefix], 0
        for l in lay:
            for k, (a, b) in enumerate(zip(sizes(old_cap, l), sizes(self.level_cap, l))):
                part = tail[o:o + a]
                o += a
                fill = np.full(b - a, 0xFF, np.uint8) if k == 5 else np.zeros(b - a, np.uint8)   # deaths -1 = empty
                out += [part, fill]
        return np.concatenate(out)

    def drop_obs_buffers(self):
        """Free the default picture outputs (a caller that always binds its own via bind_obs)."""
        self.obs = self.emph = None

    def bind_obs(self, obs, emph):
        """Write the next pictures into these tensors ([N,4,224,240] / [N,4,224] uint8, contiguous, same device)."""
        assert obs.is_contiguous() and emph.is_contiguous() and obs.shape == (self.n, FRAMES, OBS_H, OBS_W)
        self._obs_keep = (obs, emph)
        self.lib.rlenv_bind_obs(self.h, ctypes.c_void_p(obs.data_ptr()), ctypes.c_void_p(emph.data_ptr()))

    def _stream(self):
        return ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream if self.device.type == 'cuda' else 0)

    def set_episode_level_end(self, level):
        self.lib.rlenv_set_episode_level_end(self.h, int(level))

    def start(self):
        if self.lib.rlenv_start(self.h, self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())

    def step(self):
        if self.lib.rlenv_step(self.h, self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())

    def step_game(self, gi, obs, emph):
        """Step only the gi-th game (of self.games) on the current stream, its pictures into obs/emph ([N,...] whole
        batch tensors; only that game's rows are written). CUDA only."""
        if self.lib.rlenv_step_game(self.h, gi, ctypes.c_void_p(obs.data_ptr()), ctypes.c_void_p(emph.data_ptr()),
                                    self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())

    def step_range(self, gi, start, count, obs, emph):
        """Like step_game, for envs [start, start+count) of the gi-th game only. CUDA only."""
        if self.lib.rlenv_step_range(self.h, gi, start, count, ctypes.c_void_p(obs.data_ptr()),
                                     ctypes.c_void_p(emph.data_ptr()), self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())

    def game_slices(self):
        """Row range of each game in the batch (games in self.games order)."""
        out, a = [], 0
        for c in self.counts:
            out.append(slice(a, a + c))
            a += c
        return out

    def state_dict(self) -> bytes:
        """Exact resume state of every env (emulators, rewards, dropout, RNG counters) plus the observation outputs."""
        buf = np.empty(self.lib.rlenv_state_bytes(self.h), np.uint8)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        if self.lib.rlenv_save(self.h, buf.ctypes.data_as(ctypes.c_void_p), self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())
        outs = {k: getattr(self, k).cpu() for k in ('obs', 'emph', 'masks', 'timeline', 'done', 'reward')
                if getattr(self, k) is not None}
        return dict(env=buf, outputs=outs, games=self.games, counts=self.counts, level_cap=self.level_cap)

    def load_state_dict(self, sd):
        assert sd['games'] == self.games and sd['counts'] == self.counts, 'env layout changed'
        buf = np.ascontiguousarray(sd['env'])
        need = self.lib.rlenv_state_bytes(self.h)
        old_cap = sd.get('level_cap', 64)
        if len(buf) < need and self.level_cap > old_cap > 0:
            grown = self._grow_archive(buf, old_cap, need)
            buf = buf if grown is None else grown
        if len(buf) < need and self.level_cap > 0:
            # saved before level starts were on: the level-start state comes last; take it from this fresh env
            # (every episode started at power-on, nothing saved yet)
            buf = np.concatenate([buf, self.state_dict()['env'][len(buf):]])
        assert len(buf) == need, 'env state size changed'
        if self.lib.rlenv_load(self.h, buf.ctypes.data_as(ctypes.c_void_p), self._stream()) != 0:
            raise RuntimeError(self.lib.rlenv_error().decode())
        for k, v in sd['outputs'].items():
            if getattr(self, k) is not None:
                getattr(self, k).copy_(v)

    def __del__(self):
        if getattr(self, 'h', None):
            self.lib.rlenv_destroy(self.h)
            self.h = None
