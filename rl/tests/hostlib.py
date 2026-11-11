"""Build and load rl/build/librl_host.so: the host (CPU) compile of the headers the GPU kernel uses."""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSRC = ROOT / 'rl' / 'csrc'
LIB = ROOT / 'rl' / 'build' / 'librl_host.so'


def load() -> ctypes.CDLL:
    src = [CSRC / n for n in ('host_api.cpp', 'reward10.cuh', 'reward10_consts.h', 'controller.cuh')]
    if not LIB.exists() or LIB.stat().st_mtime < max(p.stat().st_mtime for p in src):
        LIB.parent.mkdir(parents=True, exist_ok=True)
        # -ffp-contract=off: Python never fuses a*b+c; the GPU build uses -fmad=false for the same reason
        subprocess.run(['c++', '-O2', '-std=c++17', '-ffp-contract=off', '-shared', '-fPIC', str(src[0]),
                        '-o', str(LIB)], check=True)
    lib = ctypes.CDLL(str(LIB))
    lib.rl10_replay.restype = ctypes.c_int
    lib.rlc_episode_seed.restype = ctypes.c_uint64
    lib.rlc_episode_seed.argtypes = [ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint64]
    lib.rlc_schedule.argtypes = [ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return lib


def ptr(a):
    return a.ctypes.data_as(ctypes.c_void_p)
