"""NeSLE (CPU build) behind the same interface as reward10's nesram.Nes, so boot(), search and audit run on it.

RAM is reward/debug data only, never an actor input. No poke (address verification stays on fceumm).
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
from PIL import Image

# NESLE_HOST_LIB selects another build (e.g. a mapper branch) without touching a running check
LIB = Path(os.environ.get('NESLE_HOST_LIB',
                          Path(__file__).resolve().parents[2] / '.cache/nesle-check/libnesle_host.so'))
NES_BITS = {'A': 0, 'B': 1, 'SELECT': 2, 'START': 3, 'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7}

_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(LIB))
        _lib.nh_create.restype = ctypes.c_void_p
        _lib.nh_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        _lib.nh_destroy.argtypes = [ctypes.c_void_p]
        _lib.nh_step.restype = ctypes.c_uint32
        _lib.nh_step.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_int]
        _lib.nh_ram.restype = ctypes.POINTER(ctypes.c_uint8)
        _lib.nh_ram.argtypes = [ctypes.c_void_p]
        _lib.nh_state_size.restype = ctypes.c_size_t
        _lib.nh_state_size.argtypes = [ctypes.c_void_p]
        _lib.nh_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.nh_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _lib.nh_render.restype = ctypes.POINTER(ctypes.c_uint8)
        _lib.nh_render.argtypes = [ctypes.c_void_p]
        if hasattr(_lib, 'nh_load_fcs'):
            _lib.nh_load_fcs.restype = ctypes.c_int
            _lib.nh_load_fcs.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
    return _lib


class EmulatorFault(RuntimeError):
    pass


class NesleNes:
    def __init__(self, rom):
        self.rom = str(Path(rom).resolve(strict=True))
        data = Path(self.rom).read_bytes()
        self.h = lib().nh_create(data, len(data))
        if not self.h:
            raise RuntimeError(f'NeSLE cannot load {self.rom}')
        self.frame = 0
        self._size = lib().nh_state_size(self.h)

    def __del__(self):
        if getattr(self, 'h', None):
            lib().nh_destroy(self.h)
            self.h = None

    def ram(self):
        return np.ctypeslib.as_array(lib().nh_ram(self.h), (2048,)).copy()

    def screen(self):
        return np.ctypeslib.as_array(lib().nh_render(self.h), (240, 256, 3)).copy()

    def png(self, path, scale=2):
        img = Image.fromarray(self.screen())
        img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)

    def step(self, buttons=(), frames=1, record=None):
        byte = sum(1 << NES_BITS[b] for b in buttons)
        if record is None:
            fault = lib().nh_step(self.h, byte, frames)
            self.frame += frames
        else:
            for _ in range(frames):
                fault = lib().nh_step(self.h, byte, 1)
                self.frame += 1
                record.append(self.ram())
        if fault:
            raise EmulatorFault(f'unimplemented opcode {fault & 0xFF:02X} at ${fault >> 8:04X}, frame {self.frame}')

    def tap(self, button, hold=4, release=20):
        self.step((button,), hold)
        self.step((), release)

    def save(self):
        buf = ctypes.create_string_buffer(self._size)
        lib().nh_save(self.h, buf)
        return (buf.raw, self.frame)

    def load(self, snap):
        lib().nh_load(self.h, snap[0])
        self.frame = snap[1]

    def load_fcs(self, state, dot_offset=0):
        """Resume from an FCEUmm state (stable_retro em.get_state()); the next step ends the following frame."""
        if lib().nh_load_fcs(self.h, state, len(state), dot_offset):
            raise RuntimeError('NeSLE cannot load this FCEUmm state')
