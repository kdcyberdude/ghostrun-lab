"""CPU-only NES harness for reward discovery: RAM reads, RAM pokes, savestates.

RAM is read from the core every frame for reward/audit purposes only; it must
never become an actor input. Pokes are for semantic verification of addresses.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import stable_retro as retro

BUTTONS = list(retro.get_system_info('Nes')['buttons'])


def mask(*names):
    return np.array([b in names for b in BUTTONS], dtype=np.uint8)


class Nes:
    def __init__(self, rom):
        self.rom = str(Path(rom).resolve(strict=True))
        self.em = retro.RetroEmulator(self.rom)
        self.data = retro.data.GameData()
        self.em.configure_data(self.data)
        self.frame = 0
        self._ram_off = None

    # ---- observation ----
    def ram(self):
        self.data.update_ram()
        b = self.data.memory.blocks
        return np.frombuffer(bytes(b[0]) + bytes(b[1024]), np.uint8).copy()

    def screen(self):
        return self.em.get_screen().copy()

    def png(self, path, scale=2):
        img = Image.fromarray(self.screen())
        img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)

    # ---- control ----
    def step(self, buttons=(), frames=1, record=None):
        """Hold buttons for `frames` frames. If record is a list, append RAM each frame."""
        self.em.set_button_mask(mask(*buttons), 0)
        for _ in range(frames):
            self.em.step()
            self.frame += 1
            if record is not None:
                record.append(self.ram())

    def tap(self, button, hold=4, release=20):
        self.step((button,), hold)
        self.step((), release)

    # ---- state ----
    def save(self):
        return (self.em.get_state(), self.frame)

    def load(self, snap):
        self.em.set_state(snap[0])
        self.frame = snap[1]

    def poke(self, addr, value):
        """Write one RAM byte by patching the serialized core state."""
        state = bytearray(self.em.get_state())
        ram = self.ram().tobytes()
        if self._ram_off is None or bytes(state[self._ram_off:self._ram_off + 2048]) != ram:
            off = bytes(state).find(ram)
            if off < 0:
                raise RuntimeError('RAM image not found contiguously in state')
            self._ram_off = off
        state[self._ram_off + addr] = value & 0xFF
        self.em.set_state(bytes(state))


def rom_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bcd(ram, addrs):
    """Decode BCD bytes, most significant first."""
    v = 0
    for a in addrs:
        x = int(ram[a])
        v = v * 100 + (x >> 4) * 10 + (x & 15)
    return v


def digits(ram, addrs):
    """Decode one-decimal-digit-per-byte, most significant first."""
    v = 0
    for a in addrs:
        v = v * 10 + int(ram[a])
    return v


def run_isolated(fn, args_list, workers=48):
    """Run fn(*args) in fresh spawned processes (one live core per process)."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, maxtasksperchild=1) as pool:
        return pool.starmap(fn, args_list)
