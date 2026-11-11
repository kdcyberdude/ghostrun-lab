"""fceumm through its libretro API, with the full 256x240 picture.

stable_retro crops fceumm's picture to 240x224 (the core's "fceumm_overscan" option defaults to cropping and
stable_retro sets no core options). This loads the same core binary that stable_retro ships, answers every
core option the way stable_retro does (not set) except fceumm_overscan = disabled, and so returns the
uncropped frame. Everything else (states, RAM, input order, RGB565 -> RGB888) matches stable_retro;
check_equivalence() verifies that against stable_retro itself in a separate process.

One core per process: the core keeps global state, so a second instance in the same process would share it.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

import numpy as np

CORE = Path(os.environ.get('FCEUMM_CORE', Path(__file__).resolve().parents[2] /
                           '.cache/multigame-prior-runtime-v1/lib/python3.12/site-packages/stable_retro/cores/'
                           'fceumm_libretro.so'))
# stable_retro's Nes button order -> libretro joypad ids (B=0, Y=1, SELECT=2, START=3, UP..RIGHT=4..7, A=8)
NES_BITS = {'A': 0, 'B': 1, 'SELECT': 2, 'START': 3, 'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7}
NES_TO_RETRO = {0: 8, 1: 0, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

ENV_GET_SYSTEM_DIRECTORY = 9
ENV_SET_PIXEL_FORMAT = 10
ENV_GET_VARIABLE = 15
ENV_GET_SAVE_DIRECTORY = 31


class GameInfo(C.Structure):
    _fields_ = [('path', C.c_char_p), ('data', C.c_void_p), ('size', C.c_size_t), ('meta', C.c_char_p)]


class Variable(C.Structure):
    _fields_ = [('key', C.c_char_p), ('value', C.c_char_p)]


ENV_CB = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
VIDEO_CB = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
AUDIO_CB = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AUDIO_BATCH_CB = C.CFUNCTYPE(C.c_size_t, C.c_void_p, C.c_size_t)
POLL_CB = C.CFUNCTYPE(None)
INPUT_CB = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)

_core = None


class Fceumm:
    """One fceumm core. crop=False gives 256x240; crop=True gives stable_retro's 240x224."""

    def __init__(self, rom, crop=False):
        global _core
        if _core is not None:
            raise RuntimeError('one fceumm core per process')
        _core = self
        self.crop = crop
        self.lib = C.CDLL(str(CORE))
        self.buttons = 0
        self.frame_rgb = None
        self.pixfmt = None
        self._sysdir = C.c_char_p(b'.')
        self._overscan = C.c_char_p(b'enabled' if crop else b'disabled')
        self._cbs = [ENV_CB(self._env), VIDEO_CB(self._video), AUDIO_CB(lambda l, r: None),
                     AUDIO_BATCH_CB(lambda d, n: n), POLL_CB(lambda: None), INPUT_CB(self._input)]
        L = self.lib
        L.retro_set_environment(self._cbs[0])
        L.retro_set_video_refresh(self._cbs[1])
        L.retro_set_audio_sample(self._cbs[2])
        L.retro_set_audio_sample_batch(self._cbs[3])
        L.retro_set_input_poll(self._cbs[4])
        L.retro_set_input_state(self._cbs[5])
        L.retro_init()
        self.rom = str(Path(rom).resolve(strict=True))
        self._rom_bytes = Path(self.rom).read_bytes()
        self._buf = C.create_string_buffer(self._rom_bytes, len(self._rom_bytes))
        info = GameInfo(self.rom.encode(), C.cast(self._buf, C.c_void_p), len(self._rom_bytes), None)
        L.retro_load_game.restype = C.c_bool
        if not L.retro_load_game(C.byref(info)):
            raise RuntimeError(f'fceumm cannot load {rom}')
        L.retro_serialize_size.restype = C.c_size_t
        L.retro_serialize.restype = C.c_bool
        L.retro_unserialize.restype = C.c_bool
        L.retro_get_memory_data.restype = C.c_void_p
        L.retro_get_memory_size.restype = C.c_size_t
        L.retro_run()  # stable_retro's RetroEmulator runs one frame when it loads the game
        self.frame = 0

    # ---- callbacks ----
    def _env(self, cmd, data):
        cmd &= 0xFFFF
        if cmd == ENV_SET_PIXEL_FORMAT:
            self.pixfmt = C.cast(data, C.POINTER(C.c_int)).contents.value
            return self.pixfmt == 2  # RGB565, which fceumm asks for first
        if cmd in (ENV_GET_SYSTEM_DIRECTORY, ENV_GET_SAVE_DIRECTORY):
            C.cast(data, C.POINTER(C.c_char_p))[0] = self._sysdir
            return True
        if cmd == ENV_GET_VARIABLE:
            var = C.cast(data, C.POINTER(Variable)).contents
            if var.key == b'fceumm_overscan':
                var.value = self._overscan
                return True
            return False
        return False

    def _video(self, data, width, height, pitch):
        if not data:
            return
        raw = np.ctypeslib.as_array(C.cast(data, C.POINTER(C.c_uint16)), (height, pitch // 2))[:, :width]
        r = ((raw >> 11) & 0x1F).astype(np.uint8) << 3
        g = ((raw >> 5) & 0x3F).astype(np.uint8) << 2
        b = (raw & 0x1F).astype(np.uint8) << 3
        self.frame_rgb = np.dstack([r, g, b])

    def _input(self, port, device, index, button):
        if port != 0 or device != 1:
            return 0
        for nes_bit, retro_id in NES_TO_RETRO.items():
            if retro_id == button:
                return (self.buttons >> nes_bit) & 1
        return 0

    # ---- API (same shape as nesram.Nes / neslenes.NesleNes) ----
    def step_byte(self, byte, frames=1):
        self.buttons = byte
        for _ in range(frames):
            self.lib.retro_run()
            self.frame += 1

    def step(self, buttons=(), frames=1):
        self.step_byte(sum(1 << NES_BITS[b] for b in buttons), frames)

    def screen(self):
        return self.frame_rgb.copy()

    def ram(self):
        p = self.lib.retro_get_memory_data(2)
        return np.ctypeslib.as_array(C.cast(p, C.POINTER(C.c_uint8)), (2048,)).copy()

    def get_state(self):
        n = self.lib.retro_serialize_size()
        buf = C.create_string_buffer(n)
        if not self.lib.retro_serialize(buf, n):
            raise RuntimeError('retro_serialize failed')
        return buf.raw

    def set_state(self, state):
        if not self.lib.retro_unserialize(state, len(state)):
            raise RuntimeError('retro_unserialize failed')


def _stable_retro_run(rom, buttons):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reward10'))
    from nesram import Nes
    n = Nes(rom)
    frames, rams, states = [], [], []
    for i, b in enumerate(buttons):
        n.em.set_button_mask(np.array([(b >> 1) & 1, 0, (b >> 2) & 1, (b >> 3) & 1, (b >> 4) & 1, (b >> 5) & 1,
                                       (b >> 6) & 1, (b >> 7) & 1, b & 1], np.uint8), 0)
        n.em.step()
        frames.append(n.screen())
        rams.append(n.ram())
        if i % 97 == 0:
            states.append(n.em.get_state())
    return np.stack(frames), np.stack(rams), states


def _lr_run(rom, buttons, crop):
    f = Fceumm(rom, crop=crop)
    frames, rams, states = [], [], []
    for i, b in enumerate(buttons):
        f.step_byte(int(b))
        frames.append(f.screen())
        rams.append(f.ram())
        if i % 97 == 0:
            states.append(f.get_state())
    return np.stack(frames), np.stack(rams), states


def check_equivalence(rom, buttons):
    """Same buttons through stable_retro and this frontend (separate processes). Returns a dict."""
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    with ctx.Pool(3) as pool:
        a = pool.apply_async(_stable_retro_run, (rom, buttons))
        b = pool.apply_async(_lr_run, (rom, buttons, True))
        c = pool.apply_async(_lr_run, (rom, buttons, False))
        sr, lr, full = a.get(), b.get(), c.get()
    return dict(frames=len(buttons),
                cropped_frames_equal=bool(np.array_equal(sr[0], lr[0])),
                ram_equal=bool(np.array_equal(sr[1], lr[1]) and np.array_equal(sr[1], full[1])),
                states_equal=all(x == y for x, y in zip(sr[2], lr[2])),
                full_frame_shape=list(full[0].shape[1:]),
                full_crop_equals_stable_retro=bool(np.array_equal(full[0][:, 8:232, 8:248], sr[0])))
