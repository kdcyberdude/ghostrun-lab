"""SMB level starts for all 32 levels, made by the game itself (no search, no play): from the boot state (in play at
1-1) set the world, level, area number and area pointer (from the ROM's level tables), then run the steps of the game's
own ContinueGame (what it does after Mario loses a life: reload the area from those numbers, small Mario, fresh timer).
Each level's state is saved when Mario is first in play there, in the level-start slot of that level.

    python rl/scripts/smb_level_starts.py CKPT [--out CKPT]   fill the checkpoint's EMPTY level-start slots (levels
                                                               the run has not reached yet); saved arrivals are kept
    python rl/scripts/smb_level_starts.py --sheet PNG          only make the starts and save a picture of all of them
A filled slot counts 0 deaths, so the run's own first arrival there with no lost life replaces it (rlenv_core.cuh).
"""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, rom_path  # noqa: E402

INTRO = 0x29   # the walk-into-the-pipe intro before 1-2, 2-2, 4-2, 7-2 (shown as that level)


def level_plan():
    """level index (world*4 + level) -> (world, level, area number, area pointer), from the ROM's tables."""
    prg = Path(rom_path('smb')).read_bytes()[16:16 + 32768]
    b = lambda a: prg[a - 0x8000]  # noqa: E731
    woff = [b(0x9cb4 + w) for w in range(8)] + [0x24]
    out = {}
    for w in range(8):
        lv = 0
        for k in range(woff[w + 1] - woff[w]):
            v = b(0x9cbc + woff[w] + k)
            out.setdefault(w * 4 + lv, (w, lv, k, v))
            if v != INTRO:
                lv += 1
    assert len(out) == 32
    return out


def make_starts(max_decisions=400):
    """CPU env whose env e holds level e's start in place slot e (e = 1..31; level 0 = power-on). Returns (env,
    {level: RAM at the save}, {level: picture})."""
    plan = level_plan()
    n = 32
    env = RlEnv({'smb': n}, device='cpu', sticky=False, idle_max=0, dropout=False, level_cap=64, level_slots=1)
    env.set_allow_skip(True)   # the jump from 1-1 to the set level is not a skip
    env.bind_ram(range(2048))
    env.start()
    buf = env.state_dict()['env']
    raw = buf.tobytes()
    o = raw.find(env.ram[0].numpy().tobytes())     # the console RAM item: 2 KB per env, env after env
    assert all(raw[o + e * 2048:o + (e + 1) * 2048] == env.ram[e].numpy().tobytes() for e in range(n))
    for e in range(1, n):
        w, lv, area_no, ptr = plan[e]
        pokes = {0x75F: w, 0x75C: lv, 0x760: area_no,           # WorldNumber, LevelNumber, AreaNumber
                 0x750: ptr, 0x74E: (ptr & 0x60) >> 5,           # LoadAreaPointer: AreaPointer, AreaType
                 0x75B: 0,                                       # HalfwayPage: the level's start
                 0x754: 1, 0x757: 1, 0x747: 0, 0x756: 0,         # ContinueGame: PlayerSize small, new timer, ...
                 0x0E: 0, 0x772: 0, 0x770: 1}                    # ... GameEngineSubroutine, OperMode_Task, OperMode
        for a, v in pokes.items():
            buf[o + e * 2048 + a] = v
    assert env.lib.rlenv_load(env.h, buf.ctypes.data_as(ctypes.c_void_p), None) == 0
    env.actions.zero_()
    rams, pics = {}, {}
    from fceux_palette import RGB
    pal = np.array(RGB, np.uint8)
    for _ in range(max_decisions):
        env.step()
        assert not env.done.any() and int(env.pads.max()) == 0
        R = env.ram.numpy()
        for e in range(1, n):
            r = R[e].astype(int)
            if e not in rams and r[0x770] == 1 and r[0x772] == 3 and r[0x0E] == 8 and r[0xB5] < 2:
                env.place_save(0, torch.tensor([e], dtype=torch.int32), torch.tensor([e], dtype=torch.int32))
                assert r[0x75F] * 4 + r[0x75C] == e and int(env.level_now[e]) == e, (e, r[0x75F], r[0x75C])
                rams[e] = r
                ob, em = env.obs[e, 3].numpy().astype(np.int32), env.emph[e, 3].numpy().astype(np.int32)
                pics[e] = pal[ob | (em[:, None] << 6)]
        if len(rams) == n - 1:
            return env, rams, pics
    raise RuntimeError(f'levels never in play: {sorted(set(range(1, n)) - set(rams))}')


def archive_parts(buf, lay):
    """Views of the level-start archive at the end of a one-game env state (rl/csrc/rlenv_build.hpp for_each_state)."""
    cap, stride, st, kc = lay['cap'], lay['stride'], lay['st'], lay['key_cap']
    sizes = [('mem', cap * stride), ('st', cap * st), ('keys', cap * kc * 4), ('target', cap * 8), ('frame', cap * 4),
             ('deaths', cap * 4)]
    o = len(buf) - sum(s for _, s in sizes)
    out = {}
    for name, s in sizes:
        out[name] = buf[o:o + s].reshape(cap, -1) if s else None
        o += s
    return out


def fill(dst_buf, dst_lay, src_buf, src_lay, levels):
    """Copy level slots `levels` from src's archive into dst's (same game, same console layout); 0 deaths."""
    assert (dst_lay['stride'], dst_lay['st'], dst_lay['key_cap']) == (src_lay['stride'], src_lay['st'], src_lay['key_cap'])
    d, s = archive_parts(dst_buf, dst_lay), archive_parts(src_buf, src_lay)
    for lv in levels:
        for k in ('mem', 'st', 'keys', 'target', 'frame'):
            if d[k] is not None:
                d[k][lv] = s[k][lv]
        d['deaths'][lv] = np.frombuffer(np.int32(0).tobytes(), np.uint8)
    return dst_buf


def deaths(buf, lay):
    return archive_parts(buf, lay)['deaths'].copy().view(np.int32).ravel()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='?')
    ap.add_argument('--out', default=None, help='default: overwrite CKPT')
    ap.add_argument('--sheet', default=None, help='PNG of all 31 starts (1-2 .. 8-4)')
    a = ap.parse_args()
    env, rams, pics = make_starts()
    for e, r in sorted(rams.items()):
        print(f'{r[0x75F] + 1}-{r[0x75C] + 1}: area {(r[0x74E] & 3) * 32 + (r[0x74F] & 31)}, Mario x {r[0x6D] * 256 + r[0x86]} '
              f'y {r[0xCE]}, lives {r[0x75A] + 1}, timer {r[0x7F8]}{r[0x7F9]}{r[0x7FA]}, hard mode {r[0x6CC]}')
    if a.sheet:
        from PIL import Image
        sheet = Image.new('RGB', (240 * 8, 224 * 4))
        for e, p in pics.items():
            sheet.paste(Image.fromarray(p), ((e % 8) * 240, (e // 8) * 224))
        sheet.save(a.sheet)
    if a.ckpt:
        src_buf, src_lay = env.state_dict()['env'], env.level_layout(0)
        s = torch.load(a.ckpt, map_location='cpu', weights_only=False)
        sd = s['env']
        assert sd['games'] == ['smb'] and sd.get('level_cap') == 64, 'needs a one-game SMB state with 64 level slots'
        dst = np.ascontiguousarray(sd['env'])
        dst_lay = dict(src_lay, levels=64)
        before = deaths(dst, dst_lay)
        empty = [lv for lv in range(1, 32) if before[lv] < 0]
        print('saved before:', [lv for lv in range(64) if before[lv] >= 0], 'filling:', empty)
        sd['env'] = fill(dst, dst_lay, src_buf, src_lay, empty)
        torch.save(s, a.out or a.ckpt)
        print('saved after:', [lv for lv in range(64) if deaths(sd['env'], dst_lay)[lv] >= 0], '->', a.out or a.ckpt)
