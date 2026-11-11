"""Picture check NeSLE -> FCEUX: does the model see the same picture in FCEUX as in NeSLE training?

Both emulators replay the same per-frame buttons from power-on (each game's .cache/nesle-check/traces/GAME.buttons.bin;
FCEUX gets 2 neutral frames first because it starts games 2 frames later). Frames where the 2 KB CPU RAM is equal in
both (stack page and WAIT_COUNTERS ignored) at this frame and the previous one are the "same game moment"; up to --shots of them per game
are captured in FCEUX (old PPU, power-on RAM 0xFF, emulator buffer = NES palette index per pixel) and in NeSLE, and the
pictures are compared on the model's 240x224 crop (x 8..247, y 8..231) and on the full 256x240 screen:

  index_same   NeSLE palette index (reconstructed from its RGB through fceumm's palette) == FCEUX palette index
  rgb_same     NeSLE RGB == FCEUX RGB (FCEUX's own default palette; differs where the two palettes differ)

usage: python rl/scripts/fceux_picture_check.py OUT_JSON [GAME..] [--shots 10] [--png-dir DIR]
Needs the FCEUX RAM traces from `LEAD=2 bash rl/scripts/fceux_trace_all.sh old1L2 0 1`. CPU only (xvfb).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TRACES = REPO / '.cache/nesle-check/traces'
FX_TRACES = REPO / '.cache/rl/fceux-traces'
FCEUX = REPO / '.cache/fceux-compat-build-v2/build/src/fceux'
NESLE_LIB = REPO / '.cache/nesle-check/libnesle_host.so'  # branch llm-gameplay-env 90d0cb3
os.environ.setdefault('NESLE_HOST_LIB', str(NESLE_LIB))
sys.path.insert(0, str(REPO / 'scripts/nesle_check'))
sys.path.insert(0, str(REPO / 'scripts/reward10'))
sys.path.insert(0, str(REPO / 'rl/scripts'))
import neslenes  # noqa: E402
import contract  # noqa: E402
from gen_reward_consts import GAMES  # noqa: E402

NOSTACK = np.ones(2048, bool)
NOSTACK[0x100:0x200] = False
# bytes a game counts up while it waits for the next frame (they differ every frame between emulators; measured)
WAIT_COUNTERS = {'excitebike': range(0x18, 0x20), 'contra': [0x34]}
CROP = (slice(8, 232), slice(8, 248))
LEAD, FRAMES = 2, 6000


def fceux_shots(game, rom, frames, workdir):
    home = tempfile.mkdtemp(prefix=f'fceux-{game}-')  # fresh profile: no battery save from an earlier run
    out = workdir / f'{game}.shots.bin'
    env = dict(os.environ, HOME=str(home), TRACE_BUTTONS=str(TRACES / f'{game}.buttons.bin'),
               TRACE_OUT=str(workdir / f'{game}.ram.bin'), TRACE_FRAMES=str(max(frames) + 1), TRACE_LEAD=str(LEAD),
               TRACE_SHOTS=','.join(map(str, frames)), TRACE_SHOT_OUT=str(out))
    subprocess.run(['xvfb-run', '-a', str(FCEUX), '--sound', '0', '--newppu', '0', '--ramInit', '1', '--loadlua',
                    str(REPO / 'rl/scripts/fceux_trace.lua'), rom], env=env, check=True, timeout=900,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(home)
    raw = np.fromfile(out, np.uint8).reshape(len(frames), -1)
    pal = raw[:, :256 * 240].reshape(-1, 240, 256)
    rgb = raw[:, 256 * 240:].reshape(-1, 240, 256, 3)
    return pal, rgb


def check(game, shots, workdir, png_dir):
    rom = contract.load(game).ROM
    buttons = np.fromfile(TRACES / f'{game}.buttons.bin', np.uint8)[:FRAMES]
    fx_ram = np.fromfile(FX_TRACES / f'{game}.old1L2.ram.bin', np.uint8).reshape(-1, 2048)
    keep = NOSTACK.copy()
    keep[list(WAIT_COUNTERS.get(game, []))] = False
    nes = neslenes.NesleNes(rom)
    same = []
    for i, b in enumerate(buttons[:len(fx_ram)]):
        neslenes.lib().nh_step(nes.h, int(b), 1)
        same.append(bool(((nes.ram() == fx_ram[i]) | ~keep).all()))
    same = np.array(same)
    ok = np.flatnonzero(same[1:] & same[:-1]) + 1
    ok = ok[ok >= 30]  # skip the blank power-on frames
    res = dict(game=game, frames=len(same), same_ram_frames=int(same.sum()), same_moments=int(len(ok)))
    if len(ok) == 0:
        return res
    pick = sorted(set(ok[np.linspace(0, len(ok) - 1, min(shots, len(ok))).round().astype(int)].tolist()))
    fx_pal, fx_rgb = fceux_shots(game, rom, pick, workdir)
    nes = neslenes.NesleNes(rom)
    ne_rgb = []
    want = set(pick)
    for i, b in enumerate(buttons[:max(pick) + 1]):
        neslenes.lib().nh_step(nes.h, int(b), 1)
        if i in want:
            ne_rgb.append(nes.screen())
    ne_rgb = np.array(ne_rgb)
    # NeSLE colour -> palette index: the mapping must be one index per colour (else the pictures differ).
    key = (ne_rgb[..., 0].astype(np.int32) << 16) | (ne_rgb[..., 1].astype(np.int32) << 8) | ne_rgb[..., 2]
    rows = []
    for k in range(len(pick)):
        r = dict(frame=pick[k])
        for name, sl in (('crop', CROP), ('full', (slice(None), slice(None)))):
            kk, fp = key[k][sl].ravel(), fx_pal[k][sl].ravel() & 0x3F
            # majority FCEUX index per NeSLE colour in this shot
            agree = 0
            for c in np.unique(kk):
                m = kk == c
                agree += np.bincount(fp[m]).max()
            r[f'{name}_index_same'] = round(agree / kk.size, 5)
            r[f'{name}_rgb_same'] = round(float((ne_rgb[k][sl] == fx_rgb[k][sl]).all(-1).mean()), 5)
        rows.append(r)
        if png_dir:
            from PIL import Image
            gap = np.full((240, 4, 3), 128, np.uint8)
            Image.fromarray(np.hstack([fx_rgb[k], gap, ne_rgb[k]])).save(Path(png_dir) / f'{game}_{pick[k]:05d}.png')
    res['shots'] = rows
    res['min_crop_index_same'] = min(r['crop_index_same'] for r in rows)
    res['min_crop_rgb_same'] = min(r['crop_rgb_same'] for r in rows)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('games', nargs='*')
    ap.add_argument('--shots', type=int, default=10)
    ap.add_argument('--png-dir')
    a = ap.parse_args()
    games = a.games or GAMES
    workdir = REPO / '.cache/rl/fceux-pictures'
    workdir.mkdir(parents=True, exist_ok=True)
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
    neslenes.lib()
    with ThreadPoolExecutor(12) as ex:
        results = list(ex.map(lambda g: check(g, a.shots, workdir, a.png_dir), games))
    for r in results:
        print(f"{r['game']:12s} same-RAM moments {r['same_moments']:5d}/{r['frames']}  shots {len(r.get('shots', []))}"
              f"  crop index-same min {r.get('min_crop_index_same', '-')}  crop rgb-same min {r.get('min_crop_rgb_same', '-')}")
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
