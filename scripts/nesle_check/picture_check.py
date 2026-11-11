"""Picture check: NeSLE's 256x240 frame (its CUDA batch renderer, compiled for the CPU) against fceumm's.

fceumm runs through fceumm_lr (stable_retro's core binary, uncropped). For each game's same-button trace
(.cache/nesle-check/traces/GAME.*) two comparisons are made, one fceumm pass each:

  trace  NeSLE replays the trace's buttons from power-on; a frame is compared when NeSLE's CPU RAM
         (stack page $0100-$01FF ignored) equals fceumm's after this frame and the previous one.
  state  every --every frames fceumm's state is loaded into NeSLE (fcs restore, paged boards included);
         NeSLE then runs the next two frames with the same buttons and both pictures are compared.

usage: python picture_check.py OUT_JSON GAME.. [--lib LIB] [--label L] [--every N] [--dot-offset D]
       [--png-dir DIR] [--worst K] [--max-frames F]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TRACES = REPO / '.cache/nesle-check/traces'
STACK = np.ones(2048, bool)
STACK[0x100:0x200] = False


def side_by_side(ref, got, path):
    from PIL import Image
    diff = (ref != got).any(2)
    mask = np.zeros_like(ref)
    mask[diff] = (255, 0, 255)
    gap = np.full((240, 4, 3), 128, np.uint8)
    img = Image.fromarray(np.hstack([ref, gap, got, gap, mask]))
    img.resize((img.width * 2, img.height * 2), Image.NEAREST).save(path)


class Tally:
    """Per-mode frame statistics plus the worst frames (kept as pixel pairs until the end)."""

    def __init__(self, worst):
        self.n = 0
        self.same_full = 0
        self.same_crop = 0
        self.diff_px = 0
        self.per_frame = []
        self.worst = []  # (diff, frame, tag, ref, got)
        self.k = worst

    def add(self, frame, tag, ref, got):
        d = (ref != got).any(2)
        n = int(d.sum())
        self.n += 1
        self.same_full += n == 0
        self.same_crop += not d[8:232].any()
        self.diff_px += n
        self.per_frame.append((frame, n, int(d[8:232].sum())))
        if n and (len(self.worst) < self.k or n > self.worst[-1][0]):
            self.worst.append((n, frame, tag, ref.copy(), got.copy()))
            self.worst.sort(key=lambda w: -w[0])
            del self.worst[self.k:]

    def result(self, png_dir, game, mode):
        out = dict(frames=self.n,
                   identical_full_pct=round(100 * self.same_full / self.n, 3) if self.n else None,
                   identical_lines_8_231_pct=round(100 * self.same_crop / self.n, 3) if self.n else None,
                   mean_diff_pixels=round(self.diff_px / self.n, 2) if self.n else None,
                   worst=[])
        for n, frame, tag, ref, got in self.worst:
            path = png_dir / f'{game}_{mode}_{frame:06d}{tag}.png'
            side_by_side(ref, got, path)
            out['worst'].append(dict(frame=frame, diff_pixels=n, png=os.path.relpath(path, REPO)))
        return out


def check_game(game, lib, every, dot_offset, png_dir, worst, max_frames):
    os.environ['NESLE_HOST_LIB'] = str(lib)
    sys.path.insert(0, str(HERE))
    import fceumm_lr
    import neslenes

    meta = json.loads((TRACES / f'{game}.json').read_text())
    buttons = np.fromfile(TRACES / f'{game}.buttons.bin', np.uint8)[:max_frames]
    ref = fceumm_lr.Fceumm(meta['rom'])
    trace = neslenes.NesleNes(meta['rom'])
    loaded = neslenes.NesleNes(meta['rom'])
    tallies = dict(trace=Tally(worst), state=Tally(worst))
    pending = {}  # fceumm frame index -> NeSLE picture of that frame from a loaded state
    ram_ok_prev = False
    ram_ok_frames = 0
    load_failures = 0
    t0 = time.time()
    for i, b in enumerate(buttons):
        b = int(b)
        ref.step_byte(b)
        neslenes.lib().nh_step(trace.h, b, 1)
        f_pic = ref.screen()
        ram_ok = bool(((ref.ram() == trace.ram()) | ~STACK).all())
        ram_ok_frames += ram_ok
        if ram_ok and ram_ok_prev:
            tallies['trace'].add(i, '', f_pic, trace.screen())
        ram_ok_prev = ram_ok
        if i in pending:
            tag, pic = pending.pop(i)
            tallies['state'].add(i, tag, f_pic, pic)
        if i % every == 0 and i + 2 < len(buttons):
            try:
                loaded.load_fcs(ref.get_state(), dot_offset)
            except RuntimeError:
                load_failures += 1
                continue
            for k in (1, 2):
                neslenes.lib().nh_step(loaded.h, int(buttons[i + k]), 1)
                pending[i + k] = (f'_s{k}', loaded.screen())
    res = dict(game=game, rom=Path(meta['rom']).name, frames=len(buttons), input=meta['input'],
               ram_match_frames=ram_ok_frames, state_samples_every=every, state_load_failures=load_failures,
               seconds=round(time.time() - t0, 1))
    for mode, t in tallies.items():
        res[mode] = t.result(png_dir, game, mode)
        res[mode]['diff_pixels_per_frame'] = t.per_frame
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('games', nargs='+')
    ap.add_argument('--lib', default=str(REPO / '.cache/nesle-check-render/libnesle_host.so'))
    ap.add_argument('--label', default='after')
    ap.add_argument('--every', type=int, default=30)
    ap.add_argument('--dot-offset', type=int, default=0)
    ap.add_argument('--png-dir', default=None)
    ap.add_argument('--worst', type=int, default=4)
    ap.add_argument('--max-frames', type=int, default=10 ** 9)
    a = ap.parse_args()
    png_dir = Path(a.png_dir or REPO / f'.cache/nesle-check-render/png/{a.label}')
    png_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(len(a.games), 12)) as pool:
        jobs = [pool.apply_async(check_game, (g, Path(a.lib).resolve(), a.every, a.dot_offset, png_dir, a.worst,
                                              a.max_frames)) for g in a.games]
        rows = [j.get() for j in jobs]
    for r in rows:
        print(f"{r['game']:13s} trace {r['trace']['frames']:6d} fr {r['trace']['identical_full_pct']}% "
              f"(8-231 {r['trace']['identical_lines_8_231_pct']}%) mean {r['trace']['mean_diff_pixels']} | "
              f"state {r['state']['frames']:5d} fr {r['state']['identical_full_pct']}% "
              f"(8-231 {r['state']['identical_lines_8_231_pct']}%) mean {r['state']['mean_diff_pixels']} "
              f"[{r['seconds']}s]", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(label=a.label, lib=str(a.lib), every=a.every, dot_offset=a.dot_offset,
                                           games=rows)) + '\n')


if __name__ == '__main__':
    main()
