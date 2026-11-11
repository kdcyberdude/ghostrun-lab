#!/usr/bin/env python3
"""Record the v1 policy (the frozen all-32 player) on every SMB level.

v1 plays the whole game in one deterministic greedy run from a standard reset and clears 1-1 .. 8-4 in order, so one
run covers all 32 levels; it is cut per level. v1 runs in its own emulator (an nes_py-based "reference lab"), not in
the rl/ environment, with one decision per tick, each tick held for 7 or 8 frames (8 decisions per second).

The reference lab is an external JSON-lines tool (one request per stdin line, one reply per stdout line):
  reset {start: standard} -> auto {ticks: 1} (the policy picks and applies a tick) -> diagnostics (world, stage, mode)
Each reply has the tick's live 256x240 PNG, its applied 10-bit mask and frames_advanced. Point --reference-lab
(or GHOSTRUN_REFERENCE_LAB) at its launcher, called as: LAUNCHER --rom ROM --output DIR --diagnostics.
--replay-route plays the bundled recorded route (data/routes/smb_v1_all32_route.json, v1's own applied masks) with
`step` instead of asking the policy, for when only the emulator is available.

Per level L in OUT/v1/<L>/: smb_v1_<L>.actions.json, smb_v1_<L>.pads.bin, smb_v1_<L>.mp4 (each tick's picture held
for its frames, 60 fps, 3x). OUT/v1/full_game.actions.json is the whole run; OUT/v1/summary.json the per-level cut and
whether the run reproduced the bundled route mask for mask.

usage: python scripts/recording/record_v1.py --rom SMB.nes --out OUT [--reference-lab LAUNCHER] [--replay-route]
       [--levels all] [--max-ticks 12000] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from action_log import (Mp4Writer, expand, level_name, make_log, parse_levels, run_length, self_test,  # noqa: E402
                        v1_to_nes, validate_log, write_log)

ROUTE = ROOT / 'data/routes/smb_v1_all32_route.json'
CLOCK = 'reference_lab_frame (legacy 29,781 CPU-cycle slice of the nes_py core, not a PPU frame)'


class Lab:
    def __init__(self, launcher, rom, output):
        self.p = subprocess.Popen([launcher, '--rom', rom, '--output', output, '--diagnostics'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def call(self, **req):
        self.p.stdin.write(json.dumps(req) + '\n')
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f'reference lab closed on {req}')
        r = json.loads(line)
        if not r.get('ok'):
            raise RuntimeError(f'{req}: {r.get("error")}')
        return r

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=30)


def state(lab):
    d = lab.call(op='diagnostics')['decoded_existing_observer']
    return dict(level=(d['world'] - 1) * 4 + d['stage'] - 1, mode=d['mode_raw'], x=d['player_x_raw'])


def play(args, lab_dir):
    """One full run. Returns ticks: [{mask, frames, png, level, mode, x}] (level etc. after the tick)."""
    route = json.loads(ROUTE.read_text())['actions']
    lab = Lab(args.reference_lab, args.rom, lab_dir)
    ticks = []
    try:
        r = lab.call(op='reset', start='standard')
        first = dict(mask=None, frames=0, png=r['image']['path'], **state(lab))
        for t in range(args.max_ticks):
            if args.replay_route:
                if t >= len(route):
                    break
                r = lab.call(op='step', mask=route[t][0], ticks=1)
            else:
                r = lab.call(op='auto', ticks=1)
            s = state(lab)
            ticks.append(dict(mask=int(r['applied_masks'][-1]), frames=int(r['frames_advanced']),
                              png=r['image']['path'], **s))
            if r['terminal'] or s['mode'] == 3 or (s['mode'] == 2 and s['level'] // 4 == 7):
                break
    finally:
        lab.close()
    return first, ticks, route


def write_video(path, ticks):
    import numpy as np
    from PIL import Image
    v = Mp4Writer(path, 256, 240, fps=60)
    for t in ticks:
        img = np.asarray(Image.open(t['png']).convert('RGB'))
        for _ in range(t['frames']):
            v.write(img)
    v.close()


def record(args):
    out = Path(args.out) / 'v1'
    out.mkdir(parents=True, exist_ok=True)
    keep = args.keep_lab_output
    lab_dir = str(out / 'reference-lab') if keep else tempfile.mkdtemp(prefix='ghostrun-v1-lab-')
    try:
        first, ticks, route = play(args, lab_dir)
        masks = [t['mask'] for t in ticks]
        diverge = next((i for i, (a, b) in enumerate(zip(masks, (m for m, _ in route))) if a != b), None)
        frames_all = [v1_to_nes(t['mask']) for t in ticks for _ in range(t['frames'])]
        mode = 'route_replay' if args.replay_route else 'greedy'
        won = bool(ticks) and ticks[-1]['mode'] == 2 and ticks[-1]['level'] // 4 == 7
        full = make_log(level=0, model=args.model_name, mode=mode, frames=frames_all, start='reference_lab_standard_reset',
                        clock=CLOCK, result=dict(ticks=len(ticks), frames=len(frames_all), won=won,
                                                 last_level=level_name(ticks[-1]['level']) if ticks else None),
                        scope='full game (level field = first level)', ticks=run_length(t['mask'] for t in ticks),
                        ticks_bits='v1 10-bit masks per tick: up, down, left, right, a, b, x, y, start, select')
        write_log(out / 'full_game.actions.json', full)
        # cut per level: a tick belongs to the level shown after it; level k's segment starts at its first tick
        wanted = set(parse_levels(args.levels))
        segs, offset = {}, 0
        for i, t in enumerate(ticks):
            lv = t['level'] if t['mode'] == 1 else (ticks[i - 1]['level'] if i else first['level'])
            segs.setdefault(lv, dict(ticks=[], offset=offset))['ticks'].append(t)
            offset += t['frames']
        summary = []
        for lv in sorted(segs):
            if lv not in wanted:
                continue
            seg, name = segs[lv], level_name(lv)
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            frames = [v1_to_nes(t['mask']) for t in seg['ticks'] for _ in range(t['frames'])]
            nxt = max(segs) > lv
            result = dict(cleared=nxt or (lv == 31 and won), ticks=len(seg['ticks']), frames=len(frames),
                          max_x_px=max(t['x'] for t in seg['ticks']))
            log = make_log(level=lv, model=args.model_name, mode=mode, frames=frames,
                           start='reference_lab_standard_reset', clock=CLOCK, result=result,
                           route_offset_frames=seg['offset'],
                           note='frames are this level only; prepend the first route_offset_frames frames of '
                                'full_game.actions.json to replay it from the standard reset',
                           video=f'smb_v1_{name}.mp4')
            write_log(d / f'smb_v1_{name}.actions.json', log)
            write_video(d / f'smb_v1_{name}.mp4', seg['ticks'])
            summary.append(dict(level=name, **result, route_offset_frames=seg['offset']))
            print(f'[record-v1] {name}: {len(seg["ticks"])} ticks, {len(frames)} frames', flush=True)
        missing = [level_name(lv) for lv in sorted(wanted - set(segs))]
        (out / 'summary.json').write_text(json.dumps(dict(
            model=args.model_name, mode=mode, ticks=len(ticks), won=won, levels=summary, levels_not_reached=missing,
            route_ticks=len(route), first_divergence_from_route_tick=diverge,
            reproduced_route=diverge is None and len(masks) == len(route)), indent=1))
        print(f'[record-v1] {len(ticks)} ticks, won={won}, reproduced bundled route: '
              f'{diverge is None and len(masks) == len(route)}, missing levels: {missing}', flush=True)
    finally:
        if not keep:
            shutil.rmtree(lab_dir, ignore_errors=True)


def route_log(out_dir):
    """Dry-run check on real data: the bundled v1 route -> a full-game action log through the same writer."""
    route = json.loads(ROUTE.read_text())
    frames = [v1_to_nes(m) for m in expand(route['actions'])]
    assert len(frames) == route['nativeFrames'], (len(frames), route['nativeFrames'])
    log = make_log(level=0, model='ghostrun-smb-policy-v1', mode='recorded_route', frames=frames,
                   start='reference_lab_standard_reset', clock=CLOCK, result=dict(ticks=route['ticks']),
                   scope='full game, from data/routes (dry run)')
    path = Path(out_dir) / 'v1_route.actions.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_log(path, log)
    validate_log(json.loads(path.read_text()))
    # the route's own masks use the same first 8 bits as the native-trace order when x/y are unused
    assert log['actions'][0][0] == route['actions'][0][0] & 0xFF
    return dict(route_ticks=route['ticks'], route_frames=len(frames), log=str(path))


def dry_run(args):
    lab = args.reference_lab
    checks = {
        'reference lab launcher': (lab, bool(lab) and Path(lab).exists()),
        'rom': (args.rom, bool(args.rom) and Path(args.rom).exists()),
        'ffmpeg': (shutil.which('ffmpeg'), shutil.which('ffmpeg') is not None),
        'route': (str(ROUTE), ROUTE.exists()),
    }
    for k, (v, ok) in checks.items():
        print(f'[record-v1] {"ok  " if ok else "MISS"} {k}: {v}')
    levels = parse_levels(args.levels)
    print(f'[record-v1] plan: one {"route replay" if args.replay_route else "greedy policy"} run from a standard reset '
          f'(<= {args.max_ticks} ticks), cut into {len(levels)} levels -> MP4 + action logs under {Path(args.out) / "v1"}')
    print('[record-v1] self-test:', json.dumps(self_test(Path(args.out) / 'dry-run' / 'v1')))
    print('[record-v1] route check:', json.dumps(route_log(Path(args.out) / 'dry-run' / 'v1')))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference-lab', default=os.environ.get('GHOSTRUN_REFERENCE_LAB', ''))
    ap.add_argument('--rom', default=os.environ.get('GHOSTRUN_SMB_ROM', ''))
    ap.add_argument('--out', default='recordings')
    ap.add_argument('--levels', default='all')
    ap.add_argument('--max-ticks', type=int, default=12000, help='v1 needs 10,626 ticks for the whole game')
    ap.add_argument('--replay-route', action='store_true')
    ap.add_argument('--keep-lab-output', action='store_true', help='keep the lab journal and every tick PNG')
    ap.add_argument('--model-name', default='ghostrun-smb-policy-v1')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.dry_run:
        dry_run(a)
        return
    if not (a.reference_lab and a.rom):
        raise SystemExit('--reference-lab and --rom are required (or GHOSTRUN_REFERENCE_LAB / GHOSTRUN_SMB_ROM)')
    record(a)


if __name__ == '__main__':
    main()
