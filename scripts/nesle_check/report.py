"""Summarize a NeSLE-vs-fceumm RAM check: frame equality and the reward10 reward on both traces.

usage: python report.py TRACE_DIR OUT.json GAME..   (runs nesle_trace with fill ff, start 0, vblank boundary)
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'reward10'))
import compare as C  # noqa: E402
from reward import RewardTracker  # noqa: E402


def reward_run(g, rams):
    tr = RewardTracker(g)
    total, events = 0.0, []
    for t, ram in enumerate(rams):
        r, done, ev = tr.update(ram, t, frames=1)
        total += r
        events += [(t, k) for k, _ in ev]
        if done:
            break
    return round(total, 3), events, tr.end_reason


def main():
    trace_dir, out, games = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    rows = []
    for game in games:
        meta, p, ref, got = C.run(trace_dir, game, 'ff', 0, 'vblank')
        n = min(len(ref), len(got))
        diff = ref[:n] != got[:n]
        ns = diff.copy()
        ns[:, 0x100:0x200] = False
        g = importlib.import_module(f'games.{game}')
        r_ref, e_ref, end_ref = reward_run(g, ref[:n])
        r_got, e_got, end_got = reward_run(g, got[:n])
        first_ev = next((i for i, (a, b) in enumerate(zip(e_ref, e_got)) if a != b), None)
        rows.append(dict(
            game=game, mapper_rom=Path(meta['rom']).name, frames=n, input=meta['input'],
            nesle_exit=p.returncode, nesle_msg=p.stderr.strip().splitlines()[-1],
            identical_frames=round(float(1 - diff.any(1).mean()), 5),
            identical_frames_ignoring_stack=round(float(1 - ns.any(1).mean()), 5),
            first_non_stack_diff_frame=int(np.argmax(ns.any(1))) if ns.any() else None,
            mean_diff_bytes_last_2000=round(float(diff[-2000:].sum(1).mean()), 1),
            reward_fceumm=r_ref, reward_nesle=r_got, reward_events_fceumm=len(e_ref), reward_events_nesle=len(e_got),
            first_reward_event_mismatch=None if first_ev is None and len(e_ref) == len(e_got) else
            (e_ref[first_ev] if first_ev is not None else e_ref[min(len(e_ref), len(e_got)):][:1],
             e_got[first_ev] if first_ev is not None else e_got[min(len(e_ref), len(e_got)):][:1]),
            end_fceumm=end_ref, end_nesle=end_got))
        print(json.dumps(rows[-1]), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        what='NeSLE v0.3.1 (0b04d8b) CUDA batch console code compiled for the CPU vs fceumm (stable_retro), '
             'same per-frame buttons, CPU RAM compared after every frame',
        settings=dict(power_on_ram='0xFF (fceumm)', frame_end='vblank (scanline 241 dot 1)', start_dot=0),
        games=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
