"""Record the fceumm (stable_retro) reference: per-frame buttons and CPU RAM after every frame.

The input is the game's reward10 boot() followed by a saved search run (actions from power-on)
or, without one, seeded random actions. nesle_trace replays the same per-frame buttons.

usage: python fceumm_trace.py GAME OUT_DIR [--acts RESULT.json] [--random N] [--seed S] [--max-frames F]
writes OUT_DIR/GAME.buttons.bin (1 byte/frame), GAME.ram.bin (2048 bytes/frame), GAME.json
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reward10'))
from nesram import Nes  # noqa: E402
from search import apply_action  # noqa: E402

# NES controller order as the console reads it: bit0 A ... bit7 Right
NES_BITS = {'A': 0, 'B': 1, 'SELECT': 2, 'START': 3, 'UP': 4, 'DOWN': 5, 'LEFT': 6, 'RIGHT': 7}


class StopTrace(Exception):
    pass


class TraceNes(Nes):
    """Nes that logs the button byte and the RAM after every emulated frame."""

    def __init__(self, rom, max_frames):
        super().__init__(rom)
        self.buttons, self.rams, self.max_frames = [], [], max_frames

    def step(self, buttons=(), frames=1, record=None):
        byte = sum(1 << NES_BITS[b] for b in buttons)
        for _ in range(frames):
            if len(self.buttons) >= self.max_frames:
                raise StopTrace
            super().step(buttons, 1)
            self.buttons.append(byte)
            self.rams.append(self.ram())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game')
    ap.add_argument('out')
    ap.add_argument('--acts', help='search result JSON whose acts run from power-on (boot included)')
    ap.add_argument('--random', type=int, default=0, help='random decisions after boot (when no --acts)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-frames', type=int, default=60 * 60 * 10)
    a = ap.parse_args()
    g = importlib.import_module(f'games.{a.game}')
    n = TraceNes(g.ROM, a.max_frames)
    source = 'boot only'
    try:
        g.boot(n)
        boot_frames = len(n.buttons)
        if a.acts:
            acts = json.loads(Path(a.acts).read_text())['acts']
            source = f'{a.acts} ({len(acts)} decisions)'
            for i in acts:
                apply_action(n, g.ACTIONS[i])
        elif a.random:
            rng = random.Random(a.seed)
            source = f'random seed {a.seed} ({a.random} decisions)'
            for _ in range(a.random):
                apply_action(n, g.ACTIONS[rng.randrange(len(g.ACTIONS))])
    except StopTrace:
        pass
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.array(n.buttons, np.uint8).tofile(out / f'{a.game}.buttons.bin')
    np.stack(n.rams).astype(np.uint8).tofile(out / f'{a.game}.ram.bin')
    meta = dict(game=a.game, rom=g.ROM, rom_sha256=g.ROM_SHA256, frames=len(n.buttons), boot_frames=boot_frames,
                input=source)
    (out / f'{a.game}.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
