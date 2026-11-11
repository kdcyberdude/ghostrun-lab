"""Render reward events as screenshots so a human (or Claude) can check them.

replay: replay a search result's actions from power-on, capture each event.
random: sticky-random play from the fresh start until game over, capture each event.
Each captured frame is annotated with the step, event and current signals, and all
frames go on one contact sheet. A signals timeline is written as JSONL.

usage: python audit.py <game> replay --result DIR/seed0.json --out DIR
       python audit.py <game> random --episodes 4 --steps 3000 --out DIR
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nesram import Nes, run_isolated  # noqa: E402
from reward import RewardTracker  # noqa: E402
from search import apply_action, load  # noqa: E402


def caption(img, text):
    im = Image.fromarray(img).resize((256, 240), Image.NEAREST)
    out = Image.new('RGB', (256, 240 + 28), 'white')
    out.paste(im, (0, 28))
    d = ImageDraw.Draw(out)
    for i, line in enumerate(text.split('\n')[:2]):
        d.text((2, 2 + 12 * i), line[:44], fill='black')
    return out


def sheet(tiles, path, cols=6):
    if not tiles:
        return
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    s = Image.new('RGB', (cols * w, rows * h), 'gray')
    for i, t in enumerate(tiles):
        s.paste(t, ((i % cols) * w, (i // cols) * h))
    s.save(path)


def sig_text(sig):
    s = sig
    return (f"L{s.level} p{s.progress:.0f} lv{s.lives} hp{'' if s.health is None else round(s.health, 2)} "
            f"sc{s.score} {'PLAY' if s.in_play else 'noplay'}{' DYING' if s.dying else ''}"
            f"{' GO' if s.game_over else ''}{' WON' if s.won else ''}")


def run(game, acts_or_seed, steps, out, tag, max_tiles=36):
    g = load(game)
    n = Nes(g.ROM)
    g.boot(n)
    tr = RewardTracker(g)
    tr.update(n.ram(), 0)
    tiles = [caption(n.screen(), f'{tag} step0 START\n' + sig_text(tr.prev))]
    hist, ret, timeline = deque(maxlen=3), 0.0, []
    rng = random.Random(acts_or_seed) if isinstance(acts_or_seed, int) else None
    acts = acts_or_seed if rng is None else None
    a = 0
    total = len(acts) if acts is not None else steps
    pending = []  # (capture_at_step, label)
    for t in range(1, total + 1):
        if acts is not None:
            a = acts[t - 1]
        elif rng.random() > 0.8:
            a = rng.randrange(len(g.ACTIONS))
        f = apply_action(n, g.ACTIONS[a])
        r, done, ev = tr.update(n.ram(), t, frames=f)
        ret += r
        sig = tr.prev
        timeline.append(dict(t=t, r=round(r, 4), ret=round(ret, 3), **{k: v for k, v in asdict(sig).items()}))
        hist.append((t, n.screen()))
        for kind, val in ev:
            if len(tiles) < max_tiles:
                bt, bimg = hist[0]
                tiles.append(caption(bimg, f'{tag} t{bt} BEFORE {kind}\n' + sig_text(sig)))
                tiles.append(caption(n.screen(), f'{tag} t{t} {kind}={val} R={ret:.2f}\n' + sig_text(sig)))
                pending.append((t + 12, f'{kind} +12'))
        for at, label in [p for p in pending if p[0] == t]:
            if len(tiles) < max_tiles:
                tiles.append(caption(n.screen(), f'{tag} t{t} {label}\n' + sig_text(sig)))
        pending = [p for p in pending if p[0] > t]
        if done and not pending:
            break
    tiles.append(caption(n.screen(), f'{tag} t{t} END R={ret:.2f} deaths={tr.deaths}\n' + sig_text(tr.prev)))
    Path(out).mkdir(parents=True, exist_ok=True)
    sheet(tiles, Path(out) / f'{tag}.png')
    with open(Path(out) / f'{tag}.jsonl', 'w') as f:
        for row in timeline:
            f.write(json.dumps(row, default=str) + '\n')
    return dict(tag=tag, steps=t, ret=round(ret, 3), deaths=tr.deaths, events=tr.events[:40],
                final=sig_text(tr.prev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('game')
    ap.add_argument('mode', choices=['replay', 'random'])
    ap.add_argument('--result')
    ap.add_argument('--episodes', type=int, default=4)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    if a.mode == 'replay':
        acts = json.loads(Path(a.result).read_text())['acts']
        res = [run(a.game, acts, 0, a.out, 'replay_' + Path(a.result).stem)]
    else:
        res = run_isolated(run, [(a.game, s + 1000, a.steps, a.out, f'random{s}') for s in range(a.episodes)],
                           min(a.workers, a.episodes))
    for r in res:
        print(json.dumps(r, default=str))


if __name__ == '__main__':
    main()
