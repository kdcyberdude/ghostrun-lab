"""Inference-only validation (user request 2026-11-19): how well does a saved model really play, apart from training's
randomness (explorers, held random moves, mid-level starts, curriculum weights)? Only the model's own moves, sampled
from its policy as in training and test plays; no exploration; fixed seeds, so two models face the same games.

Two tests:
  levels  about P games per saved level start (SMB: 1-2 .. 8-4, the run's level-start archive), ONE LIFE each: a game
          passes if Mario reaches another level (the next one, or a warp) before his first death; a death, a stall,
          the end of the game or `cap` decisions = fail.
  full    F games from power-on (1-1), 3 lives, to the end: furthest level reached and how many won.
Writes OUT/itN.json and appends one line to OUT/history.jsonl.
usage: python rl/validate.py RUN [--model RUN/eval/itN/model.pt (default: newest)] [--per-level 32] [--full 32]
       [--device cuda] [--threads 16]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import Policy, pack  # noqa: E402
from rlenv import RlEnv, END_NAMES  # noqa: E402
import slots as slotlib  # noqa: E402
from evaluate import Mp4Writer, PAL, STRIP_H, pad_strip  # noqa: E402

LC = 64


def name(lv):
    return f'{lv // 4 + 1}-{lv % 4 + 1}'


@torch.no_grad()
def play(model, dev, env, cap, one_life, seed, vdir=None, tag=''):
    """Plays every env of `env` (already started) until its stop rule -> per env: start level, passed, end, furthest
    level, decisions. vdir: also a video (1x, H.264) of the first game from each start: TAG_from_W-L.mp4."""
    n = env.n
    use_ram = getattr(model, 'ram', False)
    gen = torch.Generator(device=dev).manual_seed(seed)
    h = torch.zeros(n, model.core, device=dev)
    prev = [torch.zeros(n, dtype=torch.long, device=dev) for _ in range(5)]
    alive = np.ones(n, bool)
    first = np.full(n, -1, np.int64)        # level of the start (known after the first decision)
    best = first.copy()
    passed = np.zeros(n, bool)
    end = np.array(['running'] * n, dtype=object)
    dec = np.zeros(n, np.int64)
    rec = {}                                 # env -> video writer (one game per start level)
    for t in range(cap):
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            z = model.encode(env.obs.to(dev), env.emph.to(dev), env.timeline.to(dev), prev, None,
                             env.ram.to(dev) if use_ram else None)
        h = model.gru(z.float(), h)
        parts, _ = model.act(h, env.masks.to(dev), generator=gen)
        env.actions.copy_(pack(parts).cpu())
        env.step()
        prev = parts
        done = env.done.cpu().numpy().astype(bool)
        lv = env.level_now.cpu().numpy()
        death = env.parts[:, 2].cpu().numpy() != 0
        info = env.info.cpu().numpy()
        dec[alive] += 1
        new = (first < 0) & (lv >= 0) & alive & ~done
        first = np.where(new, lv, first)
        if vdir is not None:
            taken = {int(first[i]) for i in rec}
            for i in np.flatnonzero(new):
                if int(first[i]) not in taken:
                    taken.add(int(first[i]))
                    rec[i] = Mp4Writer(Path(vdir) / f'{tag}_from_{name(int(first[i]))}.mp4', h=224 + STRIP_H, scale=1)
            pd = env.pads.numpy()
            for i, wr in rec.items():
                if alive[i] and not done[i]:
                    o, e = env.obs[i, 3].numpy().astype(np.int32), env.emph[i, 3].numpy().astype(np.int32)
                    wr.write(np.concatenate([PAL[o | (e[:, None] << 6)], pad_strip(int(pd[i][-1]))], 0))
        best = np.where(alive & ~done & (lv >= 0), np.maximum(best, lv), best)
        if one_life:
            moved = alive & ~done & (lv >= 0) & (first >= 0) & (lv != first)
            passed |= moved
            end[moved] = 'passed'
            end[alive & ~moved & death] = 'death'
            stop = moved | death | done
        else:
            stop = done
        for i in np.flatnonzero(alive & done & (end == 'running')):
            end[i] = END_NAMES.get(int(info[i, 2]), '?')
            best[i] = max(best[i], int(info[i, 4]))
        alive &= ~stop
        if not alive.any():
            break
    end[alive] = 'cap'
    for wr in rec.values():
        wr.close()
    return [dict(start=int(first[i]), passed=bool(passed[i]), end=str(end[i]), best=int(best[i]), decisions=int(dec[i]))
            for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--model', default='')
    ap.add_argument('--per-level', type=int, default=32)
    ap.add_argument('--full', type=int, default=32)
    ap.add_argument('--level-cap', type=int, default=3000, help='decisions per one-life level game')
    ap.add_argument('--full-cap', type=int, default=40000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--out', default='')
    ap.add_argument('--videos', default='', help='folder: one video per start level + one full game from 1-1')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    run = Path(a.run)
    mp = Path(a.model) if a.model else sorted(run.glob('eval/it*/model.pt'))[-1]
    out = Path(a.out) if a.out else run / 'validation'
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    s = torch.load(mp, map_location=dev, weights_only=False)
    cfg = s['cfg']
    m = Policy(core=cfg['core'], moods=cfg.get('moods', 0), ram=bool(cfg.get('ram_io', 0))).to(dev)
    m.load_state_dict(s['model'])
    m.eval()
    game = cfg['games'].split(',')[0]
    arch = run / f'level_starts_{game}.npy'
    t0 = time.time()
    kw = dict(device='cpu', sticky=False, idle_max=0, dropout=bool(cfg.get('dropout', 1)), level_cap=LC)

    # levels: P games per saved level, one life (every game starts at a saved level start)
    probe = RlEnv({game: 1}, **kw)
    slotlib.load_entrances(probe, 0, str(arch))
    saved = sorted(lv for lv in probe.saved_levels().get(game, {}) if lv > 0)
    del probe
    # all levels in one batch: every game starts at a saved level drawn uniformly (fixed seed: the same games for
    # every model), about P per level
    env = RlEnv({game: a.per_level * len(saved)}, seed=a.seed, **kw)
    env.set_allow_skip(True)
    slotlib.load_entrances(env, 0, str(arch))
    w = np.zeros(LC)
    w[saved] = 1.0 / len(saved)
    env.set_level_plan(0, 1.0, w)
    env.bind_ram(range(2048))
    env.start()
    tag = f"it{s.get('it', 0):06d}"
    vd = None
    if a.videos:
        vd = Path(a.videos) / tag
        vd.mkdir(parents=True, exist_ok=True)
    rows = play(m, dev, env, a.level_cap, True, a.seed, vd, tag)
    del env
    t1 = time.time()
    # full games from power-on, 3 lives
    env = RlEnv({game: a.full}, seed=a.seed, **kw)
    env.set_allow_skip(True)
    env.bind_ram(range(2048))
    env.start()
    full = play(m, dev, env, a.full_cap, False, a.seed, vd, tag + '_full')
    del env
    per = {}
    for r in rows:
        p = per.setdefault(name(r['start']), [0, 0])
        p[0] += r['passed']
        p[1] += 1
    rate = {k: v[0] / v[1] for k, v in per.items()}
    bests = [r['best'] for r in full]
    res = dict(it=s.get('it'), model=str(mp), time=time.time(), level_pass=rate,
               level_pass_mean=float(np.mean(list(rate.values()))) if rate else 0.0,
               levels_at_70=int(sum(v >= 0.7 for v in rate.values())),
               full_best_mean=float(np.mean(bests)), full_best_median=float(np.median(bests)),
               full_best_max=int(max(bests)), full_won=int(sum(r['end'] == 'won' for r in full)),
               full_best=bests, levels_seconds=round(t1 - t0), full_seconds=round(time.time() - t1),
               per_level_games=a.per_level, full_games=a.full)
    (out / f"it{res['it']:06d}.json").write_text(json.dumps(dict(res, level_games=rows, full_games_rows=full), indent=1))
    with open(out / 'history.jsonl', 'a') as f:
        f.write(json.dumps(res) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'full_best'}), flush=True)


if __name__ == '__main__':
    main()
