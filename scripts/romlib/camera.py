"""Stage 4 of a per-game RAM map: the CAMERA (scroll position in level pixels), found the same way for every game.

Truth from the picture: between two frames the background slides by the camera's move (camera right by c pixels ->
the picture's content moves left by c). The picture is cut into bands (16-pixel rows for x, 16-pixel columns for y);
each band that has texture votes the shift in -MAXS..MAXS that makes it match the previous frame best; a status bar
votes 0, so the shift with the most votes wins (a tie goes to the move). No band matches at all = a scene cut (a new
room, a fade, a send-back redraw): the camera track starts a new segment there.

Reader from RAM: the byte whose frame-to-frame change (mod 256, signed) equals the picture's shift (at a lag of up to
LAG frames: games write their scroll before or after the frame shows it), most often on frames where either moves;
then the byte that steps by +-1 exactly when that low byte wraps (the page). Both are plain RAM bytes, found by
statistics; nothing here knows a game.

Map: the playfield of one segment pasted at its camera position (a strip of the level, out/maps/SLUG.png): if the
camera is right, the strip is seamless.

usage: python camera.py OUT.jsonl [--only SLUG ...] [--workers 12] [--maps N]   (games from rl/games_table.json)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')     # one thread per worker process (many workers run at once)
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
sys.path.insert(0, str(HERE))
ROM_DIR = Path('/home/kd/Desktop/roms')
EPISODES, FRAMES = 3, 1500
MAXS = 8                  # largest shift per frame looked for (pixels)
BAND = 16
ROWS = 4                  # every ROWS-th line of a band is compared (full detail along the shift)
MATCH = 0.97              # a band matches at a shift when this share of its pixels are equal
LAG = 2
OK_SCORE = 0.9           # a reader that agrees with the picture less often is reported, not trusted
MIN_MOVE = 60             # frames with a picture shift needed to name a reader


def shift_votes(a, b, axis):
    """Shift c of b relative to a along axis (1: x, 0: y) per band: b[.., i] == a[.., i + c], judged on the middle of
    the band (the part every shift in -MAXS..MAXS keeps). Returns the voted c, or None when no textured band matches
    (a scene cut)."""
    if axis == 0:
        a, b = a.T, b.T
    H, W = a.shape
    nb = H // BAND
    A, B = a[:nb * BAND].reshape(nb, BAND, W)[:, ::ROWS], b[:nb * BAND].reshape(nb, BAND, W)[:, ::ROWS]
    flat = (A == A[:, :, :1]).all((1, 2))     # flat band (sky): matches any shift, no vote
    if flat.all():
        return 0
    win = np.lib.stride_tricks.sliding_window_view(A, W - 2 * MAXS, axis=2)     # [nb, BAND, 2 MAXS + 1, W - 2 MAXS]
    m = (win == B[:, :, None, MAXS:W - MAXS]).mean((1, 3))                     # [nb, shifts]
    srt = np.sort(m, 1)
    ok = ~flat & (srt[:, -1] >= MATCH) & (srt[:, -1] - srt[:, -2] > 0.005)
    if not ok.any():
        return None
    vals, cnt = np.unique(m.argmax(1)[ok] - MAXS, return_counts=True)
    top = vals[cnt == cnt.max()]
    return int(top[np.argmax(np.abs(top))])


def track(screens):
    """[T, H, W, 3] uint8 -> dx [T], dy [T] (shift at t vs t-1; 0 at t=0), cut [T] (True: no match vs t-1)."""
    code = (screens[..., 0] & 0xE0) | ((screens[..., 1] >> 5) << 2) | (screens[..., 2] >> 6)    # one byte per pixel
    T = len(code)
    dx, dy, cut = np.zeros(T, int), np.zeros(T, int), np.zeros(T, bool)
    for t in range(1, T):
        a, b = code[t - 1], code[t]
        x = shift_votes(a, b, 1)                  # sideways first, then up/down on the lined-up overlap
        y = None if x is None else shift_votes(*overlap(a, b, x, 1), 0)
        if y is None:                              # or the other way round (a mostly vertical move)
            y = shift_votes(a, b, 0)
            x = None if y is None else shift_votes(*overlap(a, b, y, 0), 1)
        if x is None or y is None:
            cut[t] = True
        else:
            dx[t], dy[t] = x, y
    return dx, dy, cut


def overlap(a, b, c, axis):
    """a and b cut to the part that overlaps after a shift c along axis (b[i] == a[i + c])."""
    n = a.shape[axis]
    ia, ib = (slice(c, n), slice(0, n - c)) if c >= 0 else (slice(0, n + c), slice(-c, n))
    return (a[:, ia], b[:, ib]) if axis == 1 else (a[ia], b[ib])


def signed(d):
    return ((d.astype(int) + 128) % 256) - 128


def find_reader(ram, d, cut):
    """ram [T, 2048], d [T] picture shift per frame -> (low byte, lag, score, moved frames), (page byte, score)."""
    best = (None, 0, 0.0, 0)
    for lag in range(-LAG, LAG + 1):
        # RAM change between t-1 and t, compared with the picture shift at t + lag
        dr = signed(np.diff(ram.astype(int), axis=0))           # [T-1, 2048]: change into frame t (t = 1..)
        t = np.arange(1, len(ram))
        tt = t + lag
        ok = (tt >= 1) & (tt < len(d))
        t, tt = t[ok], tt[ok]
        good = ~cut[tt] & ~cut[t]
        t, tt = t[good], tt[good]
        pd = d[tt]
        r = dr[t - 1]
        act = (pd != 0)[:, None] | (r != 0)
        n = act.sum(0)
        hit = ((r == pd[:, None]) & act).sum(0)
        moved = int((pd != 0).sum())
        sc = np.where(n >= MIN_MOVE, hit / np.maximum(n, 1), 0.0)
        a = int(sc.argmax())
        if sc[a] > best[2]:
            best = (a, lag, float(sc[a]), moved)
    a, lag, score, moved = best
    if a is None or moved < MIN_MOVE:
        return None
    lo = ram[:, a].astype(int)
    wrap = signed(np.diff(lo))
    small = np.abs(wrap) <= 2 * MAXS                             # a scroll step, not a reset (a death, a new room)
    up = np.flatnonzero((np.diff(lo) < 0) & (wrap > 0) & small) + 1     # low byte wrapped 255 -> 0 moving forward
    dn = np.flatnonzero((np.diff(lo) > 0) & (wrap < 0) & small) + 1
    page, ps = None, 0.0
    if len(up) + len(dn) >= 2:
        dh = signed(np.diff(ram.astype(int), axis=0))
        for pl in range(-LAG, LAG + 1):            # the page may be written a frame or two apart from the low byte
            exp = np.zeros(len(ram) - 1, int)
            u, w = up - 1 + pl, dn - 1 + pl
            exp[u[(u >= 0) & (u < len(exp))]] = 1
            exp[w[(w >= 0) & (w < len(exp))]] = -1
            act = ((exp != 0)[:, None] | (dh != 0)) & (small & ~cut[1:])[:, None]   # resets and cuts: not judged
            sc = ((dh == exp[:, None]) & act).sum(0) / np.maximum(act.sum(0), 1)
            sc[a] = 0
            b = int(sc.argmax())
            if sc[b] >= 0.9 and sc[b] > ps:
                page, ps = b, float(sc[b])
    return dict(lo=a, lag=lag, score=round(score, 3), ok=score >= OK_SCORE, moved=moved, page=page,
                page_score=round(ps, 3), wraps=int(len(up) + len(dn)))


def player_kind(ram, cam, addr):
    """Is the table's player byte a level coordinate ('world': it moves with the camera while the camera scrolls, the
    player held on screen) or a screen one ('screen': it stays put)? Judged on frames where the camera byte steps a
    little; None without a trusted camera or a player byte."""
    if not cam or not cam['ok'] or addr is None or addr < 0:
        return None
    dc = signed(np.diff(ram[:, cam['lo']].astype(int)))
    dp = signed(np.diff(ram[:, addr].astype(int)))
    m = (dc != 0) & (np.abs(dc) <= 2 * MAXS)
    if m.sum() < MIN_MOVE:
        return None
    same, still = float((dp[m] == dc[m]).mean()), float((dp[m] == 0).mean())
    kind = 'world' if same >= 0.6 else 'screen' if still >= 0.6 else 'mixed'
    return dict(kind=kind, same=round(same, 3), still=round(still, 3))


def play(rom, boot_pads, seed=11):
    from nesram import Nes
    from readers import buttons, random_pads
    nes = Nes(rom)
    for p in boot_pads:
        nes.step(buttons(p), 1)
    start = nes.save()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(EPISODES):
        nes.load(start)
        rams, scr = [], []
        for p in random_pads(rng, FRAMES):
            nes.step(buttons(p), 1)
            rams.append(nes.ram())
            scr.append(nes.screen())
        out.append((np.stack(rams), np.stack(scr)))
    return out


def strip(screens, dx, cut, path, max_w=8192):
    """Paste one segment's frames at their camera x (the longest uncut run of episode 0)."""
    from PIL import Image
    seg = [x for x in np.split(np.arange(len(dx)), np.flatnonzero(cut)) if len(x)]
    s = max(seg, key=lambda x: np.abs(dx[x[1:]]).sum() if len(x) > 1 else 0)
    cx = np.cumsum(dx[s] * (np.arange(len(s)) > 0))
    cx -= cx.min()
    H, W = screens.shape[1:3]
    w = min(int(cx.max()) + W, max_w)
    canvas = np.zeros((H, w, 3), np.uint8)
    for i, t in enumerate(s):
        x0 = int(cx[i])
        if x0 + W <= w:
            canvas[:, x0:x0 + W] = screens[t]
    Image.fromarray(canvas).save(path)


def one(args):
    slug, row, maps_dir = args
    try:
        eps = play(ROM_DIR / row['rom'], list(bytes.fromhex(row['boot_pads'])))
        rams = np.concatenate([r for r, _ in eps])
        dxs, dys, cuts = [], [], []
        for k, (_, s) in enumerate(eps):
            dx, dy, cut = track(s)
            cut[0] = True                       # episodes are not joined
            dxs.append(dx); dys.append(dy); cuts.append(cut)
            if k == 0 and maps_dir:
                strip(s, dx, cut, Path(maps_dir) / f'{slug}.png')
        dx, dy, cut = np.concatenate(dxs), np.concatenate(dys), np.concatenate(cuts)
        x, y = find_reader(rams, dx, cut), find_reader(rams, dy, cut)
        t = row.get('table') or {}
        res = dict(slug=slug, rom=row['rom'], scrolled_x=int(np.abs(dx).sum()), scrolled_y=int(np.abs(dy).sum()),
                   cuts=int(cut.sum()) - EPISODES, x=x, y=y,
                   player_x=player_kind(rams, x, t.get('x_addr', -1)), player_y=player_kind(rams, y, t.get('y_addr', -1)))
    except Exception as e:  # noqa: BLE001 - one bad ROM must not stop the sweep
        res = dict(slug=slug, rom=row.get('rom'), error=repr(e)[:300])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--maps', default='', help='folder for the strip pictures')
    a = ap.parse_args()
    table = json.loads((ROOT / 'rl/games_table.json').read_text())
    slugs = a.only or sorted(table)
    done = set()
    if Path(a.out).exists():
        done = {json.loads(l)['slug'] for l in open(a.out)}
    todo = [(s, table[s], a.maps) for s in slugs if s not in done]
    if a.maps:
        Path(a.maps).mkdir(parents=True, exist_ok=True)
    import multiprocessing as mp
    with open(a.out, 'a') as f, mp.get_context('spawn').Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(one, todo):
            f.write(json.dumps(r) + '\n')
            f.flush()


if __name__ == '__main__':
    main()
