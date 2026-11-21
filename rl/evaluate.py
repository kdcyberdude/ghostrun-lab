"""Greedy evaluation from fixed starts with fixed seeds: one env per game (a model with K moods: K envs per game,
env i plays mood i, so the moods' different ways of playing can be compared), no sticky actions, no idle frames, button
dropout ON with a fixed schedule (rl/csrc/controller.cuh episode seeds from --seed). Each game plays until its first
episode ends (game over, won, skipped level, stall); `steps` is only a safety cap. Writes to OUT/:
  summary.json             per game (and mood): return, decisions, end reason, deaths, max level, reward events,
                           the return split into reward parts (progress, level, death, damage, score, win, stall)
  GAME.mp4 (in --video-dir, named TAG_GAME[_moodK].mp4)  the last frame of every decision, full 240x224 model
                           picture (FCEUX colours) shown 3x with sharp pixels, at real speed (15 decisions/s), H.264
  GAME.pads.bin            pad byte of every frame from power-on (boot + 4 neutral + the AI's frames): replays in
                           FCEUX (rl/scripts/fceux_trace.lua with TRACE_LEAD=2) or as an FCEUX movie
  GAME.decisions.csv       one row per decision: the buttons pressed (each of its 4 frames) and, for SMB, the state
                           after it (world-level, area, camera x, Mario's x/y, lives, new camera position or not)
usage (standalone): python rl/evaluate.py CKPT OUT [--device cpu] [--games smb] [--threads 16] [--video-dir D] [--tag T]
                    [--starts 15,31 --archive LEVEL_STARTS.npy]
--starts: instead of power-on, one play from each of these saved level starts (level index, SMB 15 = 4-4), taken from
the run's level-start archive (RUN/level_starts_GAME.npy, written by the trainer at every checkpoint); the files are
then named ..._from_W-L (SMB) or ..._from_LN, and summary.json lists every start. (The .pads.bin of such a play does not
replay from power-on.)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fceux_palette import RGB  # noqa: E402
from model import Policy, pack  # noqa: E402
from rlenv import RlEnv, GAMES, END_NAMES, boot_pads, PART_NAMES, NUM_PARTS  # noqa: E402
from coverage import Coverage  # noqa: E402

PAL = np.array(RGB, np.uint8)
PAD_NAMES = ((4, 'UP'), (5, 'DN'), (6, 'LT'), (7, 'RT'), (0, 'A'), (1, 'B'), (3, 'ST'), (2, 'SL'))
STRIP_H = 16
_strips = {}


def pad_names(pad):
    return '+'.join(nm for bit, nm in PAD_NAMES if (pad >> bit) & 1) or '-'


def pad_strip(pad, w=240):
    """[STRIP_H, w, 3] bar under the picture showing the buttons the model held this frame (lit = pressed)."""
    if pad not in _strips:
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (w, STRIP_H), (16, 16, 16))
        d = ImageDraw.Draw(img)
        bw = w // len(PAD_NAMES)
        for k, (bit, name) in enumerate(PAD_NAMES):
            on = (pad >> bit) & 1
            d.rectangle([k * bw + 1, 1, (k + 1) * bw - 2, STRIP_H - 2], fill=(230, 60, 40) if on else (60, 60, 60))
            d.text((k * bw + 3, 3), name, fill=(255, 255, 255) if on else (150, 150, 150))
        _strips[pad] = np.asarray(img)
    return _strips[pad]


class Mp4Writer:
    """Streams RGB frames [H, W, 3] uint8 into an H.264 MP4 through ffmpeg (nearest-neighbour upscale keeps NES
    pixels sharp; frames go straight to disk, so a long episode needs no memory)."""

    def __init__(self, path, w=240, h=224, fps=15, scale=3):
        import subprocess
        self.path = path
        self.p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                   '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                                   '-vf', f'scale={w * scale}:{h * scale}:flags=neighbor', '-c:v', 'libx264',
                                   '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                   str(path)], stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, f):
        self.p.stdin.write(np.ascontiguousarray(f).tobytes())
        self.frames += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f'ffmpeg failed writing {self.path}')


@torch.no_grad()
def evaluate(model, dev, out, steps=100000, seed=12345, amp_dtype=torch.float16, games=None, video_dir=None, tag='',
             greedy=True, dropout=True, allow_skip=False, start_level=None, archive=None):
    """Plays until every env's first episode ends (game over, won, skipped level, stall) - `steps` is only a safety
    cap. Videos go to video_dir (default: out) as TAG_NAME.mp4. A RAM-input model (Policy ram=True) gets the RAM; SMB
    plays also report `cells`, the camera positions reached (rl/coverage.py)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    vdir = Path(video_dir) if video_dir else out
    vdir.mkdir(parents=True, exist_ok=True)
    games = games or GAMES
    # The CPU build (bit-identical to the GPU env) plays the envs: a GPU step costs the same ~0.45 s however few
    # envs it has. The model runs on `dev`.
    K = getattr(model, 'moods', 0)
    per = max(1, K)
    lc = 64 if start_level is not None else 0
    env = RlEnv({g: per for g in games}, device='cpu', seed=seed, sticky=False, idle_max=0, dropout=dropout, level_cap=lc)
    if start_level is not None:     # every episode starts from this saved level start (the run's archive)
        import slots as slotlib
        assert len(env.games) == 1, 'level-start plays take one game (one archive)'
        slotlib.load_entrances(env, 0, archive)
        w = np.zeros(lc)
        w[start_level] = 1.0
        env.set_level_plan(0, 1.0, w)
    if allow_skip:
        env.set_allow_skip(True)
    use_ram = getattr(model, 'ram', False)
    if use_ram or 'smb' in games:
        env.bind_ram(range(2048))
    env.start()
    cov = Coverage(env.n, 'cpu') if 'smb' in games else None
    if cov is not None:
        cov.begin(env.ram)
    was_training = model.training
    model.eval()
    n = env.n
    h = torch.zeros(n, model.core, device=dev)
    prev = [torch.zeros(n, dtype=torch.long, device=dev) for _ in range(5)]
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    names = [g if not K else f'{g}_mood{z}' for g in env.games for z in range(per)]
    if start_level is not None:
        where = f'{start_level // 4 + 1}-{start_level % 4 + 1}' if env.games[0] == 'smb' else f'L{start_level}'
        names = [f'{nm}_from_{where}' for nm in names]
    gnames = [g for g in env.games for _ in range(per)]
    mood = torch.arange(n, device=dev) % per if K else None
    pre = f'{tag}_' if tag else ''
    videos = [Mp4Writer(vdir / f'{pre}{name}.mp4', h=224 + STRIP_H) for name in names]
    pads = [bytearray(boot_pads(g)) + bytes(4) for g in gnames]
    smb_cols = ',world_level,area,camera_x,mario_x,mario_y,lives,new_position,positions_so_far'
    rows = [['decision,buttons_frame1,buttons_frame2,buttons_frame3,buttons_frame4' + (smb_cols if g == 'smb' else '')]
            for g in gnames]
    res = [dict(game=g, ret=0.0, decisions=0, end='running', deaths=0, level=0, events=0) for g in gnames]
    if start_level is not None:
        for r in res:
            r['start_level'] = start_level
    if K:
        for i, r in enumerate(res):
            r['mood'] = i % per
    parts_sum = np.zeros((n, NUM_PARTS))
    for t in range(steps):
        with torch.autocast(torch.device(dev).type, dtype=amp_dtype, enabled=amp_dtype != torch.float32):
            z = model.encode(env.obs.to(dev), env.emph.to(dev), env.timeline.to(dev), prev, mood,
                             env.ram.to(dev) if use_ram else None)
        h = model.gru(z.float(), h)
        parts, _ = model.act(h, env.masks.to(dev), greedy=greedy)
        env.actions.copy_(pack(parts).cpu())
        env.step()
        prev = parts
        a = alive.cpu().numpy()
        rew, done, info = env.reward.cpu().numpy(), env.done.cpu().numpy(), env.info.cpu().numpy()
        if cov is not None:
            found_before = cov.found.clone()
            cov.step(env.ram, env.done.bool())
        pd, ev = env.pads.cpu().numpy(), env.events.cpu().numpy()
        last = env.obs[:, 3].numpy(), env.emph[:, 3].numpy()
        for i in range(n):
            if not a[i]:
                continue
            r = res[i]
            r['ret'] += float(rew[i]); r['decisions'] += 1; r['events'] += int(ev[i] != 0)
            parts_sum[i] += env.parts[i].numpy()
            pads[i] += bytes(pd[i].tolist())
            row = f"{r['decisions']}," + ','.join(pad_names(int(x)) for x in pd[i])
            if gnames[i] == 'smb' and getattr(env, 'ram', None) is not None and not done[i]:
                rm = env.ram[i].numpy().astype(int)
                row += (f",{rm[0x75F] + 1}-{rm[0x75C] + 1},{(rm[0x74E] & 3) * 32 + (rm[0x74F] & 31)},"
                        f"{rm[0x71A] * 256 + rm[0x71C]},{rm[0x6D] * 256 + rm[0x86]},{rm[0xCE]},{rm[0x75A] + 1},"
                        + (f"{int(cov.found[i] > found_before[i])},{int(cov.found[i])}" if cov is not None else ','))
            rows[i].append(row)
            if done[i]:
                r.update(end=END_NAMES.get(int(info[i, 2]), '?'), deaths=int(info[i, 3]), level=int(info[i, 4]),
                         ret=float(info[i, 0]))
                if cov is not None and gnames[i] == 'smb':
                    r['cells'] = int(found_before[i])
                alive[i] = False
            else:  # (after an end the pictures are already the next episode's start)
                o, e = last[0][i].astype(np.int32), last[1][i].astype(np.int32)
                videos[i].write(np.concatenate([PAL[o | (e[:, None] << 6)], pad_strip(int(pd[i][-1]))], 0))
        if not alive.any():
            break
    for i, name in enumerate(names):
        if cov is not None and gnames[i] == 'smb' and 'cells' not in res[i]:
            res[i]['cells'] = int(cov.found[i])    # still playing at the step cap
        res[i]['parts'] ={k: round(float(v), 4) for k, v in zip(PART_NAMES, parts_sum[i])}
        res[i]['video'] = str(videos[i].path)
        videos[i].close()
        (out / f'{name}.pads.bin').write_bytes(bytes(pads[i]))
        (out / f'{name}.decisions.csv').write_text('\n'.join(rows[i]) + '\n')
    (out / 'summary.json').write_text(json.dumps(res, indent=1))
    if was_training:
        model.train()
    del env
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=100000, help='safety cap; play stops when every episode has ended')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--games', default=None, help='comma list (default: all 12)')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--video-dir', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--starts', default='', help='comma list of saved level starts to play from (instead of power-on)')
    ap.add_argument('--archive', default='', help='level-start archive .npy (the run\'s level_starts_GAME.npy)')
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    s = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = s['cfg']
    m = Policy(core=cfg['core'], moods=cfg.get('moods', 0), ram=bool(cfg.get('ram_io', 0))).to(dev)
    m.load_state_dict(s['model'])
    amp = torch.bfloat16 if dev.type == 'cuda' else torch.float32
    kw = dict(amp_dtype=amp, games=a.games.split(',') if a.games else None, video_dir=a.video_dir, tag=a.tag,
              greedy=not cfg.get('coverage', 0), dropout=bool(cfg.get('dropout', 1)),
              allow_skip=bool(cfg.get('coverage', 0)))
    if a.starts:
        res = []
        for lv in (int(x) for x in a.starts.split(',')):
            res += evaluate(m, dev, a.out, a.steps, start_level=lv, archive=a.archive, **kw)
        (Path(a.out) / 'summary.json').write_text(json.dumps(res, indent=1))
    else:
        res = evaluate(m, dev, a.out, a.steps, **kw)
    for r in res:
        print(r, flush=True)
