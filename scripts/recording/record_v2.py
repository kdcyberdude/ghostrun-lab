#!/usr/bin/env python3
"""Record the v2 policy (rl/ trainer, K moods) on every SMB level: per level, every mood plays from the level's
entrance; only the mood with the greatest progress is kept (cleared first, then max x, then the longest play).

Per level L (1-1 .. 8-4) in OUT/v2/<L>/:
  smb_v2_<L>.actions.json   ghostrun-action-log-v1 (scripts/recording/action_log.py): frames = NES pad byte per frame
  smb_v2_<L>.pads.bin       the same frames as raw bytes (rl/evaluate.py's *.pads.bin format)
  smb_v2_<L>.mp4            every emulated frame of the kept mood (240x224 model view, FCEUX colours, 3x, 60 fps)
  modes.json                every mood's result, so the choice can be checked
and OUT/v2/summary.json over all levels.

Starts. 1-1 starts at power-on (frames = boot pads + 4 neutral reset frames + play, exactly like rl/evaluate.py, so the
log replays from power-on anywhere). Every other level starts from the training run's saved level entrance
(rl/csrc/rlenv_core.cuh "level starts"), the same start the trainer and the per-level checks used; frames then begin
after the 4 neutral reset frames of that entrance. The entrance archive comes from --starts: either a .npy archive or a
training checkpoint (ckpt_rank0_latest.pt), from which the level-start slots are cut out and cached as .npy.

Play: greedy, FP32, no sticky actions, no idle frames, button dropout off unless --dropout, zero GRU state, one env per
mood. The emulator is the CPU build (bit-identical to the GPU build); the policy runs on --device (cuda when present).
A second pass replays the kept mood's recorded moves through a fresh identical env, checks every pad byte matches, and
renders its video; a mismatch stops the run.

usage: python scripts/recording/record_v2.py --policy MODEL.pt --starts CKPT_OR_NPY --out OUT [--levels all|1-1,4-4]
       [--device cuda] [--cap 6000] [--dropout] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from action_log import Mp4Writer, level_name, make_log, parse_levels, self_test, sha256_file, write_log  # noqa: E402

PROGRESS_UNIT = 3000.0   # scripts/reward10/games/smb.py: env.progress_now is in these units (pixels / 3000)
GAME = 'smb'


# ---------------------------------------------------------------- entrances
def archive_unit(env):
    """Bytes per level-start slot of each archive array, in the order rlenv saves them (rlenv.py _grow_archive)."""
    lay = env.level_layout(0)
    return [lay['stride'], lay['st'], lay['key_cap'] * 4, 8, 4, 4]


def load_entrances(starts: Path, cache_dir: Path):
    """-> (archive bytes as uint8 array, slots per archive, sha256 of the archive). A checkpoint is cut down to its
    level-start slots once and cached as .npy next to the outputs."""
    import numpy as np
    from rlenv import RlEnv
    if starts.suffix == '.npy':
        arc = np.load(starts)
    else:
        cached = cache_dir / f'entrances-{sha256_file(starts)[:16]}.npy'
        if cached.exists():
            arc = np.load(cached)
        else:
            import torch
            s = torch.load(starts, map_location='cpu', weights_only=False, mmap=True)
            sd, cfg = s['env'], s['cfg']
            assert sd['games'] == [GAME], f'{starts}: env games {sd["games"]}, expected [{GAME!r}]'
            cap, slots = int(sd['level_cap']), int(cfg['level_cap'])
            probe = RlEnv({GAME: 1}, device='cpu', sticky=False, idle_max=0, dropout=False, level_cap=slots)
            unit = archive_unit(probe)
            del probe
            buf = np.asarray(sd['env'], dtype=np.uint8)
            tail = buf[len(buf) - sum(u * cap for u in unit):]
            parts, o = [], 0
            for u in unit:   # each array holds `cap` entries; the first `slots` are the level starts, the rest places
                parts.append(tail[o:o + u * slots])
                o += u * cap
            arc = np.concatenate(parts)
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cached, arc)
            print(f'[record-v2] level entrances cut from {starts} -> {cached}', flush=True)
    probe = RlEnv({GAME: 1}, device='cpu', sticky=False, idle_max=0, dropout=False, level_cap=1)
    per_slot = sum(archive_unit(probe))
    del probe
    assert len(arc) % per_slot == 0, 'entrance archive does not match this environment build'
    slots = len(arc) // per_slot
    import hashlib
    return arc, slots, hashlib.sha256(arc.tobytes()).hexdigest()


def saved_levels(arc, slots):
    """Levels that have a saved entrance (deaths array, the last one, is -1 for an empty slot)."""
    import numpy as np
    deaths = np.frombuffer(arc[len(arc) - 4 * slots:].tobytes(), dtype=np.int32)
    return {i for i, d in enumerate(deaths) if d >= 0}


def make_env(level, n, seed, dropout, arc=None, slots=0):
    """n envs at the start of `level`: power-on for 1-1, else the saved entrance. Returns (env, warmup pad bytes per env).
    Entrance starts follow reports' evaluate_levels.entrance_env: load the archive into a fresh env, plan only this
    level, restart, then neutral decisions until the tracker reports the level."""
    import numpy as np
    from rlenv import RlEnv
    cap = max(slots, 1)
    env = RlEnv({GAME: n}, device='cpu', seed=seed, sticky=False, idle_max=0, dropout=dropout, level_cap=cap)
    env.start()
    warm = [bytearray() for _ in range(n)]
    if level == 0:
        return env, warm
    sd = env.state_dict()
    assert sum(u * slots for u in archive_unit(env)) == len(arc)
    sd['env'][-len(arc):] = arc
    env.load_state_dict(sd)
    w = np.zeros(cap)
    w[level] = 1
    env.set_level_plan(0, 1.0, w)
    env.start()                       # every env restarts from the entrance (+ 4 neutral frames)
    env.actions.zero_()
    for _ in range(300):
        env.step()
        for i in range(n):
            warm[i] += bytes(env.pads[i].tolist())
        if (env.level_now == level).all():
            break
    assert (env.level_now == level).all(), (level, env.level_now.tolist())
    return env, warm


# ---------------------------------------------------------------- play
def play_level(m, dev, level, args, arc, slots):
    """Pass 1: every mood plays; returns per-mood rows, moves (packed actions per decision) and pad bytes."""
    import numpy as np
    import torch
    from model import pack
    from rlenv import END_NAMES
    K = max(1, m.moods)
    env, warm = make_env(level, K, args.seed, args.dropout, arc, slots)
    h = torch.zeros(K, m.core, device=dev)
    prev = [torch.zeros(K, dtype=torch.long, device=dev) for _ in range(5)]
    mood = torch.arange(K, device=dev) if m.moods else None
    alive = np.ones(K, dtype=bool)
    rows = [dict(mode=i, cleared=False, end='cap', decisions=0, deaths=None, max_x_px=None, warmup_frames=len(warm[i]))
            for i in range(K)]
    pads = [bytearray(w) for w in warm]
    moves = []
    for _ in range(args.cap):
        z = m.encode(env.obs.to(dev), env.emph.to(dev), env.timeline.to(dev), prev, mood)
        h = m.gru(z.float(), h)
        prev, _ = m.act(h, env.masks.to(dev), greedy=True)
        a = pack(prev).cpu()
        moves.append(a.clone())
        env.actions.copy_(a)
        env.step()
        for i in np.flatnonzero(alive):
            r = rows[i]
            r['decisions'] += 1
            pads[i] += bytes(env.pads[i].tolist())
            p = float(env.progress_now[i])
            if np.isfinite(p):
                x = round(p * PROGRESS_UNIT)
                r['max_x_px'] = x if r['max_x_px'] is None else max(r['max_x_px'], x)
            won = bool(env.done[i]) and int(env.info[i, 2]) == 2
            advanced = float(env.parts[i, 1]) > 0 and int(env.level_now[i]) == level + 1
            if won or advanced:
                r.update(cleared=True, end='won' if won else 'level_clear')
                alive[i] = False
            elif env.done[i]:
                r.update(end=END_NAMES.get(int(env.info[i, 2]), '?'), deaths=int(env.info[i, 3]))
                alive[i] = False
        if not alive.any():
            break
    return rows, moves, pads


def best_mode(rows):
    """Greatest progress: a clear first, then max x, then the longest play (decisions), then the lowest mood."""
    return max(range(len(rows)), key=lambda i: (rows[i]['cleared'], rows[i]['max_x_px'] or -1, rows[i]['decisions'], -i))


def replay_video(level, k, rows, moves, pads, args, arc, slots, path):
    """Pass 2: the same env layout fed the recorded moves; checks the kept mood's pad bytes and writes its video."""
    import numpy as np
    from fceux_palette import RGB
    pal = np.array(RGB, np.uint8)
    n = len(rows)
    env, warm = make_env(level, n, args.seed, args.dropout, arc, slots)
    got = bytearray(warm[k])
    video = Mp4Writer(path, 240, 224, fps=60)
    for t in range(rows[k]['decisions']):
        env.actions.copy_(moves[t])
        env.step()
        got += bytes(env.pads[k].tolist())
        if t == rows[k]['decisions'] - 1 and env.done[k]:
            break                              # after an end the pictures are already the next episode's start
        o, e = env.obs[k].numpy().astype(np.int32), env.emph[k].numpy().astype(np.int32)
        for f in range(o.shape[0]):           # all 4 frames of the decision: real speed at 60 fps
            video.write(pal[o[f] | (e[f][:, None] << 6)])
    video.close()
    want = bytes(pads[k][len(warm[k]):]) if level else bytes(pads[k])
    have = bytes(got[len(warm[k]):]) if level else bytes(got)
    if have != want:
        raise RuntimeError(f'{level_name(level)} mood {k}: replay pads differ from the recorded play (non-deterministic env?)')


def record(args):
    import torch
    sys.path.insert(0, str(ROOT / 'rl'))
    from model import Policy
    from rlenv import boot_pads
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    s = torch.load(args.policy, map_location='cpu', weights_only=False)
    m = Policy(core=s['cfg']['core'], moods=s['cfg'].get('moods', 0))
    m.load_state_dict(s['model'])
    m = m.eval().to(dev)
    out = Path(args.out) / 'v2'
    out.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(args.levels)
    arc, slots, arc_sha = (None, 0, None)
    if any(lv > 0 for lv in levels):
        arc, slots, arc_sha = load_entrances(Path(args.starts), out / '.entrances')
        have = saved_levels(arc, slots)
        missing = [level_name(lv) for lv in levels if lv > 0 and lv not in have]
        if missing:
            raise SystemExit(f'no saved entrance for {missing} in {args.starts}')
    policy_sha = sha256_file(args.policy)
    boot = bytes(boot_pads(GAME)) + bytes(4)
    summary = []
    for lv in levels:
        name = level_name(lv)
        d = out / name
        log_path = d / f'smb_v2_{name}.actions.json'
        if log_path.exists() and not args.force:
            print(f'[record-v2] {name}: done already, skipped', flush=True)
            summary.append(json.loads(log_path.read_text())['result'] | dict(level=name))
            continue
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with torch.inference_mode():
            rows, moves, pads = play_level(m, dev, lv, args, arc, slots)
        k = best_mode(rows)
        replay_video(lv, k, rows, moves, pads, args, arc, slots, d / f'smb_v2_{name}.mp4')
        frames = (boot + bytes(pads[k])) if lv == 0 else bytes(pads[k])
        result = dict(rows[k], frames=len(frames))
        mode = rows[k]['mode'] if m.moods else 'greedy'
        log = make_log(level=lv, model=args.model_name, mode=mode, frames=frames,
                       start='power_on' if lv == 0 else 'level_entrance', clock='ppu_frame', result=result,
                       model_sha256=policy_sha, iteration=s.get('it'), moods=m.moods,
                       emulator='NeSLE CPU build of rl/ (bit-identical to the GPU training env)',
                       start_state=None if lv == 0 else dict(
                           source='training level-start archive (rl/csrc/rlenv_core.cuh)', archive_sha256=arc_sha,
                           slot=lv, reset_neutral_frames=4, warmup_frames=rows[k]['warmup_frames'],
                           note='frames begin after the entrance restore and its 4 neutral reset frames'),
                       selection=dict(rule='cleared, then max x (px), then decisions; ties -> lowest mood',
                                      dropout=bool(args.dropout), greedy=True, cap_decisions=args.cap, seed=args.seed),
                       video=f'smb_v2_{name}.mp4')
        write_log(log_path, log)
        (d / 'modes.json').write_text(json.dumps(dict(level=name, kept=k, modes=rows), indent=1))
        summary.append(dict(result, level=name, mode=mode))
        print(f'[record-v2] {name}: kept mood {k} ({rows[k]["end"]}, max x {rows[k]["max_x_px"]}, '
              f'{rows[k]["decisions"]} decisions) in {time.time() - t0:.0f}s', flush=True)
    (out / 'summary.json').write_text(json.dumps(dict(model=args.model_name, policy_sha256=policy_sha,
                                                      levels=summary), indent=1))


def dry_run(args):
    """No model, no emulator: check inputs and tools, print the plan, run the format self-test."""
    levels = parse_levels(args.levels)
    checks = {
        'policy': (args.policy, bool(args.policy) and Path(args.policy).exists()),
        'starts': (args.starts, bool(args.starts) and Path(args.starts).exists()),
        'ffmpeg': (shutil.which('ffmpeg'), shutil.which('ffmpeg') is not None),
        'rl/model.py': (str(ROOT / 'rl/model.py'), (ROOT / 'rl/model.py').exists()),
        'NES_ROM_DIR or RL_ROM_DIR': (os.environ.get('RL_ROM_DIR') or os.environ.get('NES_ROM_DIR'),
                                      bool(os.environ.get('RL_ROM_DIR') or os.environ.get('NES_ROM_DIR'))),
    }
    for k, (v, ok) in checks.items():
        print(f'[record-v2] {"ok  " if ok else "MISS"} {k}: {v}')
    print(f'[record-v2] plan: {len(levels)} levels x {args.moods} moods, cap {args.cap} decisions each, '
          f'keep 1 mood per level -> {len(levels)} MP4 + action logs under {Path(args.out) / "v2"}')
    print('[record-v2] self-test:', json.dumps(self_test(Path(args.out) / 'dry-run' / 'v2')))
    return all(ok for ok in (v[1] for v in checks.values()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--policy', default=os.environ.get('GHOSTRUN_V2_POLICY', ''),
                    help='weights: eval/itN/model.pt or a training checkpoint (env GHOSTRUN_V2_POLICY)')
    ap.add_argument('--starts', default=os.environ.get('GHOSTRUN_V2_STARTS', ''),
                    help='level entrances: training checkpoint .pt or cut .npy archive (env GHOSTRUN_V2_STARTS)')
    ap.add_argument('--out', default='recordings')
    ap.add_argument('--levels', default='all')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--cap', type=int, default=6000, help='decisions per level and mood (6000 = 400 s of game time)')
    ap.add_argument('--seed', type=int, default=654321)
    ap.add_argument('--dropout', action='store_true', help='button dropout on (training condition); default off')
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--moods', type=int, default=8, help='dry-run plan only; a real run reads it from the policy')
    ap.add_argument('--model-name', default='ghostrun-smb-policy-v2')
    ap.add_argument('--force', action='store_true', help='redo levels that already have a log')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.dry_run:
        dry_run(a)
        return
    if not a.policy:
        raise SystemExit('--policy is required')
    if not a.starts and parse_levels(a.levels) != [0]:
        raise SystemExit('--starts is required for levels after 1-1')
    record(a)


if __name__ == '__main__':
    main()
