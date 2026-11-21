"""Export every ROM that is ready to train as a table game (rl/csrc/reward10.cuh TableSpec) into rl/games_table.json.

A ROM is ready when:
  * NeSLE runs it (reports/romlib/status.csv gpu in pass / matches_ref / natural; `suspect` too with --suspect;
    reports/romlib/gpu_overrides.json re-classifies suspects: pass_ntsc = matches fceumm at NTSC speed, input_drift =
    NeSLE, fceumm and FCEUX agree at power-on and drift apart only under button presses, seed_drift = the play differs
    only through a power-on random seed),
  * the readers' boot (reports/romlib/readers.jsonl boot_pads) does not stop on an unfixed menu, and
  * on NeSLE itself the player has control after the boot (reports/romlib/train_check.jsonl control_after); the boot
    is extended by control_after idle frames, so every episode starts in control,
  * a position byte is known (the reward's cells come from it): the readers' position, else the byte the game's
    button moves in stage 1 (A / B games).

Signals (a wrong one can end episodes early, so only trusted ones are used; the rest stay -1 = none):
  lives, score, game_over      high or medium confidence
  in_play                      high or medium confidence, <= 8 values (else: always in play)
  level                        medium only (the respawn test; the boot test also accepts stage-type bytes)
Progress: distinct (level, x >> 3, y >> 3) cells per episode (x, y = the readers' position bytes); PROGRESS_CELLS new
cells pay like one level of progress. Weights, idle cost and stall penalty are reward10's (scripts/reward10/reward.py).
Episode end: game over, or STALL_SECONDS without a new cell, level or death.
Start-state check (NeSLE CPU harness, NESLE_HOST_LIB): at the exported start the player has control, so a game-over
condition already true there, or an in-play condition false there, is wrong and dropped (entry 'dropped').
Sweep check (rl/scripts/table_sweep.py, 1,500 random decisions per game on an earlier export): a game over that fired 5+
times and more often than deaths ends episodes that did not end (Ghosts'n Goblins: 10, no death); dropped.

Camera (entry 'camera', scripts/romlib/camera.py -> reports/romlib/camera.jsonl; read by rl/table_keys.py for the
search, not by the reward): x / y = {lo, page} RAM bytes of a camera that agrees with the picture >= 90% of the time
(page -1: none found), player_x / player_y = whether the table's x / y byte is a level ('world') or a screen coordinate.
--camera-only adds it to the existing rl/games_table.json without exporting again.

The 12 hand-read reward10 games are not exported (they keep their own readers); their ROMs are skipped.

usage: python rl/scripts/export_games.py [--suspect] [--out rl/games_table.json] [--camera-only]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'reports/romlib'
ROM_DIR = Path('/home/kd/Desktop/roms')
STALL_SECONDS = 30
PROGRESS_CELLS = 100.0    # new 8x8 cells worth one level of progress (W_PROGRESS); SMB 1-1 is ~400 x cells
sys.path.insert(0, str(ROOT / 'scripts/reward10'))
import reward as R10  # noqa: E402
SCORE_KINDS = {'digits_hi': 0, 'digits_lo': 1, 'bin_lo': 2, 'bcd': 3, 'bcd_lo': 4}
TRUSTED = ('high', 'medium')


def addr(v):
    return int(v[1:], 16)


def field(r, k, ok=TRUSTED):
    v = r.get(k)
    return v if isinstance(v, dict) and v.get('confidence') in ok else None


def slug(name, used):
    s = re.sub(r'[^a-z0-9]+', '_', name[:-4].lower()).strip('_')[:60] or 'rom'
    base, i = s, 2
    while s in used:
        s, i = f'{base}_{i}', i + 1
    used.add(s)
    return s


def spec(r, st1):
    pos = r['position'] if isinstance(r.get('position'), dict) else {}
    if not pos:            # A / B games: the byte the button moves in stage 1 stands in for the position
        c = st1.get('a_candidates') or st1.get('x_candidates') or st1.get('y_candidates')
        pos = {'x': c[0][0]}
    t = dict(in_play_addr=-1, in_play_values=[], level_addr=-1, lives_addr=-1, game_over_addr=-1, game_over_value=0,
             score_addr=-1, score_n=0, score_kind=0, x_addr=addr(pos['x']) if pos.get('x') else addr(pos['y']),
             y_addr=addr(pos['y']) if pos.get('x') and pos.get('y') else -1, scene_addr=-1, cell_shift=3,
             progress_unit=PROGRESS_CELLS, score_unit=100.0, stall_frames=STALL_SECONDS * R10.NES_FPS,
             idle_per_frame=R10.idle_per_frame(types.SimpleNamespace(STALL_SECONDS=STALL_SECONDS)))
    if (f := field(r, 'in_play')) and len(f['values']) <= 8:
        t['in_play_addr'], t['in_play_values'] = addr(f['addr']), [int(v) for v in f['values']]
    if f := field(r, 'level', ('medium',)):
        t['level_addr'] = addr(f['addr'])
    if f := field(r, 'lives'):
        t['lives_addr'] = addr(f['addr'])
    if f := field(r, 'game_over'):
        t['game_over_addr'], t['game_over_value'] = addr(f['addr']), int(f['value'])
    if f := field(r, 'score'):
        t['score_addr'], t['score_n'], t['score_kind'] = addr(f['addr']), int(f['n']), SCORE_KINDS[f['kind']]
    return t


def boot_ram(args):
    """RAM at the start of an episode (the exported boot played on NeSLE's CPU harness, NESLE_HOST_LIB), and at a few
    later moments of idle play."""
    name, boot = args
    sys.path.insert(0, str(ROOT / 'scripts/nesle_check'))
    import neslenes
    n = neslenes.NesleNes(str(ROM_DIR / name))
    for b in boot:
        neslenes.lib().nh_step(n.h, b, 1)
    out = [n.ram().copy()]
    for _ in range(3):
        neslenes.lib().nh_step(n.h, 0, 60)
        out.append(n.ram().copy())
    return name, out


def boot_fixes(games):
    """Drop signals that contradict the verified start (the player has control there): a game over already true, or
    in play false. Both would end or freeze every episode at once."""
    import multiprocessing as mp
    by_rom = {g['rom']: g for g in games.values()}
    fixes = {}
    with mp.get_context('spawn').Pool(24) as pool:
        for name, rams in pool.imap_unordered(boot_ram, [(g['rom'], bytes.fromhex(g['boot_pads'])) for g in games.values()]):
            t = by_rom[name]['table']
            if t['game_over_addr'] >= 0 and any(int(r[t['game_over_addr']]) == t['game_over_value'] for r in rams[:1]):
                t['game_over_addr'] = -1
                fixes[name] = fixes.get(name, []) + ['game_over true at start']
            if t['in_play_addr'] >= 0 and int(rams[0][t['in_play_addr']]) not in t['in_play_values']:
                t['in_play_addr'], t['in_play_values'] = -1, []
                fixes[name] = fixes.get(name, []) + ['in_play false at start']
    return fixes


def cameras():
    """rom -> the table's camera entry (module notes), from reports/romlib/camera.jsonl."""
    out = {}
    p = R / 'camera.jsonl'
    if not p.exists():
        return out
    for line in open(p):
        r = json.loads(line)
        if 'error' in r:
            continue
        axis = lambda a: ({'lo': a['lo'], 'page': -1 if a['page'] is None else a['page']}    # noqa: E731
                          if a and a.get('ok') else None)
        kind = lambda k: k['kind'] if k else None                                          # noqa: E731
        out[r['rom']] = dict(x=axis(r['x']), y=axis(r['y']), player_x=kind(r.get('player_x')),
                             player_y=kind(r.get('player_y')))
    return out


def add_cameras(games):
    cams = cameras()
    for g in games.values():
        if g['rom'] in cams:
            g['camera'] = cams[g['rom']]
    return sum('camera' in g and (g['camera']['x'] or g['camera']['y']) is not None for g in games.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera-only', action='store_true', help='add the camera entries to the existing --out file')
    ap.add_argument('--suspect', action='store_true', help='also ROMs whose RAM diverges from fceumm for unknown reasons')
    ap.add_argument('--out', default=str(ROOT / 'rl/games_table.json'))
    ap.add_argument('--no-boot-check', action='store_true', help='skip the NeSLE start-state check (NESLE_HOST_LIB)')
    ap.add_argument('--sweep', default=str(R / 'table_sweep.jsonl'), help='random-play sweep of an earlier export')
    a = ap.parse_args()
    if a.camera_only:
        games = json.loads(Path(a.out).read_text())
        n = add_cameras(games)
        Path(a.out).write_text(json.dumps(games, indent=1) + '\n')
        print(f'camera added: {n} of {len(games)} games have one -> {a.out}')
        return
    gpu = {r['rom']: r['gpu'] for r in csv.DictReader(open(R / 'status.csv'))}
    if (R / 'gpu_overrides.json').exists():     # suspect ROMs explained later (PAL speed in fceumm; input drift only)
        gpu.update({k: v['gpu'] for k, v in json.loads((R / 'gpu_overrides.json').read_text()).items()})
    tc = {json.loads(l)['rom']: json.loads(l) for l in open(R / 'train_check.jsonl')}
    hand = {v['rom'] for v in json.loads((ROOT / 'rl/games_manifest.json').read_text()).values()}
    runs = {'pass', 'matches_ref', 'natural', 'pass_ntsc', 'input_drift', 'seed_drift'} | ({'suspect'} if a.suspect else set())
    import sys
    sys.path.insert(0, str(ROOT / 'scripts/romlib'))
    from train_check import stage1_rows
    st1 = stage1_rows()
    games, used, why = {}, set(), {}
    for line in open(R / 'readers.jsonl'):
        r = json.loads(line)
        name = r['rom']
        reason = ('hand-read reward10 game' if name in hand else
                  f"gpu {gpu.get(name)}" if gpu.get(name) not in runs else
                  'boot stops on a menu' if (r.get('boot_menu') or {}).get('status') == 'unfixed' else
                  'no control on NeSLE after the boot' if (tc.get(name) or {}).get('control_after') is None else
                  'no position byte' if not isinstance(r.get('position'), dict) and not (
                      st1.get(name, {}).get('a_candidates') or st1.get(name, {}).get('x_candidates') or
                      st1.get(name, {}).get('y_candidates')) else None)
        if reason:
            why[reason] = why.get(reason, 0) + 1
            continue
        data = (ROM_DIR / name).read_bytes()
        # train_check's boot_pads, when present: the readers' boot shifted a frame for NeSLE's power-on timing
        boot = bytes.fromhex(tc[name].get('boot_pads') or r['boot_pads']) + bytes(int(tc[name]['control_after']))
        games[slug(name, used)] = dict(rom=name, sha256=hashlib.sha256(data).hexdigest(), boot_pads=boot.hex(),
                                       gpu=gpu[name], table=spec(r, st1.get(name, {})))
    fixes = boot_fixes(games) if not a.no_boot_check else {}
    if Path(a.sweep).exists():          # rl/scripts/table_sweep.py on an earlier export: game overs without deaths
        for line in open(a.sweep):
            s = json.loads(line)
            g = games.get(s['game'])
            if g and g['table']['game_over_addr'] >= 0 and s.get('game_overs', 0) >= 5 and s['game_overs'] > s['deaths']:
                g['table']['game_over_addr'] = -1
                fixes[g['rom']] = fixes.get(g['rom'], []) + ['game over 5+ times in random play, more than deaths']
    for name, f in fixes.items():
        next(g for g in games.values() if g['rom'] == name)['dropped'] = f
    add_cameras(games)
    Path(a.out).write_text(json.dumps(games, indent=1) + '\n')
    n = len(games)
    has = {k: sum(g['table'][k] >= 0 for g in games.values()) for k in ('lives_addr', 'game_over_addr', 'score_addr',
                                                                         'level_addr', 'in_play_addr', 'y_addr')}
    print(f'{n} games -> {a.out}')
    print('with:', {k[:-5]: f'{v} ({v / max(n, 1):.0%})' for k, v in has.items()})
    print('not ready:', dict(sorted(why.items(), key=lambda x: -x[1])))
    c = {}
    for f in fixes.values():
        for x in f:
            c[x] = c.get(x, 0) + 1
    print('dropped at the start-state check:', c)


if __name__ == '__main__':
    main()
