#!/usr/bin/env python3
"""Live terminal view of a training run: how each level of each game is doing. Reads only metrics.jsonl, the pid file
and eval/*/summary.json, so it never touches the GPU or the run, and works for any game.

usage: python3 rl/watch.py [RUN_DIR] [--every SECONDS] [--once]     (default: the most recently updated run; Ctrl-C quits)
"""
import argparse
import glob
import json
import os
import shutil
import time

SPARK = '▁▂▃▄▅▆▇█'
GREEN, YELLOW, RED, DIM, BOLD, END = '\033[32m', '\033[33m', '\033[31m', '\033[2m', '\033[1m', '\033[0m'
TAIL_BYTES = 24_000_000   # enough rows for the trend, without re-reading a huge file every refresh
MIN_VISITS = 1.0          # below this a level has no real data yet
TREND_SECONDS = 900


def latest_run():
    runs = glob.glob('runs/rl/*/metrics.jsonl')
    if not runs:
        raise SystemExit('no runs/rl/*/metrics.jsonl here; pass RUN_DIR or run from the project root')
    return os.path.dirname(max(runs, key=os.path.getmtime))


def read_rows(path):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - TAIL_BYTES))
        lines = f.read().split(b'\n')[1:-1]
    rows = []
    for line in lines:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def level_name(game, level):
    return f'{level // 4 + 1}-{level % 4 + 1}' if game == 'smb' else f'L{level}'


def colour(rate):
    return GREEN if rate >= 0.7 else YELLOW if rate >= 0.3 else RED


def history(rows, game, level, width):
    """The level's clear rate over the shown rounds, one value per character (None = no data then)."""
    step = max(1, len(rows) // width)
    out = []
    for r in rows[::step][-width:]:
        d = r.get('level_plan', {}).get(game, {}).get('levels', {}).get(level)
        out.append(d['clear'] if d and d['visits'] >= MIN_VISITS else None)
    return out


def spark(values):
    return ''.join(' ' if v is None else SPARK[min(7, int(v * 8))] for v in values)


def cell(rows, game, level, d, spark_w):
    name = level_name(game, int(level))
    if d['visits'] < MIN_VISITS:
        return f'{name:>5} {DIM}{"no data yet":<{10 + 5 + 1 + spark_w + 5}}{END}'
    rate = d['clear']
    full = int(round(rate * 10))
    then = next(r for r in rows if r['time'] >= rows[-1]['time'] - TREND_SECONDS)
    old = then.get('level_plan', {}).get(game, {}).get('levels', {}).get(level)
    trend = f'{(rate - old["clear"]) * 100:+4.0f}' if old and old['visits'] >= MIN_VISITS else '    '
    c = colour(rate)
    return (f'{name:>5} {c}{"█" * full}{DIM}{"░" * (10 - full)}{END} {c}{rate * 100:3.0f}%{END} '
            f'{spark(history(rows, game, level, spark_w))} {trend}')


def header(run, rows):
    r = rows[-1]
    pid_file = os.path.join(run, 'pid')
    alive = os.path.exists(pid_file) and os.path.isdir(f'/proc/{open(pid_file).read().strip()}')
    stopped = os.path.exists(os.path.join(run, 'STOP'))
    age = time.time() - r['time']
    state = f'{GREEN}running{END}' if alive and not stopped else f'{RED}stopped{END}'
    return [f'{BOLD}{os.path.basename(run)}{END}  {state}  last update {age:.0f}s ago  {time.strftime("%H:%M:%S")}',
            f'round {r["it"]}   moves {r["decisions"] / 1e6:.1f}M   speed {r["sps"]:.0f}/s   '
            f'GPU use (MFU) {r.get("mfu", 0) * 100:.0f}%   power {r.get("power_w", 0):.0f}W',
            f'randomness (entropy) {r["ent"]:.2f}   reward prediction {r["explained_var"]:.2f}   '
            f'policy change (KL) {r["kl"]:.3f}']


def tests(run, n=4):
    out = []
    for path in sorted(glob.glob(os.path.join(run, 'eval/it*/summary.json')))[-n:]:
        try:
            games = json.load(open(path))
        except ValueError:
            continue
        name = os.path.basename(os.path.dirname(path)).replace('it00', 'round ')
        reached = ' '.join(level_name(g['game'], g['level']) for g in games)
        out.append(f'  {name}: {reached}  {DIM}(won {sum(g["end"] == "won" for g in games)}/{len(games)}){END}')
    return out


def render(run):
    rows = read_rows(os.path.join(run, 'metrics.jsonl'))
    if not rows:
        return f'{run}: no rows yet'
    width = shutil.get_terminal_size((120, 40)).columns
    spark_w = 16
    cell_w = 5 + 1 + 10 + 1 + 4 + 1 + spark_w + 5 + 4
    cols = max(1, width // cell_w)
    lines = header(run, rows)
    for game, plan in rows[-1].get('level_plan', {}).items():
        levels = sorted(plan['levels'], key=int)
        seen = sum(plan['levels'][l]['visits'] >= MIN_VISITS for l in levels)
        good = sum(plan['levels'][l]['visits'] >= MIN_VISITS and plan['levels'][l]['clear'] >= 0.7 for l in levels)
        lines += ['', f'{BOLD}{game}{END}: {good} of {len(levels)} levels cleared 70%+ of the time, '
                      f'{seen} with data   {DIM}(bar = clear rate, then its history, then change over '
                      f'{TREND_SECONDS // 60} minutes){END}']
        height = -(-len(levels) // cols)
        for i in range(height):
            lines.append('    '.join(cell(rows, game, l, plan['levels'][l], spark_w) for l in levels[i::height]))
    shown = tests(run)
    if shown:
        lines += ['', f'{BOLD}best-move test games, furthest level each{END}'] + shown
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run', nargs='?')
    p.add_argument('--every', type=float, default=5.0)
    p.add_argument('--once', action='store_true')
    a = p.parse_args()
    run = (a.run or latest_run()).rstrip('/')
    if a.once:
        print(render(run))
        return
    print('\033[?1049h\033[?25l', end='')   # own screen, hide cursor
    try:
        while True:
            print('\033[H\033[J' + render(run), end='', flush=True)
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass
    finally:
        print('\033[?25h\033[?1049l', end='')


if __name__ == '__main__':
    main()
