"""The reward10 game contract: one shape for every game, whatever the genre.

A game module (games/<slug>.py) provides:
  emulator      ROM, ROM_SHA256, boot(emu), ACTIONS (search/audit only, never the trainer's action space)
  RAM->signals  class Reader: reset(); observe(ram: 2048 bytes) -> reward.Signals
  reward scale  PROGRESS_UNIT (= one level of progress), SCORE_UNIT (= one typical scoring event),
                DEATH_FROM ('dying' | 'lives'), optional STALL_SECONDS (None = no limit)
  description   INFO: what each signal means in this game (keys below); reward_spec.json is built from it
  controller    optional START_SELECT {'START' | 'SELECT': what it does in play}: the only Start/Select presses a
                trainer passes on after boot (default: none, because Start pauses most games and pausing right
                before a death dodges the -1; measured in SMB, Gradius, Castlevania, Final Mission)
  search only   optional SEARCH_BUCKETS, SEARCH_ACTIONS, SEARCH_ALLOW_DEATHS, search_key(signals)

Trainer API (identical for every game):
  g = contract.load('smb')                                  # import + check the module
  g.boot(emu)                                               # emu: step(buttons, frames), tap(button, hold, after), ram()
  pad = contract.play_buttons(g)                            # after boot, drop every pressed button not in pad
  tr = RewardTracker(g)                                     # one per episode; tr.reset() to reuse
  reward, done, events = tr.update(emu.ram(), t, frames=8)  # after every decision
  tr.end_reason                                             # None, or 'game_over' | 'won' | 'skip' | 'stall'
All four end reasons are terminal (do not bootstrap past them); a rollout cut by the trainer is a truncation.
RAM, Signals and events are reward/debug data only, never actor inputs.

usage: python contract.py            check every module and rewrite reward_spec.json
       python contract.py --check    check only (exit 1 on a problem or a stale reward_spec.json)
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reward as R  # noqa: E402

SPEC_PATH = HERE / 'reward_spec.json'
BUTTONS = {'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START', 'SELECT'}
PAD = frozenset({'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B'})   # always usable in play; Start/Select only via START_SELECT
WIN_STATUS = {
    'verified': 'seen in a genuine run in this emulator',
    'provisional': 'detector seen only with help (RAM poke or another emulator)',
    'unverified': 'detector written but never observed',
    'none': 'no detector: the episode ends only on game over or the stall limit',
}
# INFO keys: (type, meaning)
INFO_KEYS = {
    'title': (str, 'game title'),
    'genre': (str, 'genre'),
    'level': (str, 'what one level is (Signals.level)'),
    'levels': ((int, type(None)), 'level values in one full playthrough; None = endless'),
    'progress': (str, 'what within-level progress measures; PROGRESS_UNIT of it = one level'),
    'death': (str, 'what counts as a death'),
    'game_over': (str, 'what ends the game'),
    'health': ((str, type(None)), 'what Signals.health is; None = no health bar'),
    'score': ((str, type(None)), 'what Signals.score is; None = no score'),
    'win': (str, 'what counts as finishing the game'),
    'win_status': (str, 'one of WIN_STATUS'),
    'limits': (list, 'known reward limits a trainer should know about (may be empty)'),
}


def play_buttons(g):
    """Buttons a trainer may press after boot: the d-pad, A, B and the Start/Select the game lists."""
    return PAD | frozenset(getattr(g, 'START_SELECT', {}))


def games():
    return sorted(p.stem for p in (HERE / 'games').glob('*.py') if p.stem != '__init__')


def check(g):
    """Return a list of contract problems of a game module (empty = it follows the contract)."""
    bad = []
    for name in ('ROM', 'ROM_SHA256', 'ACTIONS', 'boot', 'Reader', 'PROGRESS_UNIT', 'SCORE_UNIT', 'DEATH_FROM',
                 'INFO'):
        if not hasattr(g, name):
            bad.append(f'missing {name}')
    if bad:
        return bad
    if not re.fullmatch(r'[0-9a-f]{64}', g.ROM_SHA256):
        bad.append('ROM_SHA256 is not a sha256 hex digest')
    if not callable(g.boot) or not all(callable(getattr(g.Reader, m, None)) for m in ('reset', 'observe')):
        bad.append('boot(emu) and Reader.reset()/observe(ram) must be callable')
    if not (g.PROGRESS_UNIT > 0 and g.SCORE_UNIT > 0):
        bad.append('PROGRESS_UNIT and SCORE_UNIT must be > 0')
    if g.DEATH_FROM not in ('dying', 'lives'):
        bad.append("DEATH_FROM must be 'dying' or 'lives'")
    stall = getattr(g, 'STALL_SECONDS', R.STALL_SECONDS)
    if stall is not None and not stall > 0:
        bad.append('STALL_SECONDS must be > 0 or None')
    ss = getattr(g, 'START_SELECT', {})
    if not (isinstance(ss, dict) and set(ss) <= {'START', 'SELECT'}
            and all(isinstance(v, str) and v for v in ss.values())):
        bad.append("START_SELECT must map 'START'/'SELECT' to what the button does")
    for a in g.ACTIONS:
        steps = a if isinstance(a, list) else [(a, 1)]
        if not all(isinstance(b, tuple) and set(b) <= BUTTONS and f > 0 for b, f in steps):
            bad.append(f'bad action {a!r}')
            break
    info = g.INFO
    if set(info) != set(INFO_KEYS):
        bad.append(f'INFO keys differ: missing {sorted(set(INFO_KEYS) - set(info))}, '
                   f'extra {sorted(set(info) - set(INFO_KEYS))}')
        return bad
    for k, (typ, _) in INFO_KEYS.items():
        if not isinstance(info[k], typ):
            bad.append(f'INFO[{k!r}] has type {type(info[k]).__name__}')
    if info['win_status'] not in WIN_STATUS:
        bad.append(f"INFO['win_status'] must be one of {sorted(WIN_STATUS)}")
    return bad


def load(slug):
    g = importlib.import_module(f'games.{slug}')
    bad = check(g)
    if bad:
        raise ValueError(f'{slug} breaks the reward10 contract: {bad}')
    return g


def spec(slug, g=None):
    """The reward a training module gets for one game, as plain JSON."""
    g = g or load(slug)
    i, stall = g.INFO, getattr(g, 'STALL_SECONDS', R.STALL_SECONDS)
    ss = getattr(g, 'START_SELECT', {})
    return {
        'game': slug, 'title': i['title'], 'genre': i['genre'],
        'rom': {'file': Path(g.ROM).name, 'sha256': g.ROM_SHA256},
        'start': 'fresh standard 1-player game, first controllable frame (boot)',
        'controller': {'after_boot': sorted(play_buttons(g)),
                       'start': ss.get('START', 'blocked after boot'),
                       'select': ss.get('SELECT', 'blocked after boot')},
        'levels': {'meaning': i['level'], 'count': i['levels']},
        'rewards': {
            'progress': {'value': R.W_PROGRESS, 'per': f'{g.PROGRESS_UNIT:g} units = one level',
                         'units': i['progress'], 'rule': 'new high-water progress in the current level only'},
            'next_level': {'value': R.W_LEVEL,
                           'rule': 'reaching the next level in order; a skip ends the episode with no bonus'},
            'death': {'value': R.W_DEATH, 'when': i['death']},
            'damage': None if i['health'] is None else {'value': R.W_DAMAGE, 'per': 'full health bar lost',
                                                        'health': i['health']},
            'score': None if i['score'] is None else {
                'value': R.W_SCORE, 'per': f'{g.SCORE_UNIT:g} points', 'score': i['score'],
                'cap_per_step': R.SCORE_CAP, 'cap_per_level': R.SCORE_LEVEL_BUDGET},
            'win': {'value': R.W_WIN, 'when': i['win'], 'status': i['win_status']},
            'waiting': None if stall is None else {
                'value': R.W_IDLE * getattr(g, 'IDLE_SCALE', 1.0), 'per': f'{stall:g} s in play without moving forward'},
            'rollback': {'value': R.W_ROLLBACK, 'drop_fraction': R.ROLLBACK_DROP,
                         'rule': 'one abrupt progress drop in valid same-level play, excluding death/transition/win'},
        },
        'episode_ends': {
            'game_over': i['game_over'],
            'won': i['win'] if i['win_status'] != 'none' else None,
            'skip': 'a level is skipped (no bonus)',
            'stall': None if stall is None else
            f'{stall:g} s of game time without new progress, a new level or a death (penalty {R.W_STALL:g}, same as a death)',
        },
        'no_reward_when': 'not in play (title, menus, demo, pause, transitions)',
        'limits': i['limits'],
    }


def all_specs():
    return {
        'contract': 'scripts/reward10/contract.py (module shape, trainer API); generated, do not edit by hand',
        'shared': {'progress': R.W_PROGRESS, 'next_level': R.W_LEVEL, 'death': R.W_DEATH, 'stall': R.W_STALL, 'waiting_per_stall_limit': R.W_IDLE, 'damage': R.W_DAMAGE,
                   'rollback': R.W_ROLLBACK, 'rollback_drop_fraction': R.ROLLBACK_DROP,
                   'score': R.W_SCORE, 'score_cap_per_step': R.SCORE_CAP,
                   'score_cap_per_level': R.SCORE_LEVEL_BUDGET, 'win': R.W_WIN,
                   'stall_seconds_default': R.STALL_SECONDS,
                   'start_select': 'blocked after boot unless the game lists them '
                                   '(pausing before a death dodges the death penalty)'},
        'win_status': WIN_STATUS,
        'games': [spec(s) for s in games()],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    problems = {s: check(importlib.import_module(f'games.{s}')) for s in games()}
    problems = {s: p for s, p in problems.items() if p}
    for s, p in problems.items():
        print(s, p)
    if problems:
        sys.exit(1)
    text = json.dumps(all_specs(), indent=1, ensure_ascii=False) + '\n'
    if a.check:
        if not SPEC_PATH.exists() or SPEC_PATH.read_text() != text:
            print('reward_spec.json is stale: run python contract.py')
            sys.exit(1)
    else:
        SPEC_PATH.write_text(text)
    print(f'{len(games())} games follow the contract' + ('' if a.check else f'; wrote {SPEC_PATH.name}'))


if __name__ == '__main__':
    main()
