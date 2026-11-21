"""Python twin of rl/csrc/reward10.cuh observe_table: a reward10 game module built from one rl/games_table.json entry
(the table spec of a ROM without a hand-written reader). The env's GPU/CPU reward for that ROM must equal
reward.RewardTracker(table.game(entry)) on the same RAM (rl/tests/test_table_games.py).
"""
from __future__ import annotations

import types

from reward import Signals

SCORE = {0: 'digits_hi', 1: 'digits_lo', 2: 'bin_lo', 3: 'bcd', 4: 'bcd_lo'}


def score_value(ram, t):
    a, n, kind = t['score_addr'], t['score_n'], t['score_kind']
    order = range(a, a + n) if kind in (0, 3) else range(a + n - 1, a - 1, -1)   # most significant byte first
    v = 0
    for i in order:
        x = int(ram[i])
        v = v * 10 + x if kind in (0, 1) else v * 256 + x if kind == 2 else v * 100 + (x >> 4) * 10 + (x & 15)
    return v


def game(entry: dict):
    t = entry['table']

    class Reader:
        def reset(self):
            self.cells = set()

        def observe(self, ram):
            in_play = t['in_play_addr'] < 0 or int(ram[t['in_play_addr']]) in t['in_play_values'][:8]
            level = int(ram[t['level_addr']]) if t['level_addr'] >= 0 else 0
            progress, valid = 0.0, False
            if t['x_addr'] >= 0:
                if in_play:
                    scene = int(ram[t['scene_addr']]) if t['scene_addr'] >= 0 else 0
                    y = int(ram[t['y_addr']]) >> t['cell_shift'] if t['y_addr'] >= 0 else 0
                    self.cells.add((level, scene, int(ram[t['x_addr']]) >> t['cell_shift'], y))
                progress, valid = float(len(self.cells)), in_play
            lives = None
            if t['lives_addr'] >= 0:
                lv = int(ram[t['lives_addr']])
                lives = lv - 256 if lv > 127 else lv
            return Signals(in_play=in_play, level=level, progress=progress, progress_valid=valid, lives=lives,
                           score=score_value(ram, t) if t['score_addr'] >= 0 else None,
                           game_over=t['game_over_addr'] >= 0 and int(ram[t['game_over_addr']]) == t['game_over_value'])

    stall = t['stall_frames'] / 60.0 if t['stall_frames'] >= 0 else None
    return types.SimpleNamespace(Reader=Reader, PROGRESS_UNIT=t['progress_unit'], SCORE_UNIT=t['score_unit'],
                                 STALL_SECONDS=stall, DEATH_FROM='lives' if t['lives_addr'] >= 0 else 'dying',
                                 START_SELECT={})
