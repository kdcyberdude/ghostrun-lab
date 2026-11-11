"""Shared reward design for the reward10 games; per-game modules only read RAM.

A game module (scripts/reward10/games/<slug>.py) defines:
  ROM            absolute ROM path;  ROM_SHA256  pinned identity
  ACTIONS        list of button tuples used by search/audit (not by the actor)
  boot(nes)      scripted fresh start from power-on to the first controllable frame
  class Reader:  reset(); observe(ram) -> Signals   (may keep per-episode state)
  PROGRESS_UNIT  progress units that count as one full level of progress
  SCORE_UNIT     score points that count as one typical scoring event
  DEATH_FROM     'dying' (rising edge of Signals.dying) or 'lives' (lives decrement)

What is optimized (identical for every game, weights below):
  + new within-level progress (high-water mark, so oscillating cannot farm it)
  + reaching the next level/stage/opponent/track (only +1 steps; a skip such as a warp zone
    ends the episode with no bonus, so every level must be played in order)
  - each death (lives lost or a new death sequence), - damage taken
  + small score tiebreaker, capped per step and per level (so farming respawning enemies or
    end-of-level tallies cannot outweigh progress)
  + large bonus for finishing the game; episode ends on game over or win.
  The episode also ends, with penalty W_STALL (same as a death), after STALL_SECONDS of game time (in play or not, so pausing
  or idling on a menu counts) without new progress, a new level or a death (deaths restart the clock
  because respawning behind the high-water mark is normal). A module may set its own STALL_SECONDS
  (None = no limit) when normal play has longer quiet stretches.
  Waiting also costs a little at every update: W_IDLE spread over the game's stall limit, charged while in play with
  valid progress that did not move forward since the last update. Walking forward behind the high-water mark after a
  respawn is free, so trying and dying stays better than standing still. Without this, standing cost nothing and the
  stall penalty, 120 s away, vanished under the discount. A module may set IDLE_SCALE when progress is naturally slow.
RAM is reward/audit data only and must never be an actor input.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

W_PROGRESS = 1.0      # per full level of new progress (PROGRESS_UNIT)
W_LEVEL = 1.0         # per newly reached level
W_DEATH = -0.1        # per death (2026-11-10: was -1.0, which made standing still beat moving on; see DECISIONS.md)
W_DAMAGE = -0.25      # per full health bar lost
W_SCORE = 0.02        # per SCORE_UNIT of score, capped per step and per level
SCORE_CAP = 0.1       # max score reward per step
SCORE_LEVEL_BUDGET = 0.2  # max score reward per level (resets only when a new level is reached, not on death)
W_WIN = 5.0           # finishing the game
W_STALL = -0.1        # the no-progress end (2026-11-10: was free, so idling until it was the safest play)
W_IDLE = -1.8         # waiting for a whole stall limit (2026-11-11; SMB: -0.001 per decision, ~-0.2 discounted > a death)
W_ROLLBACK = -0.02    # one abrupt loss of progress; ordinary small backward movements are allowed
ROLLBACK_DROP = 0.25  # fraction of PROGRESS_UNIT lost in one valid, same-level update
STALL_SECONDS = 120.0  # default no-progress limit; longest quiet stretch on the saved runs was 51 s (73 s Tecmo Bowl)
NES_FPS = 60.0
DECISION_FRAMES = 8   # default emulator frames per update() call


@dataclass
class Signals:
    in_play: bool               # real gameplay (not title/menu/demo/pause/transition)
    level: int                  # ordinal of level/stage/opponent/track, increases with true progress
    progress: float             # within-level progress in game units (PROGRESS_UNIT = 1 level)
    progress_valid: bool = True # False while progress is stale (transitions, pipes, cutscenes)
    dying: bool = False         # player in a death/loss sequence
    lives: int | None = None    # remaining lives counter if the game has one
    health: float | None = None # 0..1 fraction of the health bar if the game has one
    score: int | None = None
    game_over: bool = False
    won: bool = False
    extra: dict | None = None   # game-specific debug values (not used by the reward)


def idle_per_frame(game):
    """Cost per frame of waiting (0 without a stall limit)."""
    stall = getattr(game, 'STALL_SECONDS', STALL_SECONDS)
    return 0.0 if stall is None else W_IDLE * getattr(game, 'IDLE_SCALE', 1.0) / (stall * NES_FPS)


class RewardTracker:
    def __init__(self, game):
        self.game = game
        self.reader = game.Reader()
        stall = getattr(game, 'STALL_SECONDS', STALL_SECONDS)
        self.stall_frames = None if stall is None else stall * NES_FPS
        self.idle_per_frame = idle_per_frame(game)
        self.reset()

    def reset(self):
        self.reader.reset()
        self.prev = None
        self.best_level = None
        self.best_progress = 0.0
        self.deaths = 0
        self.idle_frames = 0    # game frames since new progress, a new level or a death
        self.score_used = 0.0   # score reward paid in the current level
        self.done = False
        self.end_reason = None  # 'game_over' | 'won' | 'skip' | 'stall' once done
        self.events = []

    def update(self, ram, t=None, frames=DECISION_FRAMES):
        """Feed RAM after an emulator step of `frames` frames; returns (reward, done, events_this_step)."""
        g, s = self.game, self.reader.observe(ram)
        p, r, ev, fresh = self.prev, 0.0, [], False
        self.prev = s
        if self.done:
            return 0.0, True, ev
        if s.in_play:
            if self.best_level is None:
                self.best_level, self.best_progress, self.score_used = s.level, None, 0.0
            elif s.level > self.best_level + 1:
                # skipping levels (e.g. SMB warp zones) ends the episode without a bonus,
                # so the reward only pays for playing every level in order
                ev.append(('skip', s.level))
                self.done, self.end_reason = True, 'skip'
            elif s.level > self.best_level:
                r += W_LEVEL
                ev.append(('level', s.level))
                self.best_level, self.best_progress, self.score_used = s.level, None, 0.0
                fresh = True
            if s.level == self.best_level and s.progress_valid:
                if self.best_progress is None:      # baseline only once position is valid
                    self.best_progress = s.progress
                elif s.progress > self.best_progress:
                    r += W_PROGRESS * (s.progress - self.best_progress) / g.PROGRESS_UNIT
                    self.best_progress = s.progress
                    fresh = True
        death = False
        if p is not None:
            if g.DEATH_FROM == 'dying':
                death = s.dying and not p.dying
            else:  # 'lives'
                death = s.lives is not None and p.lives is not None and s.lives < p.lives
            if death and (s.in_play or p.in_play):
                r += W_DEATH
                self.deaths += 1
                ev.append(('death', self.deaths))
                fresh = True
            if s.in_play and p.in_play and s.health is not None and p.health is not None and s.health < p.health:
                r += W_DAMAGE * (p.health - s.health)
            if s.in_play and p.in_play and s.score is not None and p.score is not None and s.score > p.score:
                sr = min(SCORE_CAP, W_SCORE * (s.score - p.score) / g.SCORE_UNIT, SCORE_LEVEL_BUDGET - self.score_used)
                if sr > 0:
                    r += sr
                    self.score_used += sr
        if (p is not None and not self.done and not death and not s.dying and not p.dying and not s.won
                and not s.game_over and s.in_play and p.in_play and s.progress_valid and p.progress_valid
                and s.level == p.level and s.level == self.best_level
                and p.progress - s.progress >= ROLLBACK_DROP * g.PROGRESS_UNIT):
            r += W_ROLLBACK
            ev.append(('rollback', (p.progress - s.progress) / g.PROGRESS_UNIT))
        if (not fresh and not self.done and p is not None and s.in_play and p.in_play and s.progress_valid
                and p.progress_valid and s.progress <= p.progress):
            r += self.idle_per_frame * frames
        # the first observation (right after boot) follows no step, so it adds no idle time
        self.idle_frames = 0 if fresh or p is None else self.idle_frames + frames
        if s.won:
            r += W_WIN
            ev.append(('won', None))
            self.done, self.end_reason = True, 'won'
        elif s.game_over:
            ev.append(('game_over', None))
            self.done, self.end_reason = True, 'game_over'
        elif not self.done and self.stall_frames is not None and self.idle_frames >= self.stall_frames:
            r += W_STALL
            ev.append(('stall', self.idle_frames / NES_FPS))
            self.done, self.end_reason = True, 'stall'
        for e in ev:
            self.events.append((t, *e))
        return r, self.done, ev

    def __deepcopy__(self, memo):
        import copy
        new = object.__new__(RewardTracker)
        for k, v in self.__dict__.items():
            setattr(new, k, v if k == 'game' else copy.deepcopy(v, memo))
        return new

    def state(self):
        return {'best_level': self.best_level, 'best_progress': self.best_progress,
                'deaths': self.deaths, 'score_used': self.score_used, 'idle_frames': self.idle_frames,
                'done': self.done, 'end_reason': self.end_reason,
                'signals': asdict(self.prev) if self.prev else None}
