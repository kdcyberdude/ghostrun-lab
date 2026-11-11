"""Castlevania (U) (PRG 0). Addresses from the stable-retro Castlevania-Nes map (health/lives/score) plus probes.

$0018 game state: 1 title, 3/4 intro, 5 gameplay, 6 respawn (1 frame), 7 -> 13 game over (GAME OVER/CONTINUE
      screen), 8 walking through the exit door, 10 entering the castle gate
$0019 sub-state: 6 = level running (0/1/2 while a stage or a respawn is being set up)
$0022 pause flag
$0028 stage: 0 courtyard and 1 entrance hall both show HUD "STAGE 01", then n shows "STAGE 0n" (HUD redraws a
      few frames after the change)
$002A lives including the current one (HUD P-xx = $2A - 1); decrements on every death, 0 = game over
$0045 Simon health 0..64 (16 bars x 4, HUD PLAYER bar), $0044 the animated displayed value
$0040/$0041 Simon world x (low/page), $003F Simon y (stages are one screen high), $002E/$002F camera x
$046C Simon state: 5 hit/knockback, 9 recovering, 1 jump, 8 dead (also set by time-out and pit/water falls
      with health left, so `dying` = health 0 or state 8)
$01A9 ENEMY bar (boss HP 0..64), refilled to 64 on the next stage
$07FC-$07FE score BCD, least significant byte first (= HUD SCORE digits)

Progress (PROGRESS_UNIT = 3000): stages zig-zag between floors (right on the ground, stairs up, left on the
next floor...), so there is no fixed scroll direction. Progress = 32 px x number of new (32-px column, floor)
cells Simon visits in the current stage (floor changes only after a 40 px y change, so jumps are not floors),
plus 1500 x boss damage on boss stages 3, 6, 9, 12, 15, 18. Each cell counts once per stage.
PROVISIONAL: boss bar verified on the stage-3 Phantom Bat only; no win detector (Dracula not reached), so the
  episode only ends on game over; the boss term assumes $01A9 stays 64 outside boss fights on later stages.
Reward-hacking risks: novelty also rewards exploring dead ends and backtracking floors once (bounded by the
  stage size); knockback onto a lower floor visits new cells; candles/hearts give score (capped tiebreaker);
  the timer kills Simon (state 8) so stalling is punished as a death.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesram import bcd  # noqa: E402
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Castlevania (U) (PRG 0).nes')
ROM_SHA256 = '55bcab0d9efc33a9d9efc90f2c3c4ab12f3221a44b1f357e6c6202a456399aa8'
ACTIONS = [('RIGHT',), ('RIGHT', 'B'), ('RIGHT', 'A'), ('B',), ('A',), ('LEFT',), ('LEFT', 'A'),
           ('UP',), ('DOWN',), ('UP', 'RIGHT'), ('UP', 'LEFT'), ('DOWN', 'B'), ('UP', 'B'), ()]
PROGRESS_UNIT = 3000.0   # stages 0-2 explore ~700-2500 px of new (column, floor) cells; boss stages add BOSS_UNITS
BOSS_UNITS = 1500.0
BOSS_STAGES = (3, 6, 9, 12, 15, 18)   # block bosses; $01A9 is the ENEMY bar (64 = full)
CELL_W = 32          # novelty cell width (px)
FLOOR_STEP = 40      # y change that counts as reaching another floor (jumps rise 36 px)
SCORE_UNIT = 300.0
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Castlevania',
    genre='platformer',
    level='stage 0-18 (0 = courtyard)',
    levels=19,
    progress='32 per new (32 px column, floor) cell visited in the stage, + 1500 x boss damage on boss stages',
    death='Simon dies (health 0, time-out, pit or water)',
    game_over='all lives lost (GAME OVER)',
    health='Simon health bar',
    score='game score',
    win='none (Dracula not reached)',
    win_status='none',
    limits=[
        'boss bar verified on the stage-3 Phantom Bat only',
    ],
)


def boot(n):
    n.step((), 240)
    n.tap('START', 4, 4)
    for _ in range(900):
        r = n.ram()
        if r[0x18] == 5 and r[0x19] == 6:
            n.step((), 4)    # screen fades in 4 frames after control starts
            return
        n.step((), 1)
    raise RuntimeError('Castlevania boot did not reach player control')


class Reader:
    """Progress = new territory: 32-px columns visited per floor in the current stage.

    Stages zig-zag between floors (right on the ground, stairs up, left on the next floor, ...), so a
    fixed scroll direction does not exist. Each (column, floor) cell counts once per stage.
    """
    def reset(self):
        self.stage = None
        self.cells = set()
        self.floor_y = None

    def observe(self, r):
        state = int(r[0x18])
        play = state == 5 and r[0x19] == 6
        hp = int(r[0x45])
        stage, x, y = int(r[0x28]), int(r[0x41]) * 256 + int(r[0x40]), int(r[0x3F])
        if stage != self.stage:
            self.stage, self.cells, self.floor_y = stage, set(), None
        if play and hp > 0:
            if self.floor_y is None or abs(y - self.floor_y) >= FLOOR_STEP:
                self.floor_y = y
            self.cells.add((x // CELL_W, self.floor_y // FLOOR_STEP))
        boss = (64 - min(64, int(r[0x1A9]))) / 64.0 if stage in BOSS_STAGES else 0.0
        return Signals(
            in_play=bool(play and r[0x22] == 0),
            level=stage,
            progress=float(CELL_W * len(self.cells)) + boss * BOSS_UNITS,
            progress_valid=bool(play and hp > 0),
            dying=bool(state in (5, 6, 7) and (hp == 0 or r[0x46C] == 8)),
            lives=int(r[0x2A]),
            health=hp / 64.0,
            score=bcd(r, (0x7FE, 0x7FD, 0x7FC)),
            game_over=state in (7, 13),
            won=False,
            extra=dict(x=x, y=y, state=state, sstate=int(r[0x46C]), cam=int(r[0x2F]) * 256 + int(r[0x2E]), boss=int(r[0x1A9])),
        )
