"""Final Mission (J) [p1][!] (Natsume 1990; S.C.A.T. in the US), 1 player. No prior map: addresses found by
RAM tracing + pokes, verified in audit (reports/reward10/finalmission).

$0002 main mode: 1 title, 2 intro story / attract demo, 4 game      $0003 1 in intro/demo, 0 in a real game
$0038 game routine: 2/3 "STAGE n" card + loading, 4 playing, 5 dead, 7 GAME OVER card, 8 CONTINUE/END
      screen, 9/10 stage-clear tally (SCORE + LIFE), then 1, 0, 2 = next stage card
$001E stage, 0-based (poking 1 / 2 on the STAGE 1 card loads the stage-2 / stage-3 maps)
$001C life: 3 at start, -1 per hit, +1 per kill up to a cap that grows with the score (reached 4 with a poked
      score of 10000, 9 with 100000, 49 with 778810; never above 3 in unpoked random play); the HUD draws
      min(life, 4) pips and the stage tally prints it as LIFE; 0 = death. No lives: one death = GAME OVER
$0085 1 from the frame life reaches 0 (dying), until the GAME OVER card
$000E pause flag (1 while paused, screen frozen; $0049 behaves the same)
$004F/$004E scroll page / 8-px column in the page (32 per page): +1 column per 16 frames, also through the
      vertical-scroll sections; page 0 only while routine 4 draws a new stage (~10 frames); frozen at the boss
$004D scroll mode: 0 = halted (boss fight; also 0 before a stage starts), 1/3/4/5/6 scrolling
$04A0-$04BF object HP table (ordinary enemies 1-3 HP; stage-1 boss 96, stage-2 boss 64, both in slot 10;
      stage 3 has 192/255-HP objects while it still scrolls, not counted because the scroll is not halted)
$008A-$008C score / 10 (24-bit little-endian binary; shown only on the tally / GAME OVER / CONTINUE screens)

Signals: progress = scroll columns ($4F*32+$4E; stage 1 has 676 before its boss, stage 2 644) plus, while the
scroll is halted, BOSS_UNITS * the damage dealt to the largest-HP object (>= 20 HP) since the halt began
(boss HP high-water per stage and halt position). PROGRESS_UNIT = 676 + 300. level = $001E. dying = $0085
(immediate). health = min(life, 4) / 4 (HUD pips). in_play = real game, routine 4, page >= 1, not dying,
not paused. game_over = routine 7/8. The scroll advances regardless of input (survival distance), except at
the boss, which must be destroyed (it did not leave in 9500 frames of random play).

Reward-hacking risks:
- score: enemies 50 points; SCORE_UNIT 1000 makes a whole stage ~0.1-0.15, far below progress + level.
  Kills also restore life (+1 up to the cap), which only lowers future damage penalties.
- damage: hits are free while life > 4 (the HUD bar stays full), so a policy with a large life reserve
  may tank hits; death still costs -1 and ends the episode (single life).
- boss term: any object with >= 20 HP present while the scroll is halted is treated as the boss; it is a
  high-water, so it cannot be farmed, and a mid-stage halt only pre-pays scroll progress of the same stage.
- progress accrues by surviving, not by moving (autoscroll).
Unverified: bosses after stage 2 (HP table / halt behaviour; an indestructible >= 20 HP object at a halt
would leave the boss term at 0, no gain), stages 4-5, the ending (won is never set),
the exact life-cap formula, the continue option (the episode ends at GAME OVER).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Final Mission (J) [p1][!].nes')
ROM_SHA256 = '3519782dc8b9e665f5cd7ae153fb5f84c7cd8cf98e5acf25fe3058ea4a31e9ff'
_DIRS = [(), ('UP',), ('DOWN',), ('LEFT',), ('RIGHT',), ('UP', 'RIGHT'), ('DOWN', 'RIGHT'),
         ('UP', 'LEFT'), ('DOWN', 'LEFT')]
ACTIONS = [[(d + ('B',), 4), (d, 4)] for d in _DIRS]
SCROLL_UNITS = 676.0   # stage 1 scrolls 676 8-px columns before the boss
BOSS_UNITS = 300.0     # boss damage fills the last part of a stage
PROGRESS_UNIT = SCROLL_UNITS + BOSS_UNITS
SCORE_UNIT = 1000.0    # enemies are 50 points; a search run scored 6090 in stage 1 (~0.12 reward)
HUD_PIPS = 4           # the HUD draws at most 4 health pips
BOSS_MIN_HP = 20       # ordinary enemies have 1-3 HP, the stage-1 boss 96
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Final Mission (S.C.A.T.)',
    genre='shooter (horizontal and vertical scrolling)',
    level='stage 1-5',
    levels=5,
    progress='auto-scroll columns (horizontal and vertical sections) + 300 x boss damage while the scroll is halted',
    death='life reaches 0',
    game_over='the first death (single life: GAME OVER)',
    health='life pips (the HUD shows up to 4)',
    score='game score',
    win='none (ending not reached)',
    win_status='none',
    limits=[
        'progress pays for surviving, not for moving (auto-scroll)',
        'hits are free while life > 4',
        'bosses after stage 2 unverified',
    ],
)


def boot(n):
    n.step((), 500)
    n.tap('START', 4, 4)
    for _ in range(900):
        r = n.ram()
        if r[0x02] == 4 and r[0x03] == 0 and r[0x38] == 4:
            n.step((), 16)   # the first 10 frames of routine 4 draw the stage, the screen fades in after
            return
        n.step((), 1)
    raise RuntimeError('Final Mission boot did not reach player control')


class Reader:
    def reset(self):
        self.boss_key = None
        self.boss_max = 0

    def observe(self, r):
        game = r[0x02] == 4 and r[0x03] == 0
        routine = int(r[0x38])
        # page 0 = the first ~10 frames of routine 4 (stage still being drawn behind the STAGE card)
        playing = game and routine == 4 and r[0x4F] >= 1 and r[0x85] == 0 and r[0x0E] == 0
        level = int(r[0x1E])
        progress, boss = float(int(r[0x4F]) * 32 + int(r[0x4E])), None
        if game and routine == 4 and r[0x4D] == 0 and r[0x4F] >= 1:   # scrolling stopped: boss fight
            hp = int(max(r[0x4A0:0x4C0]))
            key = (level, progress)
            if self.boss_key != key:
                self.boss_key, self.boss_max = key, 0
            self.boss_max = max(self.boss_max, hp)
            if self.boss_max >= BOSS_MIN_HP:
                boss = (self.boss_max - hp) / self.boss_max
                progress += BOSS_UNITS * boss
        return Signals(
            in_play=bool(playing),
            level=level,
            progress=progress,
            progress_valid=bool(playing),
            dying=bool(game and r[0x85] != 0),
            lives=None,
            health=min(int(r[0x1C]), HUD_PIPS) / HUD_PIPS,
            score=10 * (int(r[0x8A]) | int(r[0x8B]) << 8 | int(r[0x8C]) << 16),
            game_over=bool(game and routine in (7, 8)),
            extra=dict(routine=routine, hp=int(r[0x1C]), scroll=int(r[0x4D]), boss=boss),
        )
