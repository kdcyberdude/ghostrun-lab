"""Gradius (U), 1 player. Addresses found by RAM tracing + pokes, verified in audit (reports/reward10/gradius).

$0000 main mode: 5 = game, 2 = attract demo, 0/1 title, 3/4 power-on
$000A 1 while a real game is running (0 in the demo, on the title, from the GAME OVER frame on)
$001B stage/ship state: 1-4 stage (re)start, bit 7 = stage running, low bits = stage phase
      (0x80 scrolling, 0x82 pre-boss halt: scroll stopped ~95 decisions, the stage-1 volcano eruption,
      0x84 scrolling again, 0x85 boss (Big Core): scroll stopped until it is destroyed or leaves,
      0x86 boss gone, flying into the next stage), 0xA0 ship hit / exploding (set the frame the ship is
      hit, in any phase), 0xC0 GAME OVER
$0015 pause flag (1 while paused; screen frozen)
$0019 stage, 0-based: increments on the first frame of the next stage (stages 1-5 seen in search; poking
      it during the stage-1 boss exit loads the stage it names + 1, e.g. 3 -> the stage-5 tentacle stage)
$0017 NOT the stage: it also changes when a power-up is activated (a search exploited it as a level)
$0020 lives in reserve (the HUD number; 3 at start), decremented ~120 frames after the hit when the stage
      restarts at a checkpoint, 255 at GAME OVER (Signals.lives = -1, so deaths == lives lost)
$003F/$003E stage scroll counter (high/low): +1 every 2 frames, restarts at a checkpoint after a death
      (checkpoints seen at 512, 1536, 2048); stops during 0x82/0x85. Stages 1-4: pre-boss halt at 3072,
      boss at 3328; stage 5: 2816 and 3072
$07E4-$07E6 score BCD, least significant byte first; HUD shows the value * 10

Signals: progress = scroll counter (PROGRESS_UNIT 3328 = one stage up to its boss), level = $0019,
dying = $001B in 0xA0-0xBF (immediate), lives = $0020, game_over = $001B == 0xC0, in_play = real game,
stage running, not hit, not paused. The scroll advances regardless of input, so progress measures
survival distance. search_key (search only) adds the time spent in the halted phases 0x82/0x85.

Reward-hacking risks:
- score farming: while the scroll is halted (0x82, boss 0x85) enemies/rocks keep coming and pay score
  without progress; SCORE_UNIT 2000 makes an enemy (100) worth 0.001 and all score of a stage is capped at 0.2, the
  halted phases end on a timer (the Big Core leaves), so it is bounded and far below progress + level.
- progress is a high-water mark: replaying a section after a checkpoint restart earns nothing.
- progress accrues by surviving, not by moving: idling is as good as moving (survival is the objective).
- activating power-ups is reward-neutral (it changes $0017, deliberately not used).
Unverified: stages 6-7, what $0019 does after the final stage / game loop (if it resets, no further
level or progress reward), a win/ending detector (none: won is never set), extra lives (none seen up to
47900 points; deaths use the hit edge, so an extra life could not hide a death).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesram import bcd  # noqa: E402
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Gradius (U).nes')
ROM_SHA256 = 'cda334d473ae84fbda5df431e6618b46d9541da9fc572bd9035dccfb0df4e33d'
_DIRS = [(), ('UP',), ('DOWN',), ('LEFT',), ('RIGHT',), ('UP', 'RIGHT'), ('DOWN', 'RIGHT'),
         ('UP', 'LEFT'), ('DOWN', 'LEFT')]
# A fires only on a new press: every action is a macro that presses A for 4 frames and releases it for 4
ACTIONS = [[(d + ('A',), 4), (d, 4)] for d in _DIRS] + [[(('B',), 4), ((), 4)]]
PROGRESS_UNIT = 3328.0
SCORE_UNIT = 2000.0    # enemies are 100 points; a full stage-1 run scores ~20000 (~0.2 reward)
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Gradius',
    genre='horizontal shooter',
    level='stage 1-7 in order',
    levels=7,
    progress='auto-scroll counter (+1 per 2 frames, halts at bosses): survival distance, not movement',
    death='the ship is hit (a life lost)',
    game_over='all lives lost (GAME OVER)',
    health=None,
    score='game score',
    win='none (the stage-7 brain was reached but not beaten)',
    win_status='none',
    limits=[
        'progress pays for surviving, not for moving (auto-scroll)',
        'bosses have no damage term',
    ],
)


def boot(n):
    n.step((), 150)
    n.tap('START', 4, 4)
    for _ in range(600):
        r = n.ram()
        if r[0x00] == 5 and r[0x0A] == 1 and r[0x1B] == 128:
            n.step((), 5)   # control starts while the screen is still black; it fades in 5 frames later
            return
        n.step((), 1)
    raise RuntimeError('Gradius boot did not reach player control')


def _game(r):
    return r[0x00] == 5 and r[0x0A] == 1


def search_key(s):
    """Search only: while the scroll is halted (volcano eruption 0x82, boss 0x85) the high-water progress is
    frozen, so the time spent in that phase (decisions observed, 16 per bucket) is added to the search cell."""
    phase = s.extra['state'] & 0x0F if s.in_play else None
    return (phase, s.extra['phase_t'] // 16) if phase in (2, 5) else None


class Reader:
    def reset(self):
        self.last_state, self.phase_t = None, 0

    def observe(self, r):
        game = _game(r)
        state = int(r[0x1B])
        lives = int(r[0x20])
        self.phase_t = self.phase_t + 1 if state == self.last_state else 0
        self.last_state = state
        playing = game and state & 0xE0 == 0x80 and r[0x15] == 0
        return Signals(
            in_play=bool(playing),
            level=int(r[0x19]),
            progress=float(int(r[0x3F]) * 256 + int(r[0x3E])),
            progress_valid=bool(playing),
            dying=bool(game and state & 0xE0 == 0xA0),
            lives=lives - 256 if lives > 127 else lives,
            score=bcd(r, (0x7E6, 0x7E5, 0x7E4)) * 10,
            game_over=bool(r[0x00] == 5 and state == 192),
            extra=dict(state=state, phase_t=self.phase_t),
        )
