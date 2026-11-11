"""Contra (U). Address names follow the stable-retro Contra-Nes map (level/lives/score/xscroll) plus probes.

$0018 game routine: 1 title/attract, 2 demo, 5 game        $001C demo flag (1 = attract demo)
$002C level routine: 0-3 stage intro/loading, 4 level running, 10 game over
$0025 pause flag (1 while paused)
$0030 current level 0-7
$0032 P1 lives remaining after the current one (intro "REST 3" = $32 + 1); the final death does not
      decrement it, $0038 becomes 1 instead, so Signals.lives = -1 at game over (deaths == lives lost)
$0038 P1 game over status (1 = game over; "GAME OVER" replaces the P1 medals)
$0090 P1 state: 0 dropping in / respawning, 1 normal, 2 hit and dying (rising edge = death)
$0040 level location type: 0 outdoor (scrolling), 1 indoor base rooms
$0064 screen number, $0065 scroll offset: horizontal levels count screens rightwards, the waterfall
      counts screens upwards, base levels count rooms ($64 += 1 when walking into the next room)
$0528+i enemy type, $0578+i enemy HP (16 slots). Level 1 Defense Wall: core type $11 (32 HP; its death
      ends the level), turrets type $10 (16 HP). Base wall cores: type $14.
$07E2 score (<u2) in units of 100 points (game-over screen shows "1P 500" for the word 5)

Progress (PROGRESS_UNIT = 6000 ~ level 1 = 3072 px scroll + 3000 boss; SEARCH_BUCKETS = 120 keeps search
cells 50 units wide, a coarser cell made the search much weaker):
  outdoor: $64*256+$65, plus on level 1 the Defense Wall core damage fraction * 3000 (the boss is worth as much
           progress as the whole scroll: one core hit = 94 units)
  indoor:  room index * 1000 + 500 * damage fraction of the current room's wall core
Verified (reports/reward10/contra): boot, deaths/lives/game over (12 random episodes + Level2/3/5 savestates),
  score vs HUD, level-1 core damage, the level event at STAGE 2 BASE1, and base-room progress (fresh-start
  search6 seed 4 cleared rooms 0-3 of BASE1 and shot the 16-HP room-4 core: room counter and core HP checked
  per decision).
PROVISIONAL: waterfall progress only from stable-retro Level3 savestate exploration (counter rises while
  climbing); bosses of levels 2-8 (incl. the BASE1 final room) have no damage term; there is no win detector
  (the ending was not reached), so after the final boss the game loops and the episode only ends on game over.
Reward-hacking risks: score from respawning soldiers (capped at 0.1/step and 0.2/level, tiebreaker only); progress is a
  high-water mark so scrolling back and forth earns nothing; core damage only counts for the level-1 core
  type and for type $14 cores in base rooms; weapon pickups give score only.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Contra (U).nes')
ROM_SHA256 = '26541a5550ee22deeb3d5484e4a96130219b58cff74d068fb1eb6567fa5e5519'
ACTIONS = [('RIGHT', 'B'), ('RIGHT', 'A', 'B'), ('B',), ('UP', 'B'), ('DOWN', 'B'), ('RIGHT', 'UP', 'B'),
           ('RIGHT', 'DOWN', 'B'), ('LEFT', 'B'), ('A', 'B'), ('LEFT', 'A', 'B'), ()]
PROGRESS_UNIT = 6000.0             # level 1: ~3072 px scroll + BOSS_UNITS of core damage = one level
SEARCH_BUCKETS = 120               # search cells 50 units wide (search only, not the reward)
ROOM_UNITS = 1000.0                # indoor: one room = 1000 progress units, core damage fills the first half
BOSS_UNITS = 3000.0                # level-1 Defense Wall core damage counts as one more level of progress
BOSS_CORE = {0: (0x11, 32)}        # level -> (enemy type, max HP) of the level-ending boss core
ROOM_CORE_TYPE = 0x14              # indoor wall core (sensor) type
SCORE_UNIT = 200.0
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Contra',
    genre='run-and-gun shooter',
    level='stage 1-8 in order',
    levels=8,
    progress=('scroll pixels (the waterfall counts upward); base stages: rooms x 1000 + 500 x wall-core damage; '
              'stage 1 adds 3000 x Defense Wall core damage'),
    death='a life lost',
    game_over='all lives lost (GAME OVER)',
    health=None,
    score='game score',
    win='none (the ending was not reached; after the last stage the game loops)',
    win_status='none',
    limits=[
        'bosses of stages 2-8 (incl. the last base room) have no damage term: the reward is flat during them',
    ],
)


def boot(n):
    n.step((), 240)          # title scrolls in
    n.tap('START', 4, 60)    # finish the title scroll
    n.tap('START', 4, 4)     # 1 PLAYER
    for _ in range(900):
        r = n.ram()
        if r[0x18] == 5 and r[0x1C] == 0 and r[0x2C] == 4 and r[0x90] == 1:
            return
        n.step((), 1)
    raise RuntimeError('Contra boot did not reach player control')


def _slot(r, etype):
    for i in range(16):
        if r[0x528 + i] == etype:
            return i
    return None


class Reader:
    def reset(self):
        self.core_key = None
        self.core_max = 0

    def observe(self, r):
        game = r[0x18] == 5 and r[0x1C] == 0
        running = game and r[0x2C] == 4
        over = game and r[0x38] == 1
        lives = int(r[0x32])
        level, screen = int(r[0x30]), int(r[0x64])
        progress, boss = float(screen * 256 + int(r[0x65])), None
        if r[0x40] == 1:   # base rooms: room index + damage dealt to this room's wall core
            progress = screen * ROOM_UNITS
            i = _slot(r, ROOM_CORE_TYPE)
            if i is not None:
                key = (level, screen)
                if key != self.core_key:
                    self.core_key, self.core_max = key, 0
                self.core_max = max(self.core_max, int(r[0x578 + i]))
                if self.core_max >= 2:   # a freshly spawned core slot reads HP 1 for a frame before its real HP
                    boss = 1 - int(r[0x578 + i]) / self.core_max
                    progress += 0.5 * ROOM_UNITS * boss
        elif level in BOSS_CORE:
            etype, mx = BOSS_CORE[level]
            i = _slot(r, etype)
            if i is not None:
                boss = (mx - min(mx, int(r[0x578 + i]))) / mx
                progress += BOSS_UNITS * boss
        return Signals(
            in_play=bool(running and r[0x25] == 0 and not over),
            level=level,
            progress=progress,
            progress_valid=bool(running and r[0x90] == 1),
            dying=bool(game and r[0x90] == 2),
            lives=-1 if over else lives,
            score=(int(r[0x7E2]) | int(r[0x7E3]) << 8) * 100,
            game_over=bool(over),
            won=False,
            extra=dict(routine=int(r[0x2C]), pstate=int(r[0x90]), loc=int(r[0x40]), scrolltype=int(r[0x41]),
                       screen=screen, boss=boss),
        )
