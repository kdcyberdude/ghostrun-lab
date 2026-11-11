"""Super Mario Bros (JU) (PRG 0). Addresses from the SMB disassembly, verified in audit.

$0770 OperMode: 0 title/demo, 1 game, 2 victory (axe of EVERY castle x-4: bridge collapse, toad/princess
      message, then world+1), 3 game over. Win = OperMode 2 while WorldNumber $075F == 7 (8-4 axe).
$0772 OperMode_Task: 3 = game core running
$000E player routine: 8 normal control, 4 flagpole, 5 end level, 6 lose life, 0xB death
$00B5 vertical screen position: >=2 means fell below the screen (pit death)
$075A lives (shown lives - 1; 0xFF at game over)
$075F world 0-7, $075C stage 0-3 -> level ordinal = world*4 + stage
$006D page, $0086 x within page -> progress = page*256 + x (pixels)
$07DD-$07E2 score digits (x10)
Reward-hacking risk: warp zones (1-2, 4-2) skip levels; the shared reward ends the episode on any
level skip with no bonus, so warping is always worse than finishing the level in order.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesram import digits  # noqa: E402
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Super Mario Bros (JU) (PRG 0).nes')
ROM_SHA256 = 'f61548fdf1670cffefcc4f0b7bdcdd9eaba0c226e3b74f8666071496988248de'
ACTIONS = [('RIGHT',), ('RIGHT', 'A'), ('RIGHT', 'B'), ('RIGHT', 'A', 'B'), ('A',),
           ('LEFT',), (), ('LEFT', 'A'), ('DOWN',)]
PROGRESS_UNIT = 3000.0
SCORE_UNIT = 200.0
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Super Mario Bros.',
    genre='platformer',
    level='world-level 1-1 .. 8-4 in order (world*4 + stage)',
    levels=32,
    progress='pixels moved right in the level (page*256 + x)',
    death='Mario loses a life (enemy, pit or time-out)',
    game_over='all lives lost (GAME OVER)',
    health=None,
    score='game score (enemies, coins, flagpole, time bonus)',
    win='reaching the axe of 8-4 (OperMode 2 in world 8)',
    win_status='provisional',
    limits=[
        'pause is not detected (in_play stays True); harmless: nothing is paid while paused and the stall limit '
        'ends it',
    ],
)


def boot(n):
    n.step((), 60)
    n.tap('START', 4, 4)
    for _ in range(600):
        r = n.ram()
        if r[0x770] == 1 and r[0x772] == 3 and r[0x0E] == 8:
            return
        n.step((), 1)
    raise RuntimeError('SMB boot did not reach player control')


class Reader:
    def reset(self):
        pass

    def observe(self, r):
        mode, task, routine = int(r[0x770]), int(r[0x772]), int(r[0x0E])
        lives = int(r[0x75A])
        page = int(r[0x6D])
        return Signals(
            in_play=mode == 1 and task == 3,
            level=int(r[0x75F]) * 4 + int(r[0x75C]),
            # $006D is signed here: a maze castle's wrong path (4-4) takes it below 0 (255 = -1), not 65000 pixels on
            progress=float((page - 256 if page > 127 else page) * 256 + int(r[0x86])),
            progress_valid=mode == 1 and task == 3 and routine == 8,
            # enemy/timer deaths enter routine 6/0xB; pit deaths keep routine 8 but fall below screen
            dying=mode == 1 and (routine in (6, 0xB) or (routine == 8 and int(r[0xB5]) >= 2)),
            lives=lives - 256 if lives > 127 else lives,
            score=digits(r, range(0x7DD, 0x7E3)) * 10,
            game_over=mode == 3,
            won=mode == 2 and r[0x75F] == 7,   # OperMode 2 alone also fires at the 1-4..7-4 axes
            extra=dict(routine=routine, task=task, area=int(r[0x760]), yscreen=int(r[0xB5])),
        )
