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
$00CE Mario's y (top of the 16-px row he stands on: floor 176), $0750 area pointer, $06D9/$06DA 7-4 loop
      group correct/passed checks (maze guide only)
Reward-hacking risk: warp zones (1-2, 4-2) skip levels; the shared reward ends the episode on any
level skip with no bonus, so warping is always worse than finishing the level in order.

Maze guide (2026-11-13, MAZE_GUIDE below). 4-4, 7-4 and 8-4 loop: when the game draws page P of a loop command, it
checks Mario's y and sends him 4 pages back unless it is right. The table is the game's own (ROM LoopCmdWorldNumber /
PageNumber / YPosition): 4-4 pages 5, 9 need y 64, 176; 7-4 pages 4-6 and 8-10 need 176, 128, 64 and 64, 128, 64 (all
three of a group, else the third loops); 8-4 pages 6, 11, 16 need y 240, which no standing Mario has, so only the
right pipe passes them. Measured on the it002907 policy: the check fires near Mario's x (P - 1) * 256; 4-4
looped at page 9 on y 64/128, 7-4 at page 6 on y 176 every time, 8-4 at x 1273 and 2553. 8-4 pipes, read from the
block buffer (enterable top metatile $10) and entered: x 272-303 (stand y 96; there only after the first loop, which
lands Mario at x ~255) -> x 1848; floating x 2432-2463 (y 64) ->
x 3128 (reached from the block at x 2400, stand y 112); x 2112 (y 144) is a trap back to x 312; in section 3 (floor y 128) x 3392-3423 (y 80) is a trap back to x 312 and x 3648-3679 (y 96) leads to the water part
($074E 0), whose one exit comes out at x 4152 before the axe.
x alone pays nothing for being on the right path, so the policy took the floor every time. The guide is a potential
(Signals.guide), shaped as gamma * next - previous with the learner's discount and zero terminal potential, so
discounting cannot make a closed loop profitable: GUIDE_W * (segments behind Mario + reach * height match), where reach climbs from the segment
start to the target x (a pipe's: falls again over 128 px past it) and height match = 1 - |y - target| / 128 (>= 0;
for a pipe only being lower counts, since Mario jumps up before moving onto it).
Other areas or area types of the level (8-4's water part) count as every segment passed.
4-4's completed segments use CurrentPageLoc ($0725) and LoopCommand ($0745), so crossing the nominal x does not
give full credit or change the required height before the ROM actually accepts that check.
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

# 8-4 progress by part (measured 2026-11-13): castle x < 4096, then the water part (area type $074E 0, x 56..1090,
# one exit pipe) and the castle again from x 4152 to the axe. Plain x would pay nothing from the water to x 3650
# again, so the water part counts from WATER_BASE and the last castle part is shifted past it.
SMB84_WATER_BASE = 4096
SMB84_FINAL_SHIFT = 1280
GUIDE_W = 0.25          # potential per maze segment (a whole level of progress is 1.0)
GUIDE_Y_SPAN = 128.0    # height match falls to 0 this many pixels from the target y
GUIDE_PIPE_PAST = 128.0  # a pipe segment's reach falls to 0 this many pixels past the pipe
# level -> (castle area pointer $0750, segments (start x, target x, target y, pipe 0/1, via x, via y)); a segment runs
# from its start to the next one's (the last: to its target, or 1024 px for a pipe segment). 8-4's segments start
# where the previous pipe comes out (x 1848, 3128), not at the loop checks (1280, 2560, 3840): walking on foot to a
# check must not count as passing a segment, since the game loops Mario back from there. Via (0 = none): the target
# height is via y up to via x, then moves linearly to target y at target x (8-4: the block at x 2400 that Mario stands on
# at y 112 to reach the floating pipe's top at y 64; from the floor at 176 it is out of reach)
MAZE_GUIDE = {
    15: (0x61, ((0, 1024, 64, 0, 0, 0), (1024, 2048, 176, 0, 0, 0))),
    27: (0x64, ((0, 768, 176, 0, 0, 0), (768, 1024, 128, 0, 0, 0), (1024, 1280, 64, 0, 0, 0), (1280, 1792, 64, 0, 0, 0),
                (1792, 2048, 128, 0, 0, 0), (2048, 2304, 64, 0, 0, 0))),
    31: (0x65, ((0, 288, 96, 1, 0, 0), (1792, 2448, 64, 1, 2408, 112), (3072, 3656, 96, 1, 3600, 128))),
}


def maze_guide(level, x, y, area, wrong=0, atype=3, loop_page=None, loop_pending=0):
    """Potential of the maze guide (None outside 4-4/7-4/8-4). Same arithmetic order as rl/csrc/reward10.cuh.
    wrong: checks of the current 7-4 group passed at the wrong height ($06DA passes - $06D9 correct); each takes its
    segment's credit back at once instead of when the group's last check loops.
    loop_page/loop_pending: 4-4's renderer page and native loop command; omit for the spatial-only helper."""
    if level not in MAZE_GUIDE:
        return None
    ptr, segs = MAZE_GUIDE[level]
    n = len(segs)
    # The ROM uses bits 5-6 for area type and bits 0-4 for its index; $E5 and $65 are the same castle.
    if (area & 0x7F) != ptr or atype != 3:  # another area of the level (8-4's water part): every segment passed
        return GUIDE_W * n
    native = level == 15 and loop_page is not None
    k = -1
    if native:
        # The renderer checks one page beyond each target. Keep the height target until the ROM clears LoopCommand.
        k = sum(loop_page > tx // 256 + 1 or (loop_page == tx // 256 + 1 and not loop_pending)
                for _, tx, _, _, _, _ in segs)
        wrong = 0
        if k == n:
            return GUIDE_W * n
    else:
        for i in range(n):
            if x >= segs[i][0]:
                k = i
    if k < 0:
        return 0.0
    start, tx, ty, pipe, vx, vy = segs[k]
    end = segs[k + 1][0] if k + 1 < n else (start + 1024 if pipe else tx)
    if not native and x >= end:
        return GUIDE_W * (k + 1 - wrong)
    if native:
        reach = min(1.0, max(0.0, (x - start) / (tx - start)))
    elif x <= tx:
        reach = (x - start) / (tx - start)
    else:
        reach = 1.0 - (x - tx) / GUIDE_PIPE_PAST
        if reach < 0.0:
            reach = 0.0
    want = float(ty)
    if vx > 0:
        if x <= vx:
            want = float(vy)
        elif x < tx:
            want = vy + (ty - vy) * ((x - vx) / (tx - vx))
    dy = y - want
    if dy < 0:
        dy = 0.0 if pipe else -dy    # a pipe is reached from above: higher than the target costs nothing
    match = 1.0 - dy / GUIDE_Y_SPAN
    if match < 0.0:
        match = 0.0
    return GUIDE_W * (k - wrong + reach * match)


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
        level = int(r[0x75F]) * 4 + int(r[0x75C])
        # $006D is signed here: a maze castle's wrong path (4-4) takes it below 0 (255 = -1), not 65000 pixels on
        x = (page - 256 if page > 127 else page) * 256 + int(r[0x86])
        atype = int(r[0x74E])
        prog = x
        if level == 31:
            if atype == 0:
                prog = SMB84_WATER_BASE + x
            elif x >= SMB84_WATER_BASE:
                prog = x + SMB84_FINAL_SHIFT
        valid = mode == 1 and task == 3 and routine == 8
        return Signals(
            in_play=mode == 1 and task == 3,
            level=level,
            progress=float(prog),
            progress_valid=valid,
            guide=maze_guide(level, x, int(r[0xCE]), int(r[0x750]), int(r[0x6DA]) - int(r[0x6D9]), atype,
                             int(r[0x725]), int(r[0x745])) if valid else None,
            # enemy/timer deaths enter routine 6/0xB; pit deaths keep routine 8 but fall below screen
            dying=mode == 1 and (routine in (6, 0xB) or (routine == 8 and int(r[0xB5]) >= 2)),
            lives=lives - 256 if lives > 127 else lives,
            score=digits(r, range(0x7DD, 0x7E3)) * 10,
            game_over=mode == 3,
            won=mode == 2 and r[0x75F] == 7,   # OperMode 2 alone also fires at the 1-4..7-4 axes
            extra=dict(routine=routine, task=task, area=int(r[0x760]), yscreen=int(r[0xB5])),
        )
