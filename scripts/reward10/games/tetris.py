"""Tetris (U) [!], A-type, start level 0. Addresses from the NES Tetris disassembly, verified in audit.

$00C0 gameMode: 0 legal, 1 title, 2 game-type menu, 3 level menu, 4 game, 5 demo
$00C1 gameType: 0 A-type (boot asserts it)
$0048 playState (disassembly names): 1 player control, 2 lock, 3 row check, 4 line-clear animation,
      5 update lines, 6 B-type goal check, 7 unused, 8 spawn next piece, 10 game-over curtain
      (observed: 1, 5-8 between pieces, 10 from the top-out frame to the end)
$0058 curtainRow: 0xF0..0xFF delay, then 0..20 while the curtain fills; 20 = curtain complete
$00BD renderMode: 3 during gameplay, 0 while the PAUSE screen is shown (START pauses)
$0050-$0051 lines cleared, BCD, $0051 = hundreds
$0064 level number; A-type from level 0 rises every 10 lines (seen 0->1 at 10 and 1->2 at 20 lines)
$0053-$0055 score, BCD, little-endian ($0055 most significant)
$0400-$04C7 playfield 10x20, $EF = empty cell (debug only: extra.height/holes, not rewarded)

Signals: level = $0064. progress = lines - 10*level (lines inside the current level,
PROGRESS_UNIT = 10 lines). No lives, no health. dying = top-out (playState 10, the curtain
starts that frame); game_over = curtain complete (curtainRow 20, ~2.4 s after top-out).
in_play requires gameMode 4, playState 1..8 and renderMode 3, so pause/menus/demo give nothing.

Reward-hacking risks:
- soft-drop points (1 per row held DOWN) feed the score tiebreaker; capped and tiny (SCORE_UNIT 40).
- a multi-line clear that crosses a multiple of 10 (e.g. 8 -> 12 lines) pays the +1 level bonus but
  the 2 lines past the boundary become the new level's baseline (no progress credit). Small loss.
- no per-step cost: stalling (keeping the stack low without clearing) is reward-neutral, not punished.
- pausing yields no reward and no penalty (in_play False).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesram import bcd  # noqa: E402
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Tetris (U) [!].nes')
ROM_SHA256 = 'fdb764dce8175c15f5198648b119905ef2caf0a5c51fae329ebf1be9e3c035d4'
# single buttons only: A/B rotate, DOWN soft-drops; a button must be released between presses
ACTIONS = [(), ('LEFT',), ('RIGHT',), ('DOWN',), ('A',), ('B',)]
PROGRESS_UNIT = 10.0   # lines per level in A-type
SCORE_UNIT = 40.0      # one single line at level 0
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Tetris',
    genre='puzzle',
    level='speed level (A-type from level 0, +1 every 10 lines)',
    levels=None,
    progress='lines cleared in the current level',
    death='top-out (the stack reaches the top)',
    game_over='top-out curtain complete',
    health=None,
    score='game score (line clears, soft drops)',
    win='none (endless)',
    win_status='none',
    limits=[
        'lines of a multi-line clear past a level boundary earn no progress',
    ],
)


def boot(n):
    """Legal screen -> title -> A-type -> level 0 -> first falling piece."""
    n.step((), 270)                 # legal screen ignores START before ~frame 260
    for _ in range(12):             # title -> game type (A-type default) -> level (0 default) -> game
        mode = int(n.ram()[0xC0])
        if mode == 4:
            break
        n.tap('START', 2, 0)
        for _ in range(60):
            n.step((), 1)
            if n.ram()[0xC0] != mode:
                break
        n.step((), 30)
    for _ in range(120):
        r = n.ram()
        if r[0xC0] == 4 and r[0x48] == 1 and r[0xBD] == 3:
            if r[0xC1] != 0 or r[0x64] != 0:
                raise RuntimeError('Tetris boot did not select A-type level 0')
            return
        n.step((), 1)
    raise RuntimeError('Tetris boot did not reach player control')


class Reader:
    def reset(self):
        pass

    def observe(self, r):
        mode, ps, render = int(r[0xC0]), int(r[0x48]), int(r[0xBD])
        lines, level = bcd(r, (0x51, 0x50)), int(r[0x64])
        playing = mode == 4 and 1 <= ps <= 8 and render == 3
        topout = mode == 4 and ps == 10
        field = r[0x400:0x4C8].reshape(20, 10) != 0xEF
        filled = field.any(axis=0)
        tops = field.argmax(axis=0)
        holes = int(sum((~field[tops[c]:, c]).sum() for c in range(10) if filled[c]))
        return Signals(
            in_play=playing,
            level=level,
            progress=float(max(0, lines - 10 * level)),
            progress_valid=playing,
            dying=topout,
            score=bcd(r, (0x55, 0x54, 0x53)),
            game_over=topout and int(r[0x58]) == 20,
            extra=dict(lines=lines, ps=ps, mode=mode, render=render, curtain=int(r[0x58]),
                       height=int(20 - tops[filled].min()) if filled.any() else 0, holes=holes),
        )
