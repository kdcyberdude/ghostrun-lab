"""Pac-Man (U) (Namco), 1 player. Addresses found by RAM tracing + pokes, verified in audit.

$003F game state: 2 READY intro, 4 playing, 6 40-frame freeze (seen with a +200 ghost score),
      8 caught/death sequence (set the frame the ghost catches him), 10 game over,
      12 maze-cleared flashing, 0 transitions. The title/intro reuses some values, so $0048/$0047 gate it.
$0048 0 while a real game is in progress, $FF on the title/attract screens and the demo
$0047 1 in the attract-mode demo, 0 otherwise
$004A pause toggle count: +1 per START press in play, odd = paused (verified: sprites freeze
      on odd values, move on even values over two pause/unpause cycles)
$0067 lives including the Pac-Man in play (3 at start, decremented when the next life starts,
      0 at game over)
$006A dots + energizers remaining in the current maze (192 at maze start = 188 dots + 4
      energizers, counted on screen; the maze ends when it reaches 0)
$0068 round number, 0-based (increments when the next maze starts)
$0070-$0075 score digits, one decimal digit per byte, least significant first, in units of 10

Signals: progress = 192 - $006A (dots/energizers eaten in this maze, PROGRESS_UNIT = 192);
level = $0068; dying = state 8 (immediate edge); lives = $0067; game_over = state 10.
in_play = real game, not demo, not paused, state in {4, 6, 12} (12 included so the last dot of
a maze, which switches the state to 12 in the same frame, is still credited).
Ghosts and fruit are score only (SCORE_UNIT 1000, so all ghosts of a maze earn ~0.24, and all score is
capped at 0.2 per maze, vs 2.0 for clearing it).

Reward-hacking risks:
- ghost eating pays score only: at most 0.06 per energizer (200+400+800+1600 at SCORE_UNIT 1000);
  all score of a maze (ghosts, fruit, dots) is capped at 0.2 vs 2.0 for clearing it.
- fruit pays score only (within the same 0.2 per-maze budget).
- no per-step cost: wandering in cleared corridors is reward-neutral.
- an extra life (if awarded) raises $0067; deaths use the state-8 edge, not lives, so an extra
  life cannot hide a death.
Unverified: intermission cut-scenes (state value unknown, treated as not in play); the extra
life (never observed); rounds after 2 (search reached 176/192 dots of round 2, see
reports/reward10/pacman/summary.json).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Pac-Man (U) (Namco).nes')
ROM_SHA256 = '254fb93e16d0db63b36ffa3ef1bc5a9e6cf9e9b1de347712d9936fe2ab7b690d'
ACTIONS = [('UP',), ('DOWN',), ('LEFT',), ('RIGHT',)]
DOTS_PER_MAZE = 192
PROGRESS_UNIT = float(DOTS_PER_MAZE)
SCORE_UNIT = 1000.0    # ghosts 200-1600 -> 0.004-0.032 per ghost: a tiebreaker, not a pull toward ghost hunting
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Pac-Man',
    genre='maze',
    level='maze (round) number',
    levels=None,
    progress='dots and energizers eaten in the current maze (192 per maze)',
    death='caught by a ghost',
    game_over='all lives lost (GAME OVER)',
    health=None,
    score='game score (dots, ghosts, fruit)',
    win='none (endless)',
    win_status='none',
    limits=[
        ('the round counter $0068 stops at 22: from maze 24 on there is no level or progress reward, '
         'so a long episode ends at the stall limit'),
    ],
)


def search_key(s):
    """Search only: coarse maze region of Pac-Man ($001A x, $001C y; checked on screenshots, sprite offset
    ~12 px), so the last dots of a maze can be reached region by region."""
    e = s.extra or {}
    return (e.get('x', 0) // 32, e.get('y', 0) // 32)


def _real_game(r):
    return r[0x48] == 0 and r[0x47] == 0


def boot(n):
    """Title -> '1 PLAYER' (default cursor) -> START -> end of the READY intro."""
    n.step((), 360)                 # title scrolls in; cursor appears ~frame 355
    for _ in range(3):
        n.tap('START', 4, 4)
        for _ in range(600):
            r = n.ram()
            if _real_game(r) and r[0x3F] == 4 and not r[0x4A] & 1:
                return
            n.step((), 1)
    raise RuntimeError('Pac-Man boot did not reach player control')


class Reader:
    def reset(self):
        pass

    def observe(self, r):
        state, remaining = int(r[0x3F]), int(r[0x6A])
        game = bool(_real_game(r))
        playing = game and state in (4, 6, 12) and not r[0x4A] & 1
        return Signals(
            in_play=bool(playing),
            level=int(r[0x68]),
            progress=float(DOTS_PER_MAZE - remaining) if remaining <= DOTS_PER_MAZE else 0.0,
            progress_valid=bool(playing) and remaining <= DOTS_PER_MAZE,
            dying=game and state == 8,
            lives=int(r[0x67]),
            score=10 * sum(int(r[0x70 + i]) * 10 ** i for i in range(6)),
            game_over=game and state == 10,
            extra=dict(state=state, remaining=remaining, pause_count=int(r[0x4A]), x=int(r[0x1A]), y=int(r[0x1C])),
        )
