"""The Legend of Zelda (U) (PRG 0). Addresses: stable-retro LegendOfZeldaPRG0-Nes map
(exact ROM match) plus Data Crystal notes, checked in audit.

Zelda is non-linear, so the shared Signals are mapped as:
  level    = Triforce pieces held (popcount $0671) -> one per dungeon completed
  progress = discovery points: +1 per new room (namespace $0010 dungeon/overworld,
             room $00EB), +0.02 per new 32x32 cell Link stands in within a room (link x $0070,
             y $0084; ~1 room's worth per fully explored screen, finds caves/stairs),
             +10 per permanent item tier gained, +10 per heart container gained
             PROGRESS_UNIT = 20 points (one new room = 0.05 of a level)
  dying    = death sequence; Link has no lives, the episode ends at the first death
  health   = hearts / heart containers;  score = rupees ($066D)
$0012 game mode: 1 file select, 0xE register, 3/4 entering, 5 play, 6/7 screen scroll,
                 0x10 stairs/cave entry, 0xB inside cave, 0x11 death (verified 12/12 = hearts 0)
$0657 sword (verified: becomes 1 when taking the sword in the start-screen cave)
$066F hearts: high nibble containers-1, low nibble full hearts-1 with $0670 partial
Reward-hacking risks: room novelty rewards wandering (bounded: 128 overworld screens +
dungeon rooms, and items/containers/Triforce are worth far more); rupee farming is
score only and capped. Victory (Ganon/ending) is not yet verified, so won stays False.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Legend of Zelda, The (U) (PRG 0).nes')
ROM_SHA256 = '8f72dc2e98572eb4ba7c3a902bca5f69c448fc4391837e5f8f0d4556280440ac'
ACTIONS = [('UP',), ('DOWN',), ('LEFT',), ('RIGHT',), ('A',), ('B',),
           ('UP', 'A'), ('DOWN', 'A'), ('LEFT', 'A'), ('RIGHT', 'A')]
PROGRESS_UNIT = 20.0
SCORE_UNIT = 5.0
DEATH_FROM = 'dying'
START_SELECT = {'START': 'opens the item screen to choose the B item; the game is frozen while it is open, '
                          'so it can also be used to pause'}

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='The Legend of Zelda',
    genre='action-adventure (non-linear)',
    level='Triforce pieces held (one per dungeon completed)',
    levels=9,
    progress=('discovery points: +1 per new room, +0.02 per new 32x32 cell in a room, +10 per permanent item tier, '
              '+10 per heart container'),
    death='Link dies',
    game_over='the first death (no extra lives in this setup)',
    health='hearts / heart containers',
    score='rupees',
    win='none (Ganon / the ending not reached)',
    win_status='none',
    limits=[
        'non-linear: progress rewards exploring new rooms; the search never got the sword or finished dungeon 1',
    ],
)
PLAY_MODES = (5, 0xB)   # 5 overworld/dungeon play, 0xB inside a cave (controllable)
DEATH_MODE = 0x11

# permanent items: address -> it counts as tiers gained (value, capped)
ITEMS = {0x657: 3,   # sword 1 wood, 2 white, 3 magic
         0x659: 2,   # arrows 1 wood, 2 silver
         0x65A: 1, 0x65B: 2, 0x65C: 1, 0x65F: 1, 0x660: 1, 0x661: 1,  # bow candle whistle rod raft book
         0x662: 2, 0x663: 1, 0x664: 1, 0x665: 1, 0x666: 1,           # ring ladder magic key bracelet letter
         0x674: 1, 0x675: 1, 0x676: 1}                                # boomerang, magic boomerang, magic shield


def boot(n):
    n.step((), 120)
    n.tap('START', 4, 60)        # title -> file select (cursor on REGISTER YOUR NAME)
    n.tap('START', 4, 60)        # register screen, cursor on slot 1
    n.tap('A', 4, 30)            # name "A"
    for _ in range(3):
        n.tap('SELECT', 4, 20)   # heart cursor to END
    n.tap('START', 4, 60)        # back to file select, cursor on slot 1
    n.tap('START', 4, 4)         # start file 1
    for _ in range(600):
        if n.ram()[0x12] == 5:
            return
        n.step((), 1)
    raise RuntimeError('Zelda boot did not reach gameplay')


def hearts(r):
    hp = int(r[0x66F])
    part = int(r[0x670])
    full = hp & 0xF
    h = full + (1.0 if part >= 0x80 else 0.5 if part > 0 else 0.0)
    return h, (hp >> 4) + 1


class Reader:
    def reset(self):
        self.rooms = set()
        self.cells = set()
        self.base_items = None
        self.base_containers = None

    def observe(self, r):
        mode = int(r[0x12])
        play = mode in PLAY_MODES
        items = sum(min(int(r[a]), cap) for a, cap in ITEMS.items())
        h, containers = hearts(r)
        if play:
            self.rooms.add((int(r[0x10]), int(r[0xEB])))
            self.cells.add((int(r[0x10]), int(r[0xEB]), mode, int(r[0x70]) // 32, int(r[0x84]) // 32))
            if self.base_items is None:
                self.base_items, self.base_containers = items, containers
        gained = 0 if self.base_items is None else (
            max(0, items - self.base_items) + max(0, containers - self.base_containers))
        triforce = bin(int(r[0x671])).count('1')
        return Signals(
            in_play=play,
            level=triforce,
            progress=float(len(self.rooms) + 0.02 * len(self.cells) + 10 * gained),
            progress_valid=play,
            dying=mode == DEATH_MODE,
            health=min(1.0, h / containers),
            score=int(r[0x66D]),
            game_over=mode == DEATH_MODE,
            won=False,
            extra=dict(mode=mode, dungeon=int(r[0x10]), room=int(r[0xEB]), rooms=len(self.rooms), cells=len(self.cells),
                       items=items, containers=containers, hearts=h),
        )
