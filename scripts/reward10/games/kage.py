"""Kage (Japan) = Shadow of the Ninja (Natsume 1990). No lives: every death leads to GAME OVER / CONTINUE.

Ported from scripts/kage/game_info.py (replay-verified on the native FCEUmm env); this module never writes RAM.
$0002 main mode: 0 = playable; nonzero = intros, stage-clear walk/titles, death/GAME OVER ($39-$3E, $5A), ending
$0040 stage id 0..14 in play order (1-1..1-4, 2-1..2-3, 3-1..3-3, 4-1..4-3, 5-1, 5-2)
$06F0 P1 health 0..16 (HUD bar); 0 = death (no lives counter exists)
$0044 continues (NOT lives, stays 5 after a death), $00D1 pause flag (16 while paused)
$0610/$0620, $0630/$0640 player world x/y (fixed point /16); $0022/$0023, $0024/$0025 camera x/y
$06A1 player state (8 = pit fall / checkpoint reposition, costs health only)
Slot 8 object: $0708 kind, $0798 selector, $07F8 HP, $0758 phase; a boss is (stage, kind, selector), table BOSSES
Victory: game_info's one-frame native ending signature (stage 0, mode $6B, $AE=$FF, kind $88, sel $4F, HP 0,
phase $10, health > 0)

Progress (PROGRESS_UNIT = 1000 = one whole stage; route stages 0..1000, boss rooms 0..1000 by boss damage,
stages 0 and 14 = route 0..500 then boss 500..1000):
  route: camera position projected onto a per-stage polyline (ROUTES), arc length / route length. Polylines are
  the simplified camera path of the native full-game controller replay
  reports/kage-resume3h-20261031/research/native-ending-witness-v1 (13130 decisions, all 15 stages + ending).
  Matching only looks LOOKAHEAD px past the best point so geometric shortcuts cannot skip ahead.
  boss: (maxHP - HP) / maxHP while the stage boss occupies slot 8 (stages 0, 3, 6, 9, 12, 14).

Verification (reports/reward10/kage/summary.json):
  - stable_retro boot with the first idle shortened 600->599 frames is RAM-identical to the native env (one frame
    offset); the native route replays exactly through stages 0..6 in stable_retro and diverges inside stage 7
    (y differs at frame 22899: an emulator difference, not a reward issue).
  - Signals over the native full-game trace (RAM fields rebuilt per frame): levels 1..14 fire once each in order,
    every stage reaches progress 1000, won fires exactly once at the native ending frame 96840.
    (Consistency check only: the routes were derived from this same trace.)
  - deaths: dying = health 0; game_over = health 0 and mode left 0 (death animation over) ~36 decisions later,
    followed by the CONTINUE/END screen; 12/12 random episodes (reports/reward10/kage).
  - search.py kage 900 s x 8 workers: 4/8 seeds kill the stage-0 boss and enter stage 1 (1-2), 0 deaths.
PROVISIONAL: the won detector is verified on the native-core trace only (stable_retro cannot replay past stage 7);
  routes for stages 7..14 and boss gates for stages 9..14 come from the native trace only; pause flag $00D1 from
  10 pause/unpause toggles. No score display was found (Signals.score = None).
Reward-hacking risks: health pickups refill $06F0 (no reward, only less future damage penalty); pit falls cost
  health but no death; route progress uses the camera, which follows the player, so standing still earns nothing;
  bosses only count while their exact kind/selector occupies slot 8 of their own stage.
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Kage (Japan).nes')
ROM_SHA256 = 'ee3a87feddf45ec99034cd4f4a7e34224c8eb0e0d3c881665f796aa184e1c78b'
ACTIONS = [('RIGHT',), ('RIGHT', 'B'), ('RIGHT', 'A'), ('RIGHT', 'A', 'B'), ('B',), ('A',), ('A', 'B'),
           ('LEFT',), ('LEFT', 'B'), ('LEFT', 'A'), ('UP',), ('UP', 'A'), ('UP', 'B'), ('DOWN',),
           ('DOWN', 'A'), ('DOWN', 'B'), ()]
PROGRESS_UNIT = 1000.0
SCORE_UNIT = 1.0
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Kage (Shadow of the Ninja)',
    genre='action platformer',
    level='stage 1-1 .. 5-2 in play order',
    levels=15,
    progress='camera position along the stage route (from a full-game replay); boss stages: route then boss damage',
    death='health reaches 0',
    game_over='the first death (no lives: GAME OVER / CONTINUE)',
    health='health bar (16)',
    score=None,
    win="the game's ending (one-frame ending signature)",
    win_status='provisional',
    limits=[
        'routes of stages 7-14 and the win come from the native-emulator replay only',
    ],
)

# camera waypoints per stage (route stages only)
ROUTES = {
    0: [(256, 400), (2040, 400)],
    1: [(2560, 512), (3297, 512), (3329, 462), (3329, 294), (3585, 270)],
    2: [(256, 1024), (2048, 1024)],
    4: [(512, 768), (1745, 768), (1793, 718), (1793, 550), (1841, 510), (3328, 510)],
    5: [(512, 2048), (1241, 2048), (1281, 1998), (1281, 1550), (1537, 1550)],
    7: [(256, 1280), (1010, 1280), (1025, 782), (1536, 782)],
    8: [(1536, 782), (2049, 782), (2049, 558), (2105, 526), (2529, 526), (2561, 302), (2617, 262), (3201, 262)],
    10: [(256, 1024), (513, 1024), (513, 302), (568, 270), (2553, 270)],
    11: [(256, 1536), (992, 1536), (1024, 1760), (1073, 1792), (3032, 1792), (3072, 1742), (3072, 1302)],
    13: [(256, 768), (1016, 768), (1025, 270), (1080, 262), (2016, 262), (2048, 312), (2048, 480), (2105, 512),
         (3329, 512)],
    14: [(256, 1536), (1009, 1536), (1025, 264)],
}
# stage -> (kind, selector, max HP) of the stage boss (slot 8)
BOSSES = {0: (0x81, 0x14, 16), 3: (0x80, 0x10, 16), 6: (0x82, 0x1F, 16), 9: (0x85, 0x33, 32),
          12: (0x84, 0x45, 28), 14: (0x88, 0x4F, 100)}
LOOKAHEAD = 400.0   # px of route beyond the best point that may be matched
MAX_OFF = 192.0     # camera further than this from the route -> progress not valid


def boot(n):
    # native canonical boot ((0,600),(START,1),(0,120),(START,1),(0,300)); stable_retro runs one frame
    # ahead of the native env, so 599 keeps both emulators frame-aligned (identical RAM through stage 7)
    for buttons, frames in (((), 599), (('START',), 1), ((), 120), (('START',), 1), ((), 300)):
        n.step(buttons, frames)
    for _ in range(600):
        r = n.ram()
        if r[2] == 0 and r[0x40] == 0 and r[0x6F0] == 16:
            return
        n.step((), 1)
    raise RuntimeError('Kage boot did not reach player control')


def _fix(r, lo, hi):
    return ((int(r[hi]) << 8) | int(r[lo])) / 16.0


def _segments(route):
    segs, acc = [], 0.0
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        segs.append((x0, y0, x1, y1, L, acc))
        acc += L
    return segs, acc


SEGMENTS = {s: _segments(r) for s, r in ROUTES.items()}


class Reader:
    def reset(self):
        self.stage = None
        self.best_arc = 0.0

    def route_frac(self, stage, cx, cy):
        segs, total = SEGMENTS[stage]
        best = None
        for x0, y0, x1, y1, L, acc in segs:
            if acc > self.best_arc + LOOKAHEAD:
                break
            t = 0.0 if L == 0 else max(0.0, min(1.0, ((cx - x0) * (x1 - x0) + (cy - y0) * (y1 - y0)) / (L * L)))
            d = math.hypot(x0 + t * (x1 - x0) - cx, y0 + t * (y1 - y0) - cy)
            if best is None or d < best[0]:
                best = (d, acc + t * L)
        if best[0] > MAX_OFF:
            return None
        self.best_arc = max(self.best_arc, best[1])
        return best[1] / total

    def observe(self, r):
        mode, stage, hp = int(r[2]), int(r[0x40]), int(r[0x6F0])
        if stage != self.stage:
            self.stage, self.best_arc = stage, 0.0
        cx, cy = _fix(r, 0x22, 0x23), _fix(r, 0x24, 0x25)
        play = mode == 0 and stage < 15
        progress, valid = 0.0, play and hp > 0 and int(r[0x6A1]) != 8
        # route share of the stage: all of it, or half when the stage also ends with a boss fight
        route_w = 0.0 if stage not in ROUTES else (500.0 if stage in BOSSES else 1000.0)
        if play and stage in ROUTES:
            f = self.route_frac(stage, cx, cy)
            if f is None:
                valid = False
            else:
                progress = route_w * f
        boss_dmg = None
        if play and stage in BOSSES:
            kind, sel, mx = BOSSES[stage]
            if int(r[0x708]) == kind and int(r[0x798]) == sel:
                boss_dmg = (mx - min(mx, int(r[0x7F8]))) / mx
                progress = route_w + (1000.0 - route_w) * boss_dmg
        won = (stage == 0 and mode == 0x6B and r[0xAE] == 0xFF and r[0x708] == 0x88 and r[0x798] == 0x4F
               and r[0x7F8] == 0 and r[0x758] == 0x10 and hp > 0)
        return Signals(
            in_play=bool(play and hp > 0 and r[0xD1] == 0),
            level=stage,
            progress=progress,
            progress_valid=bool(valid),
            dying=bool(hp == 0 and stage < 15),
            lives=None,
            health=hp / 16.0,
            score=None,
            game_over=bool(hp == 0 and mode != 0),
            won=bool(won),
            extra=dict(mode=mode, cam=(cx, cy), x=_fix(r, 0x610, 0x620), y=_fix(r, 0x630, 0x640),
                       pstate=int(r[0x6A1]), boss=boss_dmg, continues=int(r[0x44])),
        )
