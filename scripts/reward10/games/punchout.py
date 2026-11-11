"""Mike Tyson's Punch-Out!! (U) (PRG 0), new career from the title menu (not a password).

Addresses found by RAM tracing, pokes and screenshots. stable-retro's 'heart' (>n4 at $0321)
spans $0321-$0324, but $0321-$0322 hold a transient value for one frame on every change, so only
$0323-$0324 are read; health_mac $0391, health_com $0398 and score $03E8 agree with stable-retro.
$0000 fight state: 1 fight running (also during knockdowns and the decision announcement),
      64 bell/round over, 255 corner between rounds (waits for START), 48 next-round/fight setup,
      fight lost: 249 by decision, 251 by TKO (3 knockdowns), 253 by KO (all three observed);
      fight won: 252 by TKO (poke-assisted run), 254 by KO (unassisted run, "YOU WON BY KO" screen);
      250 assumed for a decision win (unobserved)
$0001 fight index: 0 Glass Joe, 1 Von Kaiser (fight cards), 2 the minor-circuit TITLE BOUT
      (reached by search + a START tail), 3 the first fight after the title montage (poke-assisted
      only); increments during the post-win sequence. 13 would be Mike Tyson (14th fight;
      unverified, not reached).
$0005 knockdown/scene: 0 both standing, 1 opponent down, 2 Little Mac down, 3 between rounds
$0006 round number (1-3; 4 during the judges' decision announcement)
$0300 1 while the round clock runs (0 during the walk-in before "FIGHT!" and between rounds)
$0302 clock minutes (0-3)
$0391 Little Mac health (96 full; 0 = knocked down)
$0398 opponent health (96 full; refills, often partially, when he gets up and between rounds)
$03D1 opponent knockdowns in this fight (cumulative over rounds, seen 1..5; audit only)
$008F Little Mac knockdowns this round (3 = TKO loss); $03D0 Little Mac knockdowns in the fight
      (audit only; checked over two rounds: 1 in round 1, then 2,3,4 in round 2)
$0323-$0324 hearts, one decimal digit per byte (20 at the first fight)
$03E8-$03ED points, one decimal digit per byte, most significant first (matches HUD POINTS;
      reset to 0 for every fight, only increases are rewarded)
$0170-$0175 Mac's record, two digits each: $0170-1 wins, $0172-3 losses, $0174-5 KO wins

Signals: level = $0001. progress = cumulative damage dealt to the current opponent (sum of every
decrease of $0398 while $0000 == 1; refills are ignored; reset when $0001 changes);
PROGRESS_UNIT = 288 = three full health bars = the damage of a 3-knockdown TKO when the opponent
gets up with a full bar each time (refills seen in search replays: 80, 48, 96, 96, so a real
first-opponent fight is roughly 1-2 units).
dying = Little Mac down ($0005 == 2, rising edge). health = $0391 / 96. game_over = fight lost
($0000 in 249/251/253, backed up by the losses record becoming non-zero ~80 frames later).
won = fight won ($0000 in 250/252/254) while $0001 == 13 (Tyson): UNVERIFIED, never reached.
The level event fires when the next opponent's round clock starts: 110-180 decisions after the
win state (result screen, fight card, walk-in; the cards need START presses).
in_play = $0000 == 1, $0300 == 1 and $0005 in {0, 1} (round clock flag set, Mac standing; the
opponent's knockdown count counts as play so the knockdown punch's damage is credited at once;
with only $0005 == 0 a KO punch was never credited). The corner screens between rounds need START (star punch button in the ring).

Reward-hacking risks:
- progress is damage, not knockdowns: chipping damage on an opponent that recovers between
  rounds keeps paying (the high-water mark only grows), so a long 3-round decision loss can
  out-earn a quick KO of the same opponent. Bounded by the fight length; the loss ends the episode.
- points (score tiebreaker) come from landed punches and knockdowns, aligned with progress.
- hearts (stamina) and stars are not rewarded; running hearts to 0 (Mac turns pink and cannot
  punch) is only punished indirectly.
- Mac's health increases (getting up, between rounds) are never rewarded; only decreases cost.
Unverified: decision outcomes other than the one observed, Mike Tyson / game end, career end
after repeated losses (the episode already ends at the first lost fight).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesram import digits  # noqa: E402
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / "Mike Tyson's Punch-Out!! (U) (PRG 0).nes")
ROM_SHA256 = '0435f6d9bcfeb474a592c3a950530a99c03a6f6cacff793dc88f822bb8cc6952'
# A right punch, B left punch, UP+A/B to the head, LEFT/RIGHT dodge, DOWN duck/block,
# START star punch in the ring and "next round" in the corner
ACTIONS = [('A',), ('B',), ('UP', 'A'), ('UP', 'B'), ('LEFT',), ('RIGHT',), ('DOWN',), (), ('START',)]
FULL_HEALTH = 96
PROGRESS_UNIT = 3.0 * FULL_HEALTH
SCORE_UNIT = 100.0    # a landed punch is 10-ish, a knockdown 1000
DEATH_FROM = 'dying'
START_SELECT = {'START': 'star punch in the ring (needs a star); next round in the corner',
                'SELECT': 'in the corner between rounds: rest to recover some health'}

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title="Mike Tyson's Punch-Out!!",
    genre='boxing',
    level='opponent: fight 0 Glass Joe .. 13 Mike Tyson',
    levels=14,
    progress='damage dealt to the current opponent (sum of his health decreases)',
    death='Little Mac knocked down',
    game_over='losing a fight (KO, TKO or decision)',
    health='Little Mac health bar',
    score='points (reset every fight)',
    win='beating Mike Tyson (fight 14)',
    win_status='unverified',
    limits=[
        'chip damage keeps paying, so a long decision loss can out-earn a quick KO (bounded by the fight)',
    ],
)
SEARCH_ALLOW_DEATHS = True   # search only: a knockdown is part of a fight, losing it still ends the run
LAST_FIGHT = 13       # Mike Tyson (14th fight), unverified
LOSS_STATES = (249, 251, 253)   # decision, TKO, KO (all observed)
WIN_STATES = (250, 252, 254)    # 252 TKO and 254 KO observed; 250 assumed


def boot(n):
    """Title -> NEW (default) -> 'PUSH START' fight card -> walk-in -> round clock running."""
    n.step((), 300)
    for _ in range(200):
        r = n.ram()
        if r[0x00] == 1 and r[0x300] == 1 and r[0x05] == 0 and r[0x391] == FULL_HEALTH:
            return
        if r[0x00] != 1:            # menus and fight card: keep pressing START
            n.tap('START', 4, 26)
        else:                       # in the ring before the bell: wait
            n.step((), 30)
    raise RuntimeError('Punch-Out boot did not reach the first round')


class Reader:
    def reset(self):
        self.fight = None
        self.damage = 0
        self.prev_hp = None

    def observe(self, r):
        state, scene, fight = int(r[0x00]), int(r[0x05]), int(r[0x01])
        hp = int(r[0x398])
        if fight != self.fight:
            self.fight, self.damage, self.prev_hp = fight, 0, None
        if state == 1 and self.prev_hp is not None and hp < self.prev_hp:
            self.damage += self.prev_hp - hp
        self.prev_hp = hp
        # the opponent's knockdown count (scene 1) stays in play so the knockdown punch is credited
        # at once (a KO ends the fight before he would get up again)
        playing = state == 1 and scene in (0, 1) and r[0x300] == 1
        return Signals(
            in_play=bool(playing),
            level=fight,
            # capped at one unit per opponent: a long fight (or decision loss) cannot out-earn a KO
            progress=float(min(self.damage, PROGRESS_UNIT)),
            progress_valid=bool(playing),
            # Mac knocked down, or the fight lost without being down (decision/TKO) also costs a death
            dying=(state == 1 and scene == 2) or (state in LOSS_STATES and scene != 2),
            health=min(1.0, int(r[0x391]) / FULL_HEALTH),
            score=digits(r, range(0x3E8, 0x3EE)),
            game_over=state in LOSS_STATES or int(r[0x172]) * 10 + int(r[0x173]) > 0,
            won=state in WIN_STATES and fight == LAST_FIGHT,
            extra=dict(state=state, scene=scene, round=int(r[0x06]), opp_hp=hp, mac_hp=int(r[0x391]),
                       hearts=int(r[0x323]) * 10 + int(r[0x324]), opp_kd=int(r[0x3D1]), mac_kd=int(r[0x8F]),
                       mac_kd_total=int(r[0x3D0])),
        )
