"""Tecmo Bowl (U) (PRG 0), 1 PLAYER vs COM, team INDIANAPOLIS (cursor default), season mode.

Game structure (observed): pick a team, a roulette picks the first COM opponent ("FIRST WEEK"); 4 quarters of
1:30; a win shows "<TEAM>! WINS ... NEXT GAME" and the next week's opponent (beaten teams are crossed out); a loss
shows "... WINS / GAME OVER" with a password (the same opponent can be replayed). The 11th win (week index 10)
shows "VICTORY! WORLD CHAMPION" and the credits. Player 1 (IND) always attacks towards +x, in every quarter.

Addresses (found by RAM tracing, pokes and screenshots; no public RAM map exists for this game):
$003F-$0042 IND points per quarter (BCD), $0043 IND total (BCD);  $0044-$0047 COM per quarter, $0048 COM total
      (BCD; decoded totals equal the HUD "[IND-COM]" and the scoreboard/news screens)
$008D quarter index 0-3 (4 after regulation: end of game, or an extra "5 QTR" period)
$008B clock minutes, $008A clock seconds (BCD), 1:30 per quarter
$008E down - 1 (0 = 1st down)                    $008F games won this season (= week index, 0 = FIRST WEEK)
$00AB current opponent team index (0 IND, 1 MIA, 2 CLE, 3 DEN, 4 SEA, 5 LA, 6 WASH, 7 SF, 8 DAL, 9 NY, 10 CHI, 11 MINN)
$0049 play state: 0 = IND ball spotted / play-call / pre-snap, 1 = COM ball spotted; other values (bit flags)
      while the ball is live, on kickoffs and scoring plays
$00FB/$00FC line of scrimmage X (16 bit, field pixels, 12.8 px per yard = 128 px per 10 yards, the spacing of the
      field numbers); copied from the tackle spot $00FD/$00FE when $0049 returns to 0/1. GOAL_X (the IND
      goal line) is approximate, so extra.yard is debug only; the reward uses only yard differences
$002C/$002D screen graphics banks: (8,0) field, (9,9) offense play-call, (12,12) defense play-call,
      (8,24)/(8,25) TOUCHDOWN cut-ins, (10,x) scoreboard, (30|31,0) news screen, (20,20) title, (11,20) team select,
      (8,20) "FIRST WEEK", others (26-29, 15, 22) cut-ins and the ending

Signals:
  level = $008F (games won). The level event fires at the next game's kickoff (first in-play step).
  progress = 100/7 * IND points + yards gained in the current IND drive (high-water of the line of scrimmage
      on IND's offensive play-call screen (9,9) minus the drive's first such spot, in yards), so a touchdown
      drive is worth ~100 = PROGRESS_UNIT and the yards on the way are credited play by play. The drive restarts
      when the COM gets the ball or IND scores. Only play-call spots count because $0049 also passes through 0
      for a few frames while the LOS moves to the COM kickoff spot after a COM score (a fake 24-58 yd "gain"
      in the first random audit); the LOS never changed within 320 sampled play-call screens.
  dying = the COM total rises by >= 2 in one step (TD 6, FG 3, safety 2; the PAT +1 is part of the TD event);
      DEATH_FROM 'dying' (rising edge): every COM scoring play costs -1.
  in_play = field or play-call screen (the play-call screen is a real decision) before the end of regulation;
      cut-ins, scoreboard, news, menus, pre-game screens and the post-game field banners ($008D = 4 with the
      clock at 0:00, e.g. "SECOND WEEK" over the field bank) are not in play.
  game_over = after regulation ($008D >= 4) with the clock at 0:00 and COM ahead (the loss / GAME OVER screen).
  won = the same moment with IND ahead in a game played as week index 10 (11th win -> WORLD CHAMPION;
      poke-assisted only). The week is latched while in play because $008F already counts the win on the
      screens after a game whose final score is still in RAM (a week-9 win must not read as the title).
  score = IND points (tiebreaker, SCORE_UNIT 7).

Reward-hacking risks:
- yards are a per-drive high-water mark from the drive's first spot: losing yards and regaining them pays nothing,
  and after a failed drive only gains beyond the best drive since the last IND score pay again (the tracker's
  high-water mark). Gaining yards and then turning the ball over is not punished by itself (only the COM scores
  that follow are).
- a field goal (3 points = 42.9) can be worth less than the field position already credited for that drive, so
  it may pay nothing at once; the score tiebreaker still rewards it slightly.
- kick/punt/interception return yards are not credited (drives start at the first spot after the return).
- every COM score costs the same -1 (TD and FG alike); stalling the clock is not rewarded.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nesram import bcd  # noqa: E402
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Tecmo Bowl (U) (PRG 0).nes')
ROM_SHA256 = '48ea5218ee75db7262cf3d50659d2e76e00578b47d149c96b66540c232da44ad'
# play-call screen: direction + A picks RUN 1 (UP), RUN 2 (LEFT), PASS 1 (RIGHT), PASS 2 (DOWN) (same on defense);
# A snaps the ball; directions move the controlled player; B passes / dives (button roles depend on the play)
ACTIONS = [('UP', 'A'), ('LEFT', 'A'), ('RIGHT', 'A'), ('DOWN', 'A'), ('A',), ('B',),
           ('RIGHT',), ('UP', 'RIGHT'), ('DOWN', 'RIGHT'), ('UP',), ('DOWN',), ('LEFT',), ()]
PX_PER_YARD = 12.8
GOAL_X = 1456          # approximate IND goal line X (PAT spots ~2 yards out at X 1483); debug only
PROGRESS_UNIT = 100.0  # one touchdown drive (7 points = 100)
SEARCH_BUCKETS = 50    # search cells 2 progress units (2 yards) wide
SCORE_UNIT = 7.0
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Tecmo Bowl',
    genre='American football',
    level='games won this season (week 0-10)',
    levels=11,
    progress='own points x 100/7 + yards of the current drive past the best drive since the last own score',
    death='the opponent scores (TD, FG or safety)',
    game_over='losing a game',
    health=None,
    score='own points',
    win='winning the 11th game (WORLD CHAMPION)',
    win_status='provisional',
    limits=[
        'defense and failed drives earn nothing, so quiet stretches are long: STALL_SECONDS is 300',
    ],
)
STALL_SECONDS = 300.0  # defense and failed drives earn nothing: quiet stretches up to 73 s were measured in 27 min
SEARCH_ALLOW_DEATHS = True   # search only: COM scores happen in won games too; losing still ends the run
LAST_WEEK = 10         # 11th game: a win shows WORLD CHAMPION
PLAY_SCREENS = ((8, 0), (9, 9), (12, 12))


def boot(n):
    """Title -> 1 PLAYER -> INDIANAPOLIS (A) -> opponent roulette (A) -> FIRST WEEK -> scoreboard -> kickoff."""
    n.step((), 120)
    for _ in range(400):
        r = n.ram()
        if (r[0x2C], r[0x2D]) == (8, 0) and r[0x8D] == 0 and r[0x8B] == 1 and r[0x8A] == 0x30:
            return
        if (r[0x2C], r[0x2D]) == (20, 20):   # title / menu (1 PLAYER is the default): START
            n.tap('START', 4, 26)
        elif r[0x2D] == 20:                  # team select (IND default), roulette, week banner, scoreboard: A
            n.tap('A', 4, 26)
        else:
            n.step((), 1)
    raise RuntimeError('Tecmo Bowl boot did not reach the opening kickoff')


class Reader:
    def reset(self):
        self.own = None
        self.opp = None
        self.drive_start = None
        self.drive_best = None
        self.week = None      # week index of the game being played (latched while in play)

    def observe(self, r):
        own, opp = bcd(r, (0x43,)), bcd(r, (0x48,))
        screen = (int(r[0x2C]), int(r[0x2D]))
        ended = r[0x8D] >= 4 and r[0x8B] == 0 and r[0x8A] == 0
        # after regulation the field bank stays up under the "CHANGE SIDES" / "NEXT WEEK" banners: not play
        playing = screen in PLAY_SCREENS and r[0x8D] <= 4 and not ended
        if playing:
            self.week = int(r[0x8F])
        state = int(r[0x49])
        los = int(r[0xFB]) + 256 * int(r[0xFC])
        yard = (los - GOAL_X) / PX_PER_YARD
        scored = self.opp is not None and opp - self.opp >= 2
        if self.own is not None and own != self.own:
            self.drive_start = self.drive_best = None          # IND scored: the drive is over
        if playing and state == 1:
            self.drive_start = self.drive_best = None          # COM ball
        elif playing and state == 0 and screen == (9, 9):     # IND offensive play-call: a genuine spot
            if self.drive_start is None:
                self.drive_start = self.drive_best = yard
            self.drive_best = max(self.drive_best, yard)
        self.own, self.opp = own, opp
        gain = 0.0 if self.drive_start is None else max(0.0, self.drive_best - self.drive_start)
        return Signals(
            in_play=bool(playing),
            level=int(r[0x8F]),
            progress=100.0 / 7.0 * own + gain,
            progress_valid=bool(playing),
            dying=bool(scored),
            score=own,
            game_over=bool(ended and opp > own),
            won=bool(ended and own > opp and self.week == LAST_WEEK),
            extra=dict(own=own, opp=opp, quarter=int(r[0x8D]) + 1, clock='%x:%02x' % (r[0x8B], r[0x8A]),
                       down=int(r[0x8E]) + 1, state=state, screen=screen, los=los, yard=round(yard, 1),
                       drive_start=None if self.drive_start is None else round(self.drive_start, 1),
                       gain=round(gain, 1), week=int(r[0x8F]), opponent=int(r[0xAB])),
        )


def search_key(s):
    """Search only: possession phase, so COM-ball states are archived separately from IND drives."""
    return (s.extra or {}).get('state') if (s.extra or {}).get('state') in (0, 1) else None
