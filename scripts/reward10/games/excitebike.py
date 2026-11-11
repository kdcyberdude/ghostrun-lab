"""Excitebike (JU), Selection A (solo), challenge race on track 1 (menu defaults).

Game structure (observed): challenge race on the chosen track -> if the finish time beats the 3RD time
shown bottom-left, the main "EXCITEBIKE" race on the same track -> each qualifying main race leads to
the next track's main race -> after main race 5 the game repeats main race 5 forever (HUD "T= n").
Not qualifying is the game over: after a challenge race the CHALLENGE RACE track menu returns; after a
main race the results screen shows "GAME OVER" (ranking > 3) and the game returns to the title screen,
which then runs the attract demo (a race on track 1).

Addresses (found by RAM tracing, pokes and screenshots; the Data Crystal map is for another revision,
e.g. its temperature $03E3 is $03B6 here; its speed $00F3 and status $00F2 agree):
$0041 sequence state: 2/5 race screen, 3/6 results, 4/7 "EXCITEBIKE / TRACK n" title, 1 = back at the
      CHALLENGE RACE menu after a failed challenge race, 0 = title screen / attract demo (after a failed
      main race; the demo race then sets $0047/$004F like a real race); 255 from power-on until the first
      race loads
$0042 menu cursor (track 0-4), $0043 current track 0-4, $0046 0 = challenge race, 1 = main race
$0047 1 on a race screen (track loaded), 0 on results/title screens
$004F 1 once the start light turns (race clock started); stays 1 after the finish until the next race loads
$0052 1 after crossing the finish line (bike then rides on automatically)
$0057 laps remaining after the current lap (2 laps per race)
$0068/$0069/$006A race time minutes/seconds/hundredths (binary; the HUD TIME shows it 1-2 frames later)
$03B1 1 while paused (START)
$0012 + 256*$004D track camera X (9 bits, wraps); its per-frame change equals the image shift of the
      track one frame later (checked against the screen for 2500 frames, 0 mismatches)
$00F2 bike status: 0 on the ground, 1 crash (rider thrown), 2 airborne / riding on top of a ramp,
      4 landing; $009C recovery phase: 1..5 after a crash (thrown, lands, runs back, remounts),
      only 5 (remount) after an OVER HEAT, 0 otherwise
$00F3 speed (0..47 with A, 48 with B)
$03B6 engine temperature: 6 at the start, ~17 holding A, 32 = OVER HEAT (bike stops ~4 s, then resets to 5)
$03F3 main-race-5 completion counter ("T=" on the HUD): 1 for a fresh game, +1 each time main race 5
      is qualified (seen 1->2->3->4->5 in a poke-assisted run)

Signals:
  level = $0043 + $0046 (challenge track 1 = 0, main race k = k, main race 5 = 5)
  progress = camera pixels integrated since the start of the current race (signed per-step change of
      the 9-bit camera; only while racing), so the 2 laps of a race are one continuous distance.
      PROGRESS_UNIT = the track-1 challenge race distance (both laps); later tracks are longer.
  in_play = race screen, clock running, not finished, not paused
  dying = crash ($00F2 == 1 or $009C in 1..4) while racing; DEATH_FROM 'dying' (rising edge)
  health = 0.0 during OVER HEAT ($03B6 >= 32) else 1.0 -> each overheat costs W_DAMAGE once
  game_over = $0041 in (0, 1) (failed to qualify: menu or title shown again; never seen in a qualifying run,
      and it ends the episode before the attract demo could look like play)
  won = $03F3 >= 2 (main race 5 qualified = all five tracks cleared; poke-assisted observation only)
  score = None: Selection A shows no score, only the race time (extra.time); a time "score" would
      reward slowness.

Reward-hacking risks:
- progress is a per-race high-water mark of camera distance: riding back and forth cannot farm it,
  and the camera never scrolls backwards. Distance does not depend on time, so a slow but crash-free
  race earns the same progress as a fast one; only failing to qualify (game over) punishes slowness.
- crashes cost -1 each; overheating costs -0.25 each; turbo (B) itself is free.
- pausing gives nothing (in_play False) and stops the bike (speed 0) on resume.
- the post-finish auto ride and the main-race-5 loop are outside in_play; the episode ends on won.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import Signals  # noqa: E402

ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Excitebike (JU).nes')
ROM_SHA256 = '372654d4c62d214d92200288611ff176ae1727949c6f012e2fc95bf2452a33d7'
# A throttle, B turbo (heats the engine), UP/DOWN change lane, LEFT/RIGHT lean back/forward (landings)
ACTIONS = [('A',), ('B',), ('A', 'LEFT'), ('A', 'RIGHT'), ('B', 'LEFT'), ('B', 'RIGHT'),
           ('A', 'UP'), ('A', 'DOWN'), ('B', 'UP'), ('B', 'DOWN'), ('LEFT',), ('RIGHT',), ()]
PROGRESS_UNIT = 11000.0      # camera px of the track-1 challenge race (2 laps), see summary.json
SEARCH_BUCKETS = 150         # search cells ~73 px wide (search only)
SCORE_UNIT = 1.0             # unused (no score)
DEATH_FROM = 'dying'

# contract.py INFO: what each signal means in this game (reward_spec.json is built from it)
INFO = dict(
    title='Excitebike',
    genre='racing',
    level='race: 0 = track-1 challenge race, 1-5 = main races 1-5',
    levels=6,
    progress='track distance (camera pixels) in the current race',
    death='crash (rider thrown off the bike)',
    game_over='not qualifying (time slower than 3rd place)',
    health='engine temperature: each OVER HEAT costs one full bar',
    score=None,
    win='main race 5 qualified (completion counter $03F3)',
    win_status='provisional',
    limits=[
        ('distance ignores time: nothing rewards racing faster, though qualifying needs a fast time '
         '(the search finished main race 5 in 1:49.95; 1:14.00 was needed)'),
    ],
)
OVERHEAT = 32


def boot(n):
    """Title (SELECTION A default) -> CHALLENGE RACE menu (track 1 default) -> start light turns."""
    n.step((), 60)
    n.tap('START', 4, 4)          # SELECTION A
    n.step((), 100)               # menu draws
    n.tap('START', 4, 4)          # TRACK 1
    for _ in range(1200):
        r = n.ram()
        if r[0x47] == 1 and r[0x4F] == 1 and r[0x52] == 0:
            if r[0x43] != 0 or r[0x46] != 0:
                raise RuntimeError('Excitebike boot did not start the track-1 challenge race')
            return
        n.step((), 1)
    raise RuntimeError('Excitebike boot did not reach the race start')


class Reader:
    def reset(self):
        self.key = None       # (level, race-clock-started) identity of the current race
        self.cam = None       # last 9-bit camera value while racing
        self.dist = 0.0

    def observe(self, r):
        level = int(r[0x43]) + int(r[0x46])
        racing = r[0x47] == 1 and r[0x4F] == 1 and r[0x52] == 0
        cam = int(r[0x12]) + 256 * (int(r[0x4D]) & 1)
        key = (level, int(r[0x4F]))
        if key != self.key:                 # new race (or pre-start): restart the distance
            self.key, self.cam, self.dist = key, None, 0.0
        if racing:
            if self.cam is not None:
                self.dist += (cam - self.cam + 256) % 512 - 256
            self.cam = cam
        else:
            self.cam = None
        # crash: rider thrown ($F2 == 1) and running back ($9C 1-4); phase 5 (remount) also follows OVER HEAT
        crash = racing and (r[0xF2] == 1 or 1 <= r[0x9C] <= 4)
        temp = int(r[0x3B6])
        return Signals(
            in_play=bool(racing and r[0x3B1] == 0),
            level=level,
            progress=float(self.dist),
            progress_valid=bool(racing),
            dying=bool(crash),
            health=0.0 if temp >= OVERHEAT else 1.0,
            score=None,
            game_over=bool(r[0x41] in (0, 1)),
            won=bool(int(r[0x3F3]) >= 2),
            extra=dict(state=int(r[0x41]), track=int(r[0x43]), main=int(r[0x46]), laps_left=int(r[0x57]),
                       finished=int(r[0x52]), status=int(r[0xF2]), crash_phase=int(r[0x9C]),
                       speed=int(r[0xF3]), temp=temp, paused=int(r[0x3B1]), loop=int(r[0x3F3]),
                       time='%d:%02d:%02d' % (r[0x68], r[0x69], r[0x6A])),
        )
