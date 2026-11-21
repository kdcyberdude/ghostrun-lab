# reward10: verified RAM rewards for the 12 RL experiment games

ROMs are never included: set `NES_ROM_DIR` to a folder holding your own files.
Python: any interpreter with `stable_retro`, `numpy` and `Pillow`.
CPU only. One live emulator per process (`nesram.run_isolated` spawns workers).

## What the reward optimizes (shared, `reward.py`)
Identical for every game; a game module only translates RAM into `Signals`.
- `+1.0` per level of new within-level progress (high-water mark, `PROGRESS_UNIT` = one level)
- `+1.0` for reaching the next level/stage/opponent/track (only +1 steps; skipping levels, e.g. a
  warp zone, ends the episode with no bonus, so every level must be played in order)
- `-0.1` per death, `-0.25` per full health bar lost
- `-0.02` per abrupt loss of progress (at least a quarter of `PROGRESS_UNIT` in one same-level update while
  not dying); ordinary small backward moves are free. This discourages loops such as SMB 4-4's maze
- a small waiting cost while in play: `W_IDLE` (`-1.8`) spread over the game's stall limit
- `+0.02` per `SCORE_UNIT` of score, capped at `0.1` per step and `0.2` per level (tiebreaker only;
  the level budget resets on reaching a new level, not on death, so score farming cannot outweigh progress)
- `+5.0` for finishing the game; the episode ends on game over or win
- No-progress limit: the episode also ends, with a `-0.1` penalty, after `120 s` of game time (in play or not,
  so pausing counts) without new progress, a new level or a death; a module may set `STALL_SECONDS`
  (Tecmo Bowl 300 s, where defense earns nothing; longest quiet stretch on the other saved runs 51 s)
- No progress or score reward outside real gameplay (`in_play` False on title/menus/demo/pause);
  terminal rewards and carried guide potential are handled separately
- Optional shaping potential `Signals.guide` (only SMB's maze castles 4-4, 7-4, 8-4 set it): pay
  `guide_discount * next - previous`, with zero potential at episode end or after leaving the maze level.
  Hold the last potential during invalid positions and apply the discount correction on every update.
  `RewardTracker` and `RlEnv` default to discount 1 for reward-only search; the PPO trainer passes its `gamma`.
  Using an undiscounted difference with discounted PPO makes late loop repayments too cheap; see `games/smb.py` MAZE_GUIDE.
  In 4-4, completed segments follow the ROM's renderer page and loop-command flag, keeping the required-height
  signal until the check is accepted. Crossing a nominal x alone does not complete a segment.
RAM is reward/audit data only. It must never be an actor input.

## Game module contract (`contract.py`; `games/smb.py` is the reference module)
Every game has the same shape, whatever the genre. `python contract.py` checks all modules and regenerates
`reward_spec.json`: the exact reward of every game as JSON (values, units, what each term means, how the
episode ends, known limits). Never edit it by hand; `tests/test_reward10.py` fails when it is stale.
- `INFO`: title, genre, level (meaning), levels (count, None = endless), progress, death, game_over,
  health, score (None = absent), win, win_status (verified/provisional/unverified/none), limits
- optional `STALL_SECONDS`: own no-progress limit (None = no limit)
- optional `START_SELECT`: `{'START' | 'SELECT': what it does in play}`, the only Start/Select presses a
  trainer passes on after boot (default none: Start pauses most games, and pausing right before a death
  dodges the -1; seen in SMB, Gradius, Castlevania, Final Mission). Punch-Out!! lists both, Zelda lists Start
- `ROM`, `ROM_SHA256`, `ACTIONS` (button tuples held one decision, or macros: lists of
  `(buttons, frames)`, for search/audit only), `PROGRESS_UNIT`, `SCORE_UNIT`
- optional `search_key(signals)`: hashable game phase added to the search cell (search only)
- optional `SEARCH_BUCKETS`: search cells per `PROGRESS_UNIT` (default 60), so cell width is not tied to
  the reward scale (search only)
- `DEATH_FROM` = `'dying'` (rising edge of `Signals.dying`) or `'lives'` (lives decrement)
- `boot(nes)`: power-on to the first controllable frame of a standard fresh 1-player game
- `class Reader`: `reset()`, `observe(ram) -> Signals` (may keep per-episode state, e.g. visited rooms)
- `Signals`: `in_play, level, progress, progress_valid, dying, lives, health (0..1), score, game_over, won, extra`
- Module docstring lists every address with its meaning.

## Trainer API (identical for every game)
```python
g = contract.load('smb')                                  # import + check the module
g.boot(emu)                                               # emu: step(buttons, frames), tap(button, hold, after), ram()
pad = contract.play_buttons(g)                            # after boot, drop every pressed button not in pad
tr = RewardTracker(g)                                     # one per episode; tr.reset() to reuse
reward, done, events = tr.update(emu.ram(), t, frames=8)  # after every decision
tr.end_reason                                             # None, or 'game_over' | 'won' | 'skip' | 'stall'
```
All four end reasons are terminal (do not bootstrap past them); a rollout cut by the trainer is a truncation.
`ACTIONS` are for search/audit only, not the trainer's action space.

### Opt-in world-grid experiment (SMB, 2026-11-13)

`rl/train.py --grid-exploration 1 --gamma 1 --moods 8 --div-coef 0 --places 0` replaces the training reward
with +10 at native completion, -0.5 at the first death, -0.0001 per decision, and bounded cell novelty.
An attempt ends at first death, completion, or `--attempt-limit` (default 4000); the old horizontal-progress,
score, guide, rollback and stall rewards do not enter the objective. Rollout cutoffs still bootstrap.

Cells are 32x32 world pixels, keyed by level, area/type and native maze phase (4-4 accepted checks, 7-4 check
counters). The 0x80 area-pointer alias is normalized. Invalid/death/transition positions and the initial entrance
do not pay. RAM/cell IDs remain reward/debug data, never actor inputs. This first adapter supports SMB only;
other games still use their existing reward until an appropriate cell adapter is validated.

`rl/grid_reward.py` pays each cell once per attempt: `0.005 / sqrt(1 + prior_attempt_visits / history_scale)`, at most 1.0 total
before mood weighting. Familiarity counts update once per rollout (all parallel actors share the same prior
counts). Until a level reaches a recent native clear rate of at least 50% with at least 100 completed attempts,
all moods multiply discovery by 1. Only after that gate, first-quarter moods multiply discovery by 0,
next-quarter by .25, remaining-half by 1. The gate reverts to exploration if the rate falls below 50%; it uses
native outcomes, not the curriculum's 50% prior for unknown levels. The diversity
judge reward is disabled in this first experiment. Visits, remaining budget and actual episode returns carry
across rollouts and checkpoints. A native castle axe is completion; training/evaluation must start at the entrance.
`--grid-history-scale` defaults to 1 for reproducibility of the initial trial; the preparation script sets it to
the environment count, so increasing parallelism does not extinguish curiosity before comparable policy updates.
Staying in the same valid 32-pixel horizontal world-grid column for over 30 decisions (two seconds) costs an additional
.001 per decision, capped at .25 per attempt. Moving left or right resets the clock; jumping in place does not.
This stationary rule is specific to the current horizontal SMB adapter. Death/completion/limit resets
the clock and penalty budget. These counters and the per-level success gate are saved in checkpoints.

`rl/scripts/prepare_grid_run.py SOURCE RUN LEVEL` retains the source policy and its entrance archive, with fresh
optimizer/value estimates/episode state for the changed objective. Training starts with a bounded single-level
trial; no full-level claim is based on saved-place starts or a single clear.

### Regional exploration comparison

`--grid-regions 1` changes only the novelty term: first fine-cell visits pay `.001 / sqrt(1 + visits / history_scale)`;
first 128x32 world-region visits pay `.02 / (1 + region_visits / history_scale)`. The two terms share the same
1.0-per-attempt cap and mastery/mood gate. World coordinates, area and native maze phase remain in the key.
Regions preserve the fine vertical bands so the middle and lower corridors do not merge. Repeated boundary
crossings and maze loops cannot repay already-visited regions within an attempt. All histories survive resume.

`prepare_grid_run.py SOURCE RUN 15 --regions` starts from SOURCE's actor weights and prior fine-cell familiarity,
with new entrance episodes, optimizer and critic. Initial region counts use the maximum fine-cell count in each
region (a lower bound on historical region visits, since historical per-attempt unions were not retained).
The comparison is capped at 38 iterations = 9,961,472 decisions, just under the requested 10 million.
Metrics additionally report `new_region_visits`, `known_regions`, `regional_reward`, `maze_check1_entries` and
`maze_check2_entries`; the last two count recorded 4-4 phase transitions, not frames spent after a check.

### Scratch initialization comparison

`prepare_grid_run.py ENTRANCES.npy RUN 15 --regions --scratch` never reads a trained-model checkpoint. It creates
the same 512-core, 8-mood architecture with seed 1000 and Policy's standard random initialization, an empty
optimizer, zero return/episode statistics, and empty fine/regional histories. Only the emulator entrance archive
is reused. The trial budget is 76 iterations = 19,922,944 new decisions; checkpoint/video cadence stays 15 minutes
and completion. `source.json` records `initialization: random`, its seed, and the entrance archive path.

### Persistent coverage comparison

`prepare_grid_run.py ENTRANCES.npy RUN 15 --scratch --coverage` and training with
`--grid-exploration 1 --grid-coverage 1 --grid-regions 0` select the revised reward. The older variants above
remain selectable for comparisons. This trial starts from the same seed-1000 random policy, with empty histories,
and runs 76 iterations (19,922,944 decisions). No route demonstrations or maze height targets are supplied.

For each first visit to a 32x32 cell in the current attempt, add
`0.02 * (1 + 1/sqrt(1 + prior_attempt_visits/history_scale))` to the attempt's coverage mass `s`.
Pay the increase in `F(s) = 4*s/(4+s)`. Thus familiarity only reduces an extra rarity bonus; it cannot remove
the baseline incentive to explore again. Positive bonuses sum to less than 4 per attempt, yet further cells
still pay after the old hard cap would have stopped them. A visit after moving left, right, up or down is
treated identically. Initial cells, invalid cells, terminal reset observations and same-attempt revisits pay zero.
This separates episodic exploration from bounded lifetime novelty, as in the intrinsic-reward construction of
[Never Give Up](https://arxiv.org/pdf/2002.06038); this is a simpler exact-grid
implementation, not the NGU agent or a claim of its benchmark performance.

Each decision still costs .0001. After 60 decisions (four seconds) without a new cell in the attempt, every
further decision costs an additional .001 until discovery resumes. This cost does not run out. Repeated jumps,
boundary crossings, whole loops and invalid observations cannot reset the clock; finding a new cell on either
axis can. Short backtracking has a grace period; longer revisits remain possible but cost time. A first death
costs .5; the Python coverage layer also charges .5 at the attempt limit so timing out cannot avoid that cost.
Native completion pays 10. At the 4000-decision limit, even a win with no novelty and maximum time/stagnation
costs returns at least 5.6, above every failed attempt's maximum exploration bonus of 4. Coverage mode accepts
limits up to 5000 to preserve that ordering. Failure may still be better than endless stalling; death rate and
coverage must be monitored together to detect repeated short attempts. No reward guarantees discovery of a
long action sequence, and this adapter has only been validated for SMB.

Gamma remains 1; all moods retain full exploration until the measured 50% mastery gate. Coverage mass, visited
cells and the no-discovery clock survive rollout/checkpoint boundaries and reset on native attempt endings.
New metrics `stagnation_penalty`, `stagnant_fraction`, and `no_discovery_steps_mean` report wasted exploration
time; judge learning by new coverage, native maze-check entries and entrance clears, not reward mean alone.

### 8x8 exploration grid

`--grid-cell-size 8` selects 8x8 cells for coverage mode. The native CPU and CUDA adapters emit 34-bit keys:
level(5), room/type(9), native maze phase(4), x cell(11), y cell(5). The 32px mode retains its original keys.
One tracked Mario position selects one cell; positions are sampled once per four-frame decision, so cells
skipped between samples are not credited. Camera scrolling and maze loops do not create duplicate places.

The 8px raw first-visit mass is `.005 * (1 + 1/sqrt(1 + visits/history_scale))`, one quarter of the 32px value.
Four adjacent new 8px cells therefore yield the same mass as one new 32px cell. This normalizes straight-line
coverage while preserving the finer positions; it does not claim identical rewards for every 2D trajectory.
The smooth +4 exploration bound, +10 completion, failure/time costs and four-second stagnation grace stay the same.

To change the current run's resolution without discarding its learning:
`prepare_grid_run.py CHECKPOINT NEW_RUN 15 --coverage --cell-size 8 --regrid` preserves all model parameters,
optimizer state, return statistics and total decision count. Attempts restart at the entrance with empty
exploration memories because a visited 32px parent does not identify its visited 8px children. Directly resuming
a checkpoint with a different cell size is rejected. The new run continues the original decision budget.

## Tools

`--grid-persistent-visits 1` makes the existing shared lifetime history authoritative: only cells absent from
that history at the beginning of a rollout can earn bonuses. All parallel workers use the same frozen prior
for that rollout; each worker can collect a new cell at most once even across deaths within that rollout. At the next rollout it is familiar
for every worker. Level, room and maze phase remain part of the key. Death resets the attempt budget and
inactivity clock but never clears lifetime history. Enabling this on an existing checkpoint immediately suppresses
bonuses for its already-known cells; no historical visits are discarded. This stricter experiment deliberately
removes the episodic reward floor and makes discovery rewards sparser. `grid.new_cells` counts paid first visits
across workers; `grid.known_cells` is the shared unique-cell count. Neither measures reliable level completion.

The continued 8px trial uses `--grid-stagnation-quadratic 1`: after 8 decisions without a new cell, the additional
cost per decision is `.001 * min(1, (max(0, age-8)/8)^2)`. At ages 8/12/16 it is 0/.00025/.001; the .001 rate
continues thereafter without an episode penalty cap. Discovery or an attempt ending resets the clock; revisits
do not. The maximum cost rate is unchanged, preserving the existing completion-versus-exploration bound.
Model, optimizer, visit history and the current age are retained when resuming with this option.

- `nesram.Nes`: `step(buttons, frames)`, `tap`, `ram()` (2 KB), `screen()`, `png()`, `save()/load()`,
  `poke(addr, value)` (patches RAM through the savestate, use it to test what a byte means)
- `audit.py <game> random --episodes 12 --steps 3000 --workers 12 --out DIR`: random play until game over,
  contact sheet `DIR/randomN.png` with BEFORE/AT/+12 frames for every event, timeline `randomN.jsonl`
- `search.py <game> --seconds 300 --workers 20 --out DIR`: Go-Explore guided only by the reward.
  If it reaches level ends, the reward points the right way. Results keep actions from power-on and
  are ranked by (level, progress), so reaching a later level always beats more progress in an earlier one.
- `audit.py <game> replay --result DIR/seedN.json --out DIR`: replay a search result, render events
- View any PNG with the Read tool. Contact sheets are the evidence.

## Acceptance standard for a game module
1. Boot reaches real gameplay (screenshot), no reward before that.
2. Random audit, 12 episodes: detected deaths == lives lost in every episode (or, for games
   without lives, every death/loss event visibly correct), game over detected, screenshots agree.
3. Decoded score equals the HUD digits in screenshots (when the game shows a score).
4. Search reaches at least the end of the first level/stage and the level event fires at the
   right moment on screen (replay sheet). Report how far it got.
5. Known reward-hacking risks for the game are listed in the module docstring.

Evidence: `reports/reward10/<slug>/` (final contact sheets + `summary.json`).
