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
- No reward outside real gameplay (`in_play` False on title/menus/demo/pause)
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

## Tools
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
