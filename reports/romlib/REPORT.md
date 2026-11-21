# All-ROMs job: final report (2026-11-21)

The job had three parts, all for `/home/kd/Desktop/roms`:
1. Clean the folder.
2. Run every game on the GPU emulator (NeSLE).
3. Give every game a boot script and a RAM reader.

## 1. ROM folder: 2,004 ROMs kept, all ready to train

The first cleanup kept 2,345 and moved 998. On 2026-11-21 another 341 were moved out (section 5): 108 non-games, 40 dumps that hang in every emulator, and 193 games deferred because they are not ready to train (`roms_removed/deferred/`, listed in `reports/romlib/deferred_roms.csv`). Nothing was deleted. Every moved file is in `/home/kd/Desktop/roms_removed/`, with its reason in `manifest.csv` there. The same list, by category, is in `reports/romlib/removed_roms.csv`.

| why it was moved out | files |
|---|---|
| alternate / overdump / pirate / intro-hack dump of a kept game | 272 |
| non-English fan translation | 173 |
| Japanese release (English release kept) | 171 |
| near-identical copy (95%+ the same bytes) | 117 |
| duplicate English translation (a better one kept) | 92 |
| byte-identical copy | 89 |
| not an NES ROM | 53 |
| bad dump | 25 |
| blank screen in FCEUX | 3 |
| other-language release | 3 |

An audit of the first cleanup fixed three kinds of mistake. 107 files are back and 15 were moved out instead:
- Good dumps had been dropped for bad ones.
- Clean originals had been dropped for hacks.
- 92 English regional versions (mostly PAL) had been treated as duplicates.

The fixes are in `reports/romlib/cleanup_fixes.csv`.

I also rewrote the headers of 18 ROMs, using fceumm's own correction table. The headers named a board other than the one the game really uses, e.g. Gyruss is CNROM, not mapper 0. Backups are in `roms_removed/header_backup/`; the log is `reports/romlib/header_fixes.csv`.

## 2. GPU emulator

The code is on the fork `kdcyberdude/NeSLE`, branch `ghostrun-allmappers`, as PR #1 into `ghostrun`.
- About 60 boards were added, each modelled on how FCEUmm runs it.
- The commits carry your GitHub noreply address, not the Gmail one.

### RAM check

The same buttons are played from power-on for 3,600 frames on NeSLE and on fceumm, and RAM is compared every frame. Source: `reports/romlib/status.csv`.

| result | ROMs |
|---|---|
| RAM identical to fceumm | 1,776 |
| differs, but fceumm is the one that's wrong (FCEUX sides with NeSLE) | 195 |
| differs from a known timing difference that doesn't change the game | 188 |
| **runs on the GPU** | **2,159** |
| suspect: diverges, cause not known (mostly on the basic boards 0-4) | 146 |
| emulator fault (CPU jam) | 18 |
| board not supported: VS System (99) ×15, 151 ×3, 15 ×3, 95 ×1 | 22 |

### Picture check

The RAM check misses a wrong graphics bank or screen layout, so the picture is compared too. Frames are compared only while both emulators are in the same game state.

| result | ROMs |
|---|---|
| same picture as fceumm | 2,121 |
| different picture | 125 |
| never in the same state, nothing to compare | 77 |
| can't load (unsupported board) | 22 |

FCEUX was used as a third opinion on the 165 earlier picture failures:
- 100 were NeSLE's fault, 15 were fceumm's, and 50 were unclear.
- 26 of the 100 NeSLE errors are fixed.
- Of the 74 left, most are the core renderer on mappers 0, 1 and 4: raster effects, and SMB/Mario Bros hacks with wrong tiles.

The other session's renderer fixes (4 commits, local `f3205d1`) take the picture check to 2,151 same / 95 different:
- 30 more ROMs match fceumm, and none got worse.
- On the CPU build, RAM is unchanged for every ROM.
- 49 of the 100 confirmed NeSLE errors are fixed.
- These commits are **not on the fork yet**. They go there once their GPU-side gate (CUDA vs CPU smoke, full RAM check) has run. It ran out of memory the first time.

Parked:
- About 15 dot-timing cases: Zelda 2 (E), Boogerman, Bayou Billy (E) and others.
- Sesame Street 123 (CPU side).
- Hot Slot.
- 2 Mario hacks that write video memory mid-frame.
- 4 demos that need a larger per-frame log.
- Time Conquest.

Images are in `reports/romlib/screen_png_f3205d1/`; the earlier list is in `screen_failures.csv`.

## 3. Boot and RAM readers

**Boot:** 2,062 of 2,345 games boot to a point where the player can move.
- 87 of these came from a deeper, 80-press search.
- 63 of these boots may still stop on a menu. A second deep search moved 19 others past their menu.

The other 283 games never give control. About 100 of them are demos, test carts or light-gun games with nothing to steer.

**Readers:** one row per game in `reports/romlib/readers.jsonl`; the summary is `reports/romlib/readers_summary.md`. Each field has an address, a confidence level and the evidence for it. Coverage over the 2,062 games:

| field | found | high | medium | low |
|---|---|---|---|---|
| position | 84% | 1,432 | 0 | 300 |
| lives | 41% | 664 | 176 | 0 |
| score | 40% | 820 | 4 | 0 |
| level | 48% | 0 | 431 | 568 |
| death | 37% | 697 | 8 | 68 |
| in_play | 62% | 738 | 262 | 278 |
| game_over | 56% | 351 | 218 | 591 |

### How accurate

**SMB1 family.** These are 190 versions and hacks of Super Mario Bros that keep its RAM layout:

| field | reported | right | wrong |
|---|---|---|---|
| lives | 176 | 176 | 0 |
| score | 44 | 40 | 4 |
| position x | 190 | 188 | 2 |

Level is right on all 4 Nintendo versions. On 127 hacks it reports `$074E`, which is the stage *type*, not the world number. Level found without a death is therefore always marked low confidence.

**11 hand-mapped games.** Each field is right on the following number of the 11 (including "correctly reports none"):

| field | right of 11 |
|---|---|
| lives | 10 |
| in_play | 7 |
| score | 9 |
| game_over | 7 |
| death | 5 |
| level | 4 |

The misses:
- Final Mission reads health as lives.
- Gradius, Pac-Man and Kage have no level found.
- Excitebike, Tecmo Bowl, Tetris and Punch-Out have no in_play signal from d-pad control.

**Agreement within families** (versions and hacks of one game should report the same address):

| field | agreement |
|---|---|
| lives | 98% |
| score | 89% |
| game_over | 88% |
| in_play | 83% |
| death | 79% |
| level | 74% |

Rule of thumb: trust high-confidence lives, score and death, and treat low-confidence level, in_play and game_over as hints to check.

## 4. Ready to train: 2,004 games

Every ROM left in `/home/kd/Desktop/roms` is ready: 1,992 games with the same model and the same reward as the 12 hand-read games, so 2,004 in all (it was 1,708 before section 5's fixes). They train serially or all at once. The per-ROM list is `reports/romlib/train_ready.csv`.

### How it works

**One reward, the same for every game.** It uses reward10's formula and weights:
- \+ each new 8×8 cell the player reaches in an episode (100 cells count like one level)
- \+ a new level, \+ small score gains
- − death, − a small cost for standing still
- The episode ends on game over, or after 30 s without a new cell, level or death.

The difference from the 12 games is only *where* the numbers come from. A per-game address table (`rl/games_table.json`) is built from the RAM readers, instead of a hand-written reader for each game. A field is used only when it can be trusted:
- lives, score and game over at high or medium confidence;
- level only when the respawn test found it;
- in-play only at high or medium confidence.

Every game's start is checked on NeSLE itself (the emulator that training runs on): the boot gives the player control there. Wrong signals were removed in two ways:
- **At the start:** 273 in-play conditions were false at a start where the player has control.
- **During random play:** 33 game-over conditions fired with no deaths (Ghosts 'n Goblins, Mega Man 5 and its hacks, others).

### Tests (CPU build, nothing run on the GPU)

- **Table-game test** (`rl/tests/test_table_games.py`): on 3 table games, the environment's reward and episode ends match the Python reference bit for bit for 300 decisions, and the pictures match NeSLE's harness.
- **Existing 12-game test** (`rl/tests/test_env_cpu.py`): unchanged on the new build (re-run after the section 5 emulator fixes: passes).
- **Random-play sweep** (`rl/scripts/table_sweep.py`): all 1,992 games ran 1,500 random decisions each with no faults. Random play finds new cells in 1,960 games (median 30); 32 find none.

### How to start a run

```bash
rl/scripts/train_table.sh runs/rl/table-64 random:64 --envs-per-game 8
```

`GAMES` can be `all`, `first:N`, `random:N`, or a comma list. The list can also include the 12 hand-read games, e.g. `smb,contra,abadox_u`.

The script uses its own build in `.cache/rl/build-table` (already compiled for the 5090) and the all-mappers emulator. It never touches the live SMB run, which uses pinned libraries.

**Not yet done: a GPU run.** You asked me not to run GPU work beside the live SMB trainer. The CUDA build compiles, but no game has trained on the GPU yet, so speed with many different games at once is unmeasured. Each game is a separate kernel launch, so "all 2,004 games at once" will be much slower per decision than tens of games with many consoles each.

### Deferred (193 ROMs, moved to `roms_removed/deferred/`)

| reason | ROMs |
|---|---|
| the boot search never reaches a point where the player moves: quiz, mahjong, board and card games, text-heavy RPGs and adventures, some sports games | 106 |
| the boot gives control on fceumm but not on NeSLE, even shifted by 1-2 frames | 38 |
| VS System arcade games (coin slot, DIP switches; see section 5) | 25 |
| boot stops on a menu | 15 |
| emulator: RAM diverges from fceumm for an unknown reason (7), Dragon Buster (mapper 95), Fire Emblem `[hM04]` (jams a minute in) | 9 |

## 5. Making the rest ready (2026-11-21)

You asked me to remove the ~100 non-games and find the root cause for the rest. What I found, by cause:

| root cause | fix | games now ready or explained |
|---|---|---|
| Not games (demos, test carts, tools) or need an add-on (light gun, keyboard) | moved to `roms_removed/not_a_game` and `needs_addon` | 108 moved |
| The NeSLE control check pressed each button for only 16 frames; slow games move later | 45 frames | 135 |
| NeSLE powers on a frame or two off fceumm, so a short Start tap in the recorded boot fell on the wrong frame | boot shifted by one frame | 33 |
| The boot search accepted a boot after one control probe, though control was lost moments later (intros, demo modes) | the boot must keep control after 1, 3 and 6 s of idling (3 random seeds, up to 80 presses) | 70 |
| fceumm ran PAL (E) ROMs at PAL speed | compared at NTSC speed: same RAM | 29 |
| RAM drifts from fceumm only under button presses; NeSLE, fceumm and FCEUX agree at power-on | accepted as the same game | 79 |
| Only a power-on random seed differs | accepted | 12 |
| **NeSLE bug:** a 512-byte iNES trainer was copied to $7000 only on FFE boards | copied on every board, as FCEUmm does | 29 |
| **NeSLE bug:** INC/DEC/shift instructions made one write; the 6502 writes the old value first and MMC1 relies on it (Bill & Ted resets MMC1 with `INC $FFFF`) | the CPU makes that first write on cartridge addresses; MMC1 ignores the write right after a reset, as in FCEUmm | Hillsfar, Bill & Ted, Sesame Street 123 now identical to fceumm; Rocket Ranger no longer jams |
| Re-mapped dumps (`[hM04]`, `[hM15]`) or an older translation that jam | swapped for the original dump kept in `roms_removed`, which runs | Erika `[h1]`, Z Gundam `[p1]`, Ike Ike `[T-Eng1.0_BRC]` |
| Dumps that hang at power-on in fceumm **and** FCEUX too (32 SMB1 hacks of the same "No Title" base, Bomberman Collection `[p1]`, Star Wars Namco `[hM04]`, a DQ7 pirate, others) | moved to `roms_removed/bad_dump` | 40 moved |

**The emulator changes were checked on every ROM.** Old and new NeSLE builds were run on gpu_check's 3,600-frame button sequence for all 2,237 ROMs. Only 5 changed:
- the 3 intended fixes;
- Sesame Street 123, now identical to fceumm;
- Snow Brothers, which moved closer to fceumm (176 bytes differ instead of 695) but still doesn't match.

Both CPU tests (`test_env_cpu.py`, `test_table_games.py`) pass on the rebuilt training libraries in `.cache/rl/build-table`. The changes are in `.cache/nesle-fork` (`cpp/include/nesle/cpu.hpp`, `cuda/batch_cpu.hpp`, `cuda/mapper.cuh`, `src/rom.cpp`), not committed or pushed yet.

**Not done, and why:**
- **VS System (25 games).** These are arcade boards. Beyond the board (mapper 99 or 151), they need a coin slot, DIP switches, the VS palettes and some copy-protection checks. The fceumm harness (stable_retro) has no coin button, so their boots can't be searched or checked there either. It's a project of its own.
- **Dragon Buster (mapper 95).** Its mirroring switches tile by tile with the graphics bank the PPU fetched last; NeSLE's renderer can't model that without a major change. One game.
- **The 106 that never give control** are mostly games where nothing moves on its own under a d-pad: quiz shows (Jeopardy!), mahjong, Othello, solitaire, menu-driven RPGs (Dragon Warrior and its hacks, EarthBound hacks, Ultima). They would need a different idea of "progress" (e.g. menu or text state) rather than a position.
- **The 38 that boot on fceumm but not on NeSLE** still need a per-game look. (5 of the earlier 43 were the crash-fixed games above, whose check predated the fix; re-checked, all 5 give control.)

The 193 deferred games are in `roms_removed/deferred/` by reason (`no_control`, `no_control_on_nesle`, `vs_system`, `menu`, `emulator`); moving one back puts it back in the pipeline.

## Other things you should know

- **SMB trainer source is gone.** `/home/kd/Desktop/proj/llm_gameplay/.cache/nesle-src` (the live SMB trainer's emulator source) disappeared on 2026-11-20 around 17:45. This work didn't delete it, and its branch was never on GitHub. The trainer keeps running from its build. Everything it contained, plus the new boards, is on the fork branch above.
- **Gmail address in a pushed commit.** One commit I pushed to the public fork (`62b469a`, "Four-screen RAM: leave FFE boards folded") has your Gmail address as its *committer*. Every other commit uses the noreply address.
  - A clean copy (`827676b`, same content) is ready locally in `.cache/nesle-fork`.
  - Replacing it on GitHub needs a force-push of `ghostrun-allmappers`, which was blocked pending your OK.
  - GitHub may still serve the old commit by its hash even after that.
  - That clone is now set to use the noreply address.
- **Nothing is committed in ghostrun-lab.** The tools are in `scripts/romlib/` and the reports in `reports/romlib/`.
