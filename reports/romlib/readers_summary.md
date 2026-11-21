# RAM readers: coverage and truth checks

2062 controllable ROMs (stage 1, the 40-press retry, the menu fixes and the 80-press deep search) processed by scripts/romlib/readers.py; 0 errors.

## Coverage (all ROMs)

| field | found | high | medium | low | unknown |
|---|---|---|---|---|---|
| position | 1732 (84%) | 1432 | 0 | 300 | 330 |
| lives | 840 (41%) | 664 | 176 | 0 | 1222 |
| score | 824 (40%) | 820 | 4 | 0 | 1238 |
| level | 999 (48%) | 0 | 431 | 568 | 1063 |
| death | 773 (37%) | 697 | 8 | 68 | 1289 |
| in_play | 1278 (62%) | 738 | 262 | 278 | 784 |
| game_over | 1160 (56%) | 351 | 218 | 591 | 902 |

## Boots that stop on a menu (scripts/romlib/menucheck.py)

190 of 1897 boots flagged as a menu (screen ignores the d-pad: 96, cursor jumps: 75, cursor jumps+screen ignores the d-pad: 19); 111 continued to a state the check calls gameplay (boot pads in ramfind_fixed.jsonl; their readers re-run), 79 not (readers rows carry boot_menu: unfixed). 10 ROM files were gone.
Truth games: Tetris and Punch-Out flagged and fixed, the other 9 not flagged. Eye check of 16 random flagged boot ends: 12 menus/title/text screens, 3 gameplay (false flags), 1 unclear; of 16 random fixed boot ends: 13 gameplay (1 of them unclear), 3 still a menu or story text.

## Truth games (scripts/reward10/games/*.py)

Each field is compared with the game's own hand-verified Reader on the same recorded play: lives and level by address, score by value (ours x a constant = truth on 99% of in-play frames), death and game over by events (ours within -60..+400 / -300..+600 frames of the truth), in_play by balanced frame accuracy >= 0.9. ok = right; unknown-ok = the game has no such signal and none was reported; n/a = the stage-1 boot does not reach the game.

| game | lives | score | level | death | in_play | game_over |
|---|---|---|---|---|---|---|
| castlevania | ok ($002A) | ok ($07FD 2 bcd_lo x100) | ok ($0028) | ok (1/1 found, 0 false) | ok (acc 1.000 balanced 0.999) | unknown-ok |
| contra | ok ($0032) | ok ($07E2 2 bin_lo x100) | ok ($0030) | ok (9/9 found, 0 false) | ok (acc 0.947 balanced 0.952) | ok (3/3 found, 0 false) |
| excitebike | unknown-ok | unknown-ok | ok ($0043) | missed (17 true events) | wrong (acc 0.768 balanced 0.517) | unknown-ok |
| finalmission | false ($001C) | missed | wrong ($0041) | wrong (3/3 found, 140 false) | ok (acc 1.000 balanced 1.000) | ok (3/3 found, 0 false) |
| gradius | ok ($0020) | ok ($07E4 4 bcd_lo x10) | missed | ok (12/12 found, 0 false) | ok (acc 0.998 balanced 0.997) | ok (3/3 found, 0 false) |
| kage | unknown-ok | unknown-ok | missed | missed (3 true events) | ok (acc 0.901 balanced 0.901) | missed (3 true events) |
| pacman | ok ($0067) | ok ($0070 7 digits_lo x10) | missed | ok (9/9 found, 0 false) | ok (acc 0.997 balanced 0.998) | ok (3/3 found, 0 false) |
| punchout | unknown-ok | ok ($03E8 5 digits_hi x10) | missed | missed (3 true events) | missed | missed (3 true events) |
| smb | ok ($075A) | ok ($07DA 8 digits_hi x9.99999e-05) | ok ($075F) | ok (8/8 found, 0 false) | ok (acc 0.905 balanced 0.925) | ok (2/2 found, 0 false) |
| tecmobowl | unknown-ok | unknown-ok | missed | missed (1 true events) | wrong (acc 0.225 balanced 0.604) | false (3 events, none true) |
| tetris | unknown-ok | wrong ($03F5 8 digits_hi x1.44264e-05) | missed | missed (3 true events) | missed | missed (3 true events) |

| field | ok | unknown-ok | missed | wrong | false | n/a |
|---|---|---|---|---|---|---|
| lives | 5 | 5 | 0 | 0 | 1 | 0 |
| score | 6 | 3 | 1 | 1 | 0 | 0 |
| level | 4 | 0 | 6 | 1 | 0 | 0 |
| death | 5 | 0 | 5 | 1 | 0 | 0 |
| in_play | 7 | 0 | 2 | 2 | 0 | 0 |
| game_over | 5 | 2 | 3 | 0 | 1 | 0 |

## Before

ramfind2 v1 (lives + score only): lives 4/5 on the lives games (SMB wrong), score 3/5 (SMB, Pac-Man missed); no level/death/in_play/game_over. Before the menu check, Tetris and Punch-Out never reached the game (all fields n/a).

## Known weak spots

- Menu check: false flags on gameplay held by a countdown or drawn by background tiles (Excitebike VS, Banana, Master Fighter 2 in a 16-sample); a fix can stop on a later menu or story text that the check does not catch (Baseball Stars, Bo Jackson Baseball, Joy Mecha Fight). 79 flagged boots stay unfixed. An 80-press search with two seeds (ramfind_deep.jsonl) made 87 of the 261 remaining boots controllable (Zelda among them); 78 pass the menu check and have readers.
- Lives cannot tell a health counter that ends the game (Final Mission $1C) from lives; death/level then follow it.
- Level: the respawn test (medium) finds SMB's $075F; without a death, the boot test (+1 poke after the last menu press; always low) also accepts stage-type bytes: it reports SMB's area type $074E on 127 SMB1 hacks. Gradius, Pac-Man missed. Game over without lives is low confidence (Tecmo Bowl false).
- in_play comes from d-pad control probes: misses crashed-but-racing (Excitebike), play-call phases (Tecmo Bowl), boxing and falling pieces where the probed byte is not the player (Punch-Out, Tetris).
- Score byte order is fixed by one carry test; digits past the changing ones are not verified (SMB reads $07DA-$07E1, proportional to the true $07DD-$07E2). Tetris picked a wrong span.
- Truth checks: 11 games, random play only (stages 1-2, low scores).

## Families (hacks and versions of one game should agree)

| field | families checked | reports | agree with family majority |
|---|---|---|---|
| lives | 31 | 344 | 98% |
| score | 35 | 265 | 89% |
| level | 27 | 324 | 74% |
| death | 27 | 329 | 79% |
| game_over | 38 | 385 | 88% |
| in_play | 45 | 455 | 83% |

### SMB1 family: 190 ROMs (versions + SMB1 hacks), against the known map

| field | reported | right | wrong |
|---|---|---|---|
| lives | 176 of 190 | 176 | 0 |
| level (versions only) | 4 of 4 | 4 | 0 |
| score | 44 of 190 | 40 | 4 |
| position x | 190 of 190 | 188 | 2 |

