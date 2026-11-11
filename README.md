# GhostRun training lab

The research code behind GhostRun's game players. It contains:

- a GPU reinforcement-learning trainer that runs thousands of NES consoles on one graphics card
- verified reward functions for 12 NES games
- the training configs
- evaluation and recording scripts
- model cards for two Super Mario Bros. players
- result tables and plots

This repository has no game files. You supply your own ROMs.

## How it relates to the GhostRun plugin

The [GhostRun Hermes plugin](https://github.com/kdcyberdude/ghostrun) runs a pipeline: the agent works out a game's reward, trains a player, then rebuilds the game in Three.js by watching that player. The plugin includes a copy of this trainer under `engine/` so it can set up and train new games by itself.

This lab is where that trainer was developed, and it is newer than the plugin's copy:

| In this lab, not yet in the plugin | Where |
|---|---|
| Rollback penalty in the reward for abrupt progress loss, which discourages maze loops. In Python and the CUDA port | `scripts/reward10/reward.py`, `rl/csrc/reward10.cuh` |
| Go-Explore *places* inside levels, focused level training (`--only-levels`) and a native level-clear episode end (`--episode-level-end`) | `rl/places.py`, `rl/train.py`, `rl/csrc/` |
| Lossless 6-bit packing of stored frames (`--obs-pack6`), which cuts pinned host memory for long rollouts by a quarter | `rl/train.py`, `rl/csrc/rlenv.cu` |
| Live terminal dashboard of per-level clear rates | `rl/watch.py` |
| Emulator speed benchmarks and NeSLE-vs-FCEUmm parity tools | `rl/bench/`, `scripts/nesle_check/` |
| Per-level recording of trained players, with replayable controller logs | `scripts/recording/` |

The plugin has one trainer feature this lab lacks: `--restart-from`. It starts a new run from a checkpoint's weights only, for example after a reward change.

## Training architecture in brief

```
 8,192 NES consoles on the GPU (NeSLE + patch)        reward10 on the GPU (line-by-line port of the Python reward)
 4 frames per decision, button dropout, sticky moves  progress, new level, death, damage, capped score, win, stall
              │ pixels only (240x224 colour numbers)                │ reward never reaches the actor
              ▼                                                     ▼
 Policy: stem + Impoola CNN + SimBa blocks → GRU 512 → masked heads (d-pad, A, B, Start, Select), HL-Gauss value
              │
              ▼
 Recurrent PPO with per-game advantage normalisation, exact resume, background test plays (rl/evaluate.py)
 Exploration: level starts weighted by failure rate, moods with a diversity judge, optional curiosity (RND) and places
```

- **Environment** (`rl/csrc/`, `rl/rlenv.py`): one emulator per CUDA thread. The step makes 4 frames per decision and computes the reward and episode end on the device, then restarts finished games from a boot snapshot or a saved level entrance. A CPU build of the same code is bit-identical and checks the GPU build.
- **Reward** (`scripts/reward10/`): Python is the source of truth. `rl/scripts/gen_reward_consts.py` writes its constants as exact floats for `rl/csrc/reward10.cuh`, and `rl/tests/test_reward_replay.py` checks that Python and C++ agree bit for bit.
- **Policy** (`rl/model.py`): about 6.3M parameters. Its inputs are pixels, past controls and an optional mood; it never sees RAM, a game id or the reward.
- **Trainer** (`rl/train.py`): resumes exactly, including environments, RNG and optimizer. It saves a test play every N minutes (`eval/itN/`: `model.pt`, summary, pads, video).
- **Speed**: on one RTX 5090, about 8,000 to 10,000 decisions per second at 20 to 30% MFU.

## The 12 games

Reward modules are in `scripts/reward10/games/`. The exact reward for each game is in `scripts/reward10/reward_spec.json`. The ROM file names and SHA-256 hashes are in `rl/games_manifest.json`.

| Game | Genre | Levels | What a level is |
|---|---|---|---|
| Super Mario Bros. | platformer | 32 | world-level 1-1 to 8-4 in order |
| Castlevania | platformer | 19 | stage 0 to 18 |
| Contra | run-and-gun shooter | 8 | stage 1 to 8 |
| Excitebike | racing | 6 | challenge race, then main races 1 to 5 |
| Final Mission (S.C.A.T.) | shooter | 5 | stage 1 to 5 |
| Gradius | horizontal shooter | 7 | stage 1 to 7 |
| Kage (Shadow of the Ninja) | action platformer | 15 | stage 1-1 to 5-2 |
| Pac-Man | maze | endless | maze (round) |
| Mike Tyson's Punch-Out!! | boxing | 14 | opponent, Glass Joe to Mike Tyson |
| Tecmo Bowl | American football | 11 | games won this season |
| Tetris | puzzle | endless | speed level |
| The Legend of Zelda | action-adventure | 9 | Triforce pieces held |

## Setup

You need Linux, an NVIDIA GPU with CUDA, `nvcc`, a C++20 compiler and `ffmpeg`.

```bash
pip install -r requirements.txt
scripts/setup_nesle.sh                 # NeSLE at its pinned commit + third_party/nesle-extensions.patch
export NES_ROM_DIR=/path/to/your/roms  # file names as in rl/games_manifest.json (hashes are checked)
```

Run the CPU checks. The first needs no emulator. The others build the CPU environment and need ROMs:

```bash
python -m pytest tests/test_reward10.py
python rl/tests/test_env_cpu.py
python scripts/reward10/contract.py    # checks every game module, regenerates reward_spec.json
```

## Train

`configs/` holds the exact commands of the published runs. Run them from the repository root.

```bash
configs/smb_baseline.sh       # Mario only, plain recurrent PPO
configs/smb_v2_moods.sh       # the v2 player: moods, level starts, then the hard-level focus (phases listed inside)
configs/twelve_games.sh       # all 12 games in one network
```

To follow a run:

```bash
python rl/watch.py runs/rl/<run>           # live per-level clear rates
python rl/plot_run.py runs/rl/<run>        # progress.png
python rl/evaluate.py runs/rl/<run>/eval/itN/model.pt OUT --games smb   # greedy test play, video and pads
python scripts/summarize_run.py runs/rl/<run> results/tables            # the CSVs in results/
```

To stop cleanly, run `touch runs/rl/<run>/STOP`. Restarting the same command resumes exactly.

## Record the Mario players on all 32 levels

`scripts/record_all_levels.sh` writes an MP4 and a controller log for each level and each model:

- **v2**: every mood plays the level from its saved entrance, and the mood with the most progress is kept. A clear wins first, then the furthest x, then the longest play. A second pass replays the kept moves and checks every pad byte before it renders the video.
- **v1**: plays the whole game once and the run is cut per level. v1 runs in its own emulator, the nes_py-based reference lab.

```bash
export GHOSTRUN_V2_POLICY=model.pt                     # from kdcyberdude/ghostrun-smb-policy-v2
export GHOSTRUN_V2_STARTS=runs/rl/smb-20261110c-moods/ckpt_rank0_latest.pt   # level entrances (or a cut .npy)
export GHOSTRUN_REFERENCE_LAB=/path/to/run-reference-lab.sh                  # v1's emulator and player
export GHOSTRUN_SMB_ROM=$NES_ROM_DIR/"Super Mario Bros (JU) (PRG 0).nes"
scripts/record_all_levels.sh --dry-run      # checks inputs and tests the writers; loads no model
scripts/record_all_levels.sh --out recordings
```

The output goes to `recordings/v{1,2}/<level>/smb_v{1,2}_<level>.{mp4,actions.json,pads.bin}`, with a `summary.json` for each model. Each action log is JSON with these fields:

- `level`, `model`, `mode`
- `frames`: one NES controller byte per frame
- `actions`: the same stream run-length encoded in the Three.js tools' native-trace bit order, so it replays in `tools/harness/model-play.mjs`
- `start`, `clock`, `result`

See `scripts/recording/action_log.py` for the format.

## Models

| | Result | Link |
|---|---|---|
| v1 | Clears all 32 levels greedily in one run; one fixed route | [kdcyberdude/ghostrun-smb-policy-v1](https://huggingface.co/kdcyberdude/ghostrun-smb-policy-v1) |
| v2 | RL only, 8 moods, partly trained; best power-on run reached 4-4 | [kdcyberdude/ghostrun-smb-policy-v2](https://huggingface.co/kdcyberdude/ghostrun-smb-policy-v2) |

The full model cards are in [models/README.md](models/README.md). `scripts/upload_models.sh` publishes both.

## Results

- `results/plots/`: training progress for Mario v2 and the Mario baseline.
- `results/tables/`:
  - every test play (`*_eval_history.csv`)
  - training curves (`*_training.csv`)
  - v2's per-level clear rates (`smb_v2_moods_level_clear.csv`)
  - the greedy check on 4-4, 7-4, 8-3 and 8-4 (`smb_v2_hard_levels_check.json`)

## Layout

```
rl/                 trainer, model, environment (csrc/ CUDA + CPU), evaluation, ONNX player export, tests
scripts/reward10/   reward functions for the 12 games, contract checker, reward_spec.json
scripts/nesle_check NeSLE host wrapper and parity tools against FCEUmm
scripts/recording/  per-level recording of the Mario players
configs/            training commands of the published runs
models/             model cards; v1 architecture
data/routes/        v1's recorded all-32 controller route
results/            plots and tables
third_party/        NeSLE extensions patch and licences
```

## Licence

MIT, see [LICENSE](LICENSE). Third-party parts are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
