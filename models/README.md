# GhostRun Super Mario Bros. policies

Two pixel-only players for Super Mario Bros. (NES). Neither model reads RAM, level numbers or rewards. They see only screen pixels and their own past button presses.

| | v1 | v2 |
|---|---|---|
| Hugging Face | [`kdcyberdude/ghostrun-smb-policy-v1`](https://huggingface.co/kdcyberdude/ghostrun-smb-policy-v1) | [`kdcyberdude/ghostrun-smb-policy-v2`](https://huggingface.co/kdcyberdude/ghostrun-smb-policy-v2) |
| Result | Clears all 32 levels, 1-1 to 8-4, in one run | Partly trained. Reaches 4-4 at best from power-on |
| Play style | One fixed route, no variety | 8 learned play styles ("moods") |
| Training | Demonstrations, then PPO fine-tuning | Reinforcement learning only (recurrent PPO) |
| Emulator | nes_py 8.2.1 core | NeSLE on the GPU (this repo's `rl/`) |
| Size | ~2.0M parameters | ~6.3M parameters |

Both cards follow. `scripts/upload_models.sh` uploads each card as its repo's README.

<!-- card:v1 -->
## ghostrun-smb-policy-v1: the all-32 player

A small CNN policy that finishes Super Mario Bros.: 1-1 to 8-4 in order from a fresh start, greedy, in 10,626 decisions (about 22 minutes of game time). It was the reference player for the GhostRun Three.js reconstruction.

### Files

| File | Size | SHA-256 | What |
|---|---|---|---|
| `refined-000452-069000.pt` | 27,966,621 B | `633b56e5…c28302b8` | PyTorch training checkpoint (`fast-mario-robust-ppo-full-v1`): actor, value head and training state |
| `refined-000452-069000.json` | 1.4 MB | | Sidecar with the training config, progress counters and lineage |
| `actor.onnx` | 8,147,528 B | `c0fdb950…68f23f78a` | Export of the same actor used by the standalone CPU player |
| `robust_model.py` | 3 KB | | Architecture (`RobustPolicy`), copied from `models/v1/` |
| `smb_v1_all32_route.json` | 74 KB | | The run's applied controller masks, 10,626 ticks, from a standard reset |

Full checkpoint SHA-256: `633b56e5ea06520cea80b27f41edabcbde8dce24dd754e59a0a9a9d4c28302b8`.

### Model

`RobustPolicy` (`models/v1/robust_model.py`):

- Inputs: the last 4 RGB frames at 128×128, plus the last 128 applied controller masks (10 bits each, with a validity bit).
- Network: three convolutions with a 512-unit layer for the pixels, a 256-unit layer for the action history, then a 512-unit trunk.
- Outputs: 10 independent button logits (up, down, left, right, a, b, x, y, start, select) and a value.

The network is about 2.04M parameters. Actions run at 8 decisions per emulated second, and each decision holds its buttons for 7 or 8 frames.

### Training

1. Behaviour cloning: 100 epochs over about 1.07M demonstration samples.
2. PPO fine-tuning with a demonstration (BCE) term. The run had 72,530 updates, including 69,000 expert updates, over 452 rollouts and 729 episodes. The demonstration mixture covered nominal play, anchors, recoveries and historical data.

The PPO reward was +1 per new level clear, −1 per death, +5 for the final win and −0.001 per second. This model is **not** pure RL.

### Evaluation

- 3/3 fresh greedy games cleared all 32 levels in order, with native final-victory detection and 10,626 decisions each. Verified on 2026-10-31 and reproduced 3/3 on 2026-11-05.
- Fresh starts are deterministic, so repeating them only shows that execution is consistent. It is not evidence of variety.
- Stochastic play failed: 0/3 wins, each clearing one level and then dying.
- Recovery after real deaths did not qualify: 0/3, with game over at decision 573 after three deaths.

### Limits

- The model follows one route and has little diversity.
- It is sensitive to emulator timing. Its frames are the nes_py core's 29,781-cycle slices, not PPU frames. Its recorded route does not replay in NeSLE: a lead-offset sweep from 0 to 40 frames desyncs by world 1.
- The training code (the earlier `fast_mario` pipeline) is not part of this repository. Only the architecture is.

<!-- /card:v1 -->

<!-- card:v2 -->
## ghostrun-smb-policy-v2: the explorer with moods

A recurrent pixel policy trained with RL only on thousands of GPU-emulated consoles (`rl/train.py` in this repo). One network has 8 *moods*, which are learned play styles chosen by a one-hot input. A diversity judge rewards moods for playing differently. Training is **not finished**.

### Files

| File | Size | What |
|---|---|---|
| `model.pt` | 25,372,945 B | Weights and the training config from `eval/it002907/model.pt` (iteration 2907, 433,717,248 decisions). SHA-256 `2ed7aa22e30e8c14954dcc77a869830dc064885b689ff7b1ccfafdcdc70d0aeb` |
| `policy.onnx` | 25,162,811 B | Greedy one-step actor from `rl/player/export.py`, checked against PyTorch at export. SHA-256 `64ea4a46e2b2f1961728c1a747e32185e6f2d9274d10c42f96b0aa0ddf320cab` |
| `model.json` | | Export metadata: iteration, moods, core size, ONNX hash |
| `eval/` | ~100 KB | The it002907 test plays: `summary.json` and one power-on `*.pads.bin` per mood |

Load the weights with `rl/model.py`:

```python
s = torch.load('model.pt', map_location='cpu', weights_only=False)
m = Policy(core=s['cfg']['core'], moods=s['cfg']['moods'])
m.load_state_dict(s['model'])
```

Then play it with `rl/evaluate.py model.pt OUT` (power-on, all moods) or `scripts/record_all_levels.sh` (per level).

### Model

`rl/model.py` `Policy(core=512, moods=8)` has about 6.3M parameters.

- Inputs: 4 frames of 240×224 NES colour numbers with per-line emphasis, mapped through the FCEUX palette.
- Vision: a learned stem, an Impoola CNN with global average pooling, and two SimBa blocks.
- Context inputs: the controller-dropout timeline, the previous action and the mood one-hot.
- Recurrence: a GRU of 512.
- Outputs: ordered masked heads for the d-pad (9), A (3), B (3), Start (2) and Select (2).
- Value: HL-Gauss over per-game normalised returns.

A decision lasts 4 frames.

### Training

The run is `smb-20261110c-moods`, resumed from `smb-20261110b` at iteration 392. It used 8,192 Mario consoles on one RTX 5090 at about 8,000 to 9,000 decisions per second, with about 17 hours of active training. Training stopped several times so settings could change.

**Reward.** The reward is `scripts/reward10` for SMB: progress, new levels, deaths, score as a capped tiebreaker, and the win. Two late additions:

- a small waiting cost
- a rollback penalty for abrupt progress loss (4-4 loops)

**Starts.** Episodes start at saved level entrances, weighted by failure rate (`rl/level_plan.py`).

**Settings and changes during the run** (all logged):

- moods 8, with a diversity judge (`--div-coef 1e-3`)
- longer rounds (`--steps 64`)
- RND curiosity, on for part of the run and later off
- Go-Explore *places*, also on and later off
- a final four-level focus on 4-4, 7-4, 8-3 and 8-4

For the exact command, see the root README.

### Evaluation

**Power-on test play** (greedy, dropout on, one game per mood) at it002907:

| Mood | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Furthest level | 1-2 | 2-1 | 2-2 | 1-2 | 1-2 | 1-2 | 2-2 | 2-1 |

All eight games ended in game over. The furthest power-on test play over the whole run reached 4-4, at it001924 and at it002889 (mood 4 both times). See `results/tables/smb_v2_moods_eval_history.csv`.

**Level-start clear rates** (sampled play under training conditions, iteration 2914):

- 1-2 to 4-3: 48 to 89% per level
- 5-1 to 8-2, without 7-4: 13 to 80% per level
- 4-4, 7-4, 8-3 and 8-4: 0 to 1%

See `results/tables/smb_v2_moods_level_clear.csv`.

**Greedy check on the four hard levels** (8 moods each, from fixed entrances, iterations 2841 and 2877): 0/8 clears on each level. 4-4's furthest x improved from 1,027 to 2,060 px. See `results/tables/smb_v2_hard_levels_check.json`.

### Limits

- Not fully trained. It does not finish the game from power-on.
- Greedy play can lock into waiting or looping, for example in 4-4's maze.
- The level entrances for 5-1 to 8-4 used in training were built from the 4-1 entrance with the world, stage and area set in RAM. They are not natural arrivals.
- Videos and test plays come from NeSLE. FCEUX agreed on Mario's x, y and speeds frame for frame on the checked replays. Other RAM differs.

<!-- /card:v2 -->
