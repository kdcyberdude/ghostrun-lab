# DIAMOND (Alonso et al., "Diffusion for World Modeling: Visual Details Matter in Atari", NeurIPS 2024 spotlight): paper and repo profile

Research date: 2026-11-19. Primary sources: arXiv 2405.12399 v2 HTML (full text including Appendices A–M), the GitHub repo README and config files on branches `main`, `csgo` and `ddpm`, the Hugging Face model repo `eloialonso/diamond`, and all GitHub issues #1–#53 (read via the GitHub API). Dates are given for everything.

Timeline:
- arXiv v1: 2024-05-20. v2: 2024-10-30. v2 adds Section 6 (CS:GO), "added after NeurIPS acceptance" — [arXiv abs](https://arxiv.org/abs/2405.12399); [paper v2 HTML](https://arxiv.org/html/2405.12399)
- GitHub repo created 2024-05-19; last push 2024-12-06; MIT licence; 2,129 stars and 171 forks on 2026-11-19 — [GitHub API](https://api.github.com/repos/eloialonso/diamond)
- Branches: `main` (Atari), `csgo`, `ddpm`. The `ddpm` branch holds the code for the DDPM comparison in Section 5.1 — [README](https://github.com/eloialonso/diamond)
- Last commits: `main` 2024-12-06 ("Change git clone to https"); `csgo` 2024-12-06 ("Fix multi gpu"). CS:GO code landed 2024-10-11 — [GitHub commits](https://github.com/eloialonso/diamond/commits/csgo)
- Hugging Face model repo created 2024-10-10, last modified 2024-10-21 — [HF](https://huggingface.co/eloialonso/diamond)
- The repo has had no author commits since December 2024. Community PRs from 2025–2026 (#46, #48, #51, #52) are open and unmerged — [issues list](https://github.com/eloialonso/diamond/issues?q=is%3Aissue)

## Architecture: diffusion formulation, denoiser, conditioning, resolution, denoising steps, reward/termination model, actor-critic

### Takeaway
DIAMOND is a pixel-space EDM diffusion model (Karras et al. 2022). It is a small 2D U-Net (about 4M parameters for Atari) that predicts the next 64×64 RGB frame. It conditions on the last 4 frames, stacked channel-wise, and the last 4 actions, fed through adaptive group norm. Sampling takes 3 Euler denoising steps per frame. A separate CNN-LSTM predicts reward and termination, and a CNN-LSTM actor-critic (REINFORCE with a value baseline and λ-returns) is trained only on imagined 15-step rollouts. The CS:GO model scales the same design to 381M parameters: a 330M dynamics model at 56×30 plus a 51M diffusion upsampler (×5) to 280×150.

### Cited Findings
**Diffusion formulation (EDM)**
- DIAMOND uses the EDM formulation instead of DDPM. Perturbation kernel N(x, σ²(τ)I) with drift f=0. The network F_θ is preconditioned as D_θ = c_skip·x + c_out·F_θ(c_in·x, y), where y = (c_noise, past frames, past actions) — [paper §3.1](https://arxiv.org/html/2405.12399)
- Preconditioners: c_in = 1/√(σ²+σ_data²), c_out = σ·σ_data/√(σ²+σ_data²), c_noise = ¼·log σ, c_skip = σ_data²/(σ_data²+σ²), with σ_data = 0.5. Training noise is log σ ~ N(P_mean = −0.4, P_std = 1.2) — [paper App. C](https://arxiv.org/html/2405.12399)
- Why EDM: DDPM's noise-prediction target makes the network learn the identity at high noise. That gives poor early score estimates and severe compounding error at few steps (n ≤ 10). EDM's target makes the network predict the clean image at high noise, so it stays stable "even for a single denoising step" — [paper §5.1](https://arxiv.org/html/2405.12399)
- Quantitative drift over 1000-step Breakout rollouts (400 held-out reference trajectories): DDPM with 10 steps is still less stable than EDM with 1 step — [paper App. K](https://arxiv.org/html/2405.12399)
- Implementation details not in the paper:
  - Atari uses offset noise `sigma_offset_noise: 0.3` — [config/agent/default.yaml](https://github.com/eloialonso/diamond/blob/main/config/agent/default.yaml)
  - Author explanation: Gaussian noise does not destroy low frequencies, so offset noise teaches the model to create them. "Not particularly important in Atari when using the EDM objective" (2024-12-06) — [issue #36](https://github.com/eloialonso/diamond/issues/36)
  - Initial sampling noise is deliberately not scaled by σ_max. "Starting from a lower variance noise helps to mitigate the autoregressive drift" (2025-04-14) — [issue #40](https://github.com/eloialonso/diamond/issues/40)

**Denoiser network**
- A standard 2D U-Net. The previous L frames are concatenated channel-wise with the noisy next frame. Actions and diffusion time enter through adaptive group normalisation in the residual blocks — [paper §3.1, App. D](https://arxiv.org/html/2405.12399)
- Atari denoiser hyperparameters (Table 2): residual block layers [2,2,2,2], channels [64,64,64,64], conditioning dimension 256, no attention (`attn_depths: [0,0,0,0]`), `num_steps_conditioning: 4`, `img_channels: 3` — [paper App. D Table 2](https://arxiv.org/html/2405.12399); [config/agent/default.yaml](https://github.com/eloialonso/diamond/blob/main/config/agent/default.yaml)
- Parameter counts:
  - Atari diffusion model: about 4M parameters ("from 4M for our Atari models to 381M for our CS:GO model") — [paper §6](https://arxiv.org/html/2405.12399)
  - Full Atari agent (world model + reward/termination + actor-critic): 13M parameters, versus 30M for IRIS and 18M for DreamerV3 — [paper App. H Table 4](https://arxiv.org/html/2405.12399)

**Conditioning**
- Frame stack L = 4 observations and actions, with Euler sampling and 3 steps — [paper App. E Table 3](https://arxiv.org/html/2405.12399)
- The authors also tried a cross-attention architecture: a 2D U-Net that cross-attends to a history encoder. On 64×64 CS:GO it was worse than frame stacking (FVD 81.4 vs 34.8) — [paper App. M, Table 8](https://arxiv.org/html/2405.12399)

**Resolution and environment settings**
- Atari: 64×64×3 observations, frameskip 4, max no-op 30, termination on life loss (training), reward clipped to {−1, 0, 1}, discrete action space with up to 18 actions — [paper App. E Table 3](https://arxiv.org/html/2405.12399); [config/env/atari.yaml](https://github.com/eloialonso/diamond/blob/main/config/env/atari.yaml)
- CS:GO:
  - Dynamics model runs at 56×30, downsampled from 280×150. A second, smaller diffusion model upsamples to the original resolution — [paper §6](https://arxiv.org/html/2405.12399)
  - The released config has `upsampling_factor: 5` and 51 discrete actions; the env size is [150, 280] — [HF csgo env config](https://huggingface.co/eloialonso/diamond/blob/main/csgo/config/env/csgo.yaml); [HF csgo agent config](https://huggingface.co/eloialonso/diamond/blob/main/csgo/config/agent/csgo.yaml)
- CS:GO dynamics denoiser:
  - channels [128, 256, 512, 1024], depths [2, 2, 2, 2]
  - self-attention at the two lowest-resolution levels (`attn_depths [0,0,1,1]`), `cond_channels: 2048`, 4 conditioning frames
  - `noise_previous_obs: true` (noise augmentation on the context frames), `sigma_offset_noise: 0.1`
  - Upsampler: channels [64, 64, 128, 256], 1 conditioning frame, attention only at the lowest level
  - Source: [HF csgo agent config](https://huggingface.co/eloialonso/diamond/blob/main/csgo/config/agent/csgo.yaml)
- CS:GO size: 381M parameters including 51M for the upsampler — [paper §6](https://arxiv.org/html/2405.12399)
- The CS:GO denoiser is trained with `num_autoregressive_steps: 4` (Atari uses 1) and training σ distribution loc −1.2 (Atari: −0.4) — [csgo trainer.yaml](https://github.com/eloialonso/diamond/blob/csgo/config/trainer.yaml); [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml)

**Denoising steps at inference**
- Atari: n = 3 Euler steps (3 NFE per frame), versus 16 NFE per frame for IRIS — [paper §5.2, §5.3](https://arxiv.org/html/2405.12399)
- Why not n = 1: with a single step the L2-optimal prediction averages multimodal futures, which gives blurry frames. Example: the unpredictable opponent in Boxing — [paper §5.2](https://arxiv.org/html/2405.12399)
- Sampler config: `num_steps_denoising: 3`, `sigma_min 2e-3`, `sigma_max 5.0`, `rho 7`, `order 1` (Euler; 2 = Heun), `s_churn 0` (deterministic) — [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml)
- 1-step versus 3-step ablation on the top 10 games (1-step is a single seed):
  - Mean HNS falls from 3.052 to 1.962.
  - Boxing 86.9 → 41.9; Breakout 132.5 → 50.8; RoadRunner 20673 → 5084.
  - Asterix improved (3698 → 6687).
  - Source: [paper App. L Table 7](https://arxiv.org/html/2405.12399)
- CS:GO sampling, with the dynamics model kept at 3 steps:
  - The paper's videos use the "higher_quality" config: dynamics 3 Euler steps; upsampler 10 steps, stochastic (`s_churn 10`); conditioning noise `s_cond 0.005` — [paper §6](https://arxiv.org/html/2405.12399); [higher_quality.yaml](https://github.com/eloialonso/diamond/blob/csgo/config/world_model_env/higher_quality.yaml)
  - The default "fast" config uses 1 dynamics step and 1 Heun upsampler step — [fast.yaml](https://github.com/eloialonso/diamond/blob/csgo/config/world_model_env/fast.yaml)

**Reward and termination model**
- A separate model R_ψ made of CNN residual blocks (channels [32,32,32,32], depths [2,2,2,2], conditioning dimension 128, actions through AdaGN) and an LSTM with 512 units — [paper App. D Table 2](https://arxiv.org/html/2405.12399)
- Shared layers, separate heads. The LSTM is burned in on the L = 4 conditioning frames before imagination begins — [paper App. D](https://arxiv.org/html/2405.12399)
- Loss: cross-entropy on sign(r) and on termination, over sequences of length L+H = 19 — [paper App. G Algorithm 1, Table 3](https://arxiv.org/html/2405.12399)

**Actor-critic**
- A shared CNN-LSTM:
  - 4 residual blocks (channels [32,32,64,64], depths [1,1,1,1]), each followed by 2×2 max-pool
  - LSTM with 512 units; separate policy and value heads
  - Input: a single 64×64 frame
  - Source: [paper App. D](https://arxiv.org/html/2405.12399)
- Trained entirely in imagination:
  - imagination horizon H = 15, γ = 0.985, λ = 0.95, entropy weight 0.001
  - REINFORCE with a value baseline; the value is regressed on λ-returns
  - AdamW, learning rate 1e-4
  - Source: [paper App. E, F](https://arxiv.org/html/2405.12399)
- The RL pipeline follows IRIS, including IRIS's Freeway trick, which is not stated in the paper (author, 2024-12-06) — [issue #34](https://github.com/eloialonso/diamond/issues/34)

### Inferences
- With 4 RGB context frames plus the noisy target, the Atari U-Net input has about 15 channels. All conditioning beyond those 4 frames reaches the model only through what is visible in the frames.
- At frameskip 4, the 4-frame context covers about 16 emulator frames. For NES at 60 Hz that is about 0.27 s of game time.
- The reward model predicts only sign(r) as a 3-class classification. A Mario reward with graded magnitudes (distance, coins, flag) would need a different reward head.
- The CS:GO config adds stability features that Atari did not need: multi-step autoregressive training, noise on the context frames, offset noise and a two-stage upsampler. These are the ones to copy for a higher-resolution, longer-horizon model.

### Gaps
- The paper gives no exact per-module parameter breakdown for the Atari agent beyond "about 4M" (denoiser) and 13M (total). The reward/termination and actor-critic split is not stated. The ~54 MB per-game checkpoint size is consistent with 13M fp32 parameters, but I did not verify that by loading a checkpoint.
- How actions are embedded (embedding-table size and similar) is in the source code, which I did not read in full. The paper says only "adaptive group normalization".

## Training: data, what is trained jointly, GPU type and hours, CS:GO dataset and compute

### Takeaway
On Atari 100k, each run collects 100k agent steps (frameskip 4, about 2 hours of play). It alternates collection, then diffusion training, then reward/termination training, then actor-critic training in imagination, for 1000 epochs. Each run takes about 2.9 days and about 12 GB VRAM on one RTX 4090; the 130 runs (26 games × 5 seeds) total 1.03 GPU-years. The CS:GO world model (no RL) was trained on 87 h (5M frames) of human Dust II play for 12 days on one RTX 4090.

### Cited Findings
**Atari: data and schedule**
- Atari 100k: 100k actions per game, about 2 hours of human play. Unconstrained agents usually use 50M steps (500× more) — [paper §4.1](https://arxiv.org/html/2405.12399)
- Runs and compute: 5 seeds × 26 games; about 12 GB VRAM and about 2.9 days per run on one RTX 4090; 1.03 GPU-years in total — [paper §4.1](https://arxiv.org/html/2405.12399)
- Training loop (Algorithm 1): each epoch is collect, then update the diffusion model, then update the reward/termination model, then update the actor-critic. The three models are trained separately (not end-to-end), all on data collected so far — [paper §3.2, App. G](https://arxiv.org/html/2405.12399)
- Hyperparameters: 1000 epochs, 400 training steps per epoch per component, batch size 32, 100 environment steps per epoch, ε-greedy 0.01 for collection — [paper App. E Table 3](https://arxiv.org/html/2405.12399)
- Config details:
  - The first epoch collects 5k–10k steps and trains each model for 5k–10k steps.
  - Real-environment collection uses `num_envs: 1`.
  - Batches are sampled with `sample_weights: [0.1,0.1,0.1,0.7]`.
  - `compile_wm: True`.
  - Source: [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml)
  - Author: later epochs (400 steps each) give the real per-epoch cost (2024-05-23) — [issue #1](https://github.com/eloialonso/diamond/issues/1)

**Atari: time profile on RTX 4090 (App. I, Table 5)**

| Item | Time |
|---|---|
| One update, all three models | 543 ms |
| — diffusion model | 88 ms |
| — reward/termination model | 115 ms |
| — actor-critic | 340 ms (= 15 imagination steps × 20.4 ms + 34 ms loss/backward) |
| One epoch | 217 s |
| Whole run | 2.5 days training + 0.4 days collection/eval/checkpointing |

Source: [paper App. I Table 5](https://arxiv.org/html/2405.12399)

**Atari: comparison with baselines**
- Training days: DIAMOND 2.9, IRIS 4.1, DreamerV3 under 1. DIAMOND is slower to train than DreamerV3 — [paper App. H Table 4](https://arxiv.org/html/2405.12399)

**Third-party timings and multi-GPU**
- A user measured on 1× H100 with 100 envs (2024-12-17): denoiser 8.67 it/s, reward/termination 14.70 it/s, actor-critic 2.73 it/s, initial collection 1,315 it/s — [issue #37](https://github.com/eloialonso/diamond/issues/37)
- Moving to 8× H100 with DDP dropped throughput (denoiser 8.71 → 0.61 it/s). The authors had only tried 2-GPU machines and suspected data loading — [issue #37](https://github.com/eloialonso/diamond/issues/37)
- Multi-GPU support was broken and then fixed in commit 851cefb (2024-12-06) — [issue #35](https://github.com/eloialonso/diamond/issues/35)

**CS:GO data and compute**
- Data: `dataset_dm_scraped_dust2` from Pearce & Zhu (2022), 5.5M frames, 95 h, about 660 GB. Test split: 500 episodes of 1000 steps. Training: the remaining 5003 episodes, 5M frames or 87 h, captured at 16 Hz — [csgo README](https://github.com/eloialonso/diamond/tree/csgo); [paper §6](https://arxiv.org/html/2405.12399)
- "The provided configuration took 12 days on a RTX 4090" (dynamics + upsampler combined) — [csgo README](https://github.com/eloialonso/diamond/tree/csgo); [paper §6](https://arxiv.org/html/2405.12399)
- Released CS:GO trainer config:
  - denoiser batch 16 × gradient accumulation 8; upsampler batch 4 × 8
  - 400 steps per epoch; `num_final_epochs: 600`; 4 data-loader workers
  - Source: [csgo trainer.yaml](https://github.com/eloialonso/diamond/blob/csgo/config/trainer.yaml)
- The config at first ran out of memory on 24 GB GPUs; it was fixed in commit a4396eb (2024-11-22) to fit a 24 GB RTX 4090 — [issue #29](https://github.com/eloialonso/diamond/issues/29)
- The earlier 64×64 CS:GO and driving experiments (App. M) are a different setup:
  - 190k-frame "Clean" CS:GO set; 4.4 h of motorway driving
  - 120k updates at batch 64 on up to 4× A6000, 1–2 days per run
  - The authors say App. M "do[es] not correspond to the released demo model" — [paper App. M.3](https://arxiv.org/html/2405.12399); [issue #29](https://github.com/eloialonso/diamond/issues/29)

### Inferences
- 1000 epochs × 400 steps = 400k updates per component per Atari run, at batch 32. For CS:GO, 600 epochs × 400 steps would be about 240k optimizer steps at an effective batch of 128 (dynamics). I derived this from the configs; the authors do not state it.
- DIAMOND's training cost is dominated by the actor-critic's imagination rollouts (340 of 543 ms), not by world-model fitting.

### Gaps
- No total GPU-hours are reported for the CS:GO model beyond "12 days on an RTX 4090".
- No learning curves are published for the CS:GO model. A user asked for Atari training curves (Alien, 2025-08-23) and got no reply — [issue #43](https://github.com/eloialonso/diamond/issues/43)

## Results: Atari 100k per-game scores and aggregates, CS:GO, other environments, any Mario/NES result

### Takeaway
DIAMOND's Atari 100k scores (5 seeds): mean HNS 1.459, IQM 0.641 and median 0.373. It beats humans on 11 of 26 games. That was the best mean among agents trained purely in a world model at publication (2024), against IRIS 1.046, DreamerV3 1.097 and STORM 1.266. It is strong on games where small visual details matter (Breakout, Boxing, Krull, RoadRunner, Asterix, CrazyClimber). It is near random on exploration-heavy or long-horizon games (PrivateEye, BankHeist, Seaquest, Frostbite, BattleZone). No official or community DIAMOND result exists for Super Mario Bros, NES or platformers.

### Cited Findings
**Aggregates**

From [paper Table 1](https://arxiv.org/html/2405.12399):

| Method | Mean HNS | IQM | Superhuman games |
|---|---|---|---|
| SimPLe | 0.332 | 0.130 | 1 |
| TWM | 0.956 | 0.459 | 8 |
| IRIS | 1.046 | 0.501 | 10 |
| DreamerV3 | 1.097 | 0.497 | 9 |
| STORM | 1.266 | 0.636 | 10 |
| **DIAMOND** | **1.459** | **0.641** | **11** |

- Median HNS (not in DIAMOND's Table 1; reported by EDELINE's table): DIAMOND 0.373, STORM 0.580 — [EDELINE Table 1](https://arxiv.org/html/2502.00466v2). I recomputed DIAMOND's median from DIAMOND's Table 1 raw scores and got 0.373.
- Search-based and model-free methods (App. J Table 6), mean / IQM / superhuman games:
  - BBF: 2.247 / 1.139 / 12
  - EfficientZero: 1.943 / 1.047 / 14
  - SR-SPR: 1.271 / 0.700
  - DIAMOND does better than these on some games.
  - Source: [paper App. J Table 6](https://arxiv.org/html/2405.12399)

**Per-game raw scores (DIAMOND, mean of 5 seeds)**

From [paper Table 1](https://arxiv.org/html/2405.12399):

| Game | Score | Game | Score |
|---|---|---|---|
| Alien | 744.1 | Hero | 5621.8 |
| Amidar | 225.8 | Jamesbond | 427.4 |
| Assault | 1526.4 | Kangaroo | 5382.2 |
| Asterix | 3698.5 | Krull | 8610.1 |
| BankHeist | 19.7 | KungFuMaster | 18713.6 |
| BattleZone | 4702.0 | MsPacman | 1958.2 |
| Boxing | 86.9 | Pong | 20.4 |
| Breakout | 132.5 | PrivateEye | 114.3 |
| ChopperCommand | 1369.8 | Qbert | 4499.3 |
| CrazyClimber | 99167.8 | RoadRunner | 20673.2 |
| DemonAttack | 288.1 | Seaquest | 551.2 |
| Freeway | 33.3 | UpNDown | 3856.3 |
| Frostbite | 274.1 | | |
| Gopher | 5897.9 | | |

- Per-seed results are in `results/data/DIAMOND.json` — [README](https://github.com/eloialonso/diamond)

**Strong and weak games (versus baselines)**
- Strong, as stated by the authors: "particularly well on environments where capturing small details is important, such as Asterix, Breakout and Road Runner" — [paper §4.2](https://arxiv.org/html/2405.12399)
- Weak, compared with DreamerV3 and STORM — [paper Table 1](https://arxiv.org/html/2405.12399):

| Game | DIAMOND | DreamerV3 | STORM |
|---|---|---|---|
| BankHeist | 19.7 | 649 | 641 |
| BattleZone | 4702 | 12250 | 13540 |
| Frostbite | 274 | 909 | 1316 |
| Hero | 5622 | 11161 | 11044 |
| PrivateEye | 114 | 882 | 7781 |

**Visual comparison with IRIS (same 100k-frame static datasets)**
- IRIS shows frame-to-frame inconsistencies that DIAMOND does not: enemies turning into rewards in Asterix; inconsistent bricks and score in Breakout; inconsistent pellets in RoadRunner.
- DIAMOND reliably updates the Breakout score by +7 when a red brick breaks.
- Source: [paper §5.3](https://arxiv.org/html/2405.12399)

**CS:GO and other environments**
- The CS:GO model (world model only, no RL) generates stable trajectories over hundreds of timesteps. It is more likely to drift in rarely visited areas of the map — [paper §6](https://arxiv.org/html/2405.12399)
- Early 64×64 experiments on static data (App. M Table 8):

| Model | CS:GO FID / FVD / LPIPS | Driving FID / FVD / LPIPS | Parameters | Sample rate |
|---|---|---|---|---|
| DIAMOND frame-stack | 9.6 / 34.8 / 0.107 | 16.7 / 80.3 / 0.058 | 122M | 7.4 Hz |
| IRIS K=16 | 24.5 / 110.1 / 0.129 | — | 123M | 4.2 Hz |
| IRIS K=64 | 22.8 / 85.7 / 0.116 | — | — | 1.5 Hz |
| DreamerV3 | 106.8 / 509.1 / 0.173 | — | 181M | 266.7 Hz |

Sample rates are on one A6000 — [paper App. M Table 8](https://arxiv.org/html/2405.12399)

**Mario / NES / platformers**
- None of these sources contain a Mario, NES or platformer result:
  - the DIAMOND paper (Atari 26 games, CS:GO, motorway driving) — [paper](https://arxiv.org/html/2405.12399)
  - the repo branches — [GitHub](https://github.com/eloialonso/diamond)
  - the 53 issues/PRs. The closest is a user (2024-11-08) who said they were "working on training a model to simulate a SM64 stage" with the CS:GO branch; no results were posted — [issue #21](https://github.com/eloialonso/diamond/issues/21)

### Inferences
- DIAMOND's gains come from visual fidelity, not exploration. Its weakest games are those that need long-horizon exploration or memory of off-screen state (PrivateEye, BankHeist, Frostbite, Hero). Those are the properties that matter most for completing Super Mario Bros levels beyond 1-1 (mazes in 4-4/7-4/8-4, warp pipes, hidden blocks).
- Mean HNS is heavily skewed by a few very high games: Boxing 7.2, Krull 6.6, Breakout 4.5, CrazyClimber 3.5. The median (0.373) is below STORM's (0.580) and DreamerV3-class methods, so "best world-model agent" applies only to the mean.

### Gaps
- No DIAMOND result on any NES game, Mario or otherwise, was found in official sources, issues, or a GitHub repo search. A `gh search repos` query for DIAMOND + Mario/NES returned nothing.
- Fork contents were not audited; the repo has 171 forks.

## Released assets: checkpoints, licence, loading, hardware, playable demo speed

### Takeaway
Pretrained world model + policy checkpoints for all 26 Atari 100k games (about 54 MB each) and one CS:GO world model (1.53 GB, no policy) are on Hugging Face (`eloialonso/diamond`). They download automatically through `src/play.py`. The code is MIT-licensed; the HF card states no licence. The playable CS:GO demo runs at about 10 fps on an RTX 3090 in higher-quality mode (default "fast" mode is faster); the Atari player runs at a default target of 15 fps.

### Cited Findings
**Checkpoints on Hugging Face**
- `atari_100k/models/<Game>.pt` for all 26 games (Alien … UpNDown), each 54.30–54.35 MB, plus `atari_100k/config/agent/default.yaml` and `env/atari.yaml` — [HF file listing](https://huggingface.co/eloialonso/diamond/tree/main)
- `csgo/model/csgo.pt` (1,526,844,223 bytes), plus `csgo/config/...` and 7 spawn points (`csgo/spawn/0..6/` with low_res/full_res frames and actions) — [HF file listing](https://huggingface.co/eloialonso/diamond/tree/main)
- The README says one seed per game ("world model and policy pretrained on Atari 100k will be downloaded"); it does not say which seed — [README](https://github.com/eloialonso/diamond)

**Licence**
- Code: MIT — [GitHub API](https://api.github.com/repos/eloialonso/diamond)
- HF model card: tags and links only, no licence field — [HF model card](https://huggingface.co/eloialonso/diamond)
- The README warns that Atari ROMs are downloaded with the dependencies, "which means that you acknowledge that you have the license to use them" — [README](https://github.com/eloialonso/diamond)

**Loading**
- Setup: Python 3.10 via conda, then `pip install -r requirements.txt` (torch 2.4.1, gymnasium 0.29.1, ale-py 0.9.0, hydra-core 1.3, huggingface-hub 0.17.2, pygame 2.5.2) — [README](https://github.com/eloialonso/diamond); [requirements.txt](https://github.com/eloialonso/diamond/blob/main/requirements.txt)
- Atari: `python src/play.py --pretrained`. CS:GO: `git checkout csgo && python src/play.py` — [README](https://github.com/eloialonso/diamond); [csgo README](https://github.com/eloialonso/diamond/tree/csgo)
- To fine-tune or initialise from a checkpoint, `initialization.path_to_ckpt` with `load_denoiser`, `load_rew_end_model` and `load_actor_critic` flags exists in the config — [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml)
- `training.model_free=true` "unplugs" the world model to run plain model-free RL — [README](https://github.com/eloialonso/diamond)
- Checkpoints are loaded with `torch.load(...)` without `weights_only=True`, as the PyTorch FutureWarning in a user log shows (2024-11-10) — [issue #30](https://github.com/eloialonso/diamond/issues/30)

**Hardware and demo speed**
- The CS:GO model needs 1.5 GB of disk. The "fast" config "runs best on a machine with a CUDA GPU, but can also be run on CPU at reduced fps". `--compile` makes it faster but slows startup. The higher_quality config runs at "10fps on a RTX 3090" — [csgo README](https://github.com/eloialonso/diamond/tree/csgo)
- The paper says the CS:GO model runs at 10 Hz on an RTX 3090 (3 dynamics steps + 10 upsampler steps) — [paper §6](https://arxiv.org/html/2405.12399)
- The demo videos used an RTX 3090; "some people got fast generation with M3 chips"; `--size-multiplier 5` was used in the demo (author, 2024-10-13) — [issue #13](https://github.com/eloialonso/diamond/issues/13)
- One user saw about 0.05 fps when the code fell back to the CPU (i9-12900K) instead of the RTX 4090 — [issue #14](https://github.com/eloialonso/diamond/issues/14)
- `--compile` failed on an RTX A4000; the authors only tested RTX 3090/4090 — [issue #30](https://github.com/eloialonso/diamond/issues/30)
- Apple Silicon needs `PYTORCH_ENABLE_MPS_FALLBACK=1` — [README](https://github.com/eloialonso/diamond)
- The Atari player defaults: `--fps` target 15, imagination horizon 50, adjustable with ↑/↓ — [README](https://github.com/eloialonso/diamond)
- Authors' disclaimer: "We think of our repo as a way to share our research with the community. We did not design it as a fully-featured, playable, user-facing product" (2024-10-24) — [issue #23](https://github.com/eloialonso/diamond/issues/23)

**Training on your own data**
- No generic pipeline exists. Users have to convert their data to the CS:GO HDF5 episode format (`frame_i_x` image, `frame_i_y` multi-hot keys) and adapt `src/process_csgo_tar_files.py` (community guidance, 2024-11-08) — [issue #21](https://github.com/eloialonso/diamond/issues/21)
- A 2025-11-04 request for help training on another game got no answer — [issue #44](https://github.com/eloialonso/diamond/issues/44)

### Inferences
- The Atari checkpoints are trained at 64×64 on Atari pixel statistics and the ALE action set, so none transfer to NES. Only the code and recipe are reusable.
- Loading pickled `.pt` files without `weights_only` is a supply-chain risk to keep in mind if loading third-party fine-tunes.

### Gaps
- The HF card does not say which of the 5 seeds each Atari checkpoint corresponds to.
- No measured fps exists for the "fast" CS:GO config on any named GPU, and none for the Atari player beyond the 15 fps default cap.

## Inference speed and memory: frames/s per GPU, denoising steps, batch scaling

### Takeaway
The only first-party throughput numbers come from the training profile on one RTX 4090 at 64×64:
- 4.2 ms per denoising step and 12.7 ms per next frame (3 steps)
- 20.4 ms per full imagination step (next frame + reward/termination + action) for the actor-critic batch, which the config sets to 32

That implies about 1.6k imagined transitions/s and about 2.5k predicted frames/s per GPU at that batch. There is no published batch-scaling study and nothing on very large batches. Sequential single-stream generation is about 7.4 Hz for a 122M model at 20 steps (A6000) and 10 Hz for CS:GO (RTX 3090).

### Cited Findings
- Imagination step profile, RTX 4090, default hyperparameters — [paper App. I Table 5](https://arxiv.org/html/2405.12399):

| Component | Time |
|---|---|
| Full imagination step | 20.4 ms |
| — next observation prediction (3 denoising steps × 4.2 ms) | 12.7 ms |
| — reward/termination prediction | 7.0 ms |
| — action prediction | 0.7 ms |

- Actor-critic training batch size is 32 (`actor_critic.training.batch_size: 32`). The imagination environment's number of parallel envs equals the data-loader batch size (`self.num_envs = data_loader.batch_sampler.batch_size`) — [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml); [src/envs/world_model_env.py](https://github.com/eloialonso/diamond/blob/main/src/envs/world_model_env.py)
- Step counts: Atari uses 3 Euler NFE per frame. CS:GO uses 3 for dynamics plus 10 for the upsampler (higher_quality) or 1 + 1 Heun (fast). The `world_model_env.diffusion_sampler` section can be edited at inference time without retraining — [paper §5.2, §6](https://arxiv.org/html/2405.12399); [README](https://github.com/eloialonso/diamond); [issue #30](https://github.com/eloialonso/diamond/issues/30)
- Sequential sampling rates on one RTX A6000 at 64×64 (App. M): DIAMOND frame-stack, 122M parameters, 20 denoising steps: 7.4 Hz. DreamerV3: 266.7 Hz. IRIS K=16: 4.2 Hz — [paper App. M Table 8](https://arxiv.org/html/2405.12399)
- Memory: about 12 GB VRAM for a full Atari training run — [paper §4.1](https://arxiv.org/html/2405.12399). The CS:GO training config was reduced in November 2024 to fit 24 GB — [issue #29](https://github.com/eloialonso/diamond/issues/29)
- The authors' own successor work names inference cost as the main drawback: "The iterative characteristics of the diffusion process necessitate multiple forward passes throughout both training and inference phases" (EDELINE, 2025, which inherits DIAMOND's compute profile) — [EDELINE App. A](https://arxiv.org/html/2502.00466v2)

### Inferences
- From Table 5 at batch 32 on an RTX 4090, all my arithmetic:
  - 32 / 12.7 ms ≈ 2,500 next frames/s
  - 32 / 20.4 ms ≈ 1,570 full imagined transitions/s
  - 32 × 3 / 12.7 ms ≈ 7,600 U-Net forward passes/s
- Batch 32 of a 4M-parameter U-Net at 64×64 almost certainly underutilises a 4090, so throughput at batch 1k–10k would be higher. How much higher is unpublished.
- My own FLOP estimate for the 64-channel, 4-level U-Net at 64×64 is about 5 GFLOP per forward pass (≈15 GFLOP per frame at 3 NFE). This is a rough estimate, not measured.
- At that estimate, matching 13,900 frames/s would need about 0.2 PFLOP/s of sustained compute, roughly the whole dense bf16 tensor throughput of an RTX 5090. So even a 64×64 DIAMOND clone would at best match, not beat, the GPU NES emulator's 13,900 decisions/s, before counting reward model and policy cost.
- At native NES resolution (240×224 = 53,760 px, about 13× the pixels of 64×64), a convolutional U-Net's cost scales roughly with pixel count. That would make it an order of magnitude slower than the emulator.

### Gaps
- No first-party or third-party benchmark of DIAMOND's frames/s versus batch size, at large batch (≥1k), in fp16/bf16, or on an RTX 5090 was found.
- The App. I profile does not explicitly state the batch size used, so I took batch 32 from the config.

## Known limitations: authors and GitHub issues (compounding error, failure cases, memory, stochasticity, rare events)

### Takeaway
The authors name three limitations:
- evaluation only on discrete control
- frame stacking as a minimal memory of 4 frames
- separate reward/termination models

The CS:GO section and the issues add four more:
- autoregressive drift out of distribution, especially in rarely visited states
- forgetting state when the view is occluded ("generate a new weapon or area of map")
- inventing physics that rarely appears in the data (mid-air double jumps)
- defaulting to the most common action's effect when an action/state pair is rare or unseen

That last failure matters most for an agent whose bottleneck is rare button presses.

### Cited Findings
**Limitations stated in the paper**
- Section 8:
  1. Discrete control only.
  2. "Frame stacking for conditioning is a minimal mechanism to provide a memory of past observations"; an autoregressive transformer over time would give longer memory.
  3. Reward/termination prediction is not integrated into the diffusion model.
  - Source: [paper §8](https://arxiv.org/html/2405.12399)
- CS:GO failure modes, from [paper §6](https://arxiv.org/html/2405.12399):
  - The model is "more likely to drift out-of-distribution in less frequently visited areas of the map".
  - "Due to the limited memory of the model, approaching walls or losing visibility may cause the model to forget the current state and instead generate a new weapon or area of map."
  - It "wrongly enables successive jumps … since multiple jumps do not appear often enough in the training gameplay".
  - The authors expect scaling to fix many issues, "with the exception of the memory of the model".
- Offline-data issues, from [paper App. M.4](https://arxiv.org/html/2405.12399):
  - Actions are "unstable beyond a few frames" when an action sequence is unlikely under the demonstration data; the model "snaps onto another area of the map".
  - Causal confusion: "slow down" makes the traffic ahead slow down.
- Stochasticity: single-step denoising blurs multimodal futures, e.g. the opponent in Boxing. Multi-step sampling is needed to commit to one mode — [paper §5.2](https://arxiv.org/html/2405.12399)

**Limitations from GitHub issues**
- Compounding error: a CS:GO generation becomes abnormal after about 38 s. Author: "These are autoregressive generations, so the accumulating error can cause the model to be out of distribution"; press Enter to reset (2024-10-24) — [issue #23](https://github.com/eloialonso/diamond/issues/23)
- Rare and unseen actions (author, 2024-12-06): "If the world model has not seen the effect of certain actions in certain states it may lead to modeling the effect of the modal action instead." Examples:
  - a model trained on random actions then played by a human
  - a model trained on expert data then driven by random actions
  - the CS:GO model "does not accurately model random actions such as walking into a wall"
  - Continuous actions were not investigated.
  - Source: [issue #33](https://github.com/eloialonso/diamond/issues/33)
- Freeway scores 0 with the default config unless IRIS's trick is used — [issue #34](https://github.com/eloialonso/diamond/issues/34)
- Return curves on Alien are very noisy (user, 2025-08-23, unanswered) — [issue #43](https://github.com/eloialonso/diamond/issues/43)
- Engineering:
  - DDP scales badly beyond 2 GPUs — [issue #37](https://github.com/eloialonso/diamond/issues/37)
  - Open, unmerged community PRs (2026-10) titled "Condition denoising on sigma_hat after stochastic churn" (#51) and "Avoid NaN denoising loss for empty padding masks" (#52) suggest possible minor sampler/loss bugs; only the titles were read — [issues list](https://github.com/eloialonso/diamond/issues?q=is%3Apr)

**Memory limitation, measured by a successor paper**
- DIAMOND fails MiniGrid MemoryS7/S9 consistency tests — [EDELINE §3](https://arxiv.org/html/2502.00466v2)
- EDELINE scores 4.1× DIAMOND on Crafter (1M steps) — [EDELINE §6](https://arxiv.org/html/2502.00466v2)
- On BankHeist and ViZDoom DeadlyCorridor, DIAMOND's predictions lose task-critical elements (the player, enemies, armor) — [EDELINE §6](https://arxiv.org/html/2502.00466v2)

### Inferences
- For Super Mario Bros, a 4-frame (about 0.27 s) visual context cannot represent state that lives only in RAM or off-screen:
  - hidden blocks and already-hit blocks once scrolled away
  - enemies queued by the level loader
  - star/invincibility timers
  - warp-zone and pipe destinations
  - the loop-maze logic in 4-4/7-4/8-4
  - The user's policy already reads the 2 KB RAM; DIAMOND models pixels only, so RAM would have to be added as a predicted modality or the policy would be pixel-only in imagination.
- The "modal action" failure and the PlayGen mushroom/fire-flower failures (next section) are the same rare-transition problem the user's search-to-policy work has found. A DIAMOND-style model trained on the agent's own data would under-model exactly the rare presses that matter, unless the data is rebalanced toward them.

### Gaps
- There is no quantitative measure of how far DIAMOND's Atari rollouts stay faithful (in frames) for games other than Breakout (App. K).
- There is no study of rare-event fidelity.

## Follow-up work building on DIAMOND (2024–2026), especially Mario/NES or exploration

### Takeaway
The most direct successor is EDELINE (Lee et al., arXiv 2025-02-01, NeurIPS 2025). It adds a Mamba recurrent memory to a DIAMOND-style diffusion world model and reaches mean HNS 1.866 and median 0.817 on Atari 100k. For Super Mario Bros specifically, the relevant diffusion work is not DIAMOND-based:
- PlayGen (arXiv 2024-12-01): an autoregressive DiT at 128×128, 20 fps on an RTX 2060, 50M training transitions
- MarioVGG (2024-09): text-to-video, 2 actions, not real-time
No DIAMOND-based exploration method was found.

### Cited Findings
**EDELINE**
- Authors: Lee, Lin, Sun, Lee (NTHU/NTU). arXiv 2502.00466; v1 2025-02-01, v2 2025-06-15; NeurIPS 2025 — [arXiv](https://arxiv.org/abs/2502.00466); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/115933)
- What it adds to DIAMOND: a Mamba "Recurrent Embedding Module" over the full trajectory; the hidden state is fed to the next-frame predictor through AdaGN, alongside the last L frames; reward and termination become MLP heads on the shared representation. The agent is the same as DIAMOND's (REINFORCE + λ-returns) — [EDELINE §4](https://arxiv.org/html/2502.00466v2)
- Atari 100k, 3 seeds: mean 1.866, median 0.817, IQM 0.940, optimality gap 0.387, superhuman on 13 games. DIAMOND in the same table: 1.459 / 0.373 / 0.641 / 0.480 — [EDELINE Table 1](https://arxiv.org/html/2502.00466v2)
- Also: 4.1× DIAMOND on Crafter; better than DIAMOND on 5 ViZDoom scenarios; "computational efficiency equivalent to that of DIAMOND" — [EDELINE §6, App. A](https://arxiv.org/html/2502.00466v2)
- Code: github.com/LJH-coding/EDELINE — [search result summary of arXiv/NeurIPS pages](https://arxiv.org/pdf/2502.00466). The issue #34 reporter on DIAMOND's repo is also LJH-coding — [issue #34](https://github.com/eloialonso/diamond/issues/34)

**GameNGen (Valevski et al., Google)**
- arXiv 2408.14837, 2024-08-27 (v2 2025-04-24). A concurrent DOOM diffusion game engine: 20 fps on one TPU, PSNR 29.4. Its RL agent's training sessions are recorded, then a diffusion model is trained on them with conditioning augmentation. DIAMOND cites it as concurrent, "at a larger scale" — [GameNGen arXiv](https://arxiv.org/abs/2408.14837); [DIAMOND §7](https://arxiv.org/html/2405.12399)

**PlayGen ("Playable Game Generation", Yang et al.)**
- arXiv 2412.00887, 2024-12-01. Autoregressive DiT-based diffusion; validated on Super Mario Bros (via Mario-AI-Framework) and Doom — [arXiv](https://arxiv.org/abs/2412.00887)
- Super Mario Bros setup — [PlayGen HTML](https://arxiv.org/html/2412.00887v1):
  - 236M frames of transitions collected; 50M "balanced" transitions sampled for training
  - All frames at 128×128
  - 4 DDIM steps give 20 fps on an RTX 2060, single interactive stream
  - Action-accuracy drops only 0.2% after 1000 frames
- Rare transitions failed before dedicated long-tail fixes ("diverse data collection", "balanced data sampling", "self-supervised long-tailed transition learning") — [PlayGen HTML](https://arxiv.org/html/2412.00887v1):
  - Mario did not grow after eating a mushroom
  - Mario could not become Fire Mario after a fire flower
  - Left jumps were wrong because of data imbalance
  - Mario getting stuck at high pipes over-represented those states
- Code: github.com/GreatX3/Playable-Game-Generation — [arXiv](https://arxiv.org/abs/2412.00887)

**MarioVGG (Virtuals Protocol)**
- September 2024. Text-to-video diffusion for Super Mario Bros with only two actions ("run right", "run right and jump"); World 1-1 held out — [HF model card](https://huggingface.co/virtuals-protocol/mario-videogamegen); [Nintendo Life](https://nintendolife.com/news/2024/09/new-ai-model-can-recreate-super-mario-bros-footage-but-its-full-of-glitches)
- Not real-time: "taking 4 seconds to generate 6 frames on an NVIDIA RTX 4090" — [PlayGen §2](https://arxiv.org/html/2412.00887v1)
- Glitches: Mario changes colour or morphs into enemies — [Nintendo Life](https://nintendolife.com/news/2024/09/new-ai-model-can-recreate-super-mario-bros-footage-but-its-full-of-glitches)
- Training set of about 737k frames — secondary press source only — [Gigazine](https://wbgsv0a.gigazine.net/gsc_news/en/20240906-mariovgg-super-mario-bros-generative-ai)

**Hobby GameNGen replication**
- `soheilchavo/mario-world`: "A world model for the original super mario bros (level 1-1), replicating DIFFUSION MODELS ARE REAL-TIME GAME ENGINES". Seen only as a search-result title; not audited — [GitHub](https://github.com/soheilchavo/mario-world)

**Exploration**
- Searches for DIAMOND-based exploration or intrinsic-reward methods (2024–2026) returned nothing relevant. DIAMOND itself explores only with ε = 0.01 greedy collection plus policy entropy (weight 0.001) — [paper App. E](https://arxiv.org/html/2405.12399)

### Inferences
- PlayGen is the closest evidence for a diffusion model of Super Mario Bros. It needed about 50M balanced transitions and explicit long-tail handling to get power-ups right. That suggests a DIAMOND-style model of SMB trained on a policy's own skewed data would mis-model exactly the rare, high-value events.
- PlayGen's 20 fps is single-stream interactive speed on a weak GPU at 128×128. It says nothing about throughput at the 10k-environment batch scale the user runs.
- EDELINE is the natural "DIAMOND with memory" to evaluate if a diffusion world model is ever reused, since its memory addresses the main SMB-relevant gap (off-screen state). It still has the same per-frame diffusion inference cost.

### Gaps
- Semantic Scholar's citation API was rate-limited (HTTP 429), so no full list of papers citing DIAMOND was compiled, and no citation count is given.
- Other likely descendants (e.g. Microsoft's WHAM/Muse game world model, whose authors overlap with DIAMOND's Pearce and Kanervisto) were not verified in this pass and are left out.
- No paper was found that uses a diffusion world model to drive exploration (novelty or rare-event search) in NES games.

## Relevance to the user's setup (12,288 NES consoles on one RTX 5090 at about 13,900 decisions/s, 4 frames per decision, 240×224 screen, 2.75M-parameter GRU PPO policy that reads 2 KB RAM)

### Takeaway
DIAMOND was built for a regime of scarce real data and cheap compute per sample: 100k real steps and a single real env (`num_envs: 1`), with all RL done in imagination. The user's regime is the opposite: real NES steps on the GPU are cheaper than DIAMOND's imagined steps. At published speeds a DIAMOND world model would be slower than the emulator, lower in resolution, blind to RAM, and limited to 4 frames of memory. Its value here would be limited to "what-if" branching or planning from states the emulator cannot cheaply reach, not throughput or sample efficiency.

### Cited Findings
- DIAMOND collects real data with `num_envs: 1`, 100 steps per epoch, 100k steps in total — [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml)
- Imagined throughput: 20.4 ms per imagination step for a batch of 32 (one RTX 4090, 64×64, 3 NFE) — [paper App. I](https://arxiv.org/html/2405.12399); [main trainer.yaml](https://github.com/eloialonso/diamond/blob/main/config/trainer.yaml)
- Observations are 64×64 RGB; the reward model predicts sign(r); 4-frame memory — [paper App. E, G, §8](https://arxiv.org/html/2405.12399)
- The world model generalises poorly to action/state pairs absent from its data — [issue #33](https://github.com/eloialonso/diamond/issues/33)
- Its weakest Atari games are exploration and memory games — [paper Table 1](https://arxiv.org/html/2405.12399)

### Inferences
- Throughput gap: about 1.6k imagined transitions/s (batch 32, RTX 4090, published) versus about 13,900 real decisions/s (user's emulator). Even an optimistic 5–10× large-batch speed-up on a 5090 only reaches parity at 64×64, with reward model and policy cost on top.
- Pixels: NES frames have about 13× the pixels of 64×64. Downsampling to 64×64 would lose small sprites (fireballs, coins, Hammer Bros' hammers), the very "visual details" DIAMOND's title says matter. Keeping native resolution would need the CS:GO-style low-res dynamics + upsampler pipeline and much more compute per frame (CS:GO: 381M parameters, 10 Hz single stream on an RTX 3090).
- RAM: the user's policy uses the 2 KB RAM. DIAMOND has no RAM channel, so either the world model must also predict RAM (unexplored in DIAMOND), or the policy must be pixel-only when trained in imagination.
- Model sizes: the DIAMOND Atari agent (13M parameters total, 4M denoiser) is about 5× the user's 2.75M policy. Running the world model would dominate GPU time, not the policy.
- Reusable parts:
  - EDM preconditioning, which gives stable few-step (1–3 NFE) autoregressive rollouts
  - the frame-stack + AdaGN U-Net, which beat cross-attention in DIAMOND's tests
  - CS:GO stability tricks: context-frame noise augmentation, multi-step autoregressive training, offset noise, low-variance initial noise
  - the MIT-licensed training loop
  - None of the pretrained checkpoints are usable for NES.
