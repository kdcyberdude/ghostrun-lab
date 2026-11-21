# World models trained jointly with the policy: can a learned model solve dynamic pixel games without much model-free policy-gradient training?

Scope note: these are research notes for a single end-to-end agent on dynamic NES games (SMB, Contra, S.C.A.T., Adventure Island, Kage, Batman), trained on one RTX 5090 with an emulator running about 14,000 decisions/s. Numbers are quoted as reported, with benchmark and step budget. "HNS" means human-normalized score. Atari 100k means 100k agent steps, which is 400k frames, or about 2 hours of play.

## 1. What each modern world-model agent learns, how it is trained jointly with acting, and what it reports

### Takeaway
There are two families. (a) Dreamer-style agents (DreamerV3, IRIS, TWM, STORM, DIAMOND, OC-STORM, Dreamer 4) learn a generative model of observations, rewards and termination, then train an actor-critic on imagined rollouts. The actor is still trained by policy gradient (REINFORCE or PMPO), but on imagined data rather than real data. (b) MuZero-style agents (MuZero, Stochastic MuZero, EfficientZero, EZ-V2) learn a value-equivalent latent model that predicts only reward, value and policy, never pixels. They act by tree search and distill the search's visit counts into the policy, so they use no policy gradient at all. Best Atari 100k numbers: EZ-V2 mean 2.428 and median 1.286 HNS. Among pure imagination-trained agents, DIAMOND has mean 1.46 HNS. At large data, MuZero reaches median 2041.1% and mean 4999.2% HNS on 57 games at 20B frames.

### Cited Findings

**MuZero (arXiv Nov 2019; Nature 2020), DeepMind**
- The model predicts three things from a latent state: policy, value and reward. A representation function, a dynamics function and a prediction function are trained by unrolling the model and matching search policies, n-step values and observed rewards. It acts by MCTS. — [MuZero paper](https://arxiv.org/pdf/1911.08265)
- Atari (57 games), large-data setting: MuZero median 2041.1% and mean 4999.2% HNS at 20.0B frames, 12 hours training, 1M training steps. R2D2: 1920.6% / 4024.9% at 37.5B frames. Ape-X: 434.1% / 1695.6% at 22.8B frames. MuZero beat R2D2 on 42 of 57 games. — [MuZero paper, Table 1](https://arxiv.org/pdf/1911.08265)
- Small-data setting: MuZero Reanalyze reached median 731.1% and mean 2168.9% at 200M frames (12 hours, 1M training steps). Reanalyze re-runs MCTS on old trajectories with the latest network to make fresh targets. Comparisons: IMPALA 191.8% median, Rainbow 231.1%, LASER 431%. — [MuZero paper](https://arxiv.org/pdf/1911.08265)
- Compute for each Atari game: "8 TPUs for training and 32 TPUs for selfplay" (third-generation Cloud TPUs). Atari used 50 simulations per move, against 800 for board games. — [MuZero paper, appendix](https://arxiv.org/pdf/1911.08265)
- Value and reward outputs are softmax distributions over a support of size 601. — [MuZero paper, appendix](https://arxiv.org/pdf/1911.08265)

**Stochastic MuZero (ICLR 2022), DeepMind**
- It learns a stochastic model with *afterstates*, for example the backgammon board after a move and before the dice roll, together with discrete chance outcomes. Planning uses stochastic tree search. — [ICLR 2022 listing](https://mlanthology.org/iclr/2022/antonoglou2022iclr-planning); [blog summary](https://d1jw5m564kgkmn.cloudfront.net/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model)
- 2048: about 510,000 average points according to a secondary review. A Papers-with-Code-style listing gives 500,000 for Stochastic MuZero and for AlphaZero with a true simulator, and 300,000 for plain MuZero. Backgammon: "matched or exceeded state of the art" (no number found). — [Synced review](https://syncedreview.com/2022/07/21/deepmind-ucls-stochastic-muzero-achieves-sota-results-in-complex-stochastic-environments/); [PwC mirror](https://astro.paperswithcode.com/paper/planning-in-stochastic-environments-with-a)

**EfficientZero (arXiv Oct 2021; NeurIPS 2021), Tsinghua/Berkeley**
- Built on MuZero. Current arXiv abstract: 194.3% mean and 109.0% median HNS on Atari 100k, "close to DQN's performance at 200 million frames" with 500x less data. — [EfficientZero arXiv](https://arxiv.org/abs/2111.00210)
- The DreamerV3 paper's comparison table lists EfficientZero at mean 190 / median 109 and 1.2 V100 GPU-days on Atari 100k. It flags that EfficientZero uses online planning, non-uniform replay, increased resolution, life information and early resets: "EfficientZero achieves the highest scores but at the cost of complexity and changing the environment configuration." — [DreamerV3 v1, Table T.1](https://arxiv.org/pdf/2301.04104v1)

**EfficientZero V2 (arXiv Mar 2024; v2 Sep 2024, ICML keyword), Tsinghua**
- Components: representation, dynamics (with an action-embedding block), policy (a squashed Gaussian for continuous actions), value and reward. Value and reward are categorical with 51 bins. Search is a sampling-based Gumbel search: 32 simulations and 16 sampled actions in general, 16 simulations and 8 actions on Atari. Search-based value estimation is mixed with multi-step TD. Training unrolls 5 steps with a self-supervised consistency loss. Self-play workers, reanalyze workers and the learner run in parallel. — [EZ-V2 arXiv](https://arxiv.org/html/2403.00564)
- Atari 100k mean / median HNS: EZ-V2 2.428 / 1.286; BBF (model-free) 2.247 / 0.917; EfficientZero 1.945 / 1.090; DreamerV3 1.120 / 0.490. It beats DreamerV3 on 50 of 66 tasks. Vision control (20 DMC tasks): 726.1 against DreamerV3's 498.5. — [EZ-V2](https://arxiv.org/html/2403.00564)
- Compute: experiments ran on an 8x RTX 3090 server. The model has about 1.3M parameters and 4.7x10^7 FLOPs per decision. Training takes 2.7 hours per 100k training steps. — [EZ-V2](https://arxiv.org/html/2403.00564)

**DreamerV3 (arXiv Jan 2023; v2 Apr 2024; Nature 2025), Hafner et al.**
- World model: an RSSM. h_t = f(h_{t-1}, z_{t-1}, a_{t-1}); encoder q(z_t|h_t,x_t); dynamics predictor p(ẑ_t|h_t). The latents are vectors of categorical (softmax) variables with straight-through gradients and a 1% uniform mix. It predicts image reconstruction, reward (symlog two-hot) and a continue flag. Losses: prediction (weight 1), dynamics (weight 1), representation (weight 0.1), with free bits clipped below 1 nat. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Actor-critic is trained purely on imagined latent trajectories: horizon T=16, λ-returns, γ=0.997. The critic uses 255 symexp bins with an EMA regularizer. The actor uses REINFORCE with fixed entropy 3x10^-4. Returns are scaled by max(1, EMA of the 95th−5th percentile range). There is no lookahead planning at act time. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- One fixed hyperparameter set covers 8 domains and over 150 tasks. Atari: 57 games, 200M frames, sticky actions, beats MuZero, Rainbow and IQN. ProcGen: 16 games, 50M frames, matches PPG. DMLab: 30 tasks, 100M frames. Atari 100k beats IRIS, TWM, SPR and SimPLe. Minecraft diamonds from scratch within 100M env steps. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Atari 200M (v1 table, 55 games): DreamerV3 gamer-median 302% and gamer-mean 920%. DreamerV2: 219% / 1149%. Rainbow: 147% / 888%. IQN: 126% / 890%. — [DreamerV3 v1, Table V.1](https://arxiv.org/pdf/2301.04104v1)
- Atari 100k (v1 table): DreamerV3 mean 112% and median 49% HNS. IRIS: 105% / 29%. SPR: 62% / 40%. SimPLe: 33% / 13%. — [DreamerV3 v1, Table S.1](https://arxiv.org/pdf/2301.04104v1)
- Crafter (1M steps): DreamerV3 score 14.5 ± 1.6%, reward 11.7 ± 1.9. — [DreamerV3 v1, Table M.1](https://arxiv.org/pdf/2301.04104v1)
- Minecraft: a 12-item milestone sequence from sparse rewards, episodes up to 36,000 steps, the block-breaking action setting. All Dreamer agents found diamonds within 100M env steps. PPO, tuned IMPALA and tuned Rainbow topped out at the iron pickaxe. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Ablations: Dreamer "relies mostly on the unsupervised reconstruction loss rather than reward and value gradients". The KL objective matters most, then return normalization, then symexp two-hot. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Scaling: six sizes from 12M to 400M parameters. Larger models score higher *and* need fewer environment interactions. Higher replay ratios also reduce interactions needed. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Nature version: DOI s41586-025-08744-2 (2025). The page redirected to a login, so the Nature text itself was not read. — [Nature](https://www.nature.com/articles/s41586-025-08744-2)

**IRIS (arXiv Sep 2022; ICLR 2023 top-5%), Micheli, Alonso, Fleuret**
- The world model is a discrete autoencoder (VQ-VAE) plus an autoregressive GPT-style Transformer over frame tokens. The policy learns in imagination. Atari 100k: mean HNS 1.046, superhuman on 10 of 26 games. — [IRIS arXiv](https://arxiv.org/abs/2209.00588)
- A secondary description says IRIS maps each frame to 4x4 latent tokens. — [Transformer world model survey/paper](https://arxiv.org/html/2503.04416v1)
- Compute: about 7 V100 GPU-days per Atari 100k run, against 0.5 for DreamerV3 and 0.8 for TWM. — [DreamerV3 v1, Table T.1](https://arxiv.org/pdf/2301.04104v1)

**TWM (arXiv Mar 2023; ICLR 2023), Robine et al.**
- A Transformer-XL world model over compact latent states, actions and rewards. Imagined experience trains the policy. — [TWM arXiv](https://arxiv.org/abs/2303.07109)
- Atari 100k: mean 95%, median 50% HNS, 0.8 GPU-days (as tabulated in DreamerV3). — [DreamerV3 v1, Table T.1](https://arxiv.org/pdf/2301.04104v1)

**STORM (arXiv Oct 2023), Zhang et al.**
- A categorical-VAE encoder plus a GPT-like Transformer dynamics model. The policy improves by imagining with the world model. Atari 100k mean HNS 126.7%. Training takes 4.3 hours on a single RTX 3090 for 1.85 hours of real-time interaction. — [STORM arXiv](https://arxiv.org/abs/2310.09615); categorical-VAE description: [EMERALD/related-work summary](https://arxiv.org/html/2507.04075v2)
- DIAMOND's table gives STORM IQM 0.636 and mean 1.266. — [DIAMOND](https://arxiv.org/html/2405.12399)

**DIAMOND (arXiv May 2024), Alonso, Jelley, Micheli et al.**
- The world model is a pixel-space *diffusion* model: an EDM-formulation U-Net conditioned on the last 4 frames, past actions and diffusion time. A separate conv+LSTM model predicts reward and termination. The actor-critic is trained entirely in imagination with REINFORCE plus a value baseline, λ-returns and a 15-step horizon. The loop alternates collecting data, training the world model, and training RL in imagination. — [DIAMOND](https://arxiv.org/html/2405.12399)
- Atari 100k: mean HNS 1.459 and IQM 0.64, superhuman on 11 of 26 games. Comparisons: IRIS 1.046, DreamerV3 1.097, STORM 1.266. — [DIAMOND](https://arxiv.org/html/2405.12399)
- Compute: about 12 GB VRAM and about 2.9 days per run on one RTX 4090. Sampling uses 3 Euler denoising steps per frame. The model has about 13M parameters, against IRIS's 30M. — [DIAMOND](https://arxiv.org/html/2405.12399)
- A CS:GO "game engine" version has 381M parameters, runs at 10 Hz on an RTX 3090, and was trained on static gameplay data. It is not used for agent training. — [DIAMOND](https://arxiv.org/html/2405.12399)
- Weak games: BankHeist 19.7 vs TWM 466.7; Hero 5621.8 vs IRIS 7037.4; Frostbite 274.1 vs TWM 1475.6. Memory is limited to 4 stacked frames. — [DIAMOND](https://arxiv.org/html/2405.12399)

**Other Atari 100k / Crafter successors (2023–2025)**
- Hieros (arXiv Oct 2023) uses an S5 state-space world model with hierarchical imagination. It claims SOTA mean and median HNS on Atari 100k at the time. — [Hieros](https://arxiv.org/pdf/2310.05167)
- Δ-IRIS solves 16 of 22 Crafter tasks after 10M frames, matching DreamerV3. EMERALD (Jul 2025, spatial latent with MaskGIT decoding) reports better Crafter results than DreamerV3 and Δ-IRIS. — [EMERALD](https://arxiv.org/html/2507.04075v2)
- "Improving Transformer World Models for Data-Efficient RL" (Dedieu et al., Feb 2025, ICML 2025) uses Dyna with warmup (real plus imagined data), a nearest-neighbour patch tokenizer and block teacher forcing. Craftax-Classic at 1M steps: 69.66% reward, against DreamerV3 53.2% and human 65.0%. — [arXiv 2502.01591](https://arxiv.org/abs/2502.01591)

**Dreamer 4 (arXiv Sep 2025), Hafner, Yan, Lillicrap**
- A causal tokenizer compresses video into continuous latents. A block-causal interactive dynamics transformer is trained with a "shortcut forcing" objective and predicts clean latents. Policy, reward and value heads read from agent tokens. It runs real-time interactive inference on a single GPU. — [Dreamer 4](https://arxiv.org/html/2509.24527v1)
- Training: pretraining on the VPT contractor dataset (2541 hours at 20 FPS); action conditioning learned from a small labelled subset; behaviour cloning plus reward-model finetuning; then RL in imagination with PMPO against a frozen behavioural prior, with the transformer frozen. — [Dreamer 4](https://arxiv.org/html/2509.24527v1)
- Results from purely offline data (no environment interaction), over 1000 episodes of 60 minutes: over 90% reach the stone pickaxe, 29% the iron pickaxe, and 0.7% get diamonds. Baselines: VPT (finetuned) "sticks at 53%"; a Gemma-3 VLA gets the iron pickaxe 11% of the time. It uses about 100x less data than VPT. — [Dreamer 4](https://arxiv.org/html/2509.24527v1)
- Compute: about 2B parameters (400M tokenizer plus 1.6B dynamics), trained on 256–1024 TPU-v5p. — [Dreamer 4](https://arxiv.org/html/2509.24527v1)

**World models as game engines (not agent-training loops)**
- GameNGen (Aug 2024; ICLR 2025) is a diffusion model of DOOM conditioned on past frames and actions. Its training data was recorded from an RL agent's training sessions. It runs at 20 FPS on one TPU with next-frame PSNR 29.4. Raters were near chance at telling real from generated clips. The source does not report using it to train agents. — [GameNGen](https://arxiv.org/abs/2408.14837)
- Genie 3 (Aug 2025): DeepMind ran the SIMA agent inside Genie 3 worlds to pursue goals via navigation actions. Sources describe this as evaluation, not training. Training agents in "an unlimited curriculum" is a stated aim, not a reported result. SIMA 2 (Nov 2025) acted in Genie 3-generated worlds. — [SiliconANGLE](https://siliconangle.com/2025/08/05/google-deepmind-debuts-genie-3-model-generating-interactive-virtual-worlds/); [TechCrunch](https://techcrunch.com/2025/08/05/deepmind-reveals-genie-3-a-world-model-that-could-be-the-key-to-reaching-agi); [Decrypt on SIMA 2](https://decrypt.co/348569/google-deepmind-ai-agent-learns-adapts-plays-games-like-human)

### Inferences
- "No PPO" is achievable with either family. "No policy gradient" holds only for the MuZero family, which uses search plus distillation. Dreamer-family agents still run REINFORCE- or PMPO-style policy gradients, just on imagined rollouts.
- The strongest *data-efficient* Atari results come from search-based agents (EZ-V2 at 2.43 mean HNS), not pure imagination agents (DIAMOND at 1.46). MuZero is the only world-model agent with a documented result at billions of frames.

### Gaps
- I could not open the Nature DreamerV3 text, so any Nature-only numbers are missing. arXiv v1/v2 numbers are used instead.
- STORM's median HNS and its NeurIPS 2023 venue, and DIAMOND's venue (believed NeurIPS 2024), were not confirmed from the fetched pages.
- The EfficientZero v1 original-abstract numbers (sometimes cited as 190.4% / 116.0%) differ from the current abstract (194.3% / 109.0%). The current arXiv version is used here.
- I found no exact Stochastic MuZero backgammon win rates or Atari numbers.

## 2. Can planning/search inside a learned model alone solve games? Where it works and where it fails (hard exploration)

### Takeaway
Search in a learned model (MuZero, EfficientZero) is excellent on dense-reward Atari but does **not** fix hard exploration. MuZero scores **0.00 on Montezuma's Revenge and 0.00 on Pitfall** after 20B frames. DreamerV3 gets only 2512 on Montezuma at 200M frames. The world-model method with strong reported hard-exploration results, BYOL-Explore, uses the model as a *curiosity signal*, not as a planner. Minecraft diamonds were solved by DreamerV3 only with a 12-milestone reward sequence.

### Cited Findings
- MuZero at 20B frames, no-op starts (per-game Table S1) against R2D2 and human:
  - Montezuma's Revenge: **0.00**, against R2D2 2,061.30 and human 4,753.33.
  - Pitfall: 0.00 (R2D2 0.00, human 6,463.69).
  - Private Eye: 15,299.98 (R2D2 5,322.70, human 69,571.27; 22.0% normalized).
  - Venture: 0.40 (R2D2 1,970.70).
  - Solaris: 56.62 (R2D2 3,787.20).
  - Skiing: −29,968.36 (−100.9%).
  - Gravitar: 6,682.70 (204.8%).
  - Human-start evaluation (Table S2): Montezuma 57.10, Pitfall −801.10, Private Eye 5,067.59.
  - [MuZero paper, Tables S1/S2](https://arxiv.org/pdf/1911.08265)
- DreamerV3 at 200M frames (Table V.1, columns record / IQN / Rainbow / DreamerV2 / DreamerV3):
  - Montezuma's Revenge: record 1,219,200; IQN 495; Rainbow 487; DreamerV2 79; DreamerV3 **2512**.
  - Pitfall: −2 for DreamerV2, −0 for DreamerV3.
  - Private Eye: IQN 4286; Rainbow 21341; DreamerV2 2073; DreamerV3 5538.
  - Venture: DreamerV2 and DreamerV3 both 0.
  - Skiing: DreamerV3 −9623.
  - Solaris: DreamerV3 2453.
  - Gravitar: DreamerV3 560.
  - [DreamerV3 v1, Table V.1](https://arxiv.org/pdf/2301.04104v1)
- Atari 100k Private Eye (human 69,571): SimPLe 35, CURL 82, SPR 86, IRIS 100, DreamerV3 882. — [DreamerV3 v1, Table S.1](https://arxiv.org/pdf/2301.04104v1)
- DreamerV3 includes Montezuma among its 14 ablation tasks. — [DreamerV3 v1 (ablation figures D.1–D.3)](https://arxiv.org/pdf/2301.04104v1)
- BYOL-Explore (Jun 2022, DeepMind) learns "a world representation, the world dynamics, and an exploration policy all-together by optimizing a single prediction loss in the latent space". Prediction error is added to the extrinsic reward as an intrinsic bonus. It reports "superhuman performance on the ten hardest exploration games" in Atari and solves "the majority" of DM-HARD-8 tasks. — [BYOL-Explore](https://arxiv.org/abs/2206.08332)
- DreamerV3's Minecraft diamond task uses a "12-item sequence" of sparse milestone rewards and the simplified block-breaking action setting. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Dreamer 4 gets diamonds in only 0.7% of offline-trained episodes, even with 2541 hours of human video and behaviour-cloning priors. — [Dreamer 4](https://arxiv.org/html/2509.24527v1)
- MuZero search depth: Atari used 50 simulations per move, and the model was trained over 5 hypothetical (unrolled) steps. — [MuZero paper](https://arxiv.org/pdf/1911.08265)

### Inferences
- A learned model plans well only where data has already been collected. If the agent never reaches a reward or state, neither the model nor search can generalize to it. MuZero's 0 on Montezuma and Pitfall at 20B frames is the clearest evidence. For NES games, where progress needs long precise sequences (pits, enemies, jumps), search inside a learned model will not replace an exploration mechanism such as novelty or position-coverage rewards, curriculum or backward starts.
- Short planning horizons (5-step unrolls, 15–16-step imagination horizons) mean long-horizon credit flows through learned value functions, not through the model's foresight.
- Private Eye shows the same pattern. All learned-model agents remain far below human (≤22% normalized for MuZero at 20B frames).

### Gaps
- I did not obtain BYOL-Explore's per-game numbers (Montezuma, Pitfall, Private Eye), its RL algorithm or its frame budget. Only the abstract was read.
- I found no published result where pure search in a learned model (no intrinsic reward, no demonstrations) solves Montezuma or Pitfall.
- I did not research Go-Explore-style methods that use a learned model instead of emulator restore.

## 3. Handling many independently moving objects and stochasticity

### Takeaway
Reconstruction-based world models spend capacity on large background areas and drop small, fast, decision-critical sprites such as balls, bullets and enemies. DreamerV3 scores 31 on Breakout at 100k against IRIS's 84 and DIAMOND's 132.5. IRIS hallucinates enemies as rewards in Asterix. Fixes that work: higher-fidelity pixel models (DIAMOND diffusion), object-centric inputs from a pretrained segmenter (OC-STORM), and, for stochasticity, multi-step diffusion sampling or explicit chance nodes (Stochastic MuZero).

### Cited Findings
- DIAMOND attributes its gains to visual consistency on small details. Asterix 3698.5 vs IRIS 853.6; Breakout 132.5 vs 83.7; Road Runner 20673.2 vs 9614.6. IRIS imagined trajectories show "enemies rendered as rewards in Asterix", inconsistent bricks and score in Breakout, and inconsistent rewards in Road Runner. — [DIAMOND](https://arxiv.org/html/2405.12399)
- Stochasticity: in Boxing, single-step denoising blurs the unpredictable opponent, so multi-step sampling is needed (Boxing 86.9 vs IRIS 70.1). EDM stays stable with few denoising steps, while DDPM drifts. — [DIAMOND](https://arxiv.org/html/2405.12399)
- DreamerV3 Atari 100k: Breakout 31, IRIS 84. Freeway 0, IRIS 31 (IRIS used a separate exploration setting for Freeway). Asterix 932. — [DreamerV3 v1, Tables S.1/T.1](https://arxiv.org/pdf/2301.04104v1)
- DreamerV3 Atari 200M against MuZero 20B: Breakout 300 vs 864.00; Asterix 39643 vs 998,425.00. The frame budgets differ by 100x. — [DreamerV3 v1](https://arxiv.org/pdf/2301.04104v1); [MuZero](https://arxiv.org/pdf/1911.08265)
- OC-STORM (Jan 2025; ICLR 2026 under the title "Object-Centric World Models from Few-Shot Annotations for Sample-Efficient RL") motivates its design this way: L2 pixel reconstruction is "dominated by large areas", so small dynamic elements are missed. It adds object feature vectors from a frozen Cutie video-segmentation model (first 8 foreground memory features, 2048-d per object), needing 6 annotated masks per Atari game and 12 per Hollow Knight boss. These are combined with 64x64 pixels. A spatial-temporal transformer attends across objects within a timestep and causally across time, and predicts next latent, reward and termination. The actor-critic follows DreamerV3. — [OC-STORM](https://arxiv.org/html/2501.16443v1); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10007210)
- OC-STORM on Atari 100k beats the re-run STORM on 18 of 26 games. Mean HNS: OC-STORM 134.8%, STORM* 114.2%, DIAMOND 146%, DreamerV3 125%, IRIS 105%. Median: 43.8% vs STORM* 42.5%. On object-like games (Krull, Qbert, RoadRunner, KungFuMaster) the mean rose from 116.5% to 142.8%. On others (Alien, Gopher, Hero, MsPacman) it was on par or slightly worse (124.0% vs 130.0%), and Hero dropped. — [OC-STORM](https://arxiv.org/html/2501.16443v1)
- OC-STORM on Hollow Knight, at 100k samples (about 3.1 h at 9 FPS):
  - Mage Lord: 19.6 → 28.0.
  - Pure Vessel: 8.9 → 15.7, and 35.0 at 400k samples.
  - Hornet Protector: win rate 66.7% → 100%.
  - [OC-STORM](https://arxiv.org/html/2501.16443v1)
- OC-STORM's stated limitations: duplicate instances such as the two Mantis Lords can merge into one feature. Walls, tunnels and map layout are not object-like, so raw pixels are still needed. Non-contact interactions are not captured. Unsupervised slot methods detect objects poorly in noisy scenes; Nakano et al. 2024 added slot attention to STORM. — [OC-STORM](https://arxiv.org/html/2501.16443v1)
- Stochastic MuZero models environment randomness with afterstates and chance outcomes inside the tree search. — [ICLR 2022](https://mlanthology.org/iclr/2022/antonoglou2022iclr-planning)
- DreamerV3 uses categorical (discrete) stochastic latents with 1% uniform mixing to represent multimodal futures. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)

### Inferences
- NES action games (Contra bullets, S.C.A.T. enemy swarms, small projectiles on 8x8 to 16x16 sprites over a scrolling background) are exactly the regime where reconstruction-loss world models lose the decision-critical pixels. One cheap object-centric signal is available here and needs no segmenter: the emulator's RAM (object tables and positions) could act as privileged object features, much as OC-STORM uses Cutie features. This is an adaptation, not something any paper tested on NES.
- Diffusion pixel models handle multimodality better but cost far more per imagined step (3 denoising passes per frame, and 2.9 GPU-days for 100k steps).

### Gaps
- I did not obtain results for SLATE/STEVE-based agents, SOLD, or OCRL-benchmark numbers on games with many moving objects.
- I found no study of world-model accuracy specifically on bullet-hell or crowded shooters beyond Atari (Asterix, Demon Attack, Chopper Command per-game numbers exist, but no analysis).

## 4. Compute, wall-clock, and scaling to hundreds of millions or billions of environment steps

### Takeaway
Most world-model agents are built for *sample efficiency* (100k steps) and are learner-bound at high replay ratios. They process only single to tens of environment steps per second. Two exceptions are documented: DreamerV3 at 200M Atari frames (16 V100-days per game) and Minecraft at 100M steps (17 V100-days), and MuZero at 20B frames using 40 TPUs for 12 hours per game. Nothing found shows a Dreamer-style agent running at billions of steps on one GPU.

### Cited Findings
- DreamerV3 benchmark table (all on 1 V100 each):
  - Atari 100K: 400K env steps, train ratio 1024, <1 GPU-day, size S.
  - Atari 200M: 200M steps, action repeat 4, 8 env instances, train ratio 64, 16 GPU-days, size XL.
  - DMLab: 50M steps, train ratio 64, 4 GPU-days.
  - Minecraft: 100M steps, 16 instances, train ratio 16, 17 GPU-days.
  - Crafter: 1M steps, train ratio 512, 2 GPU-days.
  - Train ratio means replayed steps per policy step.
  - [DreamerV3 v1, Table A.1](https://arxiv.org/pdf/2301.04104v1)
- VPT used "720 V100 GPUs for 9 days" for a 2.5% diamond rate. DreamerV3 "learns to collect diamonds in 17 GPU days from sparse rewards and without human data". The v2 text cites one A100 and about 9 days. — [DreamerV3 v1](https://arxiv.org/pdf/2301.04104v1); [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)
- Atari 100k GPU-days (V100-equivalent): SimPLe 10, EfficientZero 1.2, SPR 0.2, IRIS 7, TWM 0.8, DreamerV3 0.5. — [DreamerV3 v1, Table T.1](https://arxiv.org/pdf/2301.04104v1)
- STORM: 4.3 h on one RTX 3090 per 100k-step run. DIAMOND: about 2.9 days on one RTX 4090 per run, about 1.03 GPU-years for all experiments. EZ-V2: 2.7 h per 100k training steps on an 8x3090 server. — [STORM](https://arxiv.org/abs/2310.09615); [DIAMOND](https://arxiv.org/html/2405.12399); [EZ-V2](https://arxiv.org/html/2403.00564)
- MuZero: 20.0B frames in 12 hours and 1M training steps per game, with 8 TPUs training and 32 TPUs self-play. — [MuZero paper](https://arxiv.org/pdf/1911.08265)
- Dreamer 4: about 2B parameters on 256–1024 TPU-v5p. Its world model runs in real time on one GPU at inference. — [Dreamer 4](https://arxiv.org/html/2509.24527v1)
- Scaling DreamerV3 from 12M to 400M parameters increases final score and reduces the environment steps needed. — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2)

### Inferences (arithmetic from the cited numbers)
- DreamerV3 Atari 200M is 50M agent steps in 16 V100-days, about 36 agent steps/s. Minecraft is 100M steps in 17 days, about 68 steps/s. STORM runs about 6.5 steps/s at 100k and DIAMOND about 0.4 steps/s. The user's emulator delivers about 14,000 decisions/s, which is about 1.2B per day. A Dreamer-style learner at published replay ratios would therefore be roughly 100–1000x slower than the emulator, even allowing several-fold speedups from a 5090 over a V100. Using the emulator's throughput needs a much lower train ratio (DreamerV3 already drops to 16 for Minecraft), a small model, and large batched env counts. None of these has published evidence at more than 1B steps.
- MuZero at 20B frames is about 460k frames/s, but on 40 TPUv3. A single-GPU MuZero/EZ-V2 is bottlenecked by MCTS per decision; EZ-V2 uses 16 simulations per Atari move.
- With a cheap ground-truth simulator, the main advantage of a world model is planning or representation, not sample savings. The DreamerV3 ablation (it relies mostly on reconstruction) suggests the representation-learning benefit is real.

### Gaps
- I found no published Dreamer/IRIS/STORM/DIAMOND run at 1B+ environment steps, and no throughput benchmark on a single consumer GPU at that scale.
- I found no GPU-hours for EZ-V2 per full run, and no 5090-specific numbers.

## 5. Generalization across levels and games; NES / Super Mario evidence

### Takeaway
DreamerV3 handles procedurally generated levels (ProcGen at 50M frames, matching PPG; Crafter 14.5% at 1M steps; Minecraft random worlds) with one hyperparameter set. Each game is still trained separately, and I found no multi-game world-model agent result on Atari. I found **no peer-reviewed world-model agent result on Super Mario Bros or any NES game**. The only Mario item is a community JEPA world-model project with no RL heads. Dreamer 4 shows that pretraining on large unlabelled video plus a little action-labelled data can transfer to control, at TPU-pod scale.

### Cited Findings
- DreamerV3: ProcGen 16 games, 50M frames, "matches PPG and beats Rainbow". Crafter (procedurally generated) 14.5 ± 1.6% at 1M. Minecraft uses randomly generated infinite worlds. DMLab 30 tasks at 100M frames "exceeds IMPALA and R2D2+ at 1B steps". — [DreamerV3 v2](https://arxiv.org/html/2301.04104v2); [DreamerV3 v1](https://arxiv.org/pdf/2301.04104v1)
- Crafter/Craftax successors: Δ-IRIS 16/22 tasks at 10M frames; EMERALD above DreamerV3 and Δ-IRIS; Dedieu et al. 69.66% on Craftax-Classic at 1M (above human 65.0%). — [EMERALD](https://arxiv.org/html/2507.04075v2); [arXiv 2502.01591](https://arxiv.org/abs/2502.01591)
- Dreamer 4 learns action conditioning from a small labelled subset and "most of its knowledge from diverse unlabeled videos". It is trained purely offline. — [Dreamer 4 abstract](https://arxiv.org/abs/2509.24527)
- Genie 3 worlds were used to *evaluate* SIMA agents on navigation goals. There is no report of training an agent end-to-end inside Genie 3. — [SiliconANGLE](https://siliconangle.com/2025/08/05/google-deepmind-debuts-genie-3-model-generating-interactive-virtual-worlds/)
- Mario: searches turned up no world-model agent paper on SMB. "LeMario" (2026 community post) trains a JEPA world model on Super Mario Bros without rewards. The author's takeaway is that predicting the game is not the same as learning to make progress, and it lacks Dreamer-style RL heads. A Stanford CS224R project on Mario from pixels compared DQN and PPO (PPO best), with no world model. — [LeMario post](https://www.promptzone.com/anika_bhat/lemario-trains-jepa-world-model-on-super-mario-bros-1d42.md); [CS224R project](https://cs224r.stanford.edu/projects/pdfs/CS224R_Project-1.pdf)

### Inferences
- For multi-level NES play, the evidence supports a world model mainly as a *representation and auxiliary-learning* component of one agent, as in DreamerV3's reliance on reconstruction. It is not evidence that a world model alone will generalize across levels or games. Cross-level transfer claims would have to be tested directly.
- No source shows a learned NES world model accurate enough to replace the emulator. Since the emulator is fast and exact, imagination rollouts mostly add cost unless used for planning or for representation learning.

### Gaps
- I found no multi-game Atari world-model agent results, and no ProcGen per-game or generalization-gap (train vs test levels) numbers for DreamerV3.
- LeMario is unverified and not peer-reviewed. I found no Contra, Adventure Island, Kage, Batman or S.C.A.T. world-model results at all.
