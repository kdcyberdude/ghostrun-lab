# Combining a learned world model with a high-throughput PPO agent on retro games (single consumer GPU): recipes, measured benefits, costs

Research date: 2026-11-19. Scope: how a world model (WM) could be used with the existing GhostRun PPO system, and what it would cost.

Local context used for the inferences (from the repo and project memory, not from the web):
- Policy: 2.75M-parameter GRU PPO model reading 240x224 frames plus, with `--ram-io`, the 2 KB RAM; an earlier auxiliary head `ram_logits` predicted which RAM bits the chosen move flips by the next decision (GRU output + move, low-rank bilinear readout, `ram_coef` 0.5 "nats per RAM byte", target `flip` or `next`) — `rl/model.py` lines 8-24, `rl/train.py` lines 72-78.
- Measured speed (memory note `training-speed.md`, 2026-11-15): training runs at 13.9k decisions/s (= 55.6k frames/s) with 12,288 consoles, MFU 24%. The emulator alone is not the bottleneck: env-only 52k decisions/s at 4,096 consoles, 101k at 8,192, 190k at 16,384 (render on). The encoder forward is ~77k decisions/s (0.9 GFLOP/decision); learning is the cap (~24k decisions/s for one epoch).

Markers: items tagged "(abstract, recalled; not re-fetched 2026-11-19)" are well-known paper abstracts that I did not re-open in this session; the URL is the primary source and should be spot-checked before quoting numbers.

## 1. Ways to use a world model with PPO, their measured benefits, and which help a strong high-throughput model-free agent versus only when samples are scarce

### Takeaway
The measured gains of "training in imagination" (SimPLe, IRIS, DreamerV3, STORM, DIAMOND) are all reported in the sample-scarce regime (Atari 100k = 400k frames); the published evidence that a model helps an already high-throughput policy-gradient agent comes from using the model as an auxiliary loss and/or to compute better learning targets (Muesli, PBL, UNREAL, ATC), not from imagined rollouts or action-time search. Prediction heads for game "measurements"/events (health, ammo, frags) have a strong competitive result (DFP, VizDoom 2016) and are the cheapest form of "predict what my move will do".

### Cited Findings
Training in imagination (Dreamer/DIAMOND/IRIS/STORM style) — all numbers are Atari 100k:
- Atari 100k human-normalized mean / median: SimPLe 33% / 13%; TWM 96% / 51%; IRIS 105% / 29%; DreamerV3 112% / 49%; STORM 126.7% / 58.4% (STORM, NeurIPS 2023, Table 2) — [STORM, arXiv 2310.09615](https://arxiv.org/html/2310.09615)
- DIAMOND (diffusion WM, arXiv May 2024): mean HNS 1.459, IQM 0.641 on Atari 100k; 64x64x3 observations; conditioned on the last 4 frames and actions; imagination horizon H=15; batch size 32; 3 denoising steps (Euler sampler). The paper lists as limitations compounding error (less with EDM than DDPM), frame-stacking as only memory, discrete-control-only evaluation, and that reward/termination are not integrated into the diffusion model — [DIAMOND, arXiv 2405.12399](https://arxiv.org/html/2405.12399)
- DreamerV3 reports that Dreamer "substantially outperforms a high-quality implementation of PPO" — this is a sample-efficiency comparison, not a wall-clock one — [DreamerV3, arXiv 2301.04104](https://arxiv.org/html/2301.04104v2)
- "When to use parametric models in RL?" (van Hasselt, Hessel, Aslanides, NeurIPS 2019): when a model is used only to generate imagined transitions from observed states for a model-free update, replay should be competitive or better; in a like-for-like Atari comparison, (data-efficient) Rainbow DQN with replay outperformed SimPLe "with less experience and computation". Models are argued to be more useful for backward planning (credit assignment) and forward planning for immediate behaviour — [arXiv 1906.05243](https://arxiv.org/pdf/1906.05243); [NeurIPS abstract](https://proceedings.neurips.cc/paper/2019/hash/1b742ae215adf18b75449c6e272fd92d-Abstract.html)

Planning / lookahead / MCTS with a learned model:
- Hamrick et al. (ICLR 2021) ablating MuZero on control, Atari and 9x9 Go: search is most useful for constructing policy-learning targets and for producing a more informative data distribution; shallow trees with simple Monte-Carlo rollouts perform about as well as complex search except on the hardest reasoning tasks; search at evaluation time only slightly improves zero-shot generalization and only with a highly accurate model — [arXiv 2011.04021](https://arxiv.org/pdf/2011.04021)
- Muesli (Hessel et al., ICML 2021): a policy-gradient agent (IMPALA network, Sebulba actor-learner, replay + Retrace) with a MuZero-style model trained as an auxiliary loss (not reconstructing observations) and used only for one-step look-ahead targets; actions are sampled from the fast policy network with no action-time search on Atari. 57 games, 200M frames: median HNS 1,041% (+/-40%) vs MuZero Reanalyse 1,047% (+/-40%). Abstract: "computation speed comparable to model-free baselines" — [Muesli, arXiv 2104.06159](https://ar5iv.labs.arxiv.org/html/2104.06159)
- Muesli ablations (Fig. 7a, no numeric magnitudes in text): performance degraded when the model was trained for only one step, and degraded further if the model was not trained to output policy predictions; replacing model-based values with model-free values in a V-trace estimator "performed worse" (Fig. 6a) — [Muesli](https://ar5iv.labs.arxiv.org/html/2104.06159)

Auxiliary prediction losses on the policy representation:
- PBL (Guo et al., ICML 2020): latent-predictive auxiliary loss on PopArt-IMPALA in multitask DMLab-30 and Atari-57 (no task ID given to agent); authors report "across-the-board improved performance" over pixel control (the prior state-of-the-art auxiliary for DMLab-30) and DRAW/Simcore baselines; both RL and representation loss train the same RNN — [arXiv 2004.14646](https://arxiv.org/pdf/2004.14646); [PMLR](https://proceedings.mlr.press/v119/guo20g.html) (exact scores not retrieved; see Gaps)
- ATC (Stooke et al., ICML 2021) with PPO on Atari (sticky actions, 16 envs, 4 epochs): ATC as an auxiliary loss or as weight initialization improved 6 of 8 games significantly; encoder trained only by ATC (RL gradients stopped) matched end-to-end RL in 5 of 8 but did poorly in Breakout and Space Invaders; on DMLab Lasertag the RL-only agent stayed at zero score while detached ATC learned — [arXiv 2009.08319](https://arxiv.org/html/2009.08319v3)
- Shelhamer et al. (2016) "Loss is its own Reward": policies trained with self-supervision "converge to the same or better return" in fewer updates — [arXiv 1612.07307](https://arxiv.org/pdf/1612.07307)
- UNREAL (Jaderberg et al., 2016; A3C + pixel control + reward prediction + value replay): 880% mean expert-human on Atari; on Labyrinth a 10x mean learning speed-up and 87% expert-human (abstract, recalled; not re-fetched 2026-11-19) — [arXiv 1611.05397](https://arxiv.org/abs/1611.05397)
- SPR (Schwarzer et al., ICLR 2021): median HNS 0.415 on Atari 100k, a 55% relative improvement over prior SOTA — a sample-scarce result with a value-based agent (abstract, recalled; not re-fetched) — [arXiv 2007.05929](https://arxiv.org/abs/2007.05929)
- CURL / RAD (2020): CURL reported gains at 100k steps on DMControl and Atari; RAD then reported that data augmentation alone matches or beats CURL, i.e. the contrastive auxiliary was not the main ingredient (abstracts, recalled; not re-fetched) — [CURL arXiv 2004.04136](https://arxiv.org/abs/2004.04136); [RAD arXiv 2004.14990](https://arxiv.org/abs/2004.14990)
- EfficientZero (NeurIPS 2021): notes that earlier work found auxiliary losses had "mixed effects" (tested on low-dimensional state tasks), whereas their self-supervised consistency loss is "critical in most environments with high dimensional observations and limited data" — [arXiv 2111.00210](https://arxiv.org/pdf/2111.00210)

Event / affordance / measurement prediction heads:
- Direct Future Prediction (Dosovitskiy & Koltun, ICLR 2017): the network predicts future values of measurements (ammo, health, frags) at several horizons given the action; the action is chosen by a goal-weighted combination of predictions (weights changeable at test time). It won the Full Deathmatch track of the 2016 Visual Doom AI Competition (bot "IntelAct"; weapon selection hard-coded) — [ICLR 2017 paper](https://ICLR.cc/archive/www/lib/exe/fetch.php%3Fmedia=iclr2017:dosovitskiy_koltun_learning_to_act_iclr_2017.pdf); [ViZDoom competitions review, arXiv 1809.03470](https://arxiv.org/pdf/1809.03470)
- A follow-up on temporally extended auxiliary tasks in VizDoom found performance "highly sensitive to the weighting placed on their losses" — [arXiv 2004.00600](https://ar5iv.labs.arxiv.org/html/2004.00600)

Curiosity / intrinsic reward from prediction error:
- Burda et al. "Large-Scale Study of Curiosity-Driven Learning" (2018, PPO, 54 envs): prediction error in random-feature space works well; pixel-space prediction is worse; prediction-error curiosity is vulnerable to stochastic "noisy TV" sources. RND (Burda et al. 2018) exceeded average human on Montezuma's Revenge without demonstrations (abstracts, recalled; not re-fetched) — [arXiv 1808.04355](https://arxiv.org/abs/1808.04355); [RND arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
- (Detailed exploration evidence for this project is in the sibling folder `research_notes/Controllable exploration for NES agents/`; project memory's 2026-11-19 verdict was that a world model alone will not find rare button presses.)

### Inferences
- Ranking by evidence that it helps a strong, data-rich policy-gradient agent: (a) model/latent-prediction as an auxiliary loss on the shared core (Muesli, PBL, UNREAL, ATC) — positive at 200M-frame / multitask scale; (b) model used to compute better learning targets (Muesli one-step look-ahead, MuZero targets per Hamrick) — positive, but needs a value/policy-predicting model, not a pixel generator; (c) action-time search — small gains except on reasoning-heavy tasks; (d) training in imagination — the gains reported are in sample-scarce settings, and van Hasselt et al. show replay already captures most of that benefit.
- With 13.9k decisions/s, GhostRun consumes the entire Atari 100k budget (100k decisions) in ~7 seconds. Every imagination-training result above was measured in a regime the project is ~4 orders of magnitude past; there is no evidence they transfer.
- For the project's failure mode (rare presses / walls, per memory), the closest validated recipe is DFP-style event heads: predict per-move, multi-horizon events read from RAM (death, damage/power-down, room/area change, score change, x-progress) from the GRU state + candidate move. This is option (3) generalised from "all 16,384 RAM bits" to a handful of semantic events, and it is the option with the strongest competitive precedent and the lowest compute.
- Muesli's ablation suggests that if a model head is kept, a multi-step unrolled latent model that predicts value/policy/reward (MuZero-style) is the variant that mattered; a one-step next-observation (or next-RAM) target is the weakest variant in that ablation family.

### Gaps
- No primary source found that reports imagination-training (Dreamer/DIAMOND) helping an agent that already has >100M cheap frames; I found none that tested it at that scale.
- Muesli's ablation magnitudes and PBL's DMLab-30 numbers were not retrieved (text only says "degraded" / "across-the-board").
- No study found that compares an event-prediction head against a full next-state head inside the same PPO agent.

## 2. Throughput: pixel world models versus a fast emulator with save/restore

### Takeaway
Published pixel world models train in hours to days per game on one GPU for only 100k real steps, and the large interactive ones generate ~10-20 frames/s per stream; none reports batched generation anywhere near the GhostRun emulator (55.6k frames/s used in training, up to ~760k frames/s env-only). Because GhostRun is limited by learning compute on the same GPU, not by the emulator, a learned simulator would compete for the bottleneck resource and cannot be faster or more exact than the real emulator with save/restore.

### Cited Findings
- Training cost per Atari-100k game (STORM Table 12): SimPLe 20 days on P100; TWM 10 h on A100 (12.5 h on RTX 3090); IRIS 7 days for two runs on A100; DreamerV3 12 h on V100; STORM 4.3 h on RTX 3090 (V100-equivalent 9.3 h). Some entries are extrapolated assuming P100 = 2x slower and A100 = 2x faster than V100 — [STORM, arXiv 2310.09615](https://arxiv.org/html/2310.09615)
- DIAMOND: ~2.9 days per game on one RTX 4090 (2.5 days training + 0.4 days collection/eval/checkpointing; "1.03 GPU years in total" across experiments); DIAMOND 13M params total (4M diffusion model), IRIS 30M params / 4.1 training days, DreamerV3 18M params / "<1" day; DIAMOND needs 3 NFE per frame vs 16 NFE per frame for IRIS; DIAMOND "trains faster than IRIS, although is slower than DreamerV3". No Atari frames-per-second figure is given — [DIAMOND, arXiv 2405.12399](https://arxiv.org/html/2405.12399)
- Note on conflicting IRIS cost: STORM lists IRIS as 7 days for two runs on A100 (168 V100-hours extrapolated); DIAMOND's Table 4 lists 4.1 days — [STORM](https://arxiv.org/html/2310.09615); [DIAMOND](https://arxiv.org/html/2405.12399)
- DIAMOND CS:GO model: 381M params (incl. 51M upsampler), generates at 56x30 upsampled to 280x150, trained 12 days on an RTX 4090, runs at 10 Hz on an RTX 3090 (source data was 16 Hz); drifts out of distribution in rarely visited areas, forgets state due to limited memory, wrongly allows successive jumps — [DIAMOND](https://arxiv.org/html/2405.12399)
- GameNGen (Google, Aug 2024): simulates DOOM interactively at over 20 frames/s on a single TPU (abstract, recalled; not re-fetched) — [arXiv 2408.14837](https://arxiv.org/abs/2408.14837)
- DIAMOND repo (MIT license): default 15 fps for interactive play, default imagination horizon 50 for playing; one command trains a game with the paper's hyperparameters on one GPU — [github.com/eloialonso/diamond](https://github.com/eloialonso/diamond)
- Reference model-free throughput: EnvPool ~1M Atari frames/s on a DGX-A100 (256 CPU cores); Sample Factory >10^5 env frames/s on one machine (abstracts, recalled; not re-fetched) — [EnvPool arXiv 2206.10558](https://arxiv.org/abs/2206.10558); [Sample Factory arXiv 2006.11751](https://arxiv.org/abs/2006.11751)
- When the true simulator is cheap: van Hasselt et al. argue a parametric model used Dyna-style (imagined transitions from real states) should not beat replay; models help for backward credit assignment and forward planning — [arXiv 1906.05243](https://arxiv.org/pdf/1906.05243). Hamrick et al. find planning's value is mainly in target construction and data distribution — [arXiv 2011.04021](https://arxiv.org/pdf/2011.04021)

### Inferences
Arithmetic from the cited numbers plus the local measurements:
- GhostRun processes 100k decisions in ~7.2 s (100,000 / 13,900). DIAMOND's full pipeline spends ~250,000 s (2.9 days) per 100k real steps; STORM ~15,500 s (4.3 h). These are designed for sample efficiency, but they show the compute envelope: per real decision consumed, these WM agents spend ~2,000x (STORM) to ~35,000x (DIAMOND) more wall-clock than the current PPO loop.
- Imagination at GhostRun's scale: generating 13.9k imagined decisions/s with a DIAMOND-style model needs ~41.7k denoiser forward passes/s (3 NFE) at minimum, plus reward/termination heads that DIAMOND lacks. At GhostRun's 240x224 frames the pixel count is ~13x DIAMOND's 64x64. Since learning already caps the GPU (~24k decisions/s for one epoch; MFU 24%), any WM forward pass costs throughput roughly in proportion to its FLOPs per frame relative to the 0.9 GFLOP/decision encoder. I found no published batched-throughput number for a pixel WM to make this exact.
- The real emulator dominates a learned one on every axis that matters here: it is exact (no compounding error, no out-of-distribution drift that DIAMOND reports), already batched to 16,384 consoles, uses no tensor-core time the learner needs, and supports exact save/restore, so branch rollouts / lookahead / backward curriculum can be done with ground truth. The only things a learned model adds are (i) gradients through dynamics (not used by PPO), (ii) a learned representation, and (iii) prediction-error signals — none of which needs a pixel generator.
- Decision implication for option (1) (fine-tune DIAMOND): expect ~3 days per game per 64x64 model on a 4090-class GPU just to replicate the paper setting, plus engineering to add reward/termination/RAM outputs and to batch it; the payoff would be a simulator that is slower and less accurate than the one already owned. Useful only as a research artifact (e.g. a visual "imagination" demo), not for training.

### Gaps
- No paper reports batched frames/s for DIAMOND, IRIS, STORM or DreamerV3 imagination on a consumer GPU; only per-game training time and single-stream interactive fps.
- No published fps for any NES / Super Mario Bros world model (the only SMB GameNGen-style repo found reports no fps; see Q5).
- No paper found that directly measures a learned simulator against a GPU-batched exact emulator with save/restore on the same hardware.
- Unverified lead (not fetched, no URL confirmed): Guo, Singh, Lee, Lewis & Wang (NeurIPS 2014, "Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning") used UCT search over the emulator with state restore and distilled it into a CNN policy; worth checking as the closest precedent for "plan with the real emulator, distil into a fast policy".

## 3. RAM / latent / object-space world models for retro games: cost and benefit versus pixel models

### Takeaway
Object/RAM-space models for Atari exist mostly as infrastructure (OCAtari reads objects from RAM) and as research on programmatic or causal world models evaluated on a few games; I found no paper training a direct RAM-to-RAM learned dynamics model for Atari or NES. Such a model would be tiny and cheap, but for GhostRun the emulator already is an exact RAM world model at 190k decisions/s, so a learned RAM model is only worth it as a representation/auxiliary or prediction-error signal.

### Cited Findings
- OCAtari (Delfosse et al., June 2023): extracts object-centric state from the emulator RAM for 24+ Atari 2600 games and can alter RAM; newer releases support both vision- and RAM-based extraction — [arXiv 2306.08649](https://arxiv.org/html/2306.08649v1); [OC_Atari releases](https://github.com/k4ntz/OC_Atari/releases)
- PoE-World (NeurIPS 2025): LLM-synthesised programmatic world models from OCAtari object boxes for Pong and Montezuma's Revenge; the authors report OCAtari does not extract objects reliably and needed per-game patches — [arXiv 2505.10819](https://arxiv.org/html/2505.10819)
- COMET (Apr 2025, "Better Decisions through the Right Causal World Model"): uses OCAtari's extractor on Pong and Freeway transitions and checks learned update rules against RAM annotations — [arXiv 2504.07257](https://arxiv.org/html/2504.07257v1)
- Mind-Studio (2026): LLM-synthesised executable Python world models over object-centric states, evaluated by next-state prediction hit rate of player position against real emulator rollouts — [arXiv 2606.16070](https://arxiv.org/pdf/2606.16070)
- AtariARI (Anand et al., NeurIPS 2019) uses RAM annotations to probe what learned representations encode (abstract, recalled; not re-fetched) — [arXiv 1906.08226](https://arxiv.org/abs/1906.08226)

### Inferences
- Cost: the NES state is 2,048 bytes (16,384 bits); a RAM dynamics head is a small readout compared with the 0.9 GFLOP/decision pixel encoder, but a full 16,384-way per-sample logit tensor is large at GhostRun batch sizes (12,288 consoles x 48 steps = ~590k samples per update -> ~9.7e9 logits if materialised at once), so its real cost depends on chunking; measure it rather than assume.
- Benefit vs pixel models: RAM prediction avoids reconstructing pixels (which ATC/Burda-type evidence suggests is not the useful part) and directly targets game-state variables. But most RAM bits are timers, RNG, sound and scratch values; a loss summed over all bytes is dominated by predictable or irrelevant bits. Selecting a few semantic variables (player x/y, lives, power state, area/room ID, enemy-killed flags, score) turns it into a DFP-style measurement head with far fewer outputs.
- Because exact RAM is available at every step for free, a learned RAM model's only roles are: auxiliary representation shaping, an action-effect signal for exploration (e.g. "did my press change any meaningful state"), and short-horizon prediction targets. Simulation itself should stay on the emulator.

### Gaps
- No paper found that trains a learned next-RAM (Atari or NES) dynamics model and measures its effect on an RL agent; OCAtari-based work focuses on object extraction and programmatic models on 1-3 games.
- No measured comparison of RAM-space vs pixel-space world models on the same game.

## 4. Evidence that auxiliary next-state prediction hurts or does not help PPO, and how to weight it

### Takeaway
I found no clean published result where an auxiliary predictive loss lowered PPO returns on Atari; the reported risks are (a) high sensitivity to the loss weight, (b) detaching or replacing RL gradients hurting some games, (c) augmentations hurting specific games, and (d) objective interference in a shared trunk. The more reliable PPO-specific positive results in long runs concern representation collapse / plasticity, not next-state prediction per se.

### Cited Findings
- ATC with PPO on Atari: no case reported where the auxiliary loss hurt relative to end-to-end RL; detached (no RL gradient) training performed poorly on Breakout and Space Invaders; random-shift augmentation hurt Gravitar, and lowering its probability from 1.0 to 0.1 reduced the harm while keeping gains elsewhere; multi-game pretraining dropped performance in 3 of 8 games, partly recovered by doubling filters — [arXiv 2009.08319](https://arxiv.org/html/2009.08319v3)
- VizDoom temporally-extended auxiliary tasks: performance "highly sensitive to the weighting placed on their losses" — [arXiv 2004.00600](https://ar5iv.labs.arxiv.org/html/2004.00600)
- EfficientZero: earlier auxiliary-loss studies showed "mixed effects" — [arXiv 2111.00210](https://arxiv.org/pdf/2111.00210)
- "Data, Auxiliary Losses, or Normalization Layers for Plasticity? A case study with PPO on Atari" (OpenReview; authors and date not verified): among interventions that require tuning, churn-reduction auxiliary losses gave the best performance and training stability — [OpenReview](https://openreview.net/forum?id=6CViR7tKj2)
- Moalla et al. ("No Representation, No Trust", NeurIPS 2024; the search tool attributed it to "Ma et al.", which I believe is wrong): PPO on ALE and MuJoCo suffers representation degradation linked to performance collapse; an auxiliary loss (PFO) regularising pre-activations mitigates it — [arXiv 2405.00662](https://arxiv.org/html/2405.00662v3)
- "Reward Scale Robustness for PPO via DreamerV3 tricks" (2023, CleanRL-based): DreamerV3's stability tricks "do not transfer as general improvements to PPO" — [arXiv 2310.17805](https://arxiv.org/abs/2310.17805v1)
- Muesli: a one-step model loss was the weaker variant; multi-step and policy-predicting model losses mattered — [arXiv 2104.06159](https://ar5iv.labs.arxiv.org/html/2104.06159)
- Recalled, not re-fetched: PPG (Cobbe et al., 2020) argues that sharing one network between policy and value objectives causes interference and separates them into phases — [arXiv 2009.04416](https://arxiv.org/abs/2009.04416)

### Inferences
- The project's earlier result ("did not clearly help, unknown if it hurt") is consistent with the literature: large-data, dense-reward settings usually show neutral-to-small effects, with weight sensitivity. GhostRun's `ram_coef` of 0.5 is applied to a per-byte NLL summed over 8 bits; relative to the PPO and HL-Gauss value losses its effective gradient share is unknown and should be logged (gradient-norm ratio at the shared trunk) before judging the head.
- Cheap protocol to decide option (3) without new research: run 3 matched seeds each of (A) no head, (B) head with stop-gradient into the trunk (diagnostic only: shows how much action-effect information the trunk already contains), (C) head with gradients at coef in {0.05, 0.2, 0.5}; compare returns at equal decisions and also track a trunk "effective rank"/dormant-unit metric (the collapse signal from Moalla et al.). If (B) predicts well, the representation already has the information and the head adds nothing; if (C) helps only at small coef, the original 0.5 was likely too high.
- If the head is kept, switch the target from all-bit flips to a few semantic, multi-horizon events (DFP style), which reduces irrelevant-bit dominance and directly encodes "what will my move do".

### Gaps
- No primary source found that tests a next-RAM or next-state auxiliary loss inside PPO on Atari/NES and reports harm or a null result with numbers.
- No accepted rule for setting auxiliary weights in PPO beyond per-task tuning; I found no source giving a principled default.

## 5. Open-source code that combines PPO (or a high-throughput model-free learner) with a world model

### Takeaway
Single-GPU world-model agents are well supported (DIAMOND, official DreamerV3 JAX), and there are PPO baselines with DreamerV3 tricks, but I found no maintained open-source system that couples a learned world model to an EnvPool/Sample-Factory-scale PPO learner; the integration would be custom.

### Cited Findings
- DIAMOND: MIT-licensed, ~2.1k stars; trains one Atari game on one GPU with paper hyperparameters (`python src/main.py env.train.id=BreakoutNoFrameskip-v4 common.devices=0`), ships pretrained Atari-100k world models and policies, interactive play at default 15 fps — [github.com/eloialonso/diamond](https://github.com/eloialonso/diamond)
- PPO-v3 (CleanRL-based PPO with DreamerV3 tricks): [arXiv 2310.17805](https://arxiv.org/abs/2310.17805v1) (code at github.com/RyanNavillus/PPO-v3, per the search result)
- Third-party code adapts DreamerV3 from the official danijar/dreamerv3 repo alongside PPO and SAC (Sliding Puzzles Gym benchmark) — [spgym experiments listing](https://awesome.ecosyste.ms/projects/github.com%2Fbryanoliveira%2Fspgym-experiments)
- mario-world: a GameNGen-style action-conditioned next-frame model for Super Mario Bros 1-1 (NES); reports no fps measurement and notes that an fps cap desynchronised the model and gameplay — [github.com/soheilchavo/mario-world](https://github.com/soheilchavo/mario-world)
- MarioVGG: a text-to-video diffusion model for SMB (not emulator-driven; no fps found) — [VideoGameGen PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf)
- OCAtari (object extraction from RAM, vision and RAM modes) — [github.com/k4ntz/OC_Atari](https://github.com/k4ntz/OC_Atari/releases)

### Inferences
- Engineering cost by option, from the evidence above:
  - (1) Fine-tune/train a pixel WM (DIAMOND): days of GPU per game at 64x64, new reward/termination heads, batching work, and it competes with the learner for the same GPU; benefit for a data-rich agent is unsupported. Highest cost, lowest expected value.
  - (2) Cheap RAM/latent WM as a separate network: small compute, moderate code (new model + training loop), but redundant with the exact emulator for simulation; value only as a curiosity/effect signal, which the exploration notes already cover.
  - (3) Auxiliary "what will my move do" head inside the policy (already built): lowest cost (a few config flags), literature supports small/neutral gains with weight sensitivity; best variant is semantic multi-horizon events (DFP) or a MuZero-style latent model predicting value/policy/reward (Muesli), not all-bit RAM flips.
  - (4) No world model: the default supported by van Hasselt et al. and Hamrick et al. when the true simulator is cheap; spend compute instead on emulator-based branching/lookahead with save/restore.
- No existing repo gives a drop-in for any option at GhostRun's scale; option (3) is the only one that reuses existing code.

### Gaps
- I could not verify sheeprl's feature set or throughput (search returned nothing on it).
- I found no repo combining Sample Factory, EnvPool, CleanRL or similar high-throughput PPO with a learned world model.
