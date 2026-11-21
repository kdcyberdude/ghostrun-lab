# RL agents that complete NES / retro platformers and shooters: results, recipes, and what transfers across levels and games

Scope note: the notes were compiled 2026-11-19. Primary papers were read in full text where possible: Sonic "Gotta Learn Fast", Burda et al. 2018, Pathak et al. 2017, Salimans & Chen 2018, and Procgen. OpenAI's Retro Contest result pages returned HTTP 403, so those figures come from search-engine extracts of the pages and are marked as such. Community GitHub claims are flagged where they have no evaluation protocol.

## Q1. Super Mario Bros.: best known RL results, whole-game claims, and the maze castles (4-4, 7-4, 8-4)

### Takeaway
Per-level PPO (uvipen/Super-mario-bros-PPO-pytorch) is the best documented community result: one model per level, 31/32 levels completed, with only 8-4 unsolved. I found no peer-reviewed or evaluated result that completes 8-4 or the whole game from power-on with one policy. The only "all 32 stages, one model" claim (dgriff777, A3C variant) has no success rates, and it shapes rewards per level, for example converting 7-4's "correct path" sound into reward. Curiosity-only agents explore many levels at scale (11 levels with 2048 parallel environments) but do not complete the game.

### Cited Findings
**Per-level PPO (community, no formal evaluation protocol)**
- uvipen/Super-mario-bros-PPO-pytorch reports that the agent completes **31 of 32 levels** with PPO. Training and testing are per level (`train.py --world W --stage S --lr ...`). The author's earlier A3C agent completed **19/32**. The repo has about 1.3k stars and 9 commits, and the README gives no date. — [uvipen README](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)
- Maze castles: the README calls 4-4 and 7-4 "puzzle" levels, where a wrong path choice forces the agent to go back. It says these were solved under an unspecified "hardcore setting", and that **8-4 "has not been solved yet."** — [uvipen README](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)
- The main per-level tuning knob is the learning rate (1e-3, 1e-4 or 1e-5; retry if stuck). **1-3 needed lr 7e-5 after about 70 failed attempts.** The README does not state the reward, preprocessing, action set, number of processes or frame counts. — [uvipen README](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)
- The same author's Contra repo (vietnh1009/Contra-PPO-pytorch, about 144 stars) shows a demo GIF for level 1 but gives **no metrics**. — [Contra-PPO-pytorch](https://github.com/vietnh1009/Contra-PPO-pytorch)

**One model for all 32 stages (unverified community claim)**
- dgriff777/SuperMarioRL claims **one model plays all 32 stages**, using "A3G", an A3C variant with per-agent GPU networks and Hogwild updates of a shared CPU model. Demo GIFs show every stage. There is **no success rate, trial count or frame count**. The only eval command runs 32 episodes, and the repo has about 5 stars. — [dgriff777/SuperMarioRL](https://github.com/dgriff777/SuperMarioRL)
- Its reward is points for each new rightward position beyond the previous maximum on the stage. It adds a custom penalty to discourage warp zones (the extract names stages "2-2 and 4-2"). **For 7-4, the game's "puzzle solved" sound is converted into reward points**, so the agent learns the correct path "with no hints" beyond that. 4-4 and 8-4 are not discussed. The author claims a level is won "often under an hour" on 1–2 GPUs with a 20-core CPU, without measurements. — [dgriff777/SuperMarioRL](https://github.com/dgriff777/SuperMarioRL)

**Curiosity / reward-free agents**
- **Pathak et al. 2017 (ICM + A3C, ICML):**
  - Training on SMB with curiosity only, the agent "can learn to cross over **30% of Level-1**". With no reward for it, the agent learned to kill and dodge enemies, because dying shows less of the game.
  - Setup: 14-action reparametrisation, 42×42 grayscale, 4-frame stack, action repeat 6 for training (none at inference), LSTM-256 policy.
  - The authors note that long jumps need "the same action up to 12 times in a row".
  - Transfer: run "as is" on their Level-3 (day, like Level-1), the Level-1 curiosity policy went far. On their Level-2 (night) it did poorly. Fine-tuning with curiosity for 1.5M iterations beat training from scratch on Level-2, even when scratch training ran 3.5M iterations.
  - — [Pathak et al. 2017, arXiv:1705.05363](https://arxiv.org/abs/1705.05363); [project page](https://pathak22.github.io/noreward-rl/)
- **Burda et al. 2018 (large-scale curiosity, PPO, Aug 2018):**
  - Setup: pure curiosity, no extrinsic reward, and the "done" signal removed ("death is not the end", an infinite-horizon setting).
  - Scaling from 128 to **2048 parallel environments** let the random-feature agent explore far more: "discovering **11 different levels** of the game, finding secret rooms and defeating bosses".
  - The x-axis was gradient steps, "not a claim about sample-efficiency". They used "an efficient version of Mario simulator" and "100x longer training" than prior work.
  - Settings: 84×84 grayscale, 4-frame stack, frameskip 4, lr 1e-4, rollout 128, 6 epochs for the Mario scaling runs.
  - — [Burda et al. 2018, arXiv:1808.04355](https://arxiv.org/abs/1808.04355)
- Burda et al. also tested transfer of a curiosity agent pre-trained on 1-1. Transfer to **1-2** was strong with both random features (RF) and learned inverse-dynamics features (IDF). Transfer to **1-3** (a day-to-night colour shift) worked only with IDF, not RF. They conclude learned features "appear to generalize better", while noting this "needs more analysis". — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)

**Generalisation to unseen Mario levels (procedural Mario, not NES)**
- **Arimura, Li & Togelius (12 Nov 2026):** PPO agents trained on increasingly large sets of procedurally generated Mario levels, tested on **1,000 held-out levels**. "Adding more distinct PCG training levels generally improves performance on unseen levels." Competent policies were then shaped into runner/killer/collector personas and fine-tuned with DRAIL on human demonstrations. The abstract page gives no numeric win rates. — [arXiv:2611.16175](https://arxiv.org/abs/2611.16175)
- SilvioBaratto/mario_rl_ppo reportedly trains one PPO policy on 1-1, 1-2 and 1-3 and measures generalisation on unseen 1-4. This comes from a search snippet only and is unverified. — [mario_rl_ppo](https://github.com/SilvioBaratto/mario_rl_ppo)

**Search for whole-game or 8-4 results**
- A targeted search for an agent beating 8-4 or the full game turned up only the community repos above and single-level academic work. One 2026 paper reports a 94.9% ± 1.5% win rate on World 1-1 only. — [arXiv:2606.29511](https://arxiv.org/html/2606.29511v1)

### Inferences
- The "31/32" headline means 31 *separately trained* per-level policies with per-level learning-rate retries, not one generalist agent. It is evidence that PPO plus x-progress can memorise almost any single SMB level, not that it generalises.
- Maze castles break plain max-x rewards: a wrong path loops Mario back, so x-progress is deceptive. The documented workarounds inject level-specific knowledge, such as dgriff777's "puzzle solved" sound as reward for 7-4 and uvipen's unexplained "hardcore setting" for 4-4/7-4. 8-4 (multiple loops plus pipes, underwater section and Bowser) has no documented solution in the sources found. This matches the user's own memory note that 8-4 is where search-to-policy breaks down.
- Curiosity results (11 levels discovered) used no done signal, 2048 environments and very long training. They show that exploration pressure alone gets far, but nobody reported a curiosity agent finishing the game.
- The extract's "warp zones in 2-2 and 4-2" may be a transcription error: SMB's warp zones are in 1-2 and 4-2 (background knowledge, not verified against the repo).

### Gaps
- No source found for a single policy beating 8-4, or the whole game from power-on (1-1 to 8-4 with lives), with an evaluation protocol. No Go-Explore or search-based SMB result with numbers was found. Go-Explore's published results are Atari only (see Q5).
- "MarioNette" (2021) is, as far as I know, an unsupervised sprite-decomposition / world-representation paper, not an RL agent. I did not re-verify it in this session, so it is not used as RL evidence.
- uvipen's reward, action set, frame counts and the exact "hardcore setting" for 4-4/7-4 are undocumented in the README. The code itself would need reading.
- The win-rate-versus-number-of-training-levels curve in arXiv:2611.16175 was not retrieved; only the abstract was read.

## Q2. OpenAI Retro Contest (2018): winning approaches, generalisation to unseen Sonic levels, effect of joint training

### Takeaway
Joint training on many levels followed by per-level fine-tuning roughly doubled PPO's test score, and every top contest entry was a tuned joint-PPO variant. Transfer was still shallow:
- Zero-shot test performance stopped improving after a few million joint-training steps.
- Test gains plateaued after about 50M steps.
- The best entry scored 4,692 of a possible ~10,000, against 7,438 for humans in the tech report.

### Cited Findings
**Benchmark design (Nichol et al., "Gotta Learn Fast", Apr 2018)**
- **Levels:** 58 save states across Sonic 1, Sonic 2 and Sonic 3 & Knuckles, with 11 test levels (one act chosen from multi-act zones). The test set has "mostly objects and textures present in the training set, but with different layouts". — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **Protocol:** each test level is played for **1M timesteps**, with no information shared between test levels. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **Frame skip and stochasticity:** frame skip 4 is required, plus a "**sticky frame skip**": each action is delayed one frame with probability **0.25**, to stop deterministic scripted solutions. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **Episodes:** episodes end at **4500 timesteps** (about 5 min). Boss fights are omitted; the level ends at a horizontal offset. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **Reward:** horizontal offset normalised so reaching the end gives **9000**, plus a completion bonus of **1000** that decays linearly to 0 at 4500 steps. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **Actions:** of the 12 Genesis buttons, "only eight essential button combinations" matter: {}, LEFT, RIGHT, LEFT+DOWN, RIGHT+DOWN, DOWN, DOWN+B, B. Rainbow and PPO used a 7-action subset. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **Baseline training reward:** the delta of *maximum* x-position reached in the episode, so the agent "is not punished for backtracking". — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- **PPO hyperparameters:** 1 worker, horizon 8192, 4 epochs, minibatch 8192, γ 0.99, λ 0.95, clip 0.2, **entropy 0.001**, reward scale 0.005, Atari CNN. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)

**Tech-report results (mean test score)**

| Method | Score |
|---|---|
| Rainbow | 2748.6 |
| JERK (scripted random-search with replay of best action sequence) | 1904.0 |
| PPO | 1488.8 |
| **Joint PPO (fine-tuned)** | **3127.9** |
| Joint Rainbow | 2969.2 |
| **Human** | **7438.2** (4 testers, 1 hour per level after 2 hours of practice) |

Source: [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- JERK beats plain PPO, "likely due to JERK's perfect memory and its tailored exploration strategy". — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)

**Joint training and its limits**
- Joint PPO setup: **188 parallel workers**, each assigned a training level, with gradients averaged so all levels train evenly. It "requires hundreds of millions of timesteps to converge". — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- "After roughly **50 million timesteps** of joint training, further improvement on the training set stops leading to better performance on the test set". "**Zero-shot performance does not increase much after the first few million timesteps**". Joint PPO is "roughly 100% better" than regular PPO. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- Joint Rainbow (32 GPUs, 8 envs each, a new level sampled per episode) did *not* beat Joint PPO. A Reptile meta-learning variant "yielded worse results". — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- The authors' caveats:
  - "our best transfer learning results are not much better than our best results learning from scratch".
  - "Mastering a Sonic level involves some degree of memorization".
  - "Sonic-specific hacks may outperform general meta-learning approaches".
  - — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)

**Contest outcome (two-month contest, results June 2018; OpenAI pages returned 403, figures from search extracts)**
- **Final leaderboard:**

| Rank | Team | Score |
|---|---|---|
| 1 | Dharmaraja (Qing Da, Jing-Cheng Shi, Anxiang Zeng, Guangda Huzhang, Run-Ze Li, Yang Yu) | **4692** |
| 2 | mistake | 4446 |
| 3 | aborg | 4430 |
| 4 | whatever | 4274 |
| 5 | Students of Plato | 4269 |
| – | Joint PPO baseline | 4070 |
| – | Joint Rainbow baseline | 3843 |

  923 teams registered and 229 submitted. The theoretical max is 10,000. — [OpenAI Retro Contest results](https://openai.com/index/retro-contest-results/); [retrospective](https://openai.com/blog/first-retro-contest-retrospective/)
- **Dharmaraja's approach:** "a variant of joint PPO … with a few improvements". The team found joint training with Rainbow "actually hurt performance in their case". — [OpenAI Retro Contest results](https://openai.com/index/retro-contest-results/)
- **aborg's approach:** also a joint-PPO variant. It added training levels from the **Game Boy Advance and Master System Sonic games**, used a different architecture, and tuned hyperparameters "for fast learning". OpenAI noted that top results "all came from tuning or extending existing algorithms such as PPO and Rainbow". — [OpenAI Retro Contest results](https://openai.com/index/retro-contest-results/)
- One search extract called aborg "second place", which conflicts with the leaderboard (mistake 4446 > aborg 4430). The leaderboard numbers are taken as authoritative. — [OpenAI Retro Contest results](https://openai.com/index/retro-contest-results/)

### Inferences
- Contest leaderboard scores (Joint PPO baseline 4070) are on a different test set from the tech report (Joint PPO 3127.9), so the two are not directly comparable.
- In this setting, joint pre-training mostly gives a better *initialisation for fine-tuning* (learned movement and jump habits), not zero-shot competence on new layouts. Its value saturates early, and the remaining gap to humans is level-specific memorisation and exploration.
- More diverse training levels, even from other Sonic games, was one of the few levers the top teams used. This is consistent with the Procgen finding in Q4.

### Gaps
- Could not fetch the retrospective's detailed write-ups for "mistake", "whatever" or "Students of Plato". Their methods are not documented here.
- I did not confirm whether the contest's hidden test levels were custom-designed or original game acts; the tech report used original acts.

## Q3. Shooters with many enemies and bullets (Contra, run-and-gun, Mega Man, bullet-hell, Atari shooters): what works and why

### Takeaway
No source found documents a published, evaluated RL agent completing Contra beyond level 1, or a full Mega Man stage plus boss.

| Effort | Method | How far it got |
|---|---|---|
| Contra baseline (Feb 2026) | PPO | Reaches the level-1 boss, then fails |
| megai_man (Mega Man) | Stage reward with distance map + enemy-free ROM curriculum | Partial; no full clear documented |
| thrl (Touhou bullet-hell) | PPO | Stage 2 after ~20 days of training |

The recurring failure points:
- action spaces that make fire and move mutually exclusive;
- boss fights, which are a different reward regime from x-progress;
- long-horizon credit assignment, such as picking a weak weapon long before the boss.

In Atari, dense-reward shooters are generally solved, while the residual hard games are sparse-reward or hard-vision ones (Gravitar, Pitfall, Skiing, Solaris).

### Cited Findings
**Contra (NES)**
- **kaihe/contra_agent baseline (2026-02-22):**
  - **Setup:** SB3 PPO, CnnPolicy, 84×84×4 grayscale, frame skip 4, **32M timesteps, 32 envs**, n_steps 2048, 10 epochs, lr 1e-4, clip 0.2→0.05, **entropy 0.1→0.005**.
  - **Actions:** Discrete(7) where **fire and movement are mutually exclusive**.
  - **Reward:** clamp(x_delta, 0, 3)×0.1, plus score delta, −0.05 idle after 20 steps, −20 per life lost, −50 timeout at 4000 steps, +100 win.
  - **Result:** the agent "navigates well, jumps obstacles, and shoots enemies" but "**consistently fails to defeat the Level 1 boss**".
  - — [Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)
- **Diagnosed causes** in the same baseline:
  - The score reward makes it pick up the **Laser** weapon, which is weak against the boss, instead of keeping Spread.
  - There is no "don't pick up" action.
  - The pickup-to-death delay is "hundreds of steps", hard for γ=0.99.
  - Planned fixes: a **constant-fire action space** for run-and-gun, and a **boss-area save state** so about 50% of episodes start near the boss.
  - — [Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)
- The project index lists 12 experiments, including "Enhance Boss Fight Training", "MC search to beat boss", "Better Action Space", "The First Setback", "Behavior Cloning: A New Baseline", "Fixed Anchor for Save and Load", "Search to Win the Game", "Behavior Data from Search" and "The Second Setback". Their outcomes were not on the index page. — [contra_agent index](https://kaihe.github.io/contra_agent/)
- gym-contra (nes-py) reward is rightward movement, with a death penalty and a boss-defeat bonus. — [gym-contra on PyPI](https://pypi.org/project/gym-contra/)

**Mega Man (NES)**
- **victorsevero/megai_man targets the Cut Man stage.**
  - **Setup:**
    - Observation 3×84×84; frame skip plus extra skipping while Mega Man is invisible after damage.
    - Actions MultiDiscrete([2 jump, 5 direction, 2 shoot]).
    - Stage reward from a precomputed **distance map** (wavefront expansion) to the boss door, plus damage taken. **Backward movement is penalised more than forward is rewarded**; the author calls this essential, otherwise the agent loops back and forth. The idea is credited to the Zelda "triforce" project.
    - Truncation after 900 steps without a new closest point.
  - **Curriculum:**
    - **Pre-train on an edited ROM with all enemies removed (25M+ steps, shooting masked).**
    - Then 30M+ steps with enemies.
    - Separate boss training (400k steps; reward for boss damage dealt and damage taken).
  - **Result:** no quantitative results. The author says a full AI-only run proved "much harder than I expected" and is unfinished.
  - — [megai_man](https://github.com/victorsevero/megai_man)
- A "General Deep RL in NES Games" paper (2021 per ResearchGate) ran DQN on Mega Man for about 2.2M timesteps. Per the search extract, Mega Man learned more slowly and less consistently than SMB. No stage clear was reported in the extract. — [paper PDF](https://assets.pubpub.org/uonw8d4k/01621566588509.pdf)

**Bullet-hell**
- Fujimoto (2024) compared learning methods and input types in a bullet-hell game and found **Categorical DQN** (C51, a Rainbow component) "particularly effective in learning global avoidance behavior". Search extract only. — [CiNii record](https://cir.nii.ac.jp/crid/1050865187563177728)
- The thrl Touhou framework reports a PPO run on TH04 (top-K object features) that reached **stage 2 and fought the boss after about 20 days** of training before a shutdown. This is a community report. — [touhourl/thrl](https://github.com/touhourl/thrl)
- Procgen includes StarPilot, "a simple side scrolling shooter game. All enemies fire…". Larger IMPALA-CNN models improve both sample efficiency and generalisation across Procgen games; the small Nature-CNN "almost completely fails to train". — [Procgen, arXiv:1912.01588](https://arxiv.org/abs/1912.01588)

**Atari shooters and hard games**
- **Agent57 (Mar 2020)** is the first agent above the human benchmark on **all 57** Atari games, with a 5th-percentile human-normalised score (HNS) of 116.67%. MuZero exceeds 100% HNS on 51 games and R2D2 on 52. Skiing and Solaris were the hold-outs because of long-term credit assignment. — [Agent57 blog](https://deepmind.google/blog/agent57-outperforming-the-human-atari-benchmark/); [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)
- Agent57 needed close to 80B frames. MEME (2022) reached the human baseline on all 57 games in about 390M frames, roughly 200× fewer. — [MEME, arXiv:2209.07550](https://arxiv.org/abs/2209.07550)
- The single-demo backward curriculum worked on Montezuma but "much less well" on **Gravitar and Pitfall**. The authors suspect "a harder vision problem" ("difficult to play from a downsampled screen ourselves"), saw improvement from "larger and deeper" networks, and could not find hyperparameters that trained the full curriculum. — [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)

### Inferences
- Run-and-gun games break the Mario recipe in three places:
  1. **Action space:** firing must be combinable with movement, or always on.
  2. **Bosses:** x-progress gives zero gradient in a boss room, so a separate boss reward or boss save-states is needed (megai_man, the Contra plan).
  3. **Item and weapon choice** creates long-delay credit assignment.

  Save-state starts near bosses and enemy-free pre-training are the practical community fixes.
- In curiosity agents, dodging and killing emerge because death reduces novelty (Pathak 2017). In dense-score shooters, standard value-based RL with distributional heads (C51/Rainbow) is reported to help avoidance. The evidence on this is thin: a single conference paper.

### Gaps
- No RL results found for S.C.A.T./Final Mission, Adventure Island, Kage/Shadow of the Ninja, or Batman (NES). No published Contra agent past level 1 with evaluation.
- No peer-reviewed analysis found of agents learning to aim versus dodge (for example saliency studies on Space Invaders, Assault or Robotank). Per-game scores for Gravitar, Robotank, Space Invaders and Assault were not retrieved.
- Outcomes of the later kaihe Contra experiments (MC search, behaviour cloning, "Search to Win the Game") were not retrieved.

## Q4. Evidence on the hypothesis "once the model knows what each button does (a world model), it can complete all levels with only light PPO fine-tuning"

### Takeaway
The large-scale evidence does not support "world knowledge, then light fine-tuning, completes everything":
- **VPT:** action-labelled video pre-training gives a strong prior, but reaching diamond tools still needed substantial RL fine-tuning and succeeded only about 2.5% of the time.
- **DreamerV3:** reaches diamonds from scratch by training a world model *inside* a reward-driven RL loop over 100M steps, not by pre-learning controls.
- **SIMA 2:** needed a reward model plus self-play iterations to master new games.
- **Generalisation data (Procgen, Sonic):** competence on unseen levels comes from training on *many diverse levels* (around 10,000 in Procgen), not from knowing button semantics. Fixed level sequences give "an illusion of meaningful progress".

### Cited Findings
**Behaviour prior from video (VPT, OpenAI, Jun 2022)**
- An inverse dynamics model (IDM) trained on a small labelled set pseudo-labels about **70k hours** of web Minecraft video. Behaviour cloning on this yields a foundation model at 20 Hz keyboard and mouse.
- Fine-tuning with BC and then RL enables "hard-exploration tasks … impossible to learn from scratch via reinforcement learning", including the first agents to craft **diamond tools**, which take humans "20 minutes (24,000 environment actions)".
- — [VPT, arXiv:2206.11795](https://arxiv.org/abs/2206.11795); [VPT paper PDF](https://cdn.openai.com/vpt/Paper.pdf)
- The RL-fine-tuned VPT reached about **2.5% diamond-pickaxe success**. JARVIS-1 (Nov 2023) reports 6.22% within 20 minutes; the two time windows differ across sources. — [JARVIS-1, arXiv:2311.05997](https://arxiv.org/abs/2311.05997)
- The DreamerV3 Nature paper contrasts VPT's **720 GPUs for 9 days** with Dreamer's compute. — [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)

**World model inside RL (DreamerV3, Nature, 17 Apr 2025; arXiv Jan 2023)**
- DreamerV3 is "the first algorithm to collect diamonds in Minecraft from scratch **without human data or curricula**". All Dreamer agents discovered diamonds **within 100M environment steps**; prior baselines topped out around the iron pickaxe.
- A secondary source puts the compute at **1 GPU for about 9 days**.
- — [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2); [arXiv:2301.04104](https://arxiv.org/abs/2301.04104); [heise summary](https://www.heise.de/en/news/Study-Google-s-artificial-intelligence-successfully-plays-Minecraft-10339154.html)

**Generative world models and agents (Genie, SIMA 2)**
- **Genie (DeepMind, 23 Feb 2024):** an **11B** "foundation world model" trained unsupervised from unlabelled internet video, with a video tokenizer, an autoregressive dynamics model and a **latent action model**. The learned latent actions "can help train agents to imitate behaviors from unseen videos". The abstract gives no hours, latent-action count or agent numbers. — [Genie, arXiv:2402.15391](https://arxiv.org/abs/2402.15391)
- **SIMA 2 (announced Nov 2025; tech report and arXiv Dec 2025):**
  - Gemini-based 3D game agent.
  - Held-out games ASKA and MineDojo: **45–75% task completion versus 15–30% for SIMA 1** (secondary report).
  - Self-improvement loop: after initial training on human-labelled video, a Gemini model proposes tasks and a **reward model scores attempts**, and the agent improves without new demonstrations.
  - The loop was also run in **Genie 3-generated worlds**, where initially failed tasks were solved after several self-practice generations.
  - — [SIMA 2, arXiv:2512.04797](https://arxiv.org/abs/2512.04797); [SIMA 2 tech report](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/SIMA_Tech_Report_2025.pdf); [The Decoder](https://the-decoder.com/deepminds-latest-ai-agent-learns-by-exploring-unfamiliar-games-and-ai-built-worlds/)

**Multi-game generalists**
- **Gato (May 2022):** 604 tasks, above 50% of expert score on more than 450. Above average human on 23 Atari games (per summaries). — [Gato, arXiv:2205.06175](https://arxiv.org/abs/2205.06175); [TechCrunch](https://techcrunch.com/2022/05/13/deepminds-new-ai-can-perform-over-600-tasks-from-playing-games-to-controlling-robots/amp)
- **Multi-Game Decision Transformer (May 2022):** one offline-trained transformer plays up to 46 Atari games at "close-to-human" level. Aggregate interquartile-mean HNS over 41 games exceeds human and exceeds its training data (data IQM 101%). — [MGDT, arXiv:2205.15241](https://arxiv.org/abs/2205.15241)
- **JAT (Feb 2024):** an open generalist reporting 31.1% average normalised Atari score versus Gato's 30.9%. — [JAT, arXiv:2402.09844](https://arxiv.org/abs/2402.09844)

**Level diversity versus memorisation (Procgen, Dec 2019)**
- With PPO for 200M steps on training sets of 100 to 100,000 levels, "agents strongly overfit to small training sets in almost all cases. To close the generalization gap, agents need access to **as many as 10,000 levels**." — [Procgen, arXiv:1912.01588](https://arxiv.org/abs/1912.01588)
- Fixed-sequence ablation: an ALE-like setup with a fixed level order, where failing ends the episode. Agents "become competent over the first several training levels … giving an **illusion of meaningful progress**", but on random level orders they "have in fact learned **almost nothing** about the underlying level distribution". — [Procgen](https://arxiv.org/abs/1912.01588)
- Larger IMPALA-CNNs (×2, ×4 channels) "significantly improve both sample efficiency and generalization". — [Procgen](https://arxiv.org/abs/1912.01588)

**Transfer evidence from Sonic and Mario**
- Sonic: zero-shot test performance plateaued after a few million joint-training steps, and the best transfer was "not much better" than the best from scratch. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- Mario: a curiosity policy from Level-1 transfers "as is" to a visually similar level but not to a night level; brief fine-tuning fixes this. — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363); [Burda et al. 2018](https://arxiv.org/abs/1808.04355)

### Inferences
- What transfers well is low-level control knowledge: moving, jumping, killing enemies, robustness to visual shifts with learned features. This makes fine-tuning faster (ICM, Burda, Sonic joint PPO). What does not transfer is level-specific route knowledge: maze-castle paths, boss patterns, timing over pits. That is exactly what blocks 8-4 or a Contra boss.
- The strongest generalist results come from **breadth of data**:
  - VPT: 70k hours of video.
  - Procgen: about 10,000 levels.
  - Gato and MGDT: dozens of games with expert trajectories.
  - SIMA 2: many games plus a reward model.

  None of these is a "know the buttons, then light PPO" pipeline. SMB's 32 fixed levels sit in the regime Procgen calls an "illusion of progress".
- A realistic reading of the hypothesis: a world or dynamics model (or an IDM / latent-action prior) is a good *initialisation and exploration aid*. Completing every level still needs per-level reward-driven RL, search plus distillation, or demonstrations, plus diversity (randomised start states, many levels) to avoid memorisation.
- DreamerV3 is evidence for model-based RL *with reward*, not for reward-free world-model knowledge being sufficient.

### Gaps
- No primary-source numbers retrieved for Genie 2 or Genie 3, or for SIMA 2's per-game success after self-improvement (secondary reports only).
- VPT's labelled-data size (believed to be about 2k hours of contractor data) and its RL-fine-tune compute were not verified in this session.
- No study found that trains a dynamics or world model on NES platformer play and then measures how much PPO fine-tuning is needed per level. This is a direct test of the hypothesis that appears not to have been published.

## Q5. Practical recipes and pitfalls: frame skip, sticky actions, action spaces, reward shaping, curricula, saved-state starts, backward curricula, entropy, compute

### Takeaway
The repeatedly validated recipe for retro platformers:
- frame skip 4 with a small, hand-picked combo action set;
- reward on max-x progress, without penalising backtracking, normalised per level;
- many parallel environments;
- for hard spots, starting episodes from saved states, either moved backward along a demonstration (Salimans & Chen) or chosen from an archive (Go-Explore "return, then explore");
- then robustifying with stochasticity (sticky actions, random no-ops).

The biggest pitfalls:
- entropy and exploration balance during long curricula;
- deceptive x-progress in mazes and boss rooms;
- determinism that lets agents memorise instead of generalise.

### Cited Findings
**Frame skip, sticky actions, stochasticity**
- Sonic benchmark: frame skip 4 (one timestep is about 1/15 s, "more than enough temporal resolution to play Sonic well") plus sticky frame skip at 0.25. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- SMB curiosity agents: frameskip 4 (Burda) and action repeat 6 during training (Pathak). — [Burda et al.](https://arxiv.org/abs/1808.04355); [Pathak et al.](https://arxiv.org/abs/1705.05363)
- Robustness test on the single-demo Montezuma policy (74,500 points):
  - Machado-style **sticky actions at 0.25 per frame cut it to about 10,000**.
  - 1% random actions cut it to 8,400.
  - 0–30 random no-ops at start "did not significantly hurt".
  - — [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)

**Action spaces**
- Sonic uses 8 essential combos (7 for the RL baselines). — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- SMB (ICM) uses 14 combos; long jumps need about 12 repeated identical actions. — [Pathak et al.](https://arxiv.org/abs/1705.05363)
- Contra: fire mutually exclusive with movement blocks run-and-gun; constant fire was proposed. — [Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)
- Mega Man: factored MultiDiscrete(jump, direction, shoot), with shooting masked in the enemy-free phase "to reduce entropy". — [megai_man](https://github.com/victorsevero/megai_man)

**Reward shaping**
- Max-x delta (no backtracking penalty) for Sonic baselines, and new-max-x for SMB (dgriff777). — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720); [dgriff777](https://github.com/dgriff777/SuperMarioRL)
- Per-level normalisation (9000 to the goal) plus a time-decaying completion bonus. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- Distance-to-goal maps with an asymmetric backward penalty, for levels that are not simply "go right" (Mega Man; idea from Zelda's triforce). — [megai_man](https://github.com/victorsevero/megai_man); [triforce](https://github.com/DarkAutumn/triforce)
- Level-specific event rewards, such as the 7-4 "correct path" sound. — [dgriff777](https://github.com/dgriff777/SuperMarioRL)
- Removing "done" so death is just a boring transition, for reward-free exploration. — [Burda et al.](https://arxiv.org/abs/1808.04355)
- Pitfall: a score reward for pickups taught the Contra agent to take the Laser that later loses the boss fight. — [Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)

**Saved-state starts and backward curriculum (Salimans & Chen, Dec 2018)**
- **Method:**
  - Each episode resets to a state from one human demo. The reset point moves backward when at least **ρ = 20%** of workers reach demo-comparable returns.
  - Each worker samples a local start uniformly from a window of D states before the central point.
  - PPO, distributed over **128 GPUs × 8 workers = 1024 rollout workers**, for about **50B frames over about 2 weeks**.
  - — [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)
- **Result:** **74,500 on Montezuma's Revenge**, beating the 71,500 demo by collecting more items and exploiting a key re-spawn the authors did not know about. Theoretical analysis: standard RL run time is exponential in the number of states between rewards; the method is quadratic. — [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)
- **Pitfalls they report:**
  - The agent "is generally unable to reach the exact state from later on in a demonstration" because of a different frameskip and action noise, so it must generalise between near-identical states.
  - "If the agent's actions are too random, it makes too many mistakes … if the actions are too deterministic, the agent stops learning". Success "required careful tuning of the coefficient of the entropy bonus … learning rate and the scaling of rewards".
  - It failed to train the full curriculum on Gravitar and Pitfall.
  - Compute made it "impossible to quantify the consistency … across runs".
  - — [Salimans & Chen 2018](https://arxiv.org/abs/1812.03381)

**Go-Explore (arXiv 2019/2020; Nature, Feb 2021)**
- Principles: remember promising states, **return to them first (by restoring the emulator state), then explore**, and **robustify** the found solution afterwards (a backward-curriculum-from-trajectory step). "Derailment" occurs when an agent tries to return and explore in the same stochastic policy. — [Go-Explore Nature](https://www.nature.com/articles/s41586-020-03157-9); [arXiv:2004.12919](https://arxiv.org/abs/2004.12919); [Uber blog](https://www.uber.com/us/en/blog/go-explore/)
- Reported policy-based Go-Explore scores: **97,728 on Montezuma** and **20,093 on Pitfall after 3.2B frames**. Domain-knowledge versions reported far higher numbers (for example over 650k on Montezuma in the 2019 preprint); sources mix configurations. — [arXiv:2004.12919](https://arxiv.org/abs/2004.12919); [Go-Explore 2019, arXiv:1901.10995](https://arxiv.org/abs/1901.10995)

**Other curricula and saved states**
- Enemy-free ROM pre-training, then enemies, then a separate boss phase (Mega Man). — [megai_man](https://github.com/victorsevero/megai_man)
- Boss-area save states for about 50% of episodes (Contra plan). — [Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)
- Procgen: train on many distinct levels sampled per episode rather than a fixed sequence. — [Procgen](https://arxiv.org/abs/1912.01588)

**Entropy and hyperparameters**
- Sonic PPO entropy coefficient 0.001. — [arXiv:1804.03720](https://arxiv.org/abs/1804.03720)
- Contra baseline annealed entropy 0.1→0.005 and clip 0.2→0.05. — [Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)
- uvipen's only per-level knob was the learning rate; 1-3 took about 70 tries and lr 7e-5. — [uvipen README](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)

**Compute reference points**

| Setup | Compute |
|---|---|
| Sonic PPO | 1M steps per test level |
| Sonic Joint PPO | 188 workers, hundreds of millions of steps |
| Sonic Joint Rainbow | 32 GPUs |
| Procgen easy | 25M steps, about 3 GPU-hours |
| Procgen generalisation | 200M steps |
| Burda Mario | 2048 envs |
| Contra baseline | 32M steps |
| Mega Man | about 55M steps across phases |
| Salimans & Chen (Montezuma) | about 50B frames on 128 GPUs |
| Agent57 | about 80B frames |
| DreamerV3 | 100M steps on 1 GPU |
| VPT | 720 GPUs × 9 days |

Sources: [Sonic](https://arxiv.org/abs/1804.03720); [Procgen](https://arxiv.org/abs/1912.01588); [Burda](https://arxiv.org/abs/1808.04355); [Contra](https://kaihe.github.io/contra_agent/experiments/1_baseline.html); [megai_man](https://github.com/victorsevero/megai_man); [Salimans & Chen](https://arxiv.org/abs/1812.03381); [MEME](https://arxiv.org/abs/2209.07550); [DreamerV3 Nature](https://www.nature.com/articles/s41586-025-08744-2)

### Inferences
- For the hard NES spots (8-4 maze loops, bosses), the documented path that worked at scale is **Salimans & Chen style backward curricula from a solved trajectory**, or Go-Explore return-then-robustify. A search or exploration phase finds a route, and RL learns to reproduce it from progressively earlier save states.
- The recurring pitfalls in those papers line up with the user's project notes:
  - the policy cannot exactly reproduce demo states;
  - entropy that is too low stalls learning, while entropy that is too high fails long segments;
  - "fluke" routes that cannot be robustified.
- Test under sticky actions or random no-ops. Deterministic NES emulation rewards memorisation, as shown by the Montezuma policy falling from 74.5k to 10k under 0.25 sticky actions.
- Splitting reward regimes (progress versus boss damage) and factoring the action space (fire independent of movement) are cheap, well-motivated changes for run-and-gun games.

### Gaps
- No controlled ablation found of SMB-specific action sets (SIMPLE_MOVEMENT versus COMPLEX_MOVEMENT) or frame-skip values on completion rate.
- No published per-level SMB compute numbers, such as frames to clear each level with PPO; community READMEs omit them.
- Go-Explore has no published NES or SMB results; the Atari numbers above mix configurations across sources.
