# World models and hard exploration: would a good next-screen predictor find rare actions in NES games?

Research date: 2026-11-19. Scope: whether learned world models (pixel/diffusion/transformer/RSSM) help RL agents discover rare, specific actions, versus only improving sample efficiency on games that are easy to explore. This file adds evidence beyond `reports/Controllable exploration for NES agents.md` (the "earlier report"). Where an earlier-report fact is needed for the argument, it is restated in one line with its original source.

## 1. World-model agents on hard-exploration Atari and on Super Mario Bros / NES: do world models alone solve them?

### Takeaway
No. On every published table, world-model agents without an explicit exploration mechanism score near zero on hard-exploration games. The only hard-exploration game in Atari 100k is Private Eye, and the best world model there (STORM) reached about 11% of human. DIAMOND reached about 0.1%. Learned-model planners score 0 on Montezuma's Revenge and Pitfall even after 20B frames. The one 2025 world-model agent that scores on Montezuma's Revenge (PoE-World) uses symbolic object inputs, LLM-written program rules and a planner. It only collects the first key, for 100 points. No peer-reviewed world-model RL result exists for Super Mario Bros or any NES game.

### Cited Findings
- **Atari 100k per-game scores (DIAMOND paper, arXiv May 2024, NeurIPS 2024).** The scores are compiled from the original papers: SimPLe 2019, TWM and IRIS 2022, DreamerV3 2023, STORM 2023, DIAMOND 2024 — [DIAMOND, arXiv 2405.12399](https://arxiv.org/html/2405.12399)
  - **Private Eye** (human 69,571.3; random 24.9): SimPLe 35.0, TWM 86.6, IRIS 100.0, DreamerV3 882.0, **STORM 7,781.0**, **DIAMOND 114.3** — [DIAMOND](https://arxiv.org/html/2405.12399)
  - **Freeway**, a sparse-reward game (human 29.6): **DreamerV3 0.0**, IRIS 31.1, STORM 33.5, DIAMOND 33.3, TWM 24.3, SimPLe 16.7 — [DIAMOND](https://arxiv.org/html/2405.12399)
  - **Hero** (human 30,826): DIAMOND 5,621.8, DreamerV3 11,161, STORM 11,044. **Frostbite** (human 4,334.7): DIAMOND 274.1, TWM 1,475.6 — [DIAMOND](https://arxiv.org/html/2405.12399)
  - **Aggregates**: DIAMOND mean human-normalised score (HNS) 1.459, IQM 0.641, superhuman on 11 of 26 games. STORM 1.266, DreamerV3 1.097, IRIS 1.046 — [DIAMOND](https://arxiv.org/html/2405.12399)
  - Computed from the table: Private Eye HNS is 0.13% for DIAMOND, 0.11% for IRIS, 1.2% for DreamerV3 and 11.2% for STORM. DIAMOND's leading mean HNS comes from dense-reward games such as Breakout (132.5 vs human 30.5), Asterix and Road Runner.
- **DIAMOND has no exploration mechanism.** Data collection is epsilon-greedy with epsilon 0.01, and the policy has an entropy term with weight 0.001 "to maintain sufficient exploration". Reward and termination come from a separate model, not the diffusion model. Each Atari 100k run took about 2.9 days on one RTX 4090 (1.03 GPU-years for all runs). The model has 13M parameters. IRIS has 30M parameters and takes 4.1 days; DreamerV3 has 18M and takes under a day — [DIAMOND](https://arxiv.org/html/2405.12399)
- **Large-budget world-model agents** (from the earlier report): MuZero scored 0.00 on Montezuma's Revenge, 0.00 on Pitfall and 0.40 on Venture after 20B frames (2019) — [MuZero Table S1](https://arxiv.org/pdf/1911.08265). DreamerV3 scored 2,512 on Montezuma's Revenge and 0 on Pitfall and Venture at 200M frames (2023) — [DreamerV3 v1](https://arxiv.org/pdf/2301.04104v1)
- **PoE-World (arXiv May 2025, NeurIPS 2025).** The world model is a weighted product of small Python "expert" programs written by an LLM. It reads OCAtari object lists, not pixels, and is fit from fewer than 1,000 demonstration frames per game. In Montezuma's Revenge, that demonstration "never achieves positive reward". The agent trains for at most 3k steps. Results:
  - Montezuma's Revenge **100 ± 0** (first key collected), and **100 ± 0 on MR-Alt**, a recomposed level with no demonstrations. WorldCoder + planner, ReAct, PPO at 100k steps and **PPO at 20M steps** all scored **0**.
  - The authors note that reaching the first reward takes "at least 100" actions, about 8^100 sequences.
  - Removing the learned hard constraints dropped Montezuma's Revenge planning successes from 5/9 to 2/9.
  - Stated limits: symbolic inputs only. OCAtari "does not reliably extract objects", so it was patched by hand per game. Exploration and reward learning are not addressed. Each run cost about $20 of API credit and about 8 hours.

  Source: [PoE-World, arXiv 2505.10819](https://arxiv.org/html/2505.10819)
- An older latent-world-model curiosity agent, "Latent World Models for Intrinsically Motivated Exploration", reports **2,276 on Montezuma's Revenge**. The search snippet did not date it; it is believed to be NeurIPS 2020 (Ermolov & Sebe), not re-verified — [review page](https://liner.com/review/latent-world-models-for-intrinsically-motivated-exploration)
- **Super Mario Bros / NES.** This session's searches found no peer-reviewed or official world-model RL result for Super Mario Bros or any NES game. The only hit was an uncited personal portfolio page describing a latent-diffusion Mario simulator with PPO trained inside it. It gives no results — [portfolio page, unverified](https://aravinds-kannappan.github.io/portfolio/01-gamengen-wm/). The earlier report reached the same conclusion, and noted that ICM's Mario agent stalled at a pit needing "a very specific sequence of 15-20 key presses" — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)

### Inferences
- The evidence splits cleanly in two. World models raise sample efficiency on Atari 100k games where random play finds reward: Breakout, Asterix, Road Runner, Boxing. They leave hard-exploration games near random. DIAMOND, the best pixel model, is no better than IRIS on Private Eye, and both are at about 0.1% of human. Better next-frame fidelity did not improve exploration.
- STORM's Private Eye score of 7,781 is far above every other model, including DIAMOND. That suggests seed or luck variance, not a systematic exploration advantage. The DIAMOND table gives no variances, so this is unconfirmed.
- DreamerV3's 0.0 on Freeway at 100k steps shows the same issue at small scale. If the policy never sees the reward, the world model has nothing to learn about it.
- Bellemare et al.'s 2016 taxonomy classes Freeway and Private Eye as sparse-reward hard-exploration games. That claim is from memory and was not re-verified this session.
- PoE-World is the closest thing to "a good world model unlocks Montezuma's Revenge". It works because the model is object-level and symbolic and comes with a planner. Even then it gets only the first 100 points. It does not show that a pixel next-screen predictor would.

### Gaps
- No world-model agent (DIAMOND, IRIS, STORM, TWM, DreamerV3, EfficientZero) has published scores on Pitfall, Venture, Gravitar or Solaris at 100k steps, because Atari 100k excludes them. Only MuZero and DreamerV3 report full-suite numbers at large budgets.
- No 2025–2026 paper was found that runs a diffusion world-model agent (DIAMOND-style) on Montezuma's Revenge or other hard-exploration games.
- Variance across seeds for STORM's Private Eye score was not checked.

## 2. Do world models predict rare events, or regress to the common outcome? Compounding error and rare-event fidelity

### Takeaway
Rare events are learned badly, and the effect is now measured directly. DIAMOND's authors report that their CS:GO model drifts out of distribution "in less frequently visited areas". Video diffusion models generalise "case-based": they copy the closest training example, matching on colour first and shape last, and scaling does not fix out-of-distribution failure. A 2026 study of a 350M-parameter world model finds hallucination concentrated in low-coverage state-action regions. It calls hallucination "fundamentally a data coverage issue". For an exception such as Down on one pipe among dozens, a model trained on the common case should predict the common outcome, "nothing happens", unless the rare transition is in the data and up-weighted.

### Cited Findings
- **DIAMOND, CS:GO model** (2024). The CS:GO model has 381M parameters and was trained on about 87 hours of Dust II play for 12 days on an RTX 4090. It "is more likely to drift out-of-distribution in less frequently visited areas of the map". It can "forget the current state" near walls or when visibility is lost, and it "wrongly enables successive jumps". The authors expect scaling to fix most issues "with the exception of the memory of the model" — [DIAMOND](https://arxiv.org/html/2405.12399)
- **DIAMOND on Atari, compounding error** (2024). With a DDPM sampler, generation "suffers from compounding error". The EDM-based model "appears much more stable over long time horizons", even with one denoising step. Partial observability makes next frames multimodal, so one step can blur them; Boxing uses 3 steps. Memory is a short frame stack, which the authors call "a minimal mechanism". Their stated strength is small visual details (Asterix, Breakout, Road Runner) — [DIAMOND](https://arxiv.org/html/2405.12399)
- **"How Far is Video Generation from World Model: A Physical Law Perspective"** (Kang et al., arXiv Nov 2024; ICML 2025). The authors trained diffusion video models on a 2D physics simulator.
  - The models are perfect in-distribution and show measurable scaling on combinations of seen factors. They fail out of distribution.
  - They "mimic the closest training example" ("case-based" generalisation). When retrieving that example, they weight attributes in the order **colour > size > velocity > shape**.
  - "Scaling alone is insufficient" for out-of-distribution generalisation.

  Source: [arXiv 2411.02385](https://arxiv.org/abs/2411.02385)
- **"Hallucination in World Models is Predictable and Preventable"** (Hansen & Wang, arXiv 25 June 2026).
  - Setup: a 350M-parameter world model trained on MMBench2, 427 hours of data from 210 visual tasks with ground-truth actions.
  - Hallucinations are rollouts "visually fluent while drifting from the ground-truth dynamics". There are three modes: perceptual, action-marginalised and scene-diverging. Hallucination concentrates in low-coverage regions of state-action space.
  - Fixes: coverage-aware sampling during training, and hallucination predictors used **as curiosity rewards** for targeted data collection. Fine-tuning to unseen environments needs as few as 50 real trajectories.
  - The abstract page gives no hallucination rates or correlation numbers.

  Source: [arXiv 2606.27326](https://arxiv.org/abs/2606.27326)
- **LucidWM** (arXiv preprint, 2026; seen only as a search snippet). It notes that "a state and an action can each be familiar while their combination is unsupported by observed transitions", and that common uncertainty readouts can miss this. That is the 8-4 pipe-plus-Down case exactly — [arXiv 2609.37156](https://arxiv.org/pdf/2609.37156)
- **Models also invent abilities that do not exist.** WorldCoder's learned model hallucinated bullet firing in Montezuma's Revenge (2025) — [PoE-World](https://arxiv.org/html/2505.10819). DIAMOND's CS:GO model "wrongly enables successive jumps" — [DIAMOND](https://arxiv.org/html/2405.12399). A 2026 preprint notes that optimising inside a learned model can find policies "optimal in imagination yet infeasible or unsafe in the environment" — [WoVR, arXiv 2602.13977](https://arxiv.org/html/2602.13977)
- **Pixel reconstruction underweights small, decision-relevant objects** (OC-STORM, 2025). L2 reconstruction "is dominated by large areas and often fails to capture decision-relevant details". STORM's Hollow Knight reconstructions missed the key characters — [OC-STORM, arXiv 2501.16443](https://arxiv.org/html/2501.16443v1)
- **Curious Replay** (Kauvar et al., ICML 2023) is a mitigation that works by changing which transitions the model trains on. It prioritises replay by visit count plus world-model loss, so rare, badly predicted transitions are replayed more. It raised DreamerV3 on Crafter from **14.5 to 19.4**. It also gave a "dramatic improvement" in interacting with a novel object, which Plan2Explore "did not quickly interact with" — [arXiv 2306.15934](https://arxiv.org/abs/2306.15934v1); [project page](https://sites.google.com/view/curious-replay)

### Inferences
- Consider a model trained on hundreds of Mario pipes where Down did nothing. By the case-based result, it retrieves visually similar pipes (same colour and size) and predicts "nothing happens" on the enterable 8-4 pipe until that transition is in its data. That is regression to the common outcome, and it matches the earlier report's OPIQ argument.
- If the agent has entered the pipe only a few times, those transitions are a tiny fraction of the replay. Uniform replay will underfit them. Coverage-aware replay (Curious Replay; Hansen & Wang) is the published fix, but it acts only after the event has been seen.
- The opposite error matters too. Models hallucinate affordances: successive jumps in CS:GO, bullets in Montezuma's Revenge. Imagination-trained policies can exploit such errors. A next-screen predictor can therefore suggest false exceptions as well as miss true ones. Either way, the emulator has to verify.
- Common events will be modelled well: enemy contact leading to death, stomping a Goomba, the mushroom turning Mario big. These recur thousands of times in ordinary play. Rarity is the problem, not event type.

### Gaps
- No paper was found that measures a pixel world model's accuracy on a transition seen only k = 1–10 times, for example a learning curve of rare-event fidelity against count. Hansen & Wang may contain this, but only the abstract was read.
- No measurement of DIAMOND, IRIS or STORM accuracy on rare Atari events (key pickup, room transitions) was found.
- LucidWM was seen only as a search snippet; its method and results are unverified.

## 3. Exploration methods that use a world model: which helped in hard-exploration or procedurally varied games, and by how much?

### Takeaway
World models helped exploration when used as a **novelty or uncertainty signal** (BYOL-Explore, latent-model curiosity, Plan2Explore, Hansen & Wang's hallucination-as-curiosity) or as a **replay prioritiser** (Curious Replay). They also helped when combined with **archives and priors** (Intelligent Go-Explore, PoE-World), or with **hierarchical latent goals** (Director). They did not help when used only as a simulator for a policy that explores by entropy. The largest hard-exploration gains come from BYOL-Explore, where the world model is a curiosity signal, not a planner. Gains in procedurally varied games (Crafter, Craftax-Classic) depend on milestone rewards and are modest.

### Cited Findings
- **BYOL-Explore** (arXiv June 2022, NeurIPS 2022). It learns "a world representation, the world dynamics, and an exploration policy" with one latent-prediction loss. It uses prediction error as intrinsic reward and claims "superhuman performance on the ten hardest exploration games in Atari". It solved "the majority" of DM-HARD-8 tasks with intrinsic plus extrinsic reward — [arXiv 2206.08332](https://arxiv.org/abs/2206.08332). The earlier report gives Montezuma's Revenge 13,518 and Pitfall 25,176. It also notes that BYOL-Explore is "trapped immediately" by press-triggered noise and flatlines on sticky-action Montezuma's Revenge — [Jarrett et al. 2022](https://arxiv.org/abs/2211.10515)
- **Director** (Hafner et al., arXiv June 2022, NeurIPS 2022). It plans in a learned world model's latent space. A high-level policy picks latent goals using task plus exploration reward, and a low-level policy reaches them. It beat exploration baselines on sparse-reward tasks, including 3D quadruped maze traversal. Plan2Explore (ensemble disagreement) "makes initial progress but fails to converge" in the maze, "struggles with five pads and completely fails with six" on pad activation — [Google Research blog](https://research.google/blog/deep-hierarchical-planning-from-pixels/); [arXiv 2206.04114](https://arxiv.org/abs/2206.04114). Director's Atari and Crafter numbers were not retrieved.
- **Curious Replay** (ICML 2023): Crafter **14.5 → 19.4** with DreamerV3, similar on DMC, and much better interaction with novel objects than Plan2Explore — [arXiv 2306.15934](https://arxiv.org/abs/2306.15934v1)
- **Transformer world model on Craftax-Classic** (Dedieu et al., arXiv Feb 2025, ICML 2025). Reward is **67.42% (v1) / 69.66% (v3)** after 1M environment steps. Human expert is 65.0% and DreamerV3 53.2%; the v3 abstract calls it the first result above human. The gains came from Dyna with warm-up (real plus imagined data), a nearest-neighbour patch tokenizer and block teacher forcing, on top of a strong model-free CNN+RNN baseline — [arXiv 2502.01591](https://arxiv.org/pdf/2502.01591); [ICML 2025](https://icml.cc/virtual/2025/poster/45728)
- **Intelligent Go-Explore (IGE)** (arXiv May 2024, ICLR 2025). A foundation model replaces Go-Explore's hand-coded rules: it picks which archived state to return to and which action to take, and judges whether a new state is "interestingly new". Results:

  | Task | Budget | IGE | Baselines |
  |---|---|---|---|
  | Game of 24 (GPT-4) | 150 operations | 100 ± 0% | ReAct 82%, Reflexion 83%, classic Go-Explore 38% |
  | BabyAI-Text "put next to" | 250 steps | 84 ± 14% | GLAM used 1.5M steps; classic Go-Explore 4% |
  | TextWorld Cooking | 240 steps | 92 ± 10% | 36 points above ReAct; classic Go-Explore 0% |
  | TextWorld Coin Collector | — | the only method to solve it | all others failed |

  Removing foundation-model action selection dropped Game of 24 to 68% and BabyAI to 24%. Visual BabyAI is described only as success "in a majority of cases". IGE was not tested on Atari, NES or long pixel games, and the authors name NetHack as future work — [arXiv 2405.15143](https://arxiv.org/html/2405.15143)
- **PoE-World plus a planner** (2025) is the only method to score on Montezuma's Revenge and MR-Alt in its comparison, at 100 points versus 0 for PPO at 20M steps. The authors say exploration itself is not addressed — [arXiv 2505.10819](https://arxiv.org/html/2505.10819)
- **Hallucination predictors as curiosity** (Hansen & Wang, June 2026). The signals that predict where the world model hallucinates are reused as curiosity rewards for targeted data collection. No exploration scores were given in the abstract — [arXiv 2606.27326](https://arxiv.org/abs/2606.27326)
- **Latent world-model curiosity** (believed NeurIPS 2020): Montezuma's Revenge 2,276 — [review page](https://liner.com/review/latent-world-models-for-intrinsically-motivated-exploration)
- **Where none of this helped** (from the earlier report). On Craftax with 4,096 environments and 1B steps, "none of the tested exploration methods" (RND, ICM, E3B) improved PPO — [Craftax](https://arxiv.org/abs/2402.16801). Ensemble disagreement stays high on any unfamiliar input, so it tracks screen novelty — [Pathak et al. 2019](https://arxiv.org/abs/1906.04161)

### Inferences
- Every world-model method that improved hard exploration turned model error or novelty into **reward or replay priority**, or used an **archive with return-to-state**. None improved it just by making imagined rollouts more realistic. That fits the DIAMOND table: higher fidelity, the same Private Eye score.
- None of these mechanisms pays for trying an action whose outcome the model already predicts confidently. Down on a familiar-looking pipe is exactly that case. BYOL-Explore, Plan2Explore and hallucination-curiosity all reward uncertainty, and a model that generalises from other pipes is certain. Only state-action counts (OPIQ-style, from the earlier report) or a prior that says "try Down on pipes" target it.
- IGE and PoE-World point to a route the earlier report did not cover: a **foundation-model prior** to rank archive states and actions. An LLM or VLM that "knows" Mario pipes can be entered, and that Down enters them, could rank (pipe-top, Down) highly before it has been tried. This prior comes from internet knowledge of the game, not from a world model. It is untested on pixel NES games. IGE's evidence is from text tasks with budgets of hundreds of steps.
- Craftax-Classic and Crafter gains come from milestone achievements that random play stumbles on. They are sample-efficiency results in procedurally varied games, not evidence of finding single rare exceptions.

### Gaps
- Director's and Plan2Explore's numbers on Atari, Crafter and DMLab were not retrieved.
- No result was found for Plan2Explore or ensemble disagreement on Montezuma's Revenge or Pitfall with a pixel world model.
- No published "model-based Go-Explore" result with a learned pixel world model on Atari was located this session. The Time-Myopic Go-Explore and Latent Go-Explore results appeared in search but were not read: [Time-Myopic Go-Explore](https://arxiv.org/pdf/2301.05635); [Latent Go-Explore listing](https://huggingface.co/papers?q=Latent+Go-Explore).
- IGE's cost in foundation-model calls was not reported in the text read.

## 4. Can a world model teach general entity knowledge that transfers across levels or games?

### Takeaway
Partly. Object-centric and symbolic world models show real compositional transfer. PoE-World transfers to a recomposed Montezuma level with no demonstrations. OC-STORM improves boss fights and object-heavy games. Pixel and video models generalise "case-based" and fail out of distribution. Large multi-game video models (Genie, 30k hours of platformer video) learn consistent controls (latent actions) that transfer to infer policies from unseen videos. No source shows a learned world model acquiring a rule like "most pipes cannot be entered, this one can". The evidence suggests models learn the majority rule and miss the exception.

### Cited Findings
- **OC-STORM** (arXiv Jan 2025). It adds per-object features from a frozen video segmenter (Cutie), seeded with 6 hand-labelled frames per Atari game and 12 per Hollow Knight boss, alongside 64×64 pixels.
  - Atari 100k: mean HNS **134.8% vs 114.2%** for the authors' STORM re-run, winning 18 of 26 games. Published DIAMOND is 146% and DreamerV3 125%.
  - Games whose key information is objects: **116.5% → 142.8%**. Other games: **130.0% → 124.0%**, worse. Hero fell from 11,434 to 6,441.
  - Hollow Knight at 100k samples (about 3.1 hours at 9 FPS), win rates: Mage Lord **5% → 48%**, Hornet Protector 66.7% → 100%, Mantis Lords 71.7% → 83.3%, God Tamer 70% → 55%, Pure Vessel 0% → 0% (13.3% at 400k).
  - Limits: identical objects merge into one vector. Background structure ("the tunnel in Atari Gopher") is hard to encode. Labelling needs prior knowledge of the game. Only 3–5 seeds were run.

  Source: [arXiv 2501.16443](https://arxiv.org/html/2501.16443v1)
- **PoE-World** (2025). Programmatic, object-level rules learned from under 1,000 frames of Montezuma's Revenge transferred to MR-Alt, which has more platforms, skulls and ladders and no demonstrations: 100 points vs 0 for every baseline. It also transferred to Pong-Alt, with three balls and three opponents: −13.67 vs PPO at 20M steps 1.33 and WorldCoder −19.00. The model's inputs are hand-patched OCAtari object lists — [arXiv 2505.10819](https://arxiv.org/html/2505.10819)
- **Genie** (arXiv Feb 2024, ICML 2024). An 11B-parameter model trained on 30,000 hours of internet video from hundreds of 2D platformer games. It learns a small discrete vocabulary of latent actions without labels; one summary says 8. These latent actions "can be used for inferring policies from unseen action-free videos" of RL environments. No hard-exploration or rare-event results were found — [arXiv 2402.15391](https://arxiv.org/html/2402.15391v1); [HF summary](https://Huggingface.co/blog/vladbogo/genie-generative-interactive-environments)
- **Video pre-training for RL.** APV (2022) and ContextWM (NeurIPS 2023) pre-train world models on action-free video, then fine-tune with actions. ContextWM "can significantly improve the sample efficiency" on Meta-World, DMC and CARLA. It notes that earlier attempts gave "underfitting for video pre-training or negligible benefits for downstream visual control tasks". In-domain video (RLBench for Meta-World) can beat in-the-wild video. All results are on dense-reward control; none are on hard exploration — [ContextWM, arXiv 2305.18499](https://arxiv.org/pdf/2305.18499); [APV, arXiv 2203.13880](https://arxiv.org/pdf/2203.13880.pdf)
- **Case-based generalisation** (ICML 2025). Video diffusion models retrieve the closest training case by colour > size > velocity > shape, and fail out of distribution even when scaled — [arXiv 2411.02385](https://arxiv.org/abs/2411.02385)
- **Procedurally generated games** (2025). On Craftax-Classic, a transformer world model reached 67–70% of reward versus 65% for human experts, at 1M steps. This is generalisation across procedurally generated worlds with consistent entity rules (zombies hurt; tables enable tools) — [arXiv 2502.01591](https://arxiv.org/pdf/2502.01591)
- **Hallucination fine-tuning** (2026). A world model pre-trained on 210 tasks could be adapted to unseen environments with as few as 50 real trajectories — [arXiv 2606.27326](https://arxiv.org/abs/2606.27326)
- **Transfer in earlier work** (from the earlier report). In Mario, an ICM policy fine-tuned on Level 2 slightly beat training from scratch (466 vs 455.5 distance). On Level 3, fine-tuning did worse than running the policy unchanged (97.5 vs 319.3) — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363). Procgen needed "as many as 10,000 levels" to close the generalisation gap — [Procgen](https://arxiv.org/abs/1912.01588)

### Inferences
- Entity knowledge that recurs often and looks consistent should transfer through an object-centric or pretrained model. Examples: Goombas kill on side contact and die when stomped; mushrooms make Mario big; fire flowers enable fireballs. OC-STORM's boss-fight gains are the closest evidence that a world model helps learn "which enemies hurt and how to defeat them" quickly. They are measured on Hollow Knight, not NES.
- A visual exception will not transfer, and may be suppressed: one enterable pipe that looks like all the others. Case-based generalisation retrieves by colour and size first, and every SMB pipe matches. The model's prior for the 8-4 pipe will be "not enterable". If there is a cue, it is level context, not appearance. In SMB, enterable pipes are not visually distinct from decorative ones; this is game knowledge, not sourced here.
- Genie-style latent actions, learned from many platformer videos, are the multi-game evidence that "what buttons do" can be learned broadly. That is the general half of the earlier report's general-versus-exception split.
- Whether to take power-ups is a value question, not a prediction question. A world model can predict that a power-up changes the player's state. Whether that is good depends on downstream reward. The earlier report's Contra example, where a score reward led to the Laser that is weak against the boss, shows reward design dominates here — [kaihe Contra baseline](https://kaihe.github.io/contra_agent/experiments/1_baseline.html)

### Gaps
- No world-model study on NetHack, MiniHack or Minecraft that tests entity-rule transfer with numbers was retrieved this session. The earlier report's Dreamer 4 and VPT numbers are milestone-reward results.
- Genie's quantitative transfer results (e.g., CoinRun behaviour cloning from latent actions) were not retrieved.
- No source tests whether a world model learns "most X do nothing, this X does" (exception learning) as a function of exception frequency.
- No multi-game NES world model was found.

## 5. Relation to the earlier report, and the direct answer to the objective question

### Takeaway
This research supports and sharpens the earlier report's verdict: "a world model alone will not find rare presses; count (place, button), branch probes, keep PPO". A good next-screen predictor, even DIAMOND-quality, would likely help the agent learn common entity dynamics faster. That covers which enemies hurt, how stomps and fireballs work, and what a power-up does, because those events are frequent. It would not find the 8-4 enterable pipe. It would confidently predict "nothing happens" for Down on that pipe, by case-based generalisation from every other pipe, and no uncertainty-based curiosity would point there. New since the earlier report: (a) direct 2024–2026 evidence that world-model error concentrates in low-coverage regions; (b) coverage-aware replay and hallucination-as-curiosity as the world-model-side fixes; (c) foundation-model priors (IGE) and object-level program models (PoE-World) as the only world-model-adjacent routes with hard-exploration wins, both on symbolic or text inputs.

### Cited Findings
- **The earlier report's conclusions**: learned models with tree search scored 0 on Montezuma's Revenge and Pitfall after 20B frames. Down on a pipe is "a sampling problem, not a modelling problem". OPIQ-style (place, button) counts and counterfactual RAM probes from save states are recommended. The emulator is already an exact world model with save and restore — [earlier report, local file] `reports/Controllable exploration for NES agents.md`; [OPIQ](https://arxiv.org/abs/2002.12174); [MuZero](https://arxiv.org/pdf/1911.08265)
- **No published fidelity gain without exploration gain**: DIAMOND has the best Atari 100k mean HNS (1.459) and scores 114.3 on Private Eye, against a human 69,571 — [DIAMOND](https://arxiv.org/html/2405.12399)
- **Rare and low-coverage regions are where models fail**: [DIAMOND CS:GO](https://arxiv.org/html/2405.12399); [Hansen & Wang 2026](https://arxiv.org/abs/2606.27326); [Kang et al. 2024/2025](https://arxiv.org/abs/2411.02385)
- **Mechanisms that help, with numbers**: BYOL-Explore superhuman on the 10 hardest Atari exploration games — [arXiv 2206.08332](https://arxiv.org/abs/2206.08332). Curious Replay on Crafter 14.5 → 19.4 — [arXiv 2306.15934](https://arxiv.org/abs/2306.15934v1). Transformer world model on Craftax-Classic 67–70% vs human 65% — [arXiv 2502.01591](https://arxiv.org/pdf/2502.01591). OC-STORM on Hollow Knight Mage Lord 5% → 48% win rate — [arXiv 2501.16443](https://arxiv.org/html/2501.16443v1). PoE-World on Montezuma's Revenge 100 vs 0 for PPO at 20M steps — [arXiv 2505.10819](https://arxiv.org/html/2505.10819). IGE on Coin Collector, the only solver — [arXiv 2405.15143](https://arxiv.org/html/2405.15143)
- **Cost**: DIAMOND needs about 2.9 RTX 4090-days per 100k-step Atari run — [DIAMOND](https://arxiv.org/html/2405.12399). The earlier report measured the lab's emulator at about 14,000 decisions per second.

### Inferences
- **Entering the one enterable pipe in 8-4**: a world model will not find it. Before the first entry, the model predicts the common outcome. After a handful of entries, uniform replay underweights them, so the model may still predict "nothing happens" there. Coverage-aware or Curious-Replay-style prioritisation would then help. The first entry has to come from (place, button) counts, counterfactual probes, or a game-knowledge prior from an LLM or VLM, which is IGE-style and untested on NES.
- **Which enemies hurt and how to defeat them**: likely yes, and faster. These are frequent, consistent transitions, exactly what world models learn well. OC-STORM's boss results suggest object-centric inputs help further. The lab already reads RAM, which is a stronger object-centric input than segmentation.
- **Whether to take power-ups**: a world model predicts what a power-up does. Whether to take one is decided by value learning and reward design. The model helps only if imagined rollouts reach the later payoff, for example the fire flower making a later enemy killable. Long-horizon imagination is where compounding error and short memory hurt (DIAMOND's frame stack; CS:GO "forget the current state").
- **Practical reading for the lab**: if a learned next-screen predictor is built, its best published uses are (1) an auxiliary representation loss, (2) a curiosity or hallucination signal, and (3) a coverage-aware replay prioritiser. It should not be a replacement simulator, because the emulator is exact and far faster. Its error map, where it hallucinates, is itself a useful signal for where to probe.

### Gaps
- No experiment anywhere measures whether a world model trained on SMB levels 1-1 to 8-3 assigns non-trivial probability to entering the 8-4 pipe before the agent tries it. This is the cheapest decisive test, and it is untested.
- No published result combines a foundation-model prior (IGE-style) with pixel NES play or (place, button) counts.
- Numerical rare-event fidelity curves (accuracy against number of times seen) for diffusion world models were not found.
