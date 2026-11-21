# Exploration under distractors: intrinsic motivation that targets untried actions and resists the "noisy TV" (for dynamic NES games)

Scope note: the notes cover what each method treats as novel, how it behaves when much of the screen moves on its own, reported numbers on hard Atari games and other benchmarks, and practical pitfalls. Most primary papers were read in full text through ar5iv/arXiv HTML. Where a number appears only in a figure and not in the text, it is marked as a gap rather than estimated. Dates are arXiv first-posting months; the venue is given where the paper states it.

---

## 1. Method catalogue: how each method defines novelty and how it reacts to background motion

### Takeaway
There are five families of novelty signal. (1) State novelty: counts, pseudo-counts, RND, and the Go-Explore archive. (2) Prediction error: pixel or feature forward models (ICM, BYOL-Explore). (3) Model disagreement: Disagreement, Plan2Explore. (4) Change in a controllable or embedded state: RIDE, NGU and E3B episodic terms on inverse-dynamics features, contingency-aware counts, causal-influence (CAI). (5) Frontier or learning-progress signals: NovelD/BeBold, LPM. Only a few methods reward an action in an already-visited state before that action has produced a new state. Go-Explore does it by construction: it returns to an archived cell, then takes random actions there. OPIQ does it with optimistic state-action counts, and CAI does it with action-influence estimates. Everything else rewards the outcome, so it pays out only after the rare press has already been found.

### Cited Findings

**State-novelty / pseudo-count family**
- **RND** (Burda, Edwards, Storkey, Klimov; arXiv Oct 2018, ICLR 2019).
  - Bonus: the error of a predictor network trained to match a fixed, randomly initialised target network on the next observation.
  - Robustness claim: "RND obviates factors 2 and 3 since the target network can be chosen to be deterministic and inside the model-class of the predictor network." This removes stochastic-transition noise and model-misspecification as error sources.
  - Normalisation: the intrinsic reward is divided "by a running estimate of the standard deviations of the intrinsic returns." Observations are whitened and clipped to [-5, 5] for the predictor and target, but not for the policy.
  - The intrinsic return is non-episodic; the extrinsic return is episodic. — [RND, arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
- **Successor-representation counts** (Machado, Bellemare, Bowling; arXiv Jul 2018). Bonus r_int = 1/||ψ(S_t)||₁, the inverse ℓ1 norm of learned successor features, with β = 0.05 on Atari. — [Count-based exploration with the SR, arXiv 1807.11622](https://arxiv.org/abs/1807.11622)
- **OPIQ** (Rashid et al., arXiv Feb 2020, ICLR 2020).
  - Mechanism: adds state-action count bonuses to DQN Q-values, applied during both action selection and bootstrapping, so that novel (s, a) pairs stay optimistic even when network generalisation would otherwise erase the optimism.
  - Result: it "manages to reliably explore 12 rooms during the 12.5mil timesteps" (50M frames) on Montezuma's Revenge, which the authors call significantly more than other methods. — [OPIQ, arXiv 2002.12174](https://arxiv.org/abs/2002.12174)

**Go-Explore family: archive of states, return, then explore from the frontier**
- **Go-Explore / "First return, then explore"** (Ecoffet, Huizinga, Lehman, Stanley, Clune).
  - Dates: arXiv Jan 2019 (1901.10995) and Apr 2020 (2004.12919); Nature 2021.
  - Cells are downscaled grayscale frames whose width, height and pixel depth are tuned dynamically to balance archive entropy against cell count. Domain-knowledge cells for Montezuma's Revenge are room, discretised x/y, keys and level.
  - Selection weight: W = 1/√(C_seen + 1).
  - After returning to a cell, it takes "random actions for 100 steps", sampled uniformly from the action set, and each action "repeats the previous one with 95% probability."
  - It names two failure modes of intrinsic-motivation agents. Detachment is "losing track of interesting areas to explore from". Derailment is when exploratory noise prevents a policy from "returning to previously visited states".
  - Policy-based Go-Explore trains a goal-conditioned PPO policy with self-imitation, so it needs no simulator restore. Its policy finds "roughly four times more cells" than random actions. — [First return, then explore, arXiv 2004.12919](https://arxiv.org/abs/2004.12919)
- **Latent Go-Explore (LGE, "cell-free")** (Gallouédec & Dellandréa; arXiv Aug 2022).
  - Replaces cells with a learned latent space: inverse dynamics, forward dynamics, or VQ-VAE (used for Atari). Density is estimated with k-nearest neighbours, and goals are sampled with a geometric distribution over density rank.
  - Stated limitations: it assumes a fixed start state, so it breaks in procedurally generated environments; it depends heavily on the representation; and its goal-achievement test is a "naive" latent-distance threshold. — [Cell-Free Latent Go-Explore, arXiv 2208.14928](https://arxiv.org/abs/2208.14928)

**Prediction-error family**
- **ICM** (Pathak, Agrawal, Efros, Darrell; arXiv May 2017, ICML 2017).
  - Bonus: r = (η/2)·||φ̂(s_{t+1}) − φ(s_{t+1})||², the forward-model error in a 288-dimensional feature space learned by an inverse-dynamics model (β = 0.2, λ = 0.1).
  - Rationale: the inverse model is meant to keep only features the agent can affect or that affect the agent. — [ICM, arXiv 1705.05363](https://arxiv.org/abs/1705.05363)
- **Large-scale curiosity study** (Burda, Edwards, Pathak, Storkey, Darrell, Efros; arXiv Aug 2018, ICLR 2019).
  - Compared forward-error curiosity in four feature spaces: pixels, random features (RF), VAE, and inverse-dynamics features (IDF).
  - "Training dynamics on raw-pixels performs bad across all the environments."
  - IDF beat RF on 55% of Atari games. IDF-curious agents beat a random agent on 75% of games, against 70% for RF. — [Large-Scale Study of Curiosity-Driven Learning, arXiv 1808.04355](https://arxiv.org/abs/1808.04355)
- **BYOL-Explore** (Guo et al., DeepMind; arXiv Jun 2022, NeurIPS 2022).
  - Bonus: the sum of open-loop multi-step latent prediction errors, measured as cosine distance to an EMA target encoder (K = 8 on Atari).
  - Normalisation: divided by an EMA standard deviation, with optional clipping r = max(ℓ/σ_r − μ, 0). Mixed as r_e + λ·r_i with λ = 0.1 on Atari.
  - The authors describe their setting as "(fairly) deterministic". They list "highly-stochastic and procedurally-generated" environments such as NetHack as open challenges. — [BYOL-Explore, arXiv 2206.08332](https://arxiv.org/abs/2206.08332)
- **BYOL-Hindsight / "Curiosity in Hindsight"** (Jarrett, Tallec, Altché, Mesnard, Munos, Valko; arXiv Nov 2022).
  - Problem: prediction error in a stochastic world converges to the entropy of the next outcome, which the paper calls a "stochastic trap".
  - Fix: a hindsight latent drawn from the realised outcome is fed to the predictor. An invariance (contrastive) loss keeps that latent independent of the state and action, so only reducible error is rewarded. The paper's theorem says the reward vanishes at optimum under realizability. — [Curiosity in Hindsight, arXiv 2211.10515](https://arxiv.org/abs/2211.10515)

**Model-disagreement family**
- **Disagreement** (Pathak, Gandhi, Gupta; arXiv Jun 2019, ICML 2019).
  - Bonus: the variance across an ensemble of 5 forward models in feature space. It is independent of x_{t+1}, which makes it differentiable with respect to the policy.
  - Feature spaces: random features for games and navigation, ImageNet-pretrained ResNet-18 features for the robot. — [Self-Supervised Exploration via Disagreement, arXiv 1906.04161](https://arxiv.org/abs/1906.04161)
- **Plan2Explore** (Sekar et al., ICML 2020). Ensemble disagreement over one-step latent predictions inside a Dreamer-style world model, with exploration planned in imagination. This description comes from the paper's abstract and was not re-fetched in this session. — [Plan2Explore, arXiv 2005.05960](https://arxiv.org/abs/2005.05960)

**Episodic novelty on controllable features**
- **NGU** (Badia et al., DeepMind; arXiv Feb 2020, ICLR 2020).
  - Episodic term: r_episodic = 1/√n(f(x_t)), a k-NN pseudo-count (k = 10) over an episodic memory that "starts completely empty" each episode.
  - Embeddings f are "controllable states" from a Siamese inverse-dynamics network.
  - Combination: multiplied by a lifelong RND modulator α_t = 1 + (err − μ)/σ, clipped to [1, L] with L = 5.
  - Stated limitation: controllable states may fail where "it might take more than two consecutive steps" to see the effect of an action. — [NGU, arXiv 2002.06038](https://arxiv.org/abs/2002.06038)
- **Agent57** (Badia et al.; arXiv Mar 2020, ICML 2020). NGU plus three changes:
  - a per-actor sliding-window UCB bandit (window τ = 160 episodes) that chooses among 32 (β, γ) exploration policies;
  - separate extrinsic and intrinsic Q-networks, Q = Q_e + β·Q_i, which "significantly increase[s] the training stability";
  - a doubled backprop-through-time window, and γ up to 0.9999. — [Agent57, arXiv 2003.13350](https://arxiv.org/abs/2003.13350)
- **E3B** (Henaff, Raileanu, Jiang, Rocktäschel; arXiv Oct 2022, NeurIPS 2022).
  - Bonus: b = φ(s)ᵀ C⁻¹ φ(s), where C = Σφφᵀ + λI is reset every episode and updated with Sherman-Morrison (767 FPS, against 256 FPS for naive inversion). λ = 0.1.
  - Features φ come from an inverse-dynamics model, described as "discarding information the agent cannot use to predict its actions."
  - β = 1 (MiniHack), 3·10⁻⁶ (VizDoom), 0.1 (Habitat). — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)

**Impact, frontier and learning-progress family**
- **RIDE** (Raileanu & Rocktäschel; arXiv Feb 2020, ICLR 2020).
  - Bonus: r = ||φ(s_{t+1}) − φ(s_t)||₂ / √N_ep(s_{t+1}), the change in an embedding trained by forward plus inverse dynamics, discounted by the episodic count.
  - Ablations: "OnlyEpisodicCounts" solved no tasks. Without the episodic count term, the agent can cycle between two states. Using raw observations instead of the learned embedding solved no tasks. — [RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)
- **BeBold / NovelD** (Zhang, Xu, Wang, Wu, Keutzer, Gonzalez, Tian; arXiv Dec 2020; published as NovelD at NeurIPS 2021).
  - Bonus: r = max(1/N(s_{t+1}) − 1/N(s_t), 0) · 1{N_e(s_{t+1}) = 1}. The agent is rewarded for crossing from familiar into novel territory, and only on the first visit within an episode. N is approximated by RND error.
  - Noisy TV: the authors say it "suffers from this problem as well." — [BeBold, arXiv 2012.08621](https://arxiv.org/abs/2012.08621)
- **AGAC** (Flet-Berliac et al., 2021). The E3B authors grouped AGAC with RIDE and NovelD as methods whose success depends on an episodic count term: "all three methods fail to learn" when it is removed. AGAC's own paper was not fetched in this session. — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
- **Learning Progress Monitoring (LPM)** (Hou, An, Du, UC Merced; arXiv Sep 2025, v2 Apr 2026).
  - Bonus: r^i = g_φ(o_t, a_t) − ε_t(o_{t+1}), the previous model's expected error minus the current model's error. It "rewards model improvements instead of prediction error or novelty."
  - Venue not stated. — [Beyond Noisy-TVs, arXiv 2509.25438](https://arxiv.org/abs/2509.25438)
- **RLE (Random Latent Exploration)** (Mahankali, Hong, Sekhari, Rakhlin, Agrawal; arXiv Jul 2024; arXiv comments say "presented at ICML 2024").
  - The policy pursues randomly sampled latent goals instead of novelty bonuses. Described as a "simple and general plug-in for existing RL algorithms", evaluated on Atari and Isaac Gym.
  - The abstract gives no exact numbers. — [RLE, arXiv 2407.13755](https://arxiv.org/abs/2407.13755)
- **Other 2023–2024 noisy-TV-robust proposals** (seen only in search results, not read in full):
  - TeCLE (arXiv Dec 2024): an action-conditioned, temporally correlated latent; claims robustness on MiniGrid and stochastic Atari. — [TeCLE, arXiv 2412.04775](https://arxiv.org/abs/2412.04775)
  - "Surprise novelty" (arXiv Aug 2023): the search summary reports only a moderate noisy-TV gain over RND (0.9 vs 0.85). — [Beyond Surprise, arXiv 2308.04836](https://arxiv.org/abs/2308.04836)

### Inferences
- The user's existing "new places" reward is a lifelong state-novelty bonus. The decisive NES moves (Down on one pipe, walking left about 70 px in a maze castle) occur in states that are already familiar. Every outcome-based bonus (RND, NGU, E3B, RIDE, NovelD) pays nothing for these moves until the effect first appears, such as a new area, a pipe transition, or the maze loop breaking. They therefore do not raise the chance of trying the press. Only three mechanisms push untried actions in familiar states directly:
  - Go-Explore's return-then-random-explore, which puts many uniformly random, 95%-sticky action sequences into every archived cell;
  - optimistic state-action counts in the OPIQ style;
  - action-influence or impact estimates (CAI, RIDE), but only once the model has seen that the action changes something.
- With Go-Explore's 95% repeat probability, an action is held for 20 steps on average. That makes "hold Down" or "hold Left for many frames" far more likely than under independent per-step sampling. This is directly relevant to the pipe and maze-castle cases, provided cells are fine enough to include the specific pipe x-position.

### Gaps
- AGAC, empowerment-based intrinsic motivation (e.g. Mohamed & Rezende 2015) and the count-with-successor-features variants beyond Machado et al. were not read in full in this session. Formulas and results for AGAC and empowerment are not given here.
- RLE's Atari numbers, including Montezuma's Revenge, were not available from the abstract page.
- No source was found that evaluates a bonus specifically on "untried action in a familiar state" in a pixel game as a controlled test.

---

## 2. Methods that reward the impact of the agent's own action, and how they compare on noisy-TV tests

### Takeaway
Two kinds of method key novelty to what the agent controls. Inverse-dynamics feature methods (ICM, NGU, E3B, RIDE, contingency-aware CoEX) learn features through action prediction. CAI estimates action influence directly, using conditional mutual information between the action and an entity's next state. The ones tested on noise survive it: ICM against a 40% white-noise patch, RIDE against a colour-changing ball. They share one known failure mode: an action that reliably changes something irrelevant (a message, a HUD, a ball's colour) is itself "controllable", so it can be farmed.

### Cited Findings
- **ICM noise test.** About 40% of the VizDoom observation was replaced with a fixed white-noise region. "The proposed ICM agent achieves a perfect score", while ICM-pixels, which had solved the same sparse task without noise, "suffers significantly". No numeric scores are given in the text. — [ICM, arXiv 1705.05363](https://arxiv.org/abs/1705.05363)
- **RIDE noisy-TV test.** A ball in MiniGrid MultiRoomN7S4 changes "its color to a randomly picked one whenever the agent takes a particular action."
  - ICM's performance dropped because the agent was attracted to the ball, and Count needed more training time. RIDE is called "robust to the well-known noisy-TV problem".
  - No numbers are given in the text; they are in a figure only. — [RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)
- **RIDE rewards interactions over movement.** Mean intrinsic reward per action in MultiRoomN7S4:

  | Action | RIDE | ICM | RND |
  |---|---|---|---|
  | Open door | 0.0490 | 0.0055 | 0.0032 |
  | Move forward | 0.0181 | 0.0056 | 0.0026 |
  | Turn left/right | 0.0071 | 0.0052 | 0.0031 |

  - The pattern is not universal. In MultiRoomN12S10, ICM's forward-move reward (0.0086) exceeds RIDE's (0.0032).
  - Stated limitation: RIDE "may be drawn to take actions that significantly change the environment, even when those actions are undesirable." — [RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)
- **E3B and distractors.** The paper argues the inverse-dynamics encoder ignores uncontrollable state, but it runs no noisy-TV or distractor experiment.
  - Reported failure: agents earn intrinsic reward by "repeatedly taking actions like PAY or FIGHT that produce new messages without changing the game state". The authors hypothesise the encoder keeps message information because it helps predict actions.
  - Ablations: E3B with a random fixed encoder, with a policy-tied encoder, or non-episodic all "perform significantly worse". — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
- **NGU controllable-state test (Random Disco Maze).** A 21x21 gridworld with wall colours randomised every step and no extrinsic reward.
  - Learned controllable states produced depth-first-search-like exploration. Random projections and an RND baseline only learned to avoid walls.
  - NGU has no explicit "noisy TV" experiment. — [NGU, arXiv 2002.06038](https://arxiv.org/abs/2002.06038)
- **Contingency-aware exploration, CoEX** (Choi et al.; arXiv Nov 2018, ICLR 2019).
  - An attentive dynamics model, an inverse-dynamics model with sparsemax spatial attention over a 9×9 grid, locates the region "predictive of the agent's actions". The argmax of the attention map gives the agent's location.
  - Counts are over ψ(s) = (x, y, visual-context cluster, cumulative extrinsic reward), with bonus 1/√#(ψ(s)).
  - The predicted location is "on average 1 or 2 grid cells away" from the RAM ground truth on Montezuma's Revenge, Seaquest, Hero and Venture. Error rises temporarily when a new scene appears.
  - Stated limitations: actions with no effect (e.g. mid-jump) give noisy labels, and a per-game clustering threshold is needed. — [Contingency-Aware Exploration, arXiv 1811.01483](https://arxiv.org/abs/1811.01483)
- **Causal Action Influence, CAI** (Seitzer, Schölkopf, Martius; arXiv Jun 2021, NeurIPS 2021).
  - Measure: C^j(s) = I(S'_j; A | S = s), estimated with a Gaussian next-state model and a mixture over K = 64 sampled actions.
  - Three uses: as a bonus; as "active action selection", picking the sampled action whose predicted outcome deviates most from the mixture; and as prioritised replay (CAI-P).
  - Results on Fetch tasks (state-based, with a known entity factorisation):
    - the bonus reaches 60% success "about four times faster than DDPG+HER";
    - active selection is about 2x more sample-efficient than random exploration;
    - CAI-P is 1.5–2.5x faster than PER;
    - detection AUC is 0.97 on FetchPickAndPlace.
  - Limitations: needs full observability and a predefined state factorisation. It is not pixel-based. — [CAI, arXiv 2106.03443](https://arxiv.org/abs/2106.03443)

### Inferences
- CAI's "active action selection" is the closest published analogue to choosing an untried action that has an effect in a familiar place. It needs a factored state, but NES RAM is already factored into entity variables (player x/y, area/sub-area ID, pipe-entry flags), so a RAM-entity version is plausible. That is an extrapolation; no source tests it on NES.
- On NES the "message spam" and "colour-changing ball" failures have direct analogues. Pausing, HUD changes, or animation triggered by a button press are action-predictable, so inverse-dynamics features will keep them. Impact-based bonuses can therefore pay for repeated harmless presses. A RAM-keyed count on the player's position and area ID avoids this; a pixel inverse-dynamics embedding does not.
- CoEX is essentially "learn where the player sprite is, then count its position". If RAM already exposes player position, CoEX's learned localiser is unnecessary. Its count key, position plus room plus cumulative reward, is a ready-made template.

### Gaps
- No head-to-head noisy-TV benchmark comparing RIDE, E3B, CAI and empowerment under the same protocol was found.
- Numeric noisy-TV results for RIDE and Disagreement exist only in figures.
- No empowerment-based method was verified on hard-exploration Atari or NES in this session.

---

## 3. Empirical findings on the noisy-TV problem: which bonuses fail and which survive

### Takeaway
- **Fail or degrade:** pixel prediction error; forward-dynamics error, including ICM when the noise is action-triggered; BYOL-Explore under sticky actions or on-demand noise; RND on on-demand pixel noise; NovelD/BeBold on stochastic tasks; observation-hash counts whenever the observation contains a timer.
- **Survive:** ensemble disagreement, which falls to near zero on pure noise; hindsight-corrected prediction error; learning-progress rewards; inverse-dynamics feature spaces against passive noise; RAM- or position-keyed counts.

### Cited Findings
- **Burda et al. Unity noisy TV.** A TV that changes channel when the agent acts "drastically slows down learning". With longer training, agents sometimes still converged to collecting the extrinsic reward.
  - Raw-pixel dynamics "performs bad across all the environments".
  - The authors argue unpredictability from partial observability or a weak model class causes the same problem as true randomness. — [Large-Scale Curiosity, arXiv 1808.04355](https://arxiv.org/abs/1808.04355)
- **RND on stochastic transitions.** The forward-dynamics baseline on Montezuma's Revenge "oscillates between two rooms", because sticky actions make the room-boundary outcome unpredictable and give "an irreducibly high prediction error". RND's deterministic target avoids this. — [RND, arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
- **Disagreement.**
  - On Noisy MNIST, disagreement "converges to almost zero intrinsic reward in both the scenarios", while prediction error "assigns more reward to the observations with higher stochasticity".
  - In the Unity maze with a TV, it "outperforms the baseline in the presence of the TV".
  - On sticky-action Atari it "outperforms previous state-of-the-art approaches".
  - Dropout-based disagreement "does not perform as well as ensemble-based". Numbers are in figures only. — [Disagreement, arXiv 1906.04161](https://arxiv.org/abs/1906.04161)
- **BYOL-Hindsight vs BYOL-Explore vs RND.**
  - Pycolab maze (100k learner steps, 3 seeds): under on-demand pixel noise, "BYOL-Explore is trapped immediately and RND suffers greatly", while BYOL-Hindsight matches its noise-free performance.
  - Montezuma's Revenge with sticky actions (3M learner steps, 3 seeds): BYOL-Explore "flatlines" when intrinsic-only. BYOL-Hindsight "explores most rooms" both intrinsic-only and mixed.
  - Ten hard Atari games with sticky actions: BYOL-Hindsight improves over BYOL-Explore in most cases, from only 1 seed. Exact scores are in figures only. — [Curiosity in Hindsight, arXiv 2211.10515](https://arxiv.org/abs/2211.10515)
- **LPM (2025).**
  - Noisy MNIST: its reward converges to zero after about 150 steps, against about 400 for AMA.
  - MiniWorld 3D maze with CIFAR-image noise: 1,347.6 states visited on average, 95.3 (7.6%) above the second-best method. The Figure 3 caption lists EDT as best in the deterministic setting (1276.1), which contradicts the text's "highest in all" claim.
  - MountainCar coverage: 76.50% deterministic and 67.04% stochastic. Ensemble disagreement drops 33.1%.
  - Atari action-noise variants: drops of 3.9% (Space Invaders) and 4.7% (UpNDown); EME drops 100%.
  - Note: these are the authors' own noise constructions and their own NGU re-implementation. — [LPM, arXiv 2509.25438](https://arxiv.org/abs/2509.25438)
- **Timer/counter makes every observation unique.** In MiniHack, "a time counter in the stats vector makes every observation unique", so standard NovelD "performs poorly". If every state is unique, an episodic count is always 1 and "carries no information". — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
- **BeBold/NovelD under noise.** The authors admit it "suffers from this problem as well" and attribute weaker results on Dynamic-Obstacles-5x5 to it. — [BeBold, arXiv 2012.08621](https://arxiv.org/abs/2012.08621)
- **Inverse-dynamics features under passive noise.** ICM survived the 40% white-noise patch where ICM-pixels did not ([ICM, arXiv 1705.05363](https://arxiv.org/abs/1705.05363)). RIDE survived the action-triggered colour-changing ball where ICM did not ([RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)).

### Inferences
- The distinction that matters is whether the noise is passive or action-triggered.
  - Passive background motion (enemies, scrolling, timers, animation) is handled by inverse-dynamics features, ensemble disagreement, RND with a deterministic target, or RAM-keyed counts.
  - Action-triggered noise (pressing a button changes something random or irrelevant) still traps prediction-error methods (ICM, BYOL-Explore). It can also trap impact-based ones, as the E3B message-spam case shows.
  - NES games have both kinds: random enemy behaviour, and sprite animation or sound effects tied to presses.
- For the user's setup, the existing reward for new camera or RAM positions is already noise-immune, because position is deterministic and agent-driven. The noisy-TV risk enters only if a pixel-based or full-RAM-based prediction-error bonus is added. Full NES RAM includes timers, RNG bytes and enemy state, so a novelty measure over full RAM behaves like an observation that contains a timer: every state looks unique, as E3B describes.

### Gaps
- No published noisy-TV evaluation on NES or Super Mario Bros was found.
- No numbers are in the text for the RIDE or Disagreement noisy-TV figures.
- The 2023–2024 alternatives (TeCLE, Surprise Novelty, hidden-state curiosity) were seen only through search snippets.

---

## 4. Results on hard-exploration Atari and other benchmarks, with step budgets, and compatibility with massively parallel PPO

### Takeaway
On Montezuma's Revenge without domain knowledge:

| Method | Montezuma's Revenge | Budget / setting |
|---|---|---|
| Go-Explore (robustified) | 43,791 | |
| NGU | 10.4k | 35B frames |
| Agent57 | 9,352 | |
| BYOL-Explore | 13,518 | max over training |
| CoEX + PPO | 11,618 | 2B frames |
| RND | 8,152 | 1.97B frames |
| RND, 1,024 envs | 10,070 | |
| BeBold | about 10,000 | 2B frames, 1 run |

Pitfall! separates the methods. RND scores −3, while NGU, Agent57, BYOL-Explore and Go-Explore score positive. RND, Burda-style curiosity, CoEX, BeBold and Craftax-style JAX PPO all run on many parallel environments. NGU and Agent57 are R2D2/actor-learner systems, and Go-Explore's best numbers rely on emulator state restore.

### Cited Findings

**RND** (PPO, 128 parallel envs, 1.97B frames; RNN policy)

| Game | RND | PPO | Dynamics |
|---|---|---|---|
| Gravitar | 3,906 | 3,426 | 3,371 |
| Montezuma's Revenge | 8,152 | 2,497 | 400 |
| Pitfall! | −3 | 0 | 0 |
| Private Eye | 8,666 | 105 | 33 |
| Solaris | 3,282 | 3,387 | 3,246 |
| Venture | 1,859 | 0 | 1,712 |

- With 1,024 envs, Montezuma's Revenge reached a "mean return of 10,070", with one run at 14,415.
- The best single run "visited all 24 rooms … achieving a best return of 17,500". The agent "often finds 22 out of the 24 rooms".
- Returns improve with more parallel envs, saturating earlier for CNN policies. The predictor's training batch was subsampled at scale (keep probability 0.25 at 128 envs, 0.03125 at 1,024).
- Stated limit: "global exploration that involves coordinated decisions over long time horizons is beyond the reach of our method." — [RND, arXiv 1810.12894](https://arxiv.org/abs/1810.12894)

**NGU** (256 actors; the text says "3535 billion frames" while Table 5 lists 35B, so a typo)
- Montezuma's Revenge 10.4k ± 1.6k; Pitfall! 8.4k ± 4.5k for N=32, or 15.2k ± 9.4k for NGU(N=1)-RND; Private Eye 100.0k; Venture 1.7k; Gravitar 14.1k; Solaris 4.9k.
- Below the best baseline on Venture (2.0k), Gravitar (15.7k) and Solaris (5.5k).
- NGU(N=1) "catastrophically fails" on Pong (−9.4). — [NGU, arXiv 2002.06038](https://arxiv.org/abs/2002.06038)

**Agent57 per-game table** (256 actors; the excerpt states no frame budget for the table, though the paper says Skiing is surpassed after 78B frames and 51 games within the first 55B frames)

| Game | Agent57 | NGU (in Agent57 paper) | MuZero |
|---|---|---|---|
| Montezuma's Revenge | 9352.01 ± 2939.78 | 19093.74 ± 12627.66 | 0.00 |
| Pitfall! | 18756.01 ± 9783.91 | 15334.30 | 0.00 |
| Private Eye | 79716.46 | 100314.44 | 15299.98 |
| Venture | 2623.71 | 1747.32 | 0.40 |
| Gravitar | 19213.96 | 14771.91 | 6682.70 |
| Solaris | 44199.93 | 7254.03 | 56.62 |
| Skiing | −4202.60 | −24271.33 | −29968.36 |

- The NGU numbers in this table differ from the NGU paper (Montezuma's Revenge 19,093 vs 10.4k), so evaluation protocols differ. — [Agent57, arXiv 2003.13350](https://arxiv.org/abs/2003.13350)

**Go-Explore** (robustified policies, sticky actions)

| Game | Go-Explore | Prior SOTA | Avg human |
|---|---|---|---|
| Montezuma's Revenge, downscaled cells | 43,791 | 11,618 | 4,753 |
| Montezuma's Revenge, domain knowledge | 1,731,645 | | |
| Pitfall!, downscaled | 6,954 | 0 | 6,463 |
| Pitfall!, domain knowledge | 102,571 | | |
| Private Eye | 95,756 | | |
| Venture | 2,281 | | |
| Gravitar | 7,588 | | |
| Solaris | 19,671 | | |
| Skiing | −3,660 | | |

- Exploration-phase budgets: 2B frames (5B for Pitfall! and Skiing, 15B for Private Eye); 1B with domain knowledge. Robustification: 10B (20B for Solaris).
- Policy-based Go-Explore with domain knowledge, without robustification: Montezuma's Revenge 97,728, Pitfall! 20,093. — [First return, then explore, arXiv 2004.12919](https://arxiv.org/abs/2004.12919)

**BYOL-Explore** (maximum score over training; the excerpt gives no total frame budget)

| Game | BYOL-Explore | RND (in BYOL-Explore paper) |
|---|---|---|
| Montezuma's Revenge | 13518.18 | 4554.37 |
| Pitfall! | 25175.53 | 0.00 |
| Private Eye | 33130.85 | 33150.09 |
| Venture | 1980.24 | 1879.68 |
| Gravitar | 17014.51 | 5885.23 |
| Solaris | 11060.24 | 5262.55 |

- Aggregate on the 10 hardest games: mean human-normalised score (HNS) 661.14 and median 331.98, against RND's 209.08 and 116.44. Superhuman on 8 of 10; the "big" variant is superhuman on all 10.
- Pure intrinsic reward on Montezuma's Revenge: "more than 20 rooms" on average, against 17 for RND. The extracted text read "2020", almost certainly a rendering artefact for 20. BYOL-Explore's intrinsic reward is episodic here, while RND's was non-episodic.
- DM-HARD-8 mean CHNS: 69.87 vs RND 14.36 and ICM 12.42. — [BYOL-Explore, arXiv 2206.08332](https://arxiv.org/abs/2206.08332)

**CoEX** (Contingency-aware exploration)

| Game | A2C (100M steps / 400M frames) | A2C+CoEX | PPO (500M steps / 2B frames, sticky) | PPO+CoEX |
|---|---|---|---|---|
| Montezuma's Revenge | 13 | 6,635 | 29 | 11,618 |
| Private Eye | 574 | 5,316 | 942 | 11,000 (cell alignment unclear) |
| Venture | 0 | 204 | 1,875 | 1,916 |
| Freeway | 7.2 | 34.0 | 34.0 | 34.0 |
| Frostbite | 1,099 | 4,260 | 7,340 | 9,076 |
| Seaquest | 2,401 | 5,169 | 2,806 | 11,794 |

- A2C results are max mean scores over 3 seeds. — [CoEX, arXiv 1811.01483](https://arxiv.org/abs/1811.01483)

**Successor-representation counts** (100M frames, sticky actions 0.25, 10 seeds; DQN_MMC^e+SR)
- Montezuma's Revenge 1395.4; Venture 1348.5; Gravitar 457.4; Private Eye 104.4; Solaris 1890.1.
- RND in the same table: Montezuma's Revenge 524.8; Venture 953.7. — [SR counts, arXiv 1807.11622](https://arxiv.org/abs/1807.11622)

**BeBold**
- Montezuma's Revenge: about 10,000 after 2B frames (CNN), against about 6,700 for RND, from a single run.
- MiniGrid: solves "the 12 most challenging environments" within 120M steps, where ICM, RIDE, RND, IMPALA and AMIGo score zero on half of them.
- NetHack: beats RND and IMPALA on all tested tasks (numbers in figures only). — [BeBold, arXiv 2012.08621](https://arxiv.org/abs/2012.08621)

**OPIQ.** "Reliably explore 12 rooms" in Montezuma's Revenge within 12.5M timesteps (50M frames). — [OPIQ, arXiv 2002.12174](https://arxiv.org/abs/2002.12174)

**Super Mario Bros.**
- ICM, curiosity only:
  - crosses "more than 30% of Level-1";
  - gets stuck at a pit at about 38% of the game that needs "a very specific sequence of 15-20 key presses";
  - action repeat 6, 20 asynchronous workers.
- The Level-1 policy transfers "surprisingly well" to Level-3 but poorly to Level-2. — [ICM, arXiv 1705.05363](https://arxiv.org/abs/1705.05363)
- Burda et al.: going from 128 to 2,048 parallel environments let the curiosity agent discover "11 levels", find secret rooms and defeat bosses. IDF features transferred from 1-1 to both 1-2 and 1-3, while random features failed on 1-3 because of the day-to-night colour change. — [Large-Scale Curiosity, arXiv 1808.04355](https://arxiv.org/abs/1808.04355)
- RIDE authors: in Mario with intrinsic reward only, "vanilla IMPALA can learn similarly good policies after training on only 1m frames". With intrinsic plus extrinsic, ICM "hurts learning". — [RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)

**MiniHack / NetHack**
- E3B "outperforms all other methods by a significant margin" on 16 MiniHack tasks at 50M steps, 5 seeds. Exact aggregates are in figures only. Skill tasks with the full 78-action space were solved by no method. — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
- A multiplicative episodic × global bonus (E3B×RND or E3B×NovelD) "significantly improve[s] median and IQM over E3B" on MiniHack, while additive combinations give no meaningful gain. On Montezuma's Revenge, a singleton MDP, RND does well and E3B does poorly. — [Global vs Episodic Bonuses, arXiv 2306.03236](https://arxiv.org/abs/2306.03236)

**Crafter** (1M steps)

| Agent | Crafter score |
|---|---|
| Human experts | 50.5 ± 6.8% |
| DreamerV2 | 10.0% |
| PPO | 4.6% |
| Rainbow | 4.3% |
| Plan2Explore (unsupervised) | 2.1% |
| RND (unsupervised) | 2.0% |
| Random | 1.6% |

— [Crafter, arXiv 2109.06780](https://arxiv.org/abs/2109.06780)

**Craftax** (JAX, 4,096 parallel workers)
- Throughput: about 405,618 steps/s on Craftax-Classic and about 266,961 steps/s on Craftax. PPO runs 1B steps in about 51 minutes on one GPU.
- With PPO, PPO-RNN, RND, ICM and E3B at 1B steps: "none of the tested exploration methods improved performance". E3B best achieved EAT_PLANT but lowered overall reward. — [Craftax, arXiv 2402.16801](https://arxiv.org/abs/2402.16801)

**Procgen.** The Henaff et al. study mentions ProcGen only as an example of a contextual MDP and reports no ProcGen results. — [Global vs Episodic Bonuses, arXiv 2306.03236](https://arxiv.org/abs/2306.03236)

**Compute shape**
- NGU and Agent57 use 256 actors feeding one GPU learner. Agent57 actors run about 260 environment steps per second each. — [Agent57, arXiv 2003.13350](https://arxiv.org/abs/2003.13350)
- E3B ran 1.5–2x slower in wall-clock time than the other methods, because an extra forward pass ran on CPU. — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)

### Inferences
- **Fit with the user's thousands-of-environments PPO on GPU:**
  - RND, ICM/IDF curiosity, and CoEX-style position counts are proven with PPO at 128–2,048 envs, and RND/ICM/E3B run at 4,096 in Craftax.
  - E3B needs a per-env d×d inverse-covariance matrix reset each episode. At d = 256 and 4,096 envs this is about 1 GB in fp32 (an arithmetic estimate, not from a source). That is feasible but not free.
  - NGU's per-env k-NN episodic memory is heavier still.
  - Go-Explore's exploration phase needs only save/restore of emulator state, which a GPU NES emulator can batch. Its robustification phase is a separate long PPO run (10B frames).
- The Pitfall! split matters for NES maze castles. Pure state-novelty methods (RND, Burda-style curiosity) scored 0 or −3 on Pitfall!. Methods that keep an explicit archive or episodic memory scored positive: Go-Explore 6,954 without domain knowledge, NGU, Agent57, BYOL-Explore. Pitfall!'s aliased rooms resemble looping maze castles; NGU notes its rooms are "frequently aliased to one another".
- The Mario pit "15–20 precise key presses" failure in ICM is the same class of problem as the pipe-Down move. An outcome-only curiosity bonus gave no signal until the sequence succeeded.

### Gaps
- Agent57's total frame budget for the per-game table, and BYOL-Explore's total frame budget, were not in the text read.
- Exact noisy-TV, Craftax and MiniHack aggregate numbers are in figures only.
- No Procgen hard-exploration result for these bonuses was located.
- No Go-Explore or E3B result on Super Mario Bros was found.

---

## 5. Practical pitfalls: bonus scale, non-stationarity, normalisation, episodic vs lifelong novelty, death and reset exploitation

### Takeaway
The recurring lessons are as follows.
- Normalise intrinsic reward by a running standard deviation of the intrinsic return, and keep separate value heads or networks with separate discounts.
- Use a non-episodic intrinsic return with lifelong novelty, but an episodic extrinsic return.
- Combine episodic and lifelong novelty multiplicatively, not additively.
- Tune β per game.
- Expect detachment, derailment, and "dancing with danger" behaviours.

### Cited Findings
- **RND recipe.**
  - Two value heads, V = V_E + V_I. Defaults γ_E = 0.999 and γ_I = 0.99. Raising γ_E from 0.99 to 0.999 "greatly improves performance", while γ_I = 0.999 "hurts" in the mixed setting.
  - Table 5 sets the extrinsic coefficient to 2 and the intrinsic coefficient to 1, while Algorithm A.1 writes r = i + e, so the paper is inconsistent.
  - Results are "dominated by effects of discrete choices", with large variance across 10 seeds. — [RND, arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
- **Death and reset exploitation.**
  - RND warns that non-episodic extrinsic rewards "could be exploited by a strategy" of restarting the game repeatedly. — [RND, arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
  - Burda et al. removed the done signal entirely. Agents still avoid dying, because death resets them to a well-known, predictable start. They warn that a survival bonus would be exploited by maximising episode length, as in Breakout. — [Large-Scale Curiosity, arXiv 1808.04355](https://arxiv.org/abs/1808.04355)
- **Dangerous play after mastery.** After mastering reliable rewards, the RND agent "keeps interacting with potentially dangerous objects", such as jumping over a moving skull or moving between laser gates. — [RND, arXiv 1810.12894](https://arxiv.org/abs/1810.12894)
- **Detachment and derailment** (definitions in section 1). Go-Explore addresses both by archiving states and returning to them before exploring. — [First return, then explore, arXiv 2004.12919](https://arxiv.org/abs/2004.12919)
- **Episodic vs global novelty** (Henaff et al. 2023).
  - Episodic bonuses work "when there is little shared structure across episodes". Global bonuses work for singleton or shared-layout environments.
  - On a singleton Corridors task, the episodic bonus scored 0.10 ± 0.68 against 0.96 ± 0.03 for the global bonus.
  - Additive combinations fail because "the global bonus decays over training", leaving the episodic term to dominate; multiplicative combinations are robust. — [Global vs Episodic Bonuses, arXiv 2306.03236](https://arxiv.org/abs/2306.03236)
- **Episodic counts are load-bearing.**
  - RIDE without the episodic term can cycle between two states. — [RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)
  - RIDE, AGAC and NovelD "fail to learn" without their episodic count term. — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
  - RND oscillates between two rooms in MiniGrid MRN7S8, where BeBold's frontier term keeps moving forward. — [BeBold, arXiv 2012.08621](https://arxiv.org/abs/2012.08621)
- **Bonus scale.**
  - E3B's β ranges over 5 orders of magnitude (1, 3·10⁻⁶, 0.1), and tuning the intrinsic coefficient was "important for best performance for all methods". — [E3B, arXiv 2210.05805](https://arxiv.org/abs/2210.05805)
  - NGU's β depends on the extrinsic reward scale. — [NGU, arXiv 2002.06038](https://arxiv.org/abs/2002.06038)
  - BYOL-Explore uses λ = 0.1 on Atari and 0.01 on DM-HARD-8. Removing reward clipping cut median HNS from 331.98 to 181.39. — [BYOL-Explore, arXiv 2206.08332](https://arxiv.org/abs/2206.08332)
- **Separate value networks** "significantly increase the training stability" in Agent57. — [Agent57, arXiv 2003.13350](https://arxiv.org/abs/2003.13350)
- **Intrinsic reward can hurt when extrinsic reward is dense.**
  - Craftax: "none of the tested exploration methods improved performance", with the authors hypothesising the intrinsic reward distracts. — [Craftax, arXiv 2402.16801](https://arxiv.org/abs/2402.16801)
  - In Mario, intrinsic plus extrinsic ICM "hurts learning". — [RIDE, arXiv 2002.12292](https://arxiv.org/abs/2002.12292)
- **Learned-embedding staleness.**
  - CoEX's localiser error rises when a new scene appears, then recovers. — [CoEX, arXiv 1811.01483](https://arxiv.org/abs/1811.01483)
  - NGU's controllable states can miss effects that take more than two steps to appear. — [NGU, arXiv 2002.06038](https://arxiv.org/abs/2002.06038)

### Inferences
- **Episodic memory can reward dying.** An episodic-novelty bonus (NGU, E3B, RIDE's episodic count, BeBold's first-visit indicator) resets on death. A level start that is cheap to reach then becomes novel again after every death, which could reward suicide-and-replay over pushing into familiar but unsolved terrain. This is an extrapolation from the reset rules above, not a reported finding. The non-episodic RND or Burda-style setup instead makes the start predictable, so death costs intrinsic reward.
- **NGU's two-step limit matters for "hold Left for about 70 px".** A move whose effect appears only after many frames (the maze castle accepting the correct path at a checkpoint) is invisible to one-step inverse-dynamics features. Count-based or archive-based mechanisms on RAM state (area ID, maze checkpoint progress) are better matched.
- **Mixing a new bonus with the existing "new places" reward:** use separate value heads and per-stream normalisation (RND, Agent57), and multiply rather than add if one term is episodic and the other lifelong (Henaff et al. 2023).

### Gaps
- No source quantifies death or reset exploitation under episodic bonuses.
- No published study was found of action-novelty bonuses for "familiar state, untried action" in NES games.
- Bonus non-stationarity fixes such as SOFE (ICLR 2024) were not read in this session.
