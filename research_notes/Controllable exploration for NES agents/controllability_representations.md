# Controllability / agent-centric representation learning for pixel-based RL (separating self-caused change from background dynamics)

Scope note: all facts below were read from the primary papers (arXiv PDFs, text-extracted). Dates are arXiv first-version dates unless a venue was visible in the PDF header. Numbers are copied as reported. Where a table had to be re-assembled from PDF column layout, that is flagged.

## 1. Inverse dynamics models and ICM (Pathak 2017) and the Burda et al. 2018 large-scale study: what the inverse model learns, why it ignores distractors, weaknesses, Mario results

### Takeaway
A one-step inverse model (predict a_t from phi(s_t), phi(s_t+1)) learns features that help predict the agent's own action. That makes it robust to action-independent pixel noise: ICM got a perfect score on VizDoom with 40% of the screen as white noise, while pixel-space curiosity failed. The same property makes it *insufficient*: by construction it can drop things the agent cannot immediately affect, including threats and reward-relevant objects. On Super Mario Bros., random features did as well as inverse-dynamics features (IDF) at training time. IDF transferred better to a new level with a new color scheme (1-1 to 1-3, day to night). Scale (2048 parallel envs) mattered more than the feature type for getting through 11 levels without reward.

### Cited Findings
**ICM: Pathak, Agrawal, Efros, Darrell, ICML 2017 (arXiv May 2017)**
- Design rationale: sources of observation change fall into three cases. (1) Things the agent controls. (2) Things it cannot control but that can affect it (e.g. a vehicle driven by another agent). (3) Things it neither controls nor is affected by (e.g. moving leaves). "A good feature space for curiosity should model (1) and (2) and be unaffected by (3)." — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Architecture and loss. An encoder phi feeds two heads. The inverse model g(phi(s_t), phi(s_t+1)) predicts a_t (loss L_I). The forward model f(phi(s_t), a_t) predicts phi(s_t+1) (loss L_F). The intrinsic reward is r^i_t = (eta/2)·||f(phi(s_t),a_t) − phi(s_t+1)||², with eta > 0 a scaling factor. Policy and ICM are trained jointly with the objective min −λ·E_π[Σ r_t] + (1−β)L_I + β·L_F, using β = 0.2 and λ = 0.1. The paper: "The inverse model learns a feature space that encodes information relevant for predicting the agent's actions only and the forward model makes predictions in this feature space." — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Training setup: A3C with 20 asynchronous workers and Adam. Inputs are the current frame stacked with the 3 previous frames. Mario was trained with action repeat, but the policy is sampled without action repeat at inference. — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Distractor test: a fixed region of white noise covering 40% of the image was added in VizDoom ("sparse" reward setup). ICM still "achieves a perfect score", while ICM-pixels (forward prediction in pixel space, no inverse model) "suffers significantly", even though it succeeded on the same task without noise. — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Super Mario Bros. with no extrinsic reward: the agent "can learn to cross over 30% of Level-1". It learned to kill and dodge enemies in order to reach new parts of the game. It could not pass "a pit at 38% of the game that requires a very specific sequence of 15-20 key presses". Falling in means death and no further signal. The authors call this "orthogonal to developing models of curiosity, but ... a challenging problem for policy learning". — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Mario transfer, Table 1 (mean distance ± stderr, curiosity only; iterations in parentheses):

  | Setting | Mean distance |
  |---|---|
  | Level-1, from scratch (1.5M) | 711 ± 59.3 |
  | Level-2, run as is (0) | 31.9 ± 4.2 |
  | Level-2, fine-tuned (1.5M) | 466 ± 37.9 |
  | Level-2, scratch (1.5M) | 399.7 ± 22.5 |
  | Level-2, scratch (3.5M) | 455.5 ± 33.4 |
  | Level-3, run as is (0) | 319.3 ± 9.7 |
  | Level-3, fine-tuned (1.5M) | 97.5 ± 17.4 |
  | Level-3, scratch (1.5M) | 11.8 ± 3.3 |
  | Level-3, scratch (5.0M) | 42.2 ± 6.4 |

  These values were re-assembled from the PDF column layout. They match the paper's prose: running "as is" fails on Level-2 (night), and fine-tuning on Level-3 is worse than running "as is". — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Failure mode, "curiosity blockade": fine-tuning on Level-3 degraded the policy. At a hard point the agent "receives almost no curiosity reward ... and the policy slowly degenerates", which the authors liken to boredom. — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)

**Burda, Edwards, Pathak, Storkey, Darrell, Efros, "Large-Scale Study of Curiosity-Driven Learning" (arXiv Aug 2018; ICLR 2019)**
- Four feature spaces were compared for forward-model prediction-error curiosity: raw pixels, random (frozen) conv features (RF), VAE features, and inverse-dynamics features (IDF). Table 1 rates each on three properties:

  | | VAE | IDF | RF | Pixels |
  |---|---|---|---|---|
  | Stable | No | No | Yes | Yes |
  | Compact | Yes | Yes | Maybe | No |
  | Sufficient | Yes | Maybe | Maybe | Yes |

  On IDF: "the features learned should correspond to aspects of the environment that are under the agent's immediate control ... A potential downside could be that the features learned may not be sufficient, that is they do not represent important aspects of the environment that the agent cannot immediately affect." — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Results:
  - Pixels "performs bad across all the environments".
  - RF "perform quite well across tasks and sometimes better than using learned features", attributed to stationarity because the features are frozen.
  - VAE "performed well but was somewhat unstable".
  - Across Atari, the IDF-curious agent beat a random agent in 75% of games and RF in 70%; IDF beat RF in 55%.

  — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Super Mario Bros.:
  - With "100x longer training" and PPO, the curious agent "is able to pass several levels".
  - Scaling from 128 to 2048 parallel environments (RF features), the agent explores "much more of the game: discovering 11 different levels of the game, finding secret rooms and defeating bosses", with no extrinsic reward.

  — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Mario generalization: pre-training with curiosity on 1-1, then fine-tuning with curiosity only.
  - 1-1 to 1-2 (both "day"): strong transfer for both RF and IDF.
  - 1-1 to 1-3 (day to night color shift): "IDF-learned features transfer in both the cases and random features ... do not transfer in the second scenario".
  - Conclusion: "while random features perform well on training environments, learned features appear to generalize better."

  — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Death handling: the 'done' signal was removed ("death is just another transition"), because done leaks reward information. Agents still avoided death, since resetting to the start is "very predictable" (boring). — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Noisy-TV: a TV with an agent action to change the channel was added to a Unity maze. "The presence of the TV drastically slows down learning" for both IDF and RF, though agents sometimes converge if run long enough. — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Implementation details:
  - Inputs: 84x84 grayscale, 4-frame stack, frameskip 4.
  - Features: a 512-d linear feature layer, with batchnorm on the embedding for Unity and Mario generalization.
  - Optimization: lr 1e-4; 128 parallel envs typical (2048 for large-scale Mario); rollout length 128; 3–6 PPO epochs.
  - The IDF and forward heads sit on the embedding network.
  - Pre-training IDF is hard "unless one has access to the internal state of the game"; random-policy data "will be biased to where the agent started".

  — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)

**Inverse-dynamics embeddings used inside later exploration bonuses**
- NGU (Badia et al., ICLR 2020) uses a Siamese inverse-dynamics embedding f, p(a|x_t,x_t+1) = h(f(x_t), f(x_t+1)), with h a one-hidden-layer MLP and softmax. It calls f's output the "controllable state". The motivation is explicit: in "a busy city with many pedestrians and vehicles" an agent could collect novelty "without taking any actions". An episodic memory of controllable states drives a kNN novelty reward. Reported a mean score of 8,400 on Pitfall! without demonstrations. — [Badia et al. 2020](https://arxiv.org/abs/2002.06038)
- E3B (Henaff et al., arXiv Oct 2022):
  - Episodic elliptical bonus b(s_t) = phi(s_t)^T C_{t-1}^{-1} phi(s_t), with C_{t-1} = Σ_{i<t} phi(s_i)phi(s_i)^T + λI, reset each episode.
  - phi is trained jointly with an inverse-dynamics head so it is "more robust to random noise or other parts of the state which are not controllable by the agent".
  - Base agent: IMPALA (TorchBeast).
  - SOTA on 16 MiniHack tasks; matches existing methods on pixel-based sparse VizDoom; outperforms on reward-free Habitat exploration.
  - Argues prior methods (RIDE, NovelD, etc.) depend critically on an episodic count term that fails in noisy, vast state spaces.

  — [Henaff et al. 2022](https://arxiv.org/abs/2210.05805)

### Inferences
- For NES games: a one-step inverse model will focus on the player sprite, and on screen scroll in side-scrollers, because scroll is caused by moving right. It will also capture the player's own bullets, since their spawn position and direction are a function of the fire and direction buttons. It will tend to drop enemies and enemy bullets, which are Pathak's case (2). Those are precisely what kills the agent in Contra-style games. IDF features are therefore a good novelty or count space for "where am I / what did I do", but not a sufficient state for the policy. Keep the policy's own encoder separate or reward-trained.
- Burda et al.'s finding that RF ≈ IDF in-distribution, with IDF better under a palette shift, is relevant: NES levels often change palette (day/night, underground, castle) while keeping sprites.
- In Mario, ICM's 38% pit and Burda's "11 levels at 2048 envs" suggest that policy optimization at scale is at least as important as the feature space.

### Gaps
- I did not extract per-game Atari numbers for RF vs IDF (only the 75%/70%/55% aggregates).
- No reported Mario experiment isolates enemy/bullet distractors specifically.

## 2. Contingency-aware exploration (Choi et al., ICLR 2019): how it finds the controllable location; results on hard Atari

### Takeaway
The Attentive Dynamics Model (ADM) is an inverse-dynamics action classifier that must route its decision through a spatial attention map over an H×W conv grid. The attention argmax becomes the agent's (x, y) location without any labels. Counting visits to (x, y, room-cluster, cumulative-reward) with a 1/sqrt(N) bonus gave Montezuma's Revenge 6,635 (A2C, 400M frames) and 11,618 (PPO, sticky actions, 2B frames).

### Cited Findings
- ADM model:
  - A conv map phi(s_t) ∈ R^{H×W×K}.
  - Per-cell action logits e_t(i,j) = MLP([phi_st(i,j) − phi_st−1(i,j); phi_st(i,j)]) ∈ R^|A|.
  - A separate MLP gives an attention mask α_t = sparsemax(MLP(phi_st(i,j))), summing to 1.
  - Prediction p(â_{t−1}|s_{t−1},s_t) = softmax(Σ_ij α_t(i,j)·e_t(i,j)).
  - "The attention probability α_t(i,j) should be high only on regions (i,j) that are predictive of the agent's actions." Sparsemax was chosen over softmax for "more stable performance".

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Loss: L_ADM = L_action + Σ_ij L^{i,j}_cell + λ_ent·L_ent.
  - L_cell is a separate cross-entropy on each cell's logits, so cells learn even when attention is wrong early on.
  - L_ent = −H(α_t) pushes attention toward uniform "whenever action prediction is not possible".

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Training issues named by the authors (the ADM is trained online, jointly with the policy):
  - Actions can be unpredictable from frame pairs, e.g. inputs during a jump in Montezuma have no effect.
  - On-policy data is non-i.i.d. and shifts over time.
  - A low-entropy policy biases the action labels. On Freeway the early agent takes mostly UP, so "the action classifier would give correct answer regardless of attention".

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Exploration bonus: (x, y) = argmax α_t, with bonus r+ = 1/sqrt(#ψ(s)) and total reward β1·r_ext + β2·r+. The full state abstraction is ψ(s) = (x, y, c, R):
  - c is a discrete "high-level visual context" (e.g. room) from a simple clustering of random projections of frames.
  - R is the cumulative extrinsic reward so far in the episode, used as a behavioral context (e.g. key picked up or not).

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Localization quality: the predicted location is "on average 1 or 2 grid cells away from the ground-truth location". Error spikes when a novel scene appears (e.g. Montezuma's second room around 10M steps) and then recovers. RAM location was used only for evaluation. — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Table 1: A2C with 16 actors, 100M env steps (400M frames), max mean score, 3 seeds.

  | Method | Freeway | Frostbite | Hero | Montezuma | PrivateEye | Qbert | Seaquest | Venture |
  |---|---|---|---|---|---|---|---|---|
  | A2C | 7.2 | 1099 | 34352 | 13 | 574 | 19620 | 2401 | 0 |
  | A2C+Pixel-SimHash | 0.0 | 829 | 28181 | 412 | 276 | 18180 | 2177 | 31 |
  | A2C+CoEX | 34.0 | 4260 | 36827 | 6635 | 5316 | 23962 | 5169 | 204 |
  | A2C+CoEX+RAM (ground-truth location and room from RAM, "roughly an upper-bound") | 34.0 | 4418 | 36765 | 6600 | 24296 | 24422 | 6113 | 1100 |

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Table 3: PPO, 128 actors, sticky actions p=0.25, 500M steps (2B frames).

  | Method | Freeway | Frostbite | Hero | Montezuma | PrivateEye | Qbert | Seaquest | Venture |
  |---|---|---|---|---|---|---|---|---|
  | PPO | 34.0 | 7340 | 36263 | 29 | 942 | 19980 | 2806 | 1875 |
  | PPO+CoEX | 34.0 | 9076 | 36664 | 11618 | 11000 | 22647 | 11794 | 1916 |

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- Limitations stated by the authors:
  - The method "mainly focuses on controllable dynamics and employs a simple clustering scheme to abstract away uncontrollable elements"; modelling the uncontrollable dynamics is left to future work.
  - "RL environments with different visual characteristics may require different forms of attention".
  - Applicability to 3D is future work.

  — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)

### Inferences
- The PrivateEye gap (5,316 learned vs 24,296 with RAM location) shows how much count-based exploration depends on an accurate self-location. A NES lab with RAM access can use RAM player-x/scroll as an upper-bound oracle, as CoEX+RAM did, to measure how far a pixel-learned locator lags.
- In side-scrollers the ADM's (x, y) is a *screen* coordinate. Mario stays near screen center while the world scrolls, so the screen position alone is a poor count key. A scroll or world-x term (CoEX's "context" analogue) would be needed.
- A single sparsemax attention peak assumes one controllable entity. Games where the player controls a sprite plus its bullets (Contra, Kage, S.C.A.T. with satellite option pods) may spread attention. The paper reports no multi-controllable-object experiments.

### Gaps
- No exact λ_ent, β1/β2 or grid-size values were extracted (they are in the appendix).
- No ADM results on scrolling platformers or shooters.

## 3. Agent-controller / controllable latent-state methods (AC-State, Ex-BMDP/PPE, ACRO, ACDF, Denoised MDPs, TIA, Iso-Dream/Iso-Dream++, later work to 2026)

### Takeaway
Two families exist.

**(a) Inverse-model / Ex-BMDP family.** Members: PPE 2021, AC-State 2022, ACRO 2022, ACDF 2024, action-bisimulation 2024, IBD 2026.
- How it separates: predicts actions over multiple steps, usually with a bottleneck. It provably discards exogenous noise under the assumption that the noise evolves independently of actions *and* never interacts with the agent's state.
- Where tested: mostly on gridworlds, mazes, a real robot arm and offline datasets, not Atari.

**(b) World-model family.** Members: TIA 2021, Denoised MDPs 2022, Iso-Dream 2022, Iso-Dream++ 2023.
- How it separates: splits a Dreamer-style latent into action-conditioned vs action-free (and/or reward-relevant vs irrelevant) branches, trained online with the policy.
- Where tested: DMC with video backgrounds, CARLA, RoboDesk and a few Atari games.

Only Iso-Dream++ explicitly models non-controllable entities that *react* to the agent ("sparse dependency").

### Cited Findings
**Ex-BMDP and PPE: Efroni, Misra, Krishnamurthy, Agarwal, Langford (arXiv Oct 2021)**
- Introduces the Exogenous Block MDP to formalize "temporally correlated noise in the observations". Shows negative results for prior methods:
  - HOMER-style contrastive methods "under-abstract": they split identical agent states by exogenous details such as the position of ducks.
  - Inverse dynamics "can ignore all information that is not relevant for action prediction, which includes all exogenous/uncontrollable information. However, it can also ignore controllable information. This may result in a failure to sufficiently explore."

  — [Efroni et al. 2021](https://arxiv.org/abs/2110.08847)
- PPE (Predictive Path Elimination) "learns a generalization of inverse dynamics". It is provably sample- and compute-efficient when endogenous dynamics are near-deterministic, with sample complexity polynomial in the endogenous latent-state count and independent of the observation and exogenous state sizes. Experiments use a combination-lock exploration problem with exogenous noise. — [Efroni et al. 2021](https://arxiv.org/abs/2110.08847)

**AC-State: Lamb, Islam, Efroni, Didolkar, Misra, Foster, Molu, Chari, Krishnamurthy, Langford (arXiv Jul 2022)**
- Objective: L = −log P(a_t | f(x_t), f(x_t+k); k).
  - It predicts the *first* action of a k-step segment from encodings of the start and end observations.
  - t is sampled over steps where the action was random, and k is uniform in [1, K_t].
  - Bottleneck: a Gaussian variational information bottleneck followed by vector quantization (discrete latent). "In the absence of a bottleneck, we could set the encoder to be the identity function ... and fail".

  — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)
- Assumption: the deterministic Ex-BMDP (Efroni et al.) "assumes that exogenous noise has no interaction with the control-endogenous state, which is stricter than the exogenous MDP definition introduced by Dietterich et al. (2018)". Endogenous dynamics are assumed deterministic with bounded diameter. — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)
- One-step inverse counterexample (Appendix B):
  - Take a 6-state cycle where a=0 moves back and a=1 moves forward.
  - Merging {s1,s4}, {s2,s5}, {s3,s6} still gives zero one-step inverse error, because every merged transition remains distinct.
  - "The 1-step inverse model is underconstrained and thus may incorrectly merge distinct states which are far apart in the MDP but have a similar local structure."
  - On the real robot, the one-step inverse model merged the arm's center-left, center and center-right positions, "precisely what the theory predicts".

  — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)
- Experiments:
  - Real robot arm: 6 hours / 14,000 samples, 4 actions, uniformly random policy. Distractors were a 4K looped video at 2x speed on a TV, drinking-bird toys, a color-changing lamp, flashing lights and people. Decoding the latent shows the arm position while the TV and lights "appeared completely blank". AC-State gives a "perfect" (diagonal) co-occurrence histogram vs autoencoder and one-step inverse failures.
  - 9-maze multi-agent world: with 3,000 samples, AC-State nearly perfectly recovers the controlled agent's state and ignores 8 other randomly acting agents (observation space c^9).
  - Maze with reset actions: a random policy reaches under 25% of the state space. AC-State plus a count-based planner reaches full coverage with 25,000 samples. The planner picks goals ∝ 1/count, uses Monte-Carlo Dijkstra over the discrete latent graph, and takes one random first action per goal-seeking trajectory as the training label.
  - Also evaluated on Matterport house navigation.

  — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)

**ACRO: Islam, Tomar, Lamb, Efroni, Zang, et al. (arXiv Nov 2022)**
- Offline RL version: the latent is z = (s, e) with T(z'|z,a) = T(s'|s,a)·T_e(e'|e). Uses multi-step inverse models to learn "Agent-Centric Representations"; distractors are ignored "because they are not predictive of the agent's actions". Releases offline visual benchmarks with per-episode background videos and other temporally correlated exogenous noise. Reports outperforming baselines across expert, medium-expert and medium datasets, and proves the representation is Bellman complete (Prop. 2.2). — [Islam et al. 2022](https://arxiv.org/abs/2211.00164)

**ACDF: Levine & Stone, "Multistep Inverse Is Not All You Need" (arXiv Mar 2024)**
- Identifies two flaws in AC-State:
  - The required horizon k "can be much larger than claimed".
  - "If the dynamics are periodic, then multistep-inverse dynamics are insufficient ... regardless of k".
- Fix (ACDF): add a latent *forward* dynamics loss so the learned endogenous states must be consistent with deterministic dynamics. Proven correct for a broad class of Ex-BMDPs. Shown in tabular and neural-encoder experiments.

— [Levine & Stone 2024](https://arxiv.org/abs/2403.11940)

**Rakelly, Gupta, Florensa, Levine, "Which Mutual-Information Representation Learning Objectives are Sufficient for Control?" (arXiv Jun 2021)**
- Inverse information J_inv = I(A_t; Z_t+k | Z_t) and state-only J_state = I(Z_t+k; Z_t) are *not* sufficient in general. Forward information is sufficient.
- In the pygame game "catcher" (paddle catches falling fruit), the representation learned via J_inv "fails to capture the fruit object accurately", and RL on top of it performs poorly.
- Protocol: 50k random-policy transitions, frozen encoder, then RL.

— [Rakelly et al. 2021](https://arxiv.org/abs/2106.07278)

**TIA: Fu, Yang, Agrawal, Jaakkola, "Learning Task Informed Abstractions" (arXiv Jun 2021)**
- Mechanism: two Dreamer-style models, a task model and a distractor model, cooperatively reconstruct the image. The distractor model is adversarially dissociated from reward.
- Critique of inverse models: when an arm pushes an object, "it is easy to capture the part that is directly controllable (e.g., the arm), capturing all controllable features (i.e., arm and the object) without imposing a reconstruction loss is non-trivial."
- Atari: evaluated on 6 games with natural distractors (e.g. Robotank's kill counter and rotating radar). TIA "significantly improves policy learning in Demon Attack, Robotank, and Yars Revenge" over DreamerV2 at 100M steps.
- Failure modes:
  - The distractor model can take over reconstruction and strip task information.
  - The dissociation "relies on informative reward signals"; with sparse, noisy early rewards the adversarial term is ineffective.
  - Sensitive to λ_Radv / λ_Os, with one set working across Atari but DMC needing tuning.

— [Fu et al. 2021](https://arxiv.org/abs/2106.15612)

**Denoised MDPs: Wang, Du, Torralba, Isola, Zhang, Tian (arXiv Jun 2022)**
- Taxonomy: information has four types (controllable/uncontrollable × reward-relevant/irrelevant). Useful information is "both controllable and reward-relevant".
- Key nuance: "uncontrollable" here means factors that "evolve independently of actions *and do not affect controllable x*". Among those, ones that "only additively affect reward" can be discarded. Uncontrollable things that affect the agent's state are not discarded ("wind directions should not be ignored while sailing").
- Implemented as a factorized Dreamer-style world model, with policy learning either by backprop through dynamics or by SAC in latent space.

— [Wang et al. 2022](https://arxiv.org/abs/2206.15477)
- DMC Table 1 (mean return averaged over Cheetah Run, Walker Walk and Reacher Easy):

  | Variant | Denoised (backprop) | TIA (backprop) | Dreamer (backprop) | Denoised (SAC-latent) | DBC | PI-SAC (no aug) | CURL | State SAC (upper bound) |
  |---|---|---|---|---|---|---|---|---|
  | Noiseless | 801.4 | 769.7 | 848.6 | 587.1 | 297.4 | 246.4 | 417.3 | 910.3 |
  | Video background | 597.7 | 407.1 | 227.8 | 309.8 | 188.0 | 131.7 | 478.0 | 910.3 |
  | Video + noisy sensor | 563.1 | 261.2 | 212.4 | 288.2 | 79.9 | 152.5 | 354.3 | 919.8 |
  | Video + camera jitter | 254.1 | 151.7 | 98.6 | 186.8 | 68.0 | 91.6 | 390.4 | 910.3 |

  CURL wins on jitter, which the authors attribute to its random-crop augmentation. — [Wang et al. 2022](https://arxiv.org/abs/2206.15477)

**Iso-Dream: Pan, He, Wang, Yang (NeurIPS 2022, arXiv May 2022)**
- Mechanism: a Dreamer-based world model with two isolated transition branches.
  - The action-conditioned branch has an "Inverse Cell" MLP(s_t−1, s_t) → ã_t−1 regressed to the true action, so it captures action-caused change.
  - The action-free branch models noncontrollable dynamics.
  - Decoders produce per-branch image components combined with learned masks.
- Policy: rolls the noncontrollable states into the future and attends over them with the current controllable state, e.g. anticipating other cars in CARLA. The replay buffer is initialized with random episodes, then trained online.

— [Pan et al. 2022](https://arxiv.org/abs/2205.13817)
- DMC video_easy results (3 seeds; columns re-assembled from PDF layout):

  | Method | walker walk | cheetah run | finger spin | hopper stand |
  |---|---|---|---|---|
  | Iso-Dream | 911±50 | 659±62 | 800±59 | 746±312 |
  | DreamerV2 | 655±47 | 475±159 | 755±92 | 260±366 |

  Also reported: outperforms DreamerV2 "by a large margin" on CARLA, and removing the Inverse Cell degrades performance. Stated limitation: "longer training time per episode" than DreamerV2. — [Pan et al. 2022](https://arxiv.org/abs/2205.13817)

**Iso-Dream++ (arXiv Mar 2023, TPAMI-format preprint)**
- "Training collapse": in the original Iso-Dream, "most useful information may collapse into the action-conditioned branch while the action-free branch learns almost nothing", and disentanglement was unstable across runs. Fix: "min-max variance constraints". The action-conditioned branch's predicted states should vary across different candidate actions; the action-free branch's outputs should not.
- Adds a "sparse dependency" of next noncontrollable state on the current controllable state, e.g. an air-hockey puck changes direction when hit, and other vehicles change direction when the ego car cuts in.
- Evaluated on CARLA and DMC, including transfer setups.

— [Pan et al. 2023](https://arxiv.org/abs/2303.14889)

**Action-bisimulation: Rudolph, Chuck, Black, et al. (RLC 2024)**
- "Myopic controllability can capture the moment right before an agent crashes into a wall, but not the control-relevance of the wall while the agent is still some distance away."
- Mechanism: extends single-step inverse-dynamics controllability with a recursive bisimulation-style invariance, giving a discounted multi-step controllability metric.
- Pre-trained on reward-free, uniformly random data. Improves sample efficiency in several environments including Habitat.

— [Rudolph et al. 2024](https://arxiv.org/abs/2403.16369)

**Exploration-bonus variants that explicitly discount non-agent novelty**
- DEIR (Wan et al., arXiv Apr 2023): "there is a gap between the novelty of an observation and an exploration, as both the stochasticity in the environment and the agent's behavior may affect the observation". The intrinsic reward contains a conditional mutual-information term that "principally scales with the novelty contributed by agent explorations". It is implemented with a discriminative forward model that learns dynamics and discriminates genuine from fake trajectories. Evaluated on MiniGrid, including noisy-observation variants, and ProcGen. — [Wan et al. 2023](https://arxiv.org/abs/2304.10770)
- BYOL-Hindsight (Jarrett et al., arXiv Nov 2022, v2 Jul 2023): adds hindsight representations to BYOL-Explore's model to "disentangle 'noise' from 'novelty'". Robust to noisy-TV-like stochasticity, including "noise actively induced by the agent". Reports "state-of-the-art results in exploring Montezuma's Revenge with sticky actions". — [Jarrett et al. 2022](https://arxiv.org/abs/2211.10515)
- InfoGating (Tomar et al., arXiv Mar 2023): differentiable signal-to-noise gating that erases input pixels or activations so as to keep minimal task information. Applied to multi-step forward and inverse dynamics models, Q-learning and BC, "to discard information not relevant for control". — [Tomar et al. 2023](https://arxiv.org/abs/2303.06121)

**Video and latent-action work (controllability learned from unlabeled frames)**
- LAPO (Schmidt & Jiang, ICLR 2024): latent IDM plus forward dynamics model (FDM) with a vector-quantized latent action, trained on observation-only data (~8M frames per Procgen game from a PPO expert). Decoding the latent policy online with PPO for 4M frames "fully recover[s] expert performance", vs PPO from scratch reaching 44% of expert. The paper states a 9-of-16-task exceedance (comparison target truncated in my extraction). — [Schmidt & Jiang 2024](https://arxiv.org/abs/2312.10812)
- Genie (Bruce et al., arXiv Feb 2024): 11B-parameter world model trained on 30,000 hours of 2D-platformer gameplay video (filtered from 200,000+ hours). Its Latent Action Model encoder sees x_1:t and x_t+1, outputs latent actions, and a decoder must predict x_t+1 from past frames plus latent actions. The VQ codebook has |A| = 8, which forces latents to capture the player-controlled change. Generated worlds show learned parallax (foreground moves more than background). The LAM is discarded at inference. — [Bruce et al. 2024](https://arxiv.org/abs/2402.15391)
- LAOM (Nikulin et al., arXiv Feb 2025): on the Distracting Control Suite, LAPO "struggle[s]" with "action-correlated distractors". LAOM improves latent-action quality 8x (linear probe). Adding ground-truth action labels for as little as 2.5% of the data during latent-action training improves downstream performance by 4.2x on average (4.3x in the body text). The authors conclude that the standard "learn LAM first, decode later" pipeline is suboptimal with distractors. — [Nikulin et al. 2025](https://arxiv.org/abs/2502.00379)
- Misra, Saran, Xie, Lamb, Langford (arXiv Mar 2024): with exogenous (temporally correlated) noise, they prove a lower bound that learning representations from *video without actions* can be exponentially worse in sample complexity than learning from action-labeled trajectories. Covers autoencoding, temporal contrastive learning and forward modeling. — [Misra et al. 2024](https://arxiv.org/abs/2403.13765)

**Benchmarks and 2026**
- DMC-VB (Ortiz et al., arXiv Sep 2024): on an offline visual-distractor benchmark, "pretrained representations do not help policy learning", and inverse dynamics (ID) pretraining specifically "do[es] not show benefits". A large pixel-vs-state gap remains. Pretraining helps when expert data is limited and suboptimal data is available, or with stochastic hidden goals. — [Ortiz et al. 2024](https://arxiv.org/abs/2409.18330)
- IBD, "Discovering What You Can Control: Interventional Boundary Discovery" (Liu, Cheng, Bogdan, arXiv Mar 2026, v3 Sep 2026): uses the agent's action channel as randomized interventions. Per-observation-dimension two-sample tests with FDR correction give a binary controllable mask. Across 12 *state-based* continuous-control settings with up to 100 distractors it matches oracle return in 11 of 12. Observational selectors (MI, forward models, gradient sensitivity) "often underperform simply passing the full observation to SAC", and "can collapse when distractors mimic controllable state variables". — [Liu et al. 2026](https://arxiv.org/abs/2603.18257)

### Inferences
- The Ex-BMDP guarantee does not hold for NES action games. Enemies home in on the player, die when shot, and bullets kill the player, so the "exogenous" process interacts with the agent's state. Multi-step inverse methods have no guarantee there. In practice they will capture an enemy only insofar as its future state is predictive of the agent's earlier action.
- Iso-Dream++'s "sparse dependency" and Denoised MDP's rule (keep uncontrollable things that affect the controllable state) are the only frameworks found that match the NES setting conceptually. Neither was tested on NES or shooter-style Atari.
- IBD (2026) is state-based, but its per-dimension interventional test maps directly onto NES RAM bytes. Randomize actions from a saved state and test which RAM addresses' distributions shift. This would give a label-free controllable/uncontrollable split of RAM for count keys or reward shaping. Not tested in the paper.
- LAOM and Misra et al. both indicate that the true controller actions (always available in an emulator) should be used. Video-only or latent-action approaches are strictly harder with distractors.

### Gaps
- Choreographer (Mazzaglia et al.), Controllability-aware Skill Discovery (CSD, Park et al. 2023) and IFactor (Liu et al. 2023) were not verified; I could not fetch them within budget.
- AC-State's Table 5 numeric comparison, Iso-Dream's CARLA numbers and BYOL-Hindsight's exact Montezuma score were not extracted.
- None of the multi-step inverse papers report Atari, Procgen or NES results.

## 4. Do these capture indirect effects (killing an enemy, enemies reacting to the agent, a pipe that changes the whole screen)? One-step vs multi-step inverse models

### Takeaway
One-step inverse models provably can miss controllable or consequential information:
- They alias distant states with similar local structure (AC-State's cycle counterexample, the robot-arm merge).
- They miss objects affected only indirectly (TIA's pushed object).
- They miss reward-relevant uncontrollable objects (Rakelly's falling fruit).

Multi-step inverse models fix the aliasing within the Ex-BMDP, but still fail for periodic dynamics and can need large k (ACDF), and they assume distractors never interact with the agent. Reactive entities are addressed only by Iso-Dream++ (sparse dependency) and by forward-model-based objectives such as Denoised MDP, ACDF and DEIR. A full-screen transition caused by an action (e.g. a pipe) is, by construction, action-predictive at the step where it happens, but no paper tests this case.

### Cited Findings
- Pathak's taxonomy explicitly wants case (2), things that affect the agent, modeled. Burda et al. note IDF may "not represent important aspects of the environment that the agent cannot immediately affect." — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363); [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- Inverse dynamics "can also ignore controllable information. This may result in a failure to sufficiently explore". — [Efroni et al. 2021](https://arxiv.org/abs/2110.08847)
- Arm-pushes-object: inverse models capture the arm easily; capturing the object "without imposing a reconstruction loss is non-trivial". — [Fu et al. 2021](https://arxiv.org/abs/2106.15612)
- J_inv is insufficient; in catcher it "fails to capture the fruit object accurately". The fruit is uncontrollable but reward-relevant, analogous to incoming bullets. — [Rakelly et al. 2021](https://arxiv.org/abs/2106.07278)
- One-step inverse merges distant states with similar local structure (6-cycle proof, robot-arm experiment). Multi-step inverse with a bottleneck recovers the full control-endogenous state under Ex-BMDP. — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)
- Multi-step inverse can still fail: the required k can be "much larger than claimed", and with periodic dynamics it fails "regardless of k". A latent forward loss fixes both. — [Levine & Stone 2024](https://arxiv.org/abs/2403.11940)
- Action-bisimulation: single-step controllability is "myopic". It sees a wall only right before collision. Its recursive metric discounts distant control-relevant features instead. — [Rudolph et al. 2024](https://arxiv.org/abs/2403.16369)
- Ex-BMDP "assumes that exogenous noise has no interaction with the control-endogenous state". — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)
- Iso-Dream++ models "sparse dependency" of noncontrollable states on the controllable state, at the moment of contact or when other agents react. — [Pan et al. 2023](https://arxiv.org/abs/2303.14889)
- Denoised MDP only discards uncontrollable factors that "do not affect controllable x" and only additively affect reward. — [Wang et al. 2022](https://arxiv.org/abs/2206.15477)
- Empirical point in favor of ICM on Mario: the curiosity agent "discovered ... how to kill and dodge enemies so that it can reach new parts of the game space". This shows enemy-related behavior can emerge even if the feature space under-represents enemies, because dying ends novelty. — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- The ADM fails to attribute control when actions have no visible effect (mid-jump in Montezuma). The ADM's attention entropy term is meant to flatten attention in those cases. — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)

### Inferences
- **Killing an enemy.** The enemy's disappearance follows a fire action within a few frames. A multi-step inverse (k of a few to tens of frames) can pick it up when the kill is reliably predictive of the earlier action. A one-step model will catch it only if the hit lands within the frame-skip window.
- **Pipe or door transitions.** These replace the whole screen after a multi-frame animation, often with inputs ignored. That makes them one-step-unpredictable, the "mid-jump" problem. Multi-step inverse models and forward-model objectives are better suited.
- **Reactive enemies and enemy bullets.** These violate the Ex-BMDP assumption, so for the policy's state, objectives with a forward or reward component (Denoised MDP, Iso-Dream++, ACDF's forward term) or plain reward-trained features are needed. Inverse-only features are best kept for the novelty or count key.

### Gaps
- No paper found empirically tests one-step vs multi-step inverse models on enemy kills, reactive enemies or screen transitions in Atari or NES. All one-step vs multi-step evidence is from theory, gridworlds and mazes, or a robot arm.

## 5. Joint (single-loop, end-to-end with RL) vs separately pre-trained; compute and implementation cost

### Takeaway
The exploration-bonus methods (ICM, Burda IDF, CoEX ADM, NGU, E3B, DEIR) train their controllability model online, in the same loop as the policy. They are cheap: a small head on a shared or sibling conv encoder. The theory-driven multi-step inverse methods (AC-State, ACRO, PPE, action-bisim, Rakelly's protocol) are pre-trained from random or offline data, or use their own count-plus-planning explorer rather than deep RL. The world-model methods (TIA, Denoised MDP, Iso-Dream) run online in a Dreamer loop, at higher cost. Latent-action methods (LAPO, Genie, LAOM) pre-train on video and decode later.

### Cited Findings
- ICM is joint and end-to-end: min −λE[Σr] + (1−β)L_I + βL_F, with A3C and 20 workers. — [Pathak et al. 2017](https://arxiv.org/abs/1705.05363)
- Burda: IDF trained online. "Learned features are not stable because their distribution changes as learning progresses". Pre-training needs representative data, which random play does not provide. Configuration: 128 envs (2048 for large-scale Mario), 512-d features, lr 1e-4. — [Burda et al. 2018](https://arxiv.org/abs/1808.04355)
- CoEX: "we train the ADM online along with the policy". A2C used 16 actors and 100M steps; PPO used 128 actors and 500M steps (2B frames). — [Choi et al. 2019](https://arxiv.org/abs/1811.01483)
- NGU: the inverse-dynamics embedding is trained by maximum likelihood within the agent. — [Badia et al. 2020](https://arxiv.org/abs/2002.06038)
- E3B: the inverse-dynamics head g is "trained jointly with phi". IMPALA (TorchBeast). The bonus requires maintaining the inverse covariance matrix each step. — [Henaff et al. 2022](https://arxiv.org/abs/2210.05805)
- AC-State: data from a uniformly random policy, or from its own count-based Dijkstra planner over the discrete latents. Sample counts: 14,000 samples for the robot, 3,000 for the 9-agent maze, 25,000 for the reset maze. Policies come from planning in the learned tabular latent MDP, not deep RL. A random policy fails "if a single bad action can cause an agent to lose many steps". — [Lamb et al. 2022](https://arxiv.org/abs/2207.08229)
- ACRO: representation pre-trained on offline datasets, then offline RL. — [Islam et al. 2022](https://arxiv.org/abs/2211.00164)
- Rakelly: 50k random transitions, encoder frozen before RL. — [Rakelly et al. 2021](https://arxiv.org/abs/2106.07278)
- Action-bisim: pre-trained on reward-free uniformly random data. — [Rudolph et al. 2024](https://arxiv.org/abs/2403.16369)
- Iso-Dream: Dreamer-style online loop with a replay buffer seeded with random episodes, and "longer training time per episode" than DreamerV2. Iso-Dream++ notes unstable disentanglement across runs ("training collapse") without its variance constraints. — [Pan et al. 2022](https://arxiv.org/abs/2205.13817); [Pan et al. 2023](https://arxiv.org/abs/2303.14889)
- TIA: online Dreamer-based. Two hyperparameters (λ_Radv, λ_Os) must be balanced, and the adversarial term is weak while rewards are sparse early in training. — [Fu et al. 2021](https://arxiv.org/abs/2106.15612)
- LAPO: pre-trained on ~8M frames per game, then 4M PPO frames to decode. Genie: 11B parameters on 30k hours of video. LAOM: supervising 2.5% of the data with true actions *during* latent-action training beats pre-training then decoding. — [Schmidt & Jiang 2024](https://arxiv.org/abs/2312.10812); [Bruce et al. 2024](https://arxiv.org/abs/2402.15391); [Nikulin et al. 2025](https://arxiv.org/abs/2502.00379)

### Inferences
- For a high-throughput PPO setup (thousands of envs), an online inverse-dynamics head (ICM/E3B/NGU-style), or a CoEX-style attention locator, adds one small network and one cross-entropy loss per batch. That is the cheapest fit with the existing loop.
- AC-State-style multi-step inverse can be added the same way: sample (t, t+k) pairs from rollout buffers and predict a_t. The theory assumes the labeled action was random, i.e. not chosen from x_t+k-correlated information. On-policy PPO actions violate that, while ε-random action injections would satisfy it. Not tested in the cited papers.

### Gaps
- Wall-clock or GPU-hour figures were not reported in the extracted text for most methods (ICM, CoEX, AC-State, Denoised MDP). Only relative statements (Iso-Dream slower than DreamerV2) and env-step budgets are available.
- No method in this survey reports results on Contra, S.C.A.T./Final Mission, Adventure Island, Kage or Batman. Super Mario Bros. (ICM, Burda) is the only NES title with published controllability-feature results.
