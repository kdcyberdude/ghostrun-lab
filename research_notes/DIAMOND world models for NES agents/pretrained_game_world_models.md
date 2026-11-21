# Pre-trained and general-purpose game world models (as of 2026-11-19), with focus on Super Mario Bros / NES

Scope note: "world model" here means an action-conditioned next-frame (or next-latent) predictor. All items dated. "Open weights" was checked against the Hugging Face (HF) API and GitHub API on 2026-11-19 where possible (file lists, licence tags, HTTP status). Context for the reader: the lab's agent runs 12,288 NES consoles on one RTX 5090 at ~13,900 decisions/s (1 decision = 4 frames, 240x224 screen), i.e. roughly 55,600 emulated frames/s.

## 1. Mario / NES-specific world models and datasets: what exists, what they predict, quality, weights

### Takeaway
There is no released, good-quality, action-conditioned world model for real NES Super Mario Bros that covers the full controller and full-resolution screen. What exists is small hobby/academic work: MarioVGG (paper only, no weights, 64x48, 2 text actions), PlayGen's Mario model (open checkpoint, but trained on the Java "Mario AI Framework" clone at 128x128), a community DIAMOND port (64x60 + upsampler, ~4M-param denoiser), and a JEPA latent model for level 1-1 only. Useful NES datasets do exist (rafaelcp smbdataset with all 256 button combinations and RAM; an 8,000-episode 1-1 dataset; a 681k-frame Mario AI Framework corpus).

### Cited Findings

**MarioVGG (Virtuals Protocol, Sept 2024)**
- Paper "Video2Game Generation: A Practical Study using Mario" by Virtuals Protocol; a text-to-video diffusion (U-Net) model that takes one initial frame plus an action text prompt and generates a short clip; chained by feeding the last generated frame back in — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf)
- Data: the public rafaelcp SMB dataset, 280 episodes (141 wins, 139 losses), 256x240 frames with per-frame action; World 1-1 held out for evaluation, all other levels for training; 8,618 "Run" + 4,634 "Jump" training sequences — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf)
- Action space cut to only two actions, "run right" and "run right and jump" (47% and 19% of recorded actions), encoded with a CLIP text encoder — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf)
- Each 35-frame sequence subsampled to 7 frames; frames downsized to 64x48 (one experiment at 128x96) — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf)
- Training: 300,000 steps, batch 8, single RTX 4090, ~48 hours; inference ~4 s to generate a 6-frame clip with 100 sampling steps on the same 4090 — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf). Press coverage says "6 seconds to generate 6 frames" and notes Mario sometimes disappears or turns into enemies — [GIGAZINE](https://wbgsv0a.gigazine.net/gsc_news/en/20240906-mariovgg-super-mario-bros-generative-ai); [Slashdot/Ars summary](https://games.slashdot.org/story/24/09/05/1958212/new-ai-model-learns-how-to-simulate-super-mario-bros-from-video-footage) (conflicts with the paper's ~4 s; the paper is the primary source).
- Weights: the HF repo `virtuals-protocol/mario-videogamegen` (created 2024-09-04, last modified 2024-09-06) contains only `VideoGameGen.pdf`, an overview PNG, README and .gitattributes; no checkpoint files and no licence tag (HF API, checked 2026-11-19) — [HF repo](https://huggingface.co/virtuals-protocol/mario-videogamegen)

**PlayGen / "Playable Game Generation" (Tencent, 1 Dec 2024) — Super Mario + DOOM**
- Authors Mingyu Yang et al. (Tencent), arXiv 2412.00887, submitted 2024-12-01 — [arXiv](https://arxiv.org/html/2412.00887v1)
- Mario data: 236M frames of transitions collected on the Mario-AI-Framework (a Java re-implementation, not the NES ROM), then 50M balanced transitions sampled for training; mix of random and expert/RL agents with random spawn positions and power-up states; 200-step samples — [arXiv](https://arxiv.org/html/2412.00887v1)
- 128x128 resolution; actions shown as left/right/jump combinations (paper does not give the action-space size) — [arXiv](https://arxiv.org/html/2412.00887v1)
- Architecture: VAE (55M) + latent DiT diffusion model (131M; depth 12, hidden 384) with an RNN-like hidden state; diffusion forcing — [arXiv](https://arxiv.org/html/2412.00887v1)
- Speed: 20 FPS with 4 DDIM steps on an RTX 2060 (10 FPS at 8 steps, 5 FPS at 16) — [arXiv](https://arxiv.org/html/2412.00887v1)
- Quality on Mario vs prediction length 1/16/64/1024 frames: PSNR 33.81 / 28.06 / 24.56 / 18.19; LPIPS 0.022 / 0.043 / 0.083 / 0.222; action accuracy ~0.80 → 0.789 at 1024 frames — [arXiv](https://arxiv.org/html/2412.00887v1)
- Release: GitHub `GreatX3/Playable-Game-Generation`, MIT licence, created 2024-11-27, last push 2025-01-07; Mario checkpoint downloaded from a Google Drive link; DOOM weights and the training module/dataset are still listed as TODO — [GitHub](https://github.com/GreatX3/Playable-Game-Generation); GitHub API check 2026-11-19.

**Super-Mario-DIAMOND (community, Nov–Dec 2024)**
- Applies DIAMOND to the gym-super-mario-bros (NES) environment; denoiser works at 64x60 (~4M params), separate upsampler (~50M trainable params) outputs 256x240 — [GitHub](https://github.com/AlexBodner/Super-Mario-DIAMOND)
- Data: human gameplay (Google Drive) plus synthetic data from a pretrained PPO agent with random-factor and NOOP-balancing scripts; frame counts not stated — [GitHub](https://github.com/AlexBodner/Super-Mario-DIAMOND)
- Training: 50 epochs, "roughly 24 hours" on one NVIDIA L4 (authors note DIAMOND itself trains 1000 epochs on twice the data); no quantitative quality metrics; install steps reference checkpoints but give no link — [GitHub](https://github.com/AlexBodner/Super-Mario-DIAMOND)
- MIT licence, created 2024-11-26, last push 2024-12-24, 5 stars (GitHub API, 2026-11-19).

**AQ-Mario (JEPA latent world model, Sept 2026)**
- HF model `maxmill/aq-mario` (mirror `Aquinlabs/aq-mario`), created 2026-09-19 / 2026-09-23: EMA-JEPA action-conditioned latent world model for SMB level 1-1; predicts future latent states, not RGB frames; 9,801,795 trainable params; 41,160 steps / 3 epochs; 140 MB checkpoint; author says it is not a solved Mario controller (x/y/scroll probes and action gate fail); code MIT, checkpoint "subject to rights in underlying game content" — [HF model](https://huggingface.co/maxmill/aq-mario); HF API 2026-11-19.
- Dataset `maxmill/aq-mario-smb1`: 8,000 level 1-1 episodes (4,000 random, 4,000 PPO), 4,877,368 observations, 512 .npz shards, ~27.4 GB; no licence listed — [HF dataset](https://huggingface.co/datasets/maxmill/aq-mario-smb1)
- A related LeMario JEPA project reports Mario could not reliably jump the first major obstacle in planning — [LeMario](https://www.benjamin-bai.com/projects/lemario) (via search snippet; not opened).

**Other Mario / Nintendo items**
- `soheilchavo/mario-world` (2026): GameNGen-style SMB 1-1 project; plans ~600 episodes / ~200k frames from a PPO agent with perturbations; VAE and diffusion model still listed as open tasks; no weights, no results — [GitHub](https://github.com/soheilchavo/mario-world)
- `SyntheticMDProductions/Super_Mario_64_Oasis_Model` (HF, 2026-08-11): community N64 (not NES) action-conditioned next-frame model, 256x144, 80 MB U-Net, 65 epochs, "experimental, short-horizon", drift accumulates; despite the name it is an 80 MB U-Net for a "Game Oasis Player" app, not a Decart Oasis-500M fine-tune — [HF](https://huggingface.co/SyntheticMDProductions/Super_Mario_64_Oasis_Model); HF API file list.

**Datasets**
- rafaelcp/smbdataset (NES SMB): ~737,000 frames, 280 episodes, 256x240 8-bit indexed PNGs, each frame tagged with one of 256 NES controller combinations, with a RAM snapshot in PNG metadata; licence CC-BY-4.0, last push 2025-07-22 — [GitHub](https://github.com/rafaelcp/smbdataset) (description via search snippet; licence via GitHub API).
- Miraiyochi/world-model-qa-mario-data (HF, collected 2026-09-19): Mario AI Framework (Java engine, not NES) corpus: 2,500 episodes over 500 levels x 5 policies (A*, random etc.), 681,047 frames at 256x256 RGB, actions as 5-bit ints (LEFT/RIGHT/DOWN/SPEED/JUMP → 0–31), 25 state signals per step, no frame skip, deterministic replay; plus 31,919 frames of human play and DINOv3 latents — [HF dataset](https://huggingface.co/datasets/Miraiyochi/world-model-qa-mario-data)
- NVIDIA NitroGen dataset (Dec 2025/Jan 2026): action labels (17 buttons + 2 sticks per frame) auto-extracted from input-overlay gameplay videos; ~30k annotated videos, ~15B frames annotated; licence CC-BY-NC-4.0; card says it is intended for behaviour cloning and world models; only labels are distributed (videos are public internet videos) — [HF dataset card](https://huggingface.co/datasets/nvidia/NitroGen). The paper reports 40,000 hours after filtering from 71,000 hours, >1,000 games, Platformer 18.4% of hours — [NitroGen arXiv](https://arxiv.org/html/2601.02427v1); [project page](https://nitrogen.minedojo.org/)
- Small/undocumented HF Mario datasets exist (`OpenGenGAME/super-mario-bros-rl-1-1`, ~2,567 rows, 2024-09; `DylanRiden/mario-human-1-1`, 2026-01) with empty cards — HF API search 2026-11-19.
- `badigadiii/retro-games-gameplay-frames-30k-512p` (2025-04, MIT tag): retro gameplay frames, no action labels documented — HF API search 2026-11-19.

### Inferences
- MarioVGG is not usable as a downloaded model: there are no weights, it runs at 64x48, and it only knows "run" and "run+jump". Its value is as evidence that ~13k short clips from 737k frames give a plausible but glitchy model in 48 h on one 4090.
- PlayGen's Mario checkpoint is the most usable "pre-trained Mario world model". But it models the Mario AI Framework clone, whose graphics and physics differ from the NES ROM, at 128x128 with a reduced action set. It would need re-training or substantial fine-tuning on real NES frames.
- None of the Mario models models the full NES controller (8 buttons) at native 256x240 with RAM-consistent state. The lab's emulator already produces frames, actions and RAM at ~55k frames/s, so the datasets above are less useful than self-generated data.

### Gaps
- Could not confirm whether the Super-Mario-DIAMOND or PlayGen-Mario checkpoints are still downloadable (Google Drive links not tested).
- No NES-wide, multi-game world model (trained on many NES/Famicom ROMs) was found as of 2026-11-19 in HF searches ("mario", "nes", "retro", "world-model") or web searches.
- I did not verify the exact action count in PlayGen's Mario model.

## 2. General game world models: catalogue (weights, licence, size, data, compute, speed, actions, fine-tuning)

### Takeaway
Open-weight game world models do exist (DIAMOND, Oasis-500M, WHAM/Muse, Matrix-Game 1.0–4.0, Hunyuan-GameCraft, MineWorld (code only now), PlayGen). Almost all target 3D first-person games with keyboard+mouse/camera actions, use 0.2B–17B+ parameters, and run at 4–60 FPS for one stream on a datacenter or high-end GPU. The DeepMind Genie family (the only one trained on 2D platformers) is closed. Fine-tuning to a new game has been shown with hundreds of hours of data (Matrix-Game 2.0: Temple Run 560 h) or, for small diffusion models, roughly 0.1M–2.5M frames (AVID, Super-Mario-DIAMOND).

### Cited Findings

| Model (date) | Weights / licence | Size | Training data | Res / speed | Actions | Source |
|---|---|---|---|---|---|---|
| DIAMOND (May 2024; CS:GO Oct 2024) | Open: 26 Atari-100k world models (~54 MB each) + CS:GO (1.5 GB) on HF `eloialonso/diamond`; code MIT | Atari ~4M U-Net (13M incl. agent); CS:GO ~381M incl. 51M upsampler | Atari: 100k actions/game; CS:GO: 5.5M frames (95 h) at 16 Hz, 87 h train | Atari 64x64, 3 denoising steps, 12.7 ms/prediction; CS:GO 56x30 → 280x150, ~10 Hz on RTX 3090 | Discrete Atari actions; CS:GO keyboard+mouse | [arXiv](https://arxiv.org/html/2405.12399); [HF](https://huggingface.co/eloialonso/diamond); [GitHub](https://github.com/eloialonso/diamond) |
| GameNGen (Aug 2024, Google) | Not released | SD v1.4 U-Net (count not stated) | 70M examples from PPO-agent DOOM play | 320x240; 20 FPS on 1 TPU-v5 with 4 DDIM steps; ~50 FPS distilled 1-step | DOOM key presses | [arXiv](https://arxiv.org/html/2408.14837) |
| Oasis (31 Oct 2024, Decart/Etched) | Oasis-500M open, MIT, gated (auto-approve) on HF `Etched/oasis-500m` (2.4 GB + 0.9 GB ViT autoencoder); larger model demo-only | 500M (open) | "millions of hours" of Minecraft gameplay (per press) | Decart: 360p at 20 FPS on H100 for served model | Minecraft keyboard/mouse | [HF](https://huggingface.co/Etched/oasis-500m); [Decart](https://decart.ai/articles/oasis-interactive-ai-video-game-model); [TechCrunch](https://techcrunch.com/2024/10/31/decarts-ai-simulates-a-real-time-playable-version-of-minecraft/) |
| Lucid v1 (Nov 2024) | Weights said to be on HF (Substack); current licence unverified | not verified | Minecraft | 20+ FPS on RTX 4090 (YC page) | Minecraft controls | [Substack](https://ramimo.substack.com/p/lucid-v1-a-world-model-that-does); [YC](https://www.ycombinator.com/companies/lucid) |
| Genie 1 (Feb 2024, DeepMind) | Not released (no checkpoints, data or examples) | 10.7B (200M tokenizer, 300M latent-action model, 10.1B dynamics) | 6.8M clips, >30k h of 2D-platformer internet video (from 244k h raw), 160x90 at 10 FPS; no action labels | ~1 FPS, 25 MaskGIT steps/frame; trained on 256 TPUv5(p), 942B tokens | 8 learned latent actions | [arXiv](https://arxiv.org/html/2402.15391) |
| Genie 2 (4 Dec 2024) | Not released | not stated | "large-scale video dataset" | consistent up to ~1 min; distilled real-time version at lower quality | keyboard+mouse, 3D | [DeepMind blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) |
| Genie 3 (Aug 2025) / Project Genie (29 Jan 2026) | Closed; Project Genie is a Labs prototype for US Google AI Ultra subscribers (18+) | not stated | not stated | 720p, 24 FPS | navigation + text "promptable events" (events absent from Project Genie) | [Google blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/); [GameWM table](https://arxiv.org/html/2611.19247) |
| Muse / WHAM (Feb 2025, Microsoft + Ninja Theory) | Open on HF `microsoft/wham` (1.6B = 18.9 GB; 200M = 3.7 GB); Microsoft Research License (academic research; not for imagery outside Bleeding Edge) | 1.6B and 200M; VQ-GAN + decoder-only transformer | ~500k Bleeding Edge matches, >1B obs/action pairs at 10 Hz (~7 years of play), 27,990 players | 300x180 fixed; context 10 obs/action pairs; tested on A6000/A100; trained on 98 H100 for 5 days | Xbox controller actions; also predicts actions (policy mode) | [HF README](https://huggingface.co/microsoft/wham); [Nature](https://www.nature.com/articles/s41586-025-08600-3) |
| MineWorld (Apr 2025, Microsoft) | Code MIT on GitHub (last push 2026-05-08); HF `microsoft/mineworld` returns 401 (checkpoints were "temporarily taken down" in May 2025 per Gitee mirror) | multiple sizes (not verified) | Minecraft only | 4–7 FPS with diagonal decoding | visual-action autoregressive transformer | [arXiv](https://arxiv.org/html/2504.08388v1); [Gitee mirror](https://gitee.com/mirrors_microsoft/mineworld); HF/GitHub API 2026-11-19 |
| GameFactory (Jan 2025, HKU + Kuaishou; ICCV 2025) | Repo moved to `KlingAIResearch/GameFactory` (no licence field); GF-Minecraft dataset released; weight release not confirmed | not verified | small first-person Minecraft action dataset, transferred to open-domain video | not verified | keyboard+mouse | [arXiv](https://arxiv.org/abs/2501.08325v1); GitHub API 2026-11-19 |
| Matrix-Game 1.0 (May 2025, Skywork) | Open, MIT, HF `Skywork/Matrix-Game` (87 GB of weights) | not verified here | Minecraft-centric | — | keyboard+mouse | HF API 2026-11-19; [GitHub](https://github.com/SkyworkAI/Matrix-Game) |
| Matrix-Game 2.0 (Aug 2025) | Open, MIT, HF `Skywork/Matrix-Game-2.0` (base, distilled, Temple Run distilled) | ~1.8B (from SkyReels-V2-I2V-1.3B / Wan2.1) | UE 615 h, Minecraft 153 h, Sekai real 85 h; plus game-driver 574 h and Temple Run 560 h for fine-tunes (paper says "~1200 h" total; numbers do not reconcile) | 352x640, 25 FPS on one H100; 120k steps batch 256 | keyboard = discrete movement (cross-attn), mouse = continuous | [arXiv](https://arxiv.org/html/2508.13009); [HF](https://huggingface.co/Skywork/Matrix-Game-2.0) |
| Matrix-Game 3.0 (27 Mar 2026) | Open, Apache-2.0, HF `Skywork/Matrix-Game-3.0` (56.6 GB) | 5B (Wan2.2-TI2V-5B); 2x14B MoE variant | UE synthetic + automated game capture + real video | 720p at 40 FPS (INT8, distilled) | keyboard+mouse, camera pose | [HF](https://huggingface.co/Skywork/Matrix-Game-3.0) |
| Matrix-Game 4.0 (15 Oct 2026) | Open, Apache-2.0; weights for 3 native scenes | 5B (Wan2.2-5B) | dual-pass UE5 capture (1,712 h) | 480p, 60 FPS on one RTX 5090; 67 FPS RTX 5080 at 864x480 | NOT a pure world model: an executable engine does gameplay/physics; model re-renders low-poly frames | [GitHub](https://github.com/SkyworkAI/Matrix-Game); [GameWM](https://arxiv.org/html/2611.19247) |
| Hunyuan-GameCraft (Jun 2025; weights 14 Aug 2025, Tencent) | Open weights, custom Tencent licence (GitHub reports NOASSERTION); 30 GB checkpoint (+ distilled) | built on HunyuanVideo | >1M recordings from >100 AAA games + synthetic fine-tune | 720p; 6.6 FPS distilled (GameWM table); min 24 GB VRAM "very slow", 80 GB recommended, tested on 8 GPUs | keyboard/mouse mapped to continuous camera space | [HF](https://huggingface.co/tencent/Hunyuan-GameCraft-1.0); [GameWM](https://arxiv.org/html/2611.19247) |
| Yan (Aug 2025, Tencent) | weights not confirmed | not stated | targets Tencent's 3D game Yuan Meng Star | claims 1080p at 60 FPS | game actions | [arXiv](https://arxiv.org/html/2508.08601v1) |
| AVID (Oct 2024; RLC 2025, Microsoft Research) | Code MIT in `microsoft/causica` (research_experiments/avid) | adapters 22M/71M on a 97M Procgen base; 11–145M on 1.4B DynamiCrafter | base trained on 15 Procgen games (no CoinRun); adapter on 100k/500k/2.5M CoinRun samples | 3 days on 1 A100 per adaptation | 15 discrete keypad actions | [arXiv](https://arxiv.org/html/2410.12822v2); [project](https://sites.google.com/view/avid-world-model-adapters/home) |
| NVIDIA Cosmos-Predict2.5 (2025) | Weights under NVIDIA Open Model License (code Apache-2.0); action-conditioned variant is for robots | 2B (action-cond), 14B others | physical-world video | action-cond: 256p, 4 FPS | robot action vectors | [HF](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B); [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5) |
| NitroGen (Dec 2025/Jan 2026, NVIDIA) | Policy, not world model; weights on HF under NVIDIA non-commercial licence | 4.93x10^8 | 40k h, >1,000 games | — | outputs 17 buttons + 2 sticks | [HF](https://huggingface.co/nvidia/NitroGen); [arXiv](https://arxiv.org/html/2601.02427v1) |

Additional 2026 game world models listed in the GameWM survey (Tencent Lightspeed, arXiv 2611.19247, 14 Nov 2026), openness not stated there: WorldPlay (720p 24 FPS on 8 GPUs, 320K clips), LingBot-World 2.0 (14B/1.3B, 720p 60 FPS multi-GPU), ABot-World-0 (720p 16 FPS), Magpie, EdenLive, StreamRender-H3, GameWM (33B first chunk 3.1 s on 8 GPUs). The survey says "Matrix-Game 4.0 shows that a 5B renderer can reach 60 FPS at 480p on one RTX 5090" and "A single H3 stream currently takes several datacenter GPUs" — [GameWM](https://arxiv.org/html/2611.19247)

Fine-tuning to a new game (evidence):
- Matrix-Game 2.0 released a separate Temple Run model (fine-tuned with 560 h) and a game-driver model (574 h); the paper gives no fine-tune GPU-hours — [arXiv](https://arxiv.org/html/2508.13009); HF file list shows `templerun_distilled_model/templerun_7dim_onlykey.safetensors` (keyboard-only, 7-dim) — [HF](https://huggingface.co/Skywork/Matrix-Game-2.0)
- AVID adapting a Procgen-pretrained video model to unseen CoinRun: with 500k action-labelled samples, AVID-71M FVD 23.1 vs from-scratch 71M 31.3; full fine-tune of the 97M base FVD 14.1; unadapted base FVD 204.0; at 2.5M samples AVID FVD 14.5 vs fine-tune 12.0; each run limited to 3 days on 1 A100 — [arXiv](https://arxiv.org/html/2410.12822v2)
- Genie: a latent-to-real action mapping learned from as few as 200 expert samples matched an oracle in CoinRun behaviour cloning — [arXiv](https://arxiv.org/html/2402.15391)
- GameNGen: a separate Chrome Dino experiment used 2K episodes and 3,000 training steps — [arXiv](https://arxiv.org/html/2408.14837)
- Community: DIAMOND → NES Mario in ~24 h on one L4 at 64x60 — [GitHub](https://github.com/AlexBodner/Super-Mario-DIAMOND)
- WHAM's card says out-of-domain image prompts "dissolve into unrecognizable blobs" or into Bleeding Edge scenes, i.e. no transfer outside its one game — [HF README](https://huggingface.co/microsoft/wham)

### Inferences
- Large open models (Matrix-Game 2/3, GameCraft, WHAM-1.6B, Oasis) are built for 3D camera navigation with mouse look. Their action encoders (continuous camera, WASD) do not match an 8-button NES pad, and their VAEs and resolutions are tuned for photoreal or 3D content. Adapting one to NES would mean replacing the action module and fine-tuning on hundreds of hours. That is a bigger job, with a bigger model, than training a small NES-specific model.
- Even the fastest open model (Matrix-Game 4.0, 60 FPS at 480p on a 5090) produces one stream at ~60 frames/s. The lab's emulator produces ~55,600 frames/s on the same GPU, about 900x more, and it is exact. A pre-trained general world model is therefore not a faster simulator. Its only possible value is as a learned representation or predictor (e.g. imagination or planning), and there the NES emulator plus RAM already gives ground truth.
- DIAMOND is the most practical code base to copy for an NES predictor. It is MIT-licensed, has small models (~4M-param Atari U-Net at 64x64), takes discrete actions, has been ported to NES Mario, and has a known compute budget.

### Gaps
- No primary source found for Oasis training data details (VPT or otherwise) or for Oasis-500M local inference speed.
- MineWorld parameter counts and whether the HF checkpoints were ever restored (API returns 401 on 2026-11-19).
- GameFactory and Yan weight availability not confirmed; Lucid's current licence not confirmed.
- Training compute (GPU-hours) is unreported for Matrix-Game 2.0/3.0, GameCraft, Oasis, Genie 2/3.

## 3. Zero-shot generalisation to unseen games/levels; evidence for NES-style games

### Takeaway
Zero-shot claims are about visual prompting (sketches, photos, concept art) and unseen levels or scenes within a trained genre, not about faithfully simulating the rules of an unseen specific game. The only family trained on 2D platformers (Genie 1) is closed and runs at ~1 FPS. No source shows any open model zero-shot simulating an unseen NES game accurately.

### Cited Findings
- Genie 1 (2D platformers) produced game-like character movement and parallax when prompted with Imagen-2 images, hand-drawn sketches and real photos. Within platformers its 8 latent actions kept consistent meanings (left, right, jump, no-op) across starting frames. The paper does not show one latent action meaning the same thing across domains. Weights were not released — [arXiv](https://arxiv.org/html/2402.15391)
- Genie 2 claims out-of-distribution generalisation (concept art, drawings, real photos become interactive), with consistent output up to ~1 minute. It is focused on 3D and closed — [DeepMind blog](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)
- Genie 3 is available only as the Project Genie prototype (US Google AI Ultra, from 29 Jan 2026, access "intentionally restricted"; promptable events missing) — [Google blog](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/project-genie/). One secondary source gives a June 23, 2026 access date that conflicts with this — [search summary of letsdatascience / others](https://www.letsdatascience.com/news/deepmind-opens-project-genie-for-interactive-worlds-eb2fec7e)
- MarioVGG trained on all SMB levels except 1-1 and generated plausible 1-1 clips (run/jump, gravity, collisions). Chained generation "synthesizes the levels" rather than reproducing the real level — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf)
- AVID: a video model pretrained on 15 Procgen games and not adapted gives FVD 204.0 on held-out CoinRun. Action-labelled CoinRun data (100k–2.5M samples) brings it to 56.0–14.5. So cross-game pretraining alone does not give action-correct predictions for a new 2D game — [arXiv](https://arxiv.org/html/2410.12822v2)
- Matrix-Game 2.0 claims generalisation to "wild scenes", but its limitations section says out-of-domain generalisation "needs to be improved", and it does not test unseen games — [arXiv](https://arxiv.org/html/2508.13009)
- SCOPE (May 2026) reports zero-shot transfer to unseen scenes for FPS world models via a multi-game CrossFPS dataset (FPS genre only) — [arXiv](https://arxiv.org/html/2605.23345)
- WHAM degrades to blobs or its training game on out-of-domain prompts — [HF README](https://huggingface.co/microsoft/wham)
- GameNGen: edited start frames can produce unseen levels and inserted characters (presented as preliminary); memory is ~3 s of history — [arXiv](https://arxiv.org/html/2408.14837)
- GameBoyWorlds (25 Sep 2026) is a benchmark of 500 tasks in unseen Game Boy games. It reports "world modelling" approaches failing, and frontier models complete <50% of tasks. It provides no released world model — [arXiv](https://arxiv.org/abs/2609.32093)

### Inferences
- For NES, "zero-shot" would at best give a plausible-looking platformer clip, not the actual SMB physics, enemy behaviour, scroll or RAM-relevant state. An agent trained against it would learn the wrong game. Any useful NES predictor has to be trained or fine-tuned on that game's own action-labelled frames.

### Gaps
- No public test of Genie 3 / Project Genie on NES screenshots was found (the prototype is closed and terms may restrict game IP).
- No study found that measures zero-shot next-frame accuracy of any open model on NES games.

## 4. Cost to train a game-specific world model from scratch for one NES game (frames, GPU-hours)

### Takeaway
Published single-game budgets range from 100k agent steps and ~2.9 RTX-4090-days (DIAMOND Atari, 64x64), through ~0.7M frames and 48 h on one 4090 (MarioVGG, weak) and 5.5M frames and 12 RTX-4090-days (DIAMOND CS:GO), up to 70M frames on 128 TPU-v5e (GameNGen DOOM) or 98 H100 x 5 days (WHAM). With the lab's emulator, data is effectively free: 70M frames is ~21 minutes of generation. The cost is GPU training time, about 1–12 single-GPU days for a DIAMOND-class model.

### Cited Findings
- DIAMOND Atari 100k: 100k actions per game (~2 h of human play), 64x64x3, ~4M-param diffusion U-Net (13M total with agent), 3 Euler denoising steps. ~2.9 days per run on one RTX 4090 (~12 GB VRAM), including RL training; 1.03 GPU-years for all 26 games x 5 seeds; mean HNS 1.46 — [arXiv](https://arxiv.org/html/2405.12399)
- Same table: IRIS 30M params, 4.1 training days, HNS 1.046, 16 NFE/frame; DreamerV3 18M params, <1 training day, HNS 1.097 — [DIAMOND arXiv Table 4](https://arxiv.org/html/2405.12399)
- DIAMOND CS:GO: 5.5M frames (95 h at 16 Hz; 87 h train), ~381M params, 56x30 + upsampler to 280x150, 12 days on one RTX 4090, ~10 Hz on RTX 3090. The paper also says "8787 hours" in two places, which conflicts with 87 h — [arXiv](https://arxiv.org/html/2405.12399)
- GameNGen (DOOM): agent trained for 50M env steps; 70M examples used for diffusion training; 700k steps, batch 128 on 128 TPU-v5e. Ablation: 1M examples renders views but fails on game logic, 10M improves, and 70M keeps improving past 700k steps — [arXiv](https://arxiv.org/html/2408.14837)
- MarioVGG: 737k-frame source dataset (280 episodes), 13,252 short training clips, 300k steps x batch 8, ~48 h on one RTX 4090, 64x48 output — [MarioVGG PDF](https://virtual-protocol.github.io/mario-videogamegen/static/pdfs/VideoGameGen.pdf); frame count from [smbdataset](https://github.com/rafaelcp/smbdataset)
- PlayGen Mario: 236M frames collected, 50M sampled for training, 186M params total; training GPU-hours not reported — [arXiv](https://arxiv.org/html/2412.00887v1)
- Super-Mario-DIAMOND (NES): ~24 h on one L4 for 50 epochs at 64x60 (+ upsampler) — [GitHub](https://github.com/AlexBodner/Super-Mario-DIAMOND)
- AQ-Mario JEPA (latent only): 9.8M params, 41,160 steps on 4.88M level 1-1 observations — [HF](https://huggingface.co/maxmill/aq-mario)
- Genie CoinRun case study: 10M random-policy transitions over 10,000 levels; runs on one mid-range TPU/GPU (16 GB) in under a week; tokenizer ~3 days on one TPU — [arXiv](https://arxiv.org/html/2402.15391)
- WHAM (3D, for scale): >1B obs/action pairs, 98 H100 x 5 days — [HF README](https://huggingface.co/microsoft/wham)
- AVID adapter on CoinRun: 3 days on one A100 with 0.1M–2.5M samples — [arXiv](https://arxiv.org/html/2410.12822v2)

### Inferences
- Data generation at ~55,600 frames/s means 5.5M frames (DIAMOND CS:GO scale) takes ~100 s and 70M frames (GameNGen scale) ~21 min (arithmetic from the lab's stated throughput). Data volume is not a constraint. Coverage is the real issue: the data needs states the agent rarely reaches. GameNGen and PlayGen both mixed random and agent policies with random spawns for this reason.
- Extrapolating from DIAMOND and MarioVGG: an NES predictor at 64x60 or 64x64 latent/pixel resolution with an upsampler or decoder should take on the order of 1–3 RTX-4090/5090-days. A full-resolution (256x240) or long-context model should take on the order of the CS:GO budget (~12 days on one 4090) or more. These are estimates from analogous papers, not measured NES numbers.
- A ~4M–50M-parameter DIAMOND-style model is small enough to batch thousands of predictions per step on a 5090, but still needs ≥3 network evaluations per frame. It will be well below the emulator's throughput. It is useful for planning or imagination, not as a replacement for the emulator.

### Gaps
- No paper reports GPU-hours for an NES-specific world model at native resolution with the full 8-button action space.
- PlayGen, Matrix-Game and Oasis do not report total training GPU-hours.
- No published RTX 5090 training benchmarks for DIAMOND-class models were found.

## 5. Bottom line for this lab: can a pre-trained model avoid training our own next-screen predictor?

### Takeaway
No, not as of 2026-11-19. No downloadable model predicts real NES SMB frames under the full NES controller at usable fidelity, and the general open models are 3D, mouse-and-keyboard, multi-billion-parameter and ~900x slower than the emulator. The cheapest credible path is to train a small DIAMOND-style or PlayGen-style model on self-generated emulator data. Published single-game budgets suggest roughly 1–12 GPU-days. Existing Mario checkpoints (PlayGen, Super-Mario-DIAMOND) are useful as starting code or as a sanity baseline, not as drop-in predictors.

### Cited Findings
- MarioVGG has no weights in its HF repo — [HF](https://huggingface.co/virtuals-protocol/mario-videogamegen) (file list via HF API 2026-11-19)
- PlayGen Mario is trained on the Mario-AI-Framework at 128x128 — [arXiv](https://arxiv.org/html/2412.00887v1)
- Super-Mario-DIAMOND gives no checkpoint link and reports no metrics — [GitHub](https://github.com/AlexBodner/Super-Mario-DIAMOND)
- Genie 1, the only 2D-platformer foundation model, is unreleased and ~1 FPS — [arXiv](https://arxiv.org/html/2402.15391)
- Fastest open interactive model: Matrix-Game 4.0 at 60 FPS at 480p on one RTX 5090, and it needs a game engine — [GameWM](https://arxiv.org/html/2611.19247); [GitHub](https://github.com/SkyworkAI/Matrix-Game)
- Cross-game pretraining without target-game action data gives poor action-conditioned predictions (AVID CoinRun FVD 204 unadapted vs 14–56 adapted) — [arXiv](https://arxiv.org/html/2410.12822v2)

### Inferences
- Licence and IP: WHAM (Microsoft Research License, academic-only, Bleeding-Edge-only use), NitroGen (NVIDIA non-commercial), and Cosmos (NVIDIA Open Model License with attribution) restrict use. MIT/Apache options are DIAMOND, Oasis-500M (gated), Matrix-Game 1–4 and PlayGen code. All Mario datasets carry the caveat that the underlying game content is Nintendo's.
- If a pre-trained component is wanted, the realistic one is a frozen image encoder (e.g. DINOv3 latents, as in the Miraiyochi dataset) or a VAE, rather than a full pretrained dynamics model.

### Gaps
- Not tested: running the PlayGen Mario checkpoint or Super-Mario-DIAMOND on real NES frames to measure transfer error.
