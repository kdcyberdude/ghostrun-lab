#!/usr/bin/env bash
# All 12 games at once (run main-20261110c): 3,072 consoles per game on one RTX 5090, 8 h limit.
# results/tables/twelve_games_*.csv. Run from the repository root; set NES_ROM_DIR (rl/games_manifest.json lists the ROMs).
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON:-python3} rl/train.py --run runs/rl/main-20261110c \
    --envs-per-game 3072 --obs-slots 1 --infer-chunk 512 --compile-mode max-autotune-no-cudagraphs \
    --steps 16 --minibatch-envs 128 --time-limit 29000 --eval-every-minutes 30 --adv-norm game
