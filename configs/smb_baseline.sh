#!/usr/bin/env bash
# Mario only, 8,192 consoles, plain recurrent PPO (run smb-20261110b, iterations 1-392). v2 continues from its checkpoint.
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON:-python3} rl/train.py --run runs/rl/smb-20261110b --games smb \
    --envs-per-game 8192 --obs-slots 1 --infer-chunk 512 --compile-mode max-autotune-no-cudagraphs \
    --steps 16 --minibatch-envs 128 --time-limit 25200 --eval-every-minutes 15 --adv-norm game
