#!/usr/bin/env bash
# The v2 policy (run smb-20261110c-moods). It started from smb-20261110b's checkpoint (copy its ckpt_rank0_latest.pt
# into the new run folder) and was stopped and resumed with changed flags as the run went on; every resume is exact
# (environments, RNG, optimizer). The phases, in order:
#   it  392  moods on:            --steps 16 --minibatch-envs 128 --moods 8 --div-coef 1e-4 --div-margin 0.1
#   later    curiosity + level starts: --div-coef 1e-3 --mood-kick 0.07 --int-coef 2e-3 --level-starts 0.5
#   it 1285  level clear counting fixed (running with --level-starts 1.0); it 1502 failure-weighted level share
#   it 1539  waiting cost in the reward; later the SMB 4-4 progress fix, and slots 5-1..8-4 added to the archive
#   it 2773  longer rounds: --steps 64 --minibatch-envs 32 --lam 0.98 --ret-beta 0.98; it 2779 --int-coef 0
#   it 2841  Go-Explore places: --places 2048 --place-k 4 --place-per-level 128; then --only-levels 15,27,30,31
#   it 2893  --places 0  <- the published snapshot it002907 was trained with this command (this repo's reward
#            also has the later rollback penalty, W_ROLLBACK in scripts/reward10/reward.py):
set -euo pipefail
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON:-python3} rl/train.py --run runs/rl/smb-20261110c-moods --games smb \
    --envs-per-game 8192 --obs-slots 2 --infer-chunk 512 --compile-mode max-autotune-no-cudagraphs \
    --steps 64 --minibatch-envs 32 --lam 0.98 --ret-beta 0.98 --time-limit 0 --eval-every-minutes 15 --adv-norm game \
    --moods 8 --div-coef 1e-3 --div-margin 0.1 --mood-kick 0.07 --int-coef 0 --level-starts 1.0 \
    --places 0 --place-k 4 --place-per-level 128 --only-levels 15,27,30,31
# Next phase after it 2913 (longer horizon, not in the published snapshot):
#   --steps 128 --minibatch-envs 16 --lam 0.99 --ret-beta 0.9604 --gamma 0.999 --obs-pack6 1, plus the reward's
#   rollback penalty (scripts/reward10/reward.py W_ROLLBACK).
