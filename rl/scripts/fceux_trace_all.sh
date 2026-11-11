#!/usr/bin/env bash
# Trace every game's saved button file in FCEUX (headless, CPU only) under one setting.
# Each run gets a fresh empty profile (HOME), so no battery save (.sav) or config from an earlier run is loaded.
# usage: [LEAD=n] rl/scripts/fceux_trace_all.sh NAME NEWPPU RAMINIT [FRAMES]   -> .cache/rl/fceux-traces/GAME.NAME.ram.bin
set -euo pipefail
cd "$(dirname "$0")/../.."
P=$PWD; NAME=$1; NEWPPU=$2; RAMINIT=$3; FRAMES=${4:-6000}
FX=${FCEUX:-.cache/fceux-compat-build-v2/build/src/fceux}
OUT=.cache/rl/fceux-traces; mkdir -p "$OUT/logs"
PY=.cache/multigame-prior-runtime-v1/bin/python
$PY -c "
import sys; sys.path.insert(0,'scripts/reward10'); sys.path.insert(0,'rl/scripts'); import contract
from gen_reward_consts import GAMES
for g in GAMES: print(g + '\t' + contract.load(g).ROM)" | { while IFS=$'\t' read -r g rom; do
  (H=$(mktemp -d); HOME=$H TRACE_BUTTONS=$P/.cache/nesle-check/traces/$g.buttons.bin TRACE_OUT=$P/$OUT/$g.$NAME.ram.bin TRACE_FRAMES=$FRAMES TRACE_LEAD=${LEAD:-0} \
    timeout 600 xvfb-run -a "$FX" --sound 0 --newppu "$NEWPPU" --ramInit "$RAMINIT" --loadlua "$P/rl/scripts/fceux_trace.lua" "$rom" \
    > "$OUT/logs/$g.$NAME.log" 2>&1 || echo "FAIL $g $NAME"; rm -rf "$H") &
done
wait; }
