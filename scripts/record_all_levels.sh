#!/usr/bin/env bash
# Record both Mario policies on all 32 levels: an MP4 and a per-frame controller log per level per model.
#   v2 (rl/ trainer, 8 moods): every mood plays each level from its entrance; the mood with the most progress is kept.
#   v1 (frozen all-32 player): one deterministic full game, cut per level.
#
# usage: scripts/record_all_levels.sh [--dry-run] [--out DIR] [--levels all|1-1,4-4] [--only v1|v2] [extra args for both]
#
# Inputs (environment variables; see models/README.md for where the files come from):
#   GHOSTRUN_V2_POLICY      v2 weights, e.g. model.pt from kdcyberdude/ghostrun-smb-policy-v2
#   GHOSTRUN_V2_STARTS      v2 level entrances: the training checkpoint (ckpt_rank0_latest.pt) or a cut .npy archive
#   GHOSTRUN_REFERENCE_LAB  v1's reference-lab launcher (called as: LAUNCHER --rom ROM --output DIR --diagnostics)
#   GHOSTRUN_SMB_ROM        your Super Mario Bros. (JU) (PRG 0) ROM, for v1
#   NES_ROM_DIR             folder holding the ROMs named in rl/games_manifest.json, for v2
#   PYTHON                  interpreter with torch, numpy, Pillow (default: python3)
# --dry-run loads no model and runs no emulator: it checks the inputs, prints the plan and tests the log/MP4 writers.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python3}"
OUT=recordings
ONLY=""
DRY=()
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=(--dry-run); shift ;;
        --out) OUT="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done
mkdir -p "$OUT"
status=0
# Bash 3.2 treats empty arrays as unset under nounset; expand optional arguments only when present.
if [[ -z "$ONLY" || "$ONLY" == v2 ]]; then
    "$PY" -u scripts/recording/record_v2.py --out "$OUT" ${DRY[@]+"${DRY[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} || status=$?
fi
if [[ -z "$ONLY" || "$ONLY" == v1 ]]; then
    "$PY" -u scripts/recording/record_v1.py --out "$OUT" ${DRY[@]+"${DRY[@]}"} ${EXTRA[@]+"${EXTRA[@]}"} || status=$?
fi
exit "$status"
