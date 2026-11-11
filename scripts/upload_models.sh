#!/usr/bin/env bash
# Upload the two Mario policies to the Hugging Face Hub with `hf upload` (log in first: `hf auth login`).
# Files are staged in a temporary folder, checked against the hashes in models/README.md, then uploaded.
#
# usage: TRAINING_DIR=/path/to/training/checkout scripts/upload_models.sh [v1|v2|both] [--dry-run]
#   TRAINING_DIR   the training working copy that holds runs/ and .cache/ (not this repository)
#   V1_REPO, V2_REPO   default kdcyberdude/ghostrun-smb-policy-v1 / -v2
#   V2_ITERATION       default 2907 (the eval snapshot with an ONNX export)
# Uploads weights, exports, the model cards and small test-play files only. No ROMs, emulator states or videos.
set -euo pipefail
LAB="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WHICH="${1:-both}"
DRY="${2:-}"
: "${TRAINING_DIR:?set TRAINING_DIR to the training working copy (holds runs/ and .cache/)}"
V1_REPO="${V1_REPO:-kdcyberdude/ghostrun-smb-policy-v1}"
V2_REPO="${V2_REPO:-kdcyberdude/ghostrun-smb-policy-v2}"
IT="$(printf '%06d' "${V2_ITERATION:-2907}")"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

check() {   # check FILE SHA256
    local got
    got="$(sha256sum "$1" | cut -d' ' -f1)"
    [[ "$got" == "$2" ]] || { echo "hash mismatch: $1 ($got, expected $2)" >&2; exit 1; }
}

card() {    # card v1|v2 -> the card section of models/README.md with Hub front matter
    printf -- '---\nlicense: mit\nlibrary_name: pytorch\ntags:\n- reinforcement-learning\n- nes\n- super-mario-bros\n- ghostrun\n---\n\n'
    awk -v s="<!-- card:$1 -->" -v e="<!-- /card:$1 -->" '$0==s{on=1;next} $0==e{on=0} on' "$LAB/models/README.md"
    printf '\nTraining code, reward functions and recording scripts: the GhostRun training lab repository.\n'
}

upload() {  # upload REPO DIR MESSAGE
    echo "== $1 <- $2"
    ls -la "$2"
    if [[ "$DRY" == --dry-run ]]; then
        echo "(dry run) hf upload $1 $2 . --repo-type model --commit-message \"$3\""
    else
        hf upload "$1" "$2" . --repo-type model --commit-message "$3"
    fi
}

if [[ "$WHICH" == v1 || "$WHICH" == both ]]; then
    d="$STAGE/v1"; mkdir -p "$d"
    src="$TRAINING_DIR/runs/fast-mario/robust-ppo-v16-active-refine69000-v1"
    cp "$src/refined-000452-069000.pt" "$src/refined-000452-069000.json" "$d/"
    cp "$TRAINING_DIR/.cache/mario-play-build-v1/assets/actor.onnx" "$d/"
    check "$d/refined-000452-069000.pt" 633b56e5ea06520cea80b27f41edabcbde8dce24dd754e59a0a9a9d4c28302b8
    check "$d/actor.onnx" c0fdb9506fafd55b0c46e857cdaeb1de7a164c92c10206fbe92022f68f23f78a
    cp "$LAB/models/v1/robust_model.py" "$LAB/data/routes/smb_v1_all32_route.json" "$d/"
    card v1 > "$d/README.md"
    upload "$V1_REPO" "$d" "v1: frozen all-32 policy (refined-000452-069000)"
fi

if [[ "$WHICH" == v2 || "$WHICH" == both ]]; then
    d="$STAGE/v2"; mkdir -p "$d/eval"
    run="$TRAINING_DIR/runs/rl/smb-20261110c-moods"
    exp="$TRAINING_DIR/runs/rl/policy-export/smb-20261110c-moods-it$IT"
    cp "$run/eval/it$IT/model.pt" "$d/"
    cp "$run/eval/it$IT/summary.json" "$run"/eval/it"$IT"/*.pads.bin "$d/eval/"
    cp "$exp/policy.onnx" "$d/"
    if [[ "$IT" == 002907 ]]; then
        check "$d/model.pt" 2ed7aa22e30e8c14954dcc77a869830dc064885b689ff7b1ccfafdcdc70d0aeb
        check "$d/policy.onnx" 64ea4a46e2b2f1961728c1a747e32185e6f2d9274d10c42f96b0aa0ddf320cab
    fi
    # model.json without the exporting machine's absolute paths
    python3 - "$exp/model.json" "$d/model.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
m.pop('source', None)
m['source'] = f"{m['run']}/eval/it{m['iteration']:06d}/model.pt"
json.dump(m, open(sys.argv[2], 'w'), indent=1)
EOF
    sed -i 's#"video": "[^"]*/#"video": "#' "$d/eval/summary.json"
    card v2 > "$d/README.md"
    upload "$V2_REPO" "$d" "v2: moods policy, iteration ${IT#00}"
fi
