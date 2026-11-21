#!/usr/bin/env bash
# Train the shared policy on table games (rl/games_table.json: every ROM ready to train, rl/scripts/export_games.py),
# optionally together with the 12 hand-read reward10 games. Same model, same reward10 formula for every game.
#
#   rl/scripts/train_table.sh RUN_DIR GAMES [train.py flags...]
#     GAMES: a comma list of slugs, "all" (every table game), or "first:N" / "random:N" (N table games)
#   e.g.   rl/scripts/train_table.sh runs/rl/table-64 random:64 --envs-per-game 8
#          rl/scripts/train_table.sh runs/rl/table-smb smb,abadox_u,contra --envs-per-game 32
#
# The emulator is the all-mappers NeSLE (.cache/nesle-fork, branch ghostrun-allmappers) and the build goes to its own
# directory, so the default rl/build (used by other runs) is never rebuilt from here.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=$1; GAMES=$2; shift 2
export RL_TABLE_GAMES=${RL_TABLE_GAMES:-$PWD/rl/games_table.json}
export NESLE_INC=${NESLE_INC:-$PWD/.cache/nesle-fork/cpp/include}
export NESLE_ROM_CPP=${NESLE_ROM_CPP:-$PWD/.cache/nesle-fork/cpp/src/rom.cpp}
export RL_BUILD=${RL_BUILD:-$PWD/.cache/rl/build-table}
export NES_ROM_DIR=${NES_ROM_DIR:-/home/kd/Desktop/roms}
export NVCC=${NVCC:-/usr/local/cuda-12.8/bin/nvcc}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
PY=${PY:-/home/kd/Desktop/proj/llm_gameplay/.cache/multigame-prior-runtime-v1/bin/python}
LIST=$("$PY" - "$GAMES" <<'EOF'
import json, os, random, sys
t = list(json.load(open(os.environ['RL_TABLE_GAMES'])))
g = sys.argv[1]
if g == 'all':
    out = t
elif g.startswith('first:'):
    out = t[:int(g[6:])]
elif g.startswith('random:'):
    out = random.Random(0).sample(t, int(g[7:]))
else:
    out = g.split(',')
print(','.join(out))
EOF
)
mkdir -p "$RUN"
echo "$PY rl/train.py --run $RUN --games <$(echo "$LIST" | tr ',' '\n' | wc -l) games> $*" > "$RUN/cmd_table.txt"
echo "$LIST" | tr ',' '\n' > "$RUN/games.txt"
exec "$PY" -u rl/train.py --run "$RUN" --games "$LIST" "$@"
