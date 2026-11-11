#!/usr/bin/env bash
# Exact resume on the CPU: 5 iterations straight vs 3 + stop + resume to 5; everything must be bit-identical.
set -euo pipefail
cd "$(dirname "$0")/../.."
S=${1:-$(mktemp -d)}; PY=${PY:-.cache/multigame-prior-runtime-v1/bin/python}
rm -rf "$S/resA" "$S/resB"
A=(--device cpu --envs-per-game 1 --steps 8 --minibatch-envs 4 --epochs 1 --eval-every-minutes 0 --deterministic 1)
export CUDA_VISIBLE_DEVICES=
$PY rl/train.py "${A[@]}" --run "$S/resA" --total-iters 5 | tail -1
$PY rl/train.py "${A[@]}" --run "$S/resB" --total-iters 3 | tail -1
$PY rl/train.py "${A[@]}" --run "$S/resB" --total-iters 5 | tail -2
$PY - "$S" <<'PYEOF'
import sys, torch
S = sys.argv[1]
a = torch.load(f'{S}/resA/ckpt_rank0_latest.pt', weights_only=False)
b = torch.load(f'{S}/resB/ckpt_rank0_latest.pt', weights_only=False)
ok = dict(it=a['it'] == b['it'] == 5,
          model=all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']),
          optimizer=all(torch.equal(x, y) for s1, s2 in zip(a['opt']['state'].values(), b['opt']['state'].values())
                        for x, y in zip(s1.values(), s2.values()) if torch.is_tensor(x)),
          env=bool((a['env']['env'] == b['env']['env']).all()), obs=torch.equal(a['obs'], b['obs']),
          h=torch.equal(a['h'], b['h']), ret_stats=torch.equal(a['ret_mean'], b['ret_mean']))
print(ok)
assert all(ok.values())
print('exact resume: identical')
PYEOF
