"""Turn a training snapshot (RUN/eval/itN/model.pt) into the folder the standalone player reads: policy.onnx + model.json.

One call = one greedy decision: the same network as rl/model.py, except that it takes RGB pictures instead of NES colour
numbers. CPU only. The folder appears complete or not at all, and only after ONNX and PyTorch agree on test inputs.
usage: python rl/player/export.py RUN/eval/itN/model.pt OUT_FOLDER
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import NEG, PARTS, Policy, SIZES  # noqa: E402

NAMES = ['pixels', 'timeline', 'prev', 'h', 'masks']


class Actor(torch.nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.p = policy

    def forward(self, pixels, timeline, prev, h, masks, mood=None):
        p = self.p
        x = F.relu(p.cnn(p.stem(pixels))).mean((2, 3))
        t = F.relu(p.timeline(timeline.flatten(1)))
        z = p.ln(p.blocks(p.inp(torch.cat([x, t, prev] + ([mood] if p.moods else []), 1))))
        # nn.GRUCell written out, because the ONNX exporter has no GRU cell
        gi, gh = F.linear(z, p.gru.weight_ih, p.gru.bias_ih), F.linear(h, p.gru.weight_hh, p.gru.bias_hh)
        ir, iz, inn = gi.chunk(3, 1)
        hr, hz, hn = gh.chunk(3, 1)
        r, u = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        h = (1 - u) * torch.tanh(inn + r * hn) + u * h
        chosen = []
        for k, (a, b) in enumerate(PARTS):
            chosen.append(p.head_logits(h, k, chosen).masked_fill(masks[:, a:b] == 0, NEG).argmax(-1))
        return torch.stack(chosen, 1), h


def example(policy: Policy, seed: int):
    g = torch.Generator().manual_seed(seed)
    args = [torch.rand(1, 12, 224, 240, generator=g), (torch.rand(1, 8, 45, generator=g) < 0.1).float(),
            torch.zeros(1, sum(SIZES)), torch.randn(1, policy.core, generator=g), torch.ones(1, sum(SIZES))]
    for a, n in PARTS:
        args[2][0, a + int(torch.randint(n - a, (1,), generator=g))] = 1
    if policy.moods:
        args.append(F.one_hot(torch.randint(policy.moods, (1,), generator=g), policy.moods).float())
    return tuple(args)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(ckpt: Path, out: Path):
    import onnxruntime as ort
    ckpt, out = Path(ckpt).resolve(), Path(out)
    s = torch.load(ckpt, map_location='cpu', weights_only=False)
    games = s['cfg']['games'].split(',')
    if 'smb' not in games:
        raise ValueError(f'{ckpt}: this model does not play smb ({games})')
    policy = Policy(core=s['cfg']['core'], moods=s['cfg'].get('moods', 0))
    policy.load_state_dict(s['model'])
    actor = Actor(policy.eval()).eval()
    tmp = out.with_name(out.name + f'.tmp{os.getpid()}')
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        names = NAMES + (['mood'] if policy.moods else [])
        with torch.no_grad():
            torch.onnx.export(actor, example(policy, 0), str(tmp / 'policy.onnx'), input_names=names,
                              output_names=['parts', 'h_next'], opset_version=17, dynamo=False)
            session = ort.InferenceSession(str(tmp / 'policy.onnx'), providers=['CPUExecutionProvider'])
            for seed in range(1, 9):
                args = example(policy, seed)
                parts, h = actor(*args)
                got = session.run(None, {n: a.numpy() for n, a in zip(names, args)})
                if not np.array_equal(got[0], parts.numpy()) or not np.allclose(got[1], h.numpy(), atol=1e-3):
                    raise RuntimeError(f'{ckpt}: ONNX and PyTorch disagree')
        info = dict(run=ckpt.parents[2].name, iteration=s['it'], decisions=s['decisions'], moods=policy.moods,
                    core=policy.core, source=str(ckpt), source_mtime=ckpt.stat().st_mtime, exported_at=time.time(),
                    onnx_sha256=digest(tmp / 'policy.onnx'))
        (tmp / 'model.json').write_text(json.dumps(info, indent=1) + '\n')
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return info


if __name__ == '__main__':
    torch.set_num_threads(2)
    print(json.dumps(export(Path(sys.argv[1]), Path(sys.argv[2]))))
