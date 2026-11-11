"""rl/level_plan.py and weighted level picks (rlenv_core.cuh). Plain rules first; then, on saved Mario level starts:
the picks follow the weights, the CPU and GPU builds agree exactly.

usage: python rl/tests/test_level_plan.py MODEL.pt STATE.npz [--gpu]
  STATE.npz is made on the first run (a policy plays on the CPU until 3 levels are saved), reused afterwards.
"""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from level_plan import LevelPlan  # noqa: E402
from model import pack  # noqa: E402
from rlenv import RlEnv  # noqa: E402
from test_level_starts import policy  # noqa: E402

N = 16
KW = dict(seed=11, sticky=True, idle_max=30, dropout=True, level_cap=64)


def rules():
    def plan_with(rows):
        """rows: level -> (visits, clears)."""
        p = LevelPlan(['smb'], 64)
        v, c = np.zeros((1, 64)), np.zeros((1, 64))
        for lv, (a, b) in rows.items():
            v[0, lv], c[0, lv] = a, b
        p.add(v, c)
        return p
    # share = failure rate, normalised: power-on 0% fails, level 1 fails 25%, level 2 fails 75%
    p = plan_with({0: (100000, 100000), 1: (100000, 75000), 2: (100000, 25000)})
    w, share = p.plan(0, {1: 0, 2: 0}, 1.0)
    assert abs(w[1] - 0.25) < 1e-3 and abs(w[2] - 0.75) < 1e-3 and w[0] == 0, w[:3]
    assert share > 0.999, share
    # power-on fails as often as level 1: each half
    w, share = plan_with({0: (100000, 50000), 1: (100000, 50000)}).plan(0, {1: 0}, 1.0)
    assert abs(share - 0.5) < 1e-3 and abs(w[1] - 1) < 1e-9, (share, w[:2])
    # little recent evidence counts as unknown, so a level no longer played is tried again
    p = plan_with({0: (1000, 1000)})
    for _ in range(600):
        p.add(np.zeros((1, 64)), np.zeros((1, 64)))
    assert abs(p.rate(0)[0] - 0.5) < 0.05, p.rate(0)[0]
    # a game whose first level index is 1: power-on is the level before the first saved one
    w, s1 = plan_with({1: (500, 0), 2: (500, 500)}).plan(0, {2: 0}, 1.0)
    assert w[2] == 1 and w[1] == 0 and s1 < 0.01, (w[:3], s1)
    assert p.plan(0, {}, 0.5) == (None, 0.5)
    print('ok: rules')


def make_state(model_path, out):
    torch.set_num_threads(8)
    m = policy(model_path)
    c = RlEnv({'smb': N}, device='cpu', **KW)
    c.start()
    h, prev = torch.zeros(N, m.core), [torch.zeros(N, dtype=torch.long) for _ in range(5)]
    mood = torch.zeros(N, dtype=torch.long) if m.moods else None
    gen = torch.Generator().manual_seed(3)
    for t in range(6000):
        with torch.no_grad():
            z = m.encode(c.obs, c.emph, c.timeline, prev, mood)
            h = m.gru(z.float(), h)
            parts, _ = m.act(h, c.masks, generator=gen)
        c.actions.copy_(pack(parts))
        c.step()
        d = c.done.bool()
        h[d] = 0
        prev = [p * (~d).long() for p in parts]
        if t % 50 == 0 and len(c.saved_levels()['smb']) >= 3:
            break
    assert len(c.saved_levels()['smb']) >= 3
    np.savez(out, env=c.state_dict()['env'])
    print('saved levels', c.saved_levels()['smb'], 'after', t + 1, 'decisions')


def load(device, blob):
    e = RlEnv({'smb': N}, device=device, **KW)
    e.load_state_dict(dict(env=blob, outputs={}, games=['smb'], counts=[N]))
    return e


def picks(model_state):
    blob = np.load(model_state)['env']
    prefix = RlEnv({'smb': N}, device='cpu', seed=11, sticky=True, idle_max=30, dropout=True).lib
    c = load('cpu', blob)
    bare = RlEnv({'smb': N}, device='cpu', seed=11, sticky=True, idle_max=30, dropout=True)
    P = bare.lib.rlenv_state_bytes(bare.h)
    del prefix
    saved = sorted(c.saved_levels()['smb'])
    a, b = saved[0], saved[-1]
    w = np.zeros(64); w[a], w[b] = 0.75, 0.25
    c.set_level_plan(0, 0.5, w)
    count = {}
    for _ in range(150):
        c.start()
        for v in np.frombuffer(c.state_dict()['env'][P:P + 4 * N].tobytes(), np.int32).tolist():
            count[v] = count.get(v, 0) + 1
    tot = sum(count.values())
    print('picks', count)
    assert set(count) == {-1, a, b}, count
    assert abs(count[-1] / tot - 0.5) < 0.05
    assert abs(count[a] / tot - 0.375) < 0.05 and abs(count[b] / tot - 0.125) < 0.05
    print('ok: picks follow the weights and the share')


def gpu(model_state):
    blob = np.load(model_state)['env']
    c, g = load('cpu', blob), load('cuda', blob)
    saved = sorted(c.saved_levels()['smb'])
    w = np.zeros(64)
    w[saved] = np.arange(1, len(saved) + 1) / sum(range(1, len(saved) + 1))
    for e in (c, g):
        e.set_level_plan(0, 0.7, w)
    for i in range(40):
        c.start(); g.start()
        for k in ('obs', 'emph', 'masks', 'timeline'):
            assert torch.equal(getattr(g, k).cpu(), getattr(c, k)), (i, k)
        assert (g.state_dict()['env'] == c.state_dict()['env']).all(), i
        g.actions.zero_(); c.actions.zero_()
        g.step(); c.step()
        assert torch.equal(g.level_now.cpu(), c.level_now), i
        assert (c.level_now >= 0).any(), 'level_now never known'
    print('ok: GPU == CPU over 40 x', N, 'weighted restarts')


if __name__ == '__main__':
    rules()
    if len(sys.argv) > 2:
        if not Path(sys.argv[2]).exists():
            make_state(sys.argv[1], sys.argv[2])
        picks(sys.argv[2])
        if '--gpu' in sys.argv:
            gpu(sys.argv[2])
