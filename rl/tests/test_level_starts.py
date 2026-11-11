"""Level starts (rlenv_core.cuh, DECISIONS.md 2026-11-11), CPU reference build:

1. a trained policy (greedy) plays Mario until the env reaches level 1 (1-2); the env saves that level start; the full
   env state right after that decision is kept (A);
2. the env then idles until its episode ends (stall), and with level-start probability 1 its next episode starts from
   the saved level start;
3. a second env loaded from A takes one neutral decision (= the 4 neutral frames of a restart) and then the same moves
   as the restored env for K decisions: pictures, rewards, reward parts, done and events must match exactly.
Also: power-on-only (probability 0) play is unchanged by having level starts enabled, and an old saved state without
level starts loads (empty archive).
Sticky actions, button dropout and idle frames are off so the two envs' episodes differ only in where they came from.

usage: python rl/tests/test_level_starts.py MODEL.pt   (an eval/itN/model.pt that reaches 1-2)
       python rl/tests/test_level_starts.py MODEL.pt --gpu   GPU build == CPU build with training settings (sticky,
       dropout, idle frames, level-start probability 0.5): every output, every saved level, for --steps decisions
"""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import Policy, pack  # noqa: E402
from rlenv import RlEnv  # noqa: E402

KW = dict(device='cpu', seed=5, sticky=False, idle_max=0, dropout=False)


def policy(path):
    s = torch.load(path, map_location='cpu', weights_only=False)
    m = Policy(core=s['cfg']['core'], moods=s['cfg'].get('moods', 0))
    m.load_state_dict(s['model'])
    return m.eval()


class Driver:
    def __init__(self, m):
        self.m, self.h, self.prev = m, torch.zeros(1, m.core), [torch.zeros(1, dtype=torch.long) for _ in range(5)]
        self.mood = torch.zeros(1, dtype=torch.long) if m.moods else None

    @torch.no_grad()
    def move(self, env):
        z = self.m.encode(env.obs, env.emph, env.timeline, self.prev, self.mood)
        self.h = self.m.gru(z.float(), self.h)
        parts, _ = self.m.act(self.h, env.masks, greedy=True)
        self.prev = parts
        return pack(parts)


def same(a, b, what):
    for k in ('obs', 'emph', 'reward', 'parts', 'done', 'events', 'masks'):
        if not torch.equal(getattr(a, k), getattr(b, k)):
            raise AssertionError(f'{what}: {k} differs')


def main(model_path):
    torch.set_num_threads(8)
    m = policy(model_path)

    # 0. probability 0: identical to an env without level starts
    a, b = RlEnv({'smb': 1}, level_cap=64, **KW), RlEnv({'smb': 1}, **KW)
    a.start(); b.start()
    d = Driver(m)
    for t in range(300):
        mv = d.move(a)
        a.actions.copy_(mv); b.actions.copy_(mv)
        a.step(); b.step()
        same(a, b, f'prob 0 step {t}')
    print('ok: level starts enabled with probability 0 play exactly as before')
    old = b.state_dict()
    a.load_state_dict(old)   # old layout (no level starts) -> prefix + empty archive
    assert a.saved_levels() == {'smb': {}}
    print('ok: a state saved without level starts loads with an empty archive')

    # 1. play until the first saved level start
    env = RlEnv({'smb': 1}, level_cap=64, **KW)
    env.start()
    d = Driver(m)
    for t in range(5000):
        env.actions.copy_(d.move(env))
        env.step()
        if env.done.any():
            d = Driver(m)
        if env.saved_levels()['smb']:
            break
    saved = env.saved_levels()['smb']
    assert saved, 'policy never reached a new level'
    lvl = next(iter(saved))
    assert int(env.level_now[0]) == lvl, (int(env.level_now[0]), lvl)
    A = env.state_dict()
    print(f'saved level start {lvl} (deaths {saved[lvl]}) after {t + 1} decisions')

    # 2. idle until the episode ends, then restart from the level start
    env.set_level_prob(1.0)
    for t in range(5000):
        env.actions.zero_()
        env.step()
        assert env.saved_levels()['smb'] == saved, 'archive changed while idling'
        if env.done.any():
            break
    assert env.done.all(), 'idle episode never ended'
    print(f'idle episode ended after {t + 1} decisions (start level {int(env.start_level[0])} -> next from level {lvl})')

    # 3. reference: A + one neutral decision, then the same moves
    ref = RlEnv({'smb': 1}, level_cap=64, **KW)
    ref.load_state_dict(A)
    ref.set_level_prob(1.0)
    ref.actions.zero_()
    ref.step()
    for k in ('obs', 'emph', 'masks', 'timeline'):
        assert torch.equal(getattr(env, k), getattr(ref, k)), f'first observation: {k} differs'
    d = Driver(m)
    K = 600
    for t in range(K):
        mv = d.move(env)
        env.actions.copy_(mv); ref.actions.copy_(mv)
        env.step(); ref.step()
        same(env, ref, f'restored vs continued, decision {t}')
        if env.done.any():
            assert int(env.start_level[0]) == lvl
            d = Driver(m)
    print(f'ok: an episode from the saved level start plays exactly like the original game continued ({K} decisions)')


def gpu_check(model_path, n=16, steps=8000):
    """Both builds, training settings, the same policy moves (computed from the CPU env's pictures)."""
    torch.set_num_threads(16)
    m = policy(model_path)
    kw = dict(seed=11, sticky=True, idle_max=30, dropout=True, level_cap=64)
    g, c = RlEnv({'smb': n}, device='cuda', **kw), RlEnv({'smb': n}, device='cpu', **kw)
    for e in (g, c):
        e.set_level_prob(0.5)
        e.start()
    h, prev = torch.zeros(n, m.core), [torch.zeros(n, dtype=torch.long) for _ in range(5)]
    mood = torch.zeros(n, dtype=torch.long) if m.moods else None
    gen = torch.Generator().manual_seed(3)
    lv_eps = 0
    for t in range(steps):
        with torch.no_grad():
            z = m.encode(c.obs, c.emph, c.timeline, prev, mood)
            h = m.gru(z.float(), h)
            parts, _ = m.act(h, c.masks, generator=gen)   # sampled, so the envs spread out
        mv = pack(parts)
        c.actions.copy_(mv); g.actions.copy_(mv.cuda())
        c.step(); g.step()
        for k in ('obs', 'emph', 'reward', 'parts', 'done', 'events', 'masks', 'timeline', 'pads'):
            if not torch.equal(getattr(g, k).cpu(), getattr(c, k)):
                raise AssertionError(f'step {t}: {k} differs')
        d = c.done.bool()
        if d.any():
            assert torch.equal(g.info.cpu()[d], c.info[d]) and torch.equal(g.start_level.cpu()[d], c.start_level[d])
            lv_eps += int((c.start_level[d] >= 0).sum())
            h[d] = 0
        prev = [p * (~d).long() for p in parts]
        if t % 50 == 0 and c.saved_levels()['smb']:   # once something is saved, every new episode starts from it
            for e in (g, c):
                e.set_level_prob(1.0)
        if lv_eps >= 2:
            break
        if t % 250 == 0 or t == steps - 1:
            sg, sc = g.saved_levels(), c.saved_levels()
            assert sg == sc, (t, sg, sc)
            print(f'step {t}: saved {sc["smb"]}, finished level-start episodes {lv_eps}', flush=True)
    sg, sc = g.saved_levels(), c.saved_levels()
    assert sg == sc, (sg, sc)
    assert lv_eps > 0, 'no level-start episode finished'
    a, b = g.state_dict()['env'], c.state_dict()['env']
    assert (a == b).all(), 'saved env states differ'
    print(f'ok: GPU == CPU for {t + 1} decisions x {n} envs with level starts ({lv_eps} level-start episodes)')


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        gpu_check(sys.argv[1])
    else:
        main(sys.argv[1])
