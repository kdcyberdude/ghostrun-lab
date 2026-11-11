"""GPU gates + speed measurement (needs a GPU; run on Kaggle or with the user's OK locally).

1. GPU env == CPU reference build, bit for bit, every output of every step (pictures, emphasis, reward, done, masks,
   dropout timeline, pad bytes, events, episode info), all 12 games, sticky + idle + dropout on, same actions.
   (The CPU build itself equals NeSLE's verified harness + Python reward10: rl/tests/test_env_cpu.py.)
2. GPU env exact resume: save state, run K steps, load, run the same K steps -> identical outputs.
3. Speed: env alone (random legal moves drawn on the GPU) for several env counts; policy forward at that batch.
Writes OUT/gpu_check.json.   usage: python rl/tests/gpu_check.py OUT [--steps 300] [--per-game 4]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlenv import RlEnv, GAMES  # noqa: E402
from actions import random_legal  # noqa: E402
from model import Policy, count_params  # noqa: E402

OUTS = ['obs', 'emph', 'reward', 'done', 'masks', 'timeline', 'pads', 'events', 'parts']


def equality(steps, per_game, seed=11):
    kw = dict(seed=seed, sticky=True, idle_max=30, dropout=True)
    g_env = RlEnv({g: per_game for g in GAMES}, device='cuda', **kw)
    c_env = RlEnv({g: per_game for g in GAMES}, device='cpu', **kw)
    g_env.start(); c_env.start()
    gen = torch.Generator().manual_seed(seed)
    first_bad, dones = None, 0
    t0 = time.time()
    for t in range(steps + 1):
        for k in OUTS:
            a, b = getattr(g_env, k).cpu(), getattr(c_env, k)
            if not torch.equal(a, b):
                first_bad = first_bad or dict(step=t, output=k, envs=int((a != b).reshape(len(a), -1).any(1).sum()))
        d = c_env.done.bool()
        if t and d.any():
            dones += int(d.sum())
            if not torch.equal(g_env.info.cpu()[d], c_env.info[d]):
                first_bad = first_bad or dict(step=t, output='info')
        if first_bad or t == steps:
            break
        act = random_legal(c_env.masks, gen)
        c_env.actions.copy_(act); g_env.actions.copy_(act.cuda())
        c_env.step(); g_env.step()
    torch.cuda.synchronize()
    return dict(identical=first_bad is None, first_difference=first_bad, steps=steps, envs=g_env.n,
                finished_episodes=dones, seconds=round(time.time() - t0, 1))


def resume(per_game, k=40, seed=3):
    env = RlEnv({g: per_game for g in GAMES}, device='cuda', seed=seed)
    env.start()
    gen = torch.Generator(device='cuda').manual_seed(seed)
    for _ in range(60):
        env.actions.copy_(random_legal(env.masks, gen)); env.step()
    sd = env.state_dict()
    gstate = gen.get_state()
    rec = []
    for _ in range(k):
        env.actions.copy_(random_legal(env.masks, gen)); env.step()
        rec.append([getattr(env, o).clone() for o in OUTS])
    env.load_state_dict(sd)
    gen.set_state(gstate)
    same = True
    for i in range(k):
        env.actions.copy_(random_legal(env.masks, gen)); env.step()
        same &= all(torch.equal(getattr(env, o), r) for o, r in zip(OUTS, rec[i]))
    return dict(identical=bool(same), steps_after_load=k, state_mb=round(len(sd['env']) / 2**20, 1))


def speed(per_game_list, steps=64):
    rows = []
    for pg in per_game_list:
        env = RlEnv({g: pg for g in GAMES}, device='cuda', seed=1)
        env.start()
        gen = torch.Generator(device='cuda').manual_seed(0)
        for _ in range(8):
            env.actions.copy_(random_legal(env.masks, gen)); env.step()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(steps):
            env.actions.copy_(random_legal(env.masks, gen)); env.step()
        torch.cuda.synchronize()
        dt = time.time() - t
        rows.append(dict(envs=env.n, decisions_per_s=round(steps * env.n / dt), frames_per_s=round(4 * steps * env.n / dt),
                         ms_per_step=round(1000 * dt / steps, 1)))
        print('env speed', rows[-1], flush=True)
        del env
        torch.cuda.empty_cache()
    return rows


def model_flops():
    """FLOPs of one sample's encoder forward (everything before the GRU; the GRU and heads are <1%)."""
    from torch.utils.flop_counter import FlopCounterMode
    m = Policy()
    obs = torch.zeros(1, 4, 224, 240, dtype=torch.uint8)
    with FlopCounterMode(display=False) as fc:
        m.encode(obs, torch.zeros(1, 4, 224, dtype=torch.uint8), torch.zeros(1, 8, 45, dtype=torch.uint8),
                 [torch.zeros(1, dtype=torch.long)] * 5)
    return fc.get_total_flops()


def cpu_env_speed(per_game=8, steps=32):
    import os
    env = RlEnv({g: per_game for g in GAMES}, device='cpu', seed=1)
    env.start()
    gen = torch.Generator().manual_seed(0)
    t = time.time()
    for _ in range(steps):
        env.actions.copy_(random_legal(env.masks, gen)); env.step()
    dt = time.time() - t
    return dict(envs=env.n, threads=os.cpu_count(), decisions_per_s=round(steps * env.n / dt))


def model_speed(batches, dtype):
    m = Policy().cuda().to(memory_format=torch.channels_last)
    rows = []
    for B in batches:
        obs = torch.randint(0, 64, (B, 4, 224, 240), dtype=torch.uint8, device='cuda')
        emph = torch.zeros(B, 4, 224, dtype=torch.uint8, device='cuda')
        tl = torch.zeros(B, 8, 45, dtype=torch.uint8, device='cuda')
        prev = [torch.zeros(B, dtype=torch.long, device='cuda') for _ in range(5)]
        for train in (False, True):
            def once():
                with torch.autocast('cuda', dtype=dtype):
                    z = m.encode(obs, emph, tl, prev)
                if train:
                    z.float().sum().backward()
            with torch.set_grad_enabled(train):
                for _ in range(3):
                    once()
                torch.cuda.synchronize()
                t = time.time()
                for _ in range(10):
                    once()
                torch.cuda.synchronize()
            dt = (time.time() - t) / 10
            rows.append(dict(batch=B, train=train, samples_per_s=round(B / dt),
                             mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)))
            print('model speed', rows[-1], flush=True)
    return dict(params_m=round(count_params(m) / 1e6, 2), rows=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--per-game', type=int, default=4)
    ap.add_argument('--speed-per-game', default='8,16,32,64')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=torch.__version__)
    res['model_fwd_gflop_per_sample'] = round(model_flops() / 1e9, 3)
    print('model flops', res['model_fwd_gflop_per_sample'], flush=True)
    res['equality'] = equality(a.steps, a.per_game)
    print('equality', res['equality'], flush=True)
    res['resume'] = resume(2)
    print('resume', res['resume'], flush=True)
    res['env_speed'] = speed([int(x) for x in a.speed_per_game.split(',')])
    res['cpu_env_speed'] = cpu_env_speed()
    print('cpu env speed', res['cpu_env_speed'], flush=True)
    bf16 = torch.cuda.get_device_capability()[0] >= 8
    res['model_speed'] = model_speed([64, 256], torch.bfloat16 if bf16 else torch.float16)
    (out / 'gpu_check.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
