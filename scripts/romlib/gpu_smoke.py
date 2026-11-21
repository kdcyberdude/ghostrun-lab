"""GPU smoke test of arbitrary ROMs through the real training env (rl/rlenv.py, CUDA build) against its CPU build.

The env slot 'smb' is borrowed only to load the ROM (its reward reads meaningless RAM here; resets happen the same way
on both builds). Boot = 120 idle frames (a pads file in RL_BOOT_DIR). Both builds get the same seeded random legal
moves; the 2 KB RAM after every decision must be identical and no env may fault.

usage: NESLE_INC=... NESLE_ROM_CPP=... RL_BUILD=... RL_BOOT_DIR=... python gpu_smoke.py ROM... [--envs 32] [--steps 200]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rl'))


def run(rom, device, envs, steps, seed=7):
    import rlenv
    from actions import random_legal
    boot = Path(os.environ['RL_BOOT_DIR'])
    boot.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(Path(rom).read_bytes()).hexdigest()[:16]
    (boot / f'smb.{key}.pads.bin').write_bytes(bytes(120))
    env = rlenv.RlEnv({'smb': envs}, device=device, sticky=False, dropout=False, roms={'smb': rom})
    env.bind_ram(range(2048))
    gen = torch.Generator().manual_seed(seed)
    rams = []
    env.step()                       # the legal-move masks are filled by the first step
    for _ in range(steps):
        env.actions.copy_(random_legal(env.masks.cpu(), gen).to(env.actions.device))
        env.step()
        rams.append(env.ram.cpu().clone())
    return torch.stack(rams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('roms', nargs='+')
    ap.add_argument('--envs', type=int, default=32)
    ap.add_argument('--steps', type=int, default=200)
    a = ap.parse_args()
    for rom in a.roms:
        try:
            g = run(rom, 'cuda', a.envs, a.steps)
            c = run(rom, 'cpu', a.envs, a.steps)
            same = (g == c).all(dim=2)                 # [steps, envs]
            moved = int((g[-1] != g[0]).any(dim=1).sum())
            print(json.dumps(dict(rom=Path(rom).name, identical=float(same.float().mean()),
                                  first_diff_step=int((~same.all(1)).nonzero()[0]) if not same.all() else None,
                                  envs_ram_changed=moved)), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps(dict(rom=Path(rom).name, error=repr(e)[:300])), flush=True)


if __name__ == '__main__':
    main()
