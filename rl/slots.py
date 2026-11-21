"""Saved-state slots of an RlEnv (level starts and places, rlenv_core.cuh) as host records, so states found in one env
(the route finder, rl/explore.py) can be kept in a file and written into another env's slots (the trainer).

A record is a dict of uint8 arrays [slots, bytes per slot]: mem (the console), st (reward tracker), keys, target, frame,
deaths. The slot arrays are the tail of the env's resume state (rlenv_build.hpp for_each_state), game after game.
"""
from __future__ import annotations

import numpy as np

FIELDS = ('mem', 'st', 'keys', 'target', 'frame', 'deaths')


def sizes(layout):
    """Bytes per slot of each field (layout: RlEnv.level_layout)."""
    return dict(mem=layout['stride'], st=layout['st'], keys=layout['key_cap'] * 4, target=8, frame=4, deaths=4)


def parse(tail, layout, cap):
    """One game's slot arrays (uint8, cap slots) -> record of views."""
    rec, o = {}, 0
    for k, b in sizes(layout).items():
        rec[k] = tail[o:o + cap * b].reshape(cap, b)
        o += cap * b
    assert o == len(tail), 'slot layout does not match'
    return rec


def _game_tail(env, buf, gi):
    lays = [env.level_layout(g) for g in range(len(env.games))]
    per = [env.level_cap * sum(sizes(l).values()) for l in lays]
    start = len(buf) - sum(per[gi:])
    return parse(buf[start:start + per[gi]], lays[gi], env.level_cap)


def read(env, gi, slots):
    """Record of game gi's slots (copies)."""
    rec = _game_tail(env, env.state_dict()['env'], gi)
    return {k: v[np.asarray(slots)].copy() for k, v in rec.items()}


def write(env, gi, slots, rec):
    """Write a record into game gi's slots (between decisions; the consoles themselves are untouched)."""
    sd = env.state_dict()
    dst = _game_tail(env, sd['env'], gi)
    for k in FIELDS:
        assert rec[k].shape == (len(slots), dst[k].shape[1]), f'{k}: saved with a different env build'
        dst[k][np.asarray(slots)] = rec[k]
    env.load_state_dict(sd)


def tail(env, gi, cap):
    """Game gi's first cap slots as one array (the entrance archive format read by load_entrances)."""
    rec = read(env, gi, np.arange(cap))
    return np.concatenate([rec[k].reshape(-1) for k in FIELDS])


def save_boot_entrance(env, gi, slot=0):
    """Level slot `slot` := the game's power-on start: the first level has no arrival to save. Restarts every env."""
    import torch
    env.set_level_plan(gi, 0.0)
    env.start()
    tmp = env.level_slots
    env.place_save(gi, torch.zeros(1, dtype=torch.int32, device=env.device),
                   torch.full((1,), tmp, dtype=torch.int32, device=env.device))
    write(env, gi, [slot], read(env, gi, [tmp]))


def load_entrances(env, gi, path):
    """An entrance archive (.npy: the slot arrays of an env with only level slots) -> this env's level slots."""
    tail = np.load(path)
    lay = env.level_layout(gi)
    cap = len(tail) // sum(sizes(lay).values())
    assert cap <= env.level_slots, 'archive has more level slots than this env'
    write(env, gi, np.arange(cap), parse(tail, lay, cap))
