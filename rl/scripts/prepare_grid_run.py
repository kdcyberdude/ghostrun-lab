"""Fresh entrance-only grid-reward run from policy weights and the source checkpoint's native entrance archive.

With --scratch, SOURCE is only an entrance .npy archive: no model checkpoint is read, all model parameters use
Policy's seeded initialization, and optimizer, value estimates and exploration histories are empty.

No policy evaluation/search is involved. Optimizer, value head, episode state, return statistics and novelty are
fresh because the objective changed. The regional experiment retains fine-cell familiarity and seeds region
counts with the max constituent-cell count, a conservative prior. The emulator state ABI is unchanged. CPU-only.
With --regrid, the current coverage model, optimizer, value statistics and step budget are preserved; only
entrance attempts and the grid's exploration memory restart because cells changed resolution.
"""
import argparse
import dataclasses
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import Config, atomic_torch_save
from model import Policy
from rlenv import RlEnv, GAMES
from grid_reward import GridReward


def prepare(source, out, level, n=4096, history_scale=None, regional=False, scratch=False, coverage=False,
            cell_size=32, regrid=False):
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    destination = out/'ckpt_rank0_latest.pt'
    if destination.exists():
        raise FileExistsError(destination)
    if coverage and regional:
        raise ValueError('coverage does not use horizontal regions')
    if regrid and (scratch or not coverage):
        raise ValueError('--regrid requires a coverage checkpoint, not --scratch')
    if scratch:
        if Path(source).suffix != '.npy':
            raise ValueError('--scratch takes an entrance archive (.npy), never a model checkpoint')
        s = None
    else:
        s = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        assert s['env']['games'] == ['smb'] and s['env'].get('level_cap') == 64
    cfg = Config(run=str(out), games='smb', envs_per_game=n, core=512 if scratch else s['cfg']['core'],
                 moods=8 if scratch else s['cfg']['moods'],
                 steps=64, gamma=1.0, grid_exploration=1, grid_regions=int(regional), grid_coverage=int(coverage),
                 grid_cell_size=cell_size,
                 grid_history_scale=n if history_scale is None else history_scale,
                 lam=.98, ret_beta=.98, minibatch_envs=32,
                 epochs=1, infer_chunk=1024, obs_slots=2, compile_mode='max-autotune-no-cudagraphs',
                 div_coef=0, int_coef=0, level_starts=1, places=0, only_levels=str(level),
                 episode_level_end=level, ckpt_minutes=15, eval_every_minutes=0,
                 total_iters=76 if scratch else 38 if regional else 20)
    if regrid:
        assert s['cfg'].get('grid_coverage') and s['cfg']['only_levels']==str(level)
        assert s['cfg']['envs_per_game']==n and s['cfg'].get('grid_cell_size',32)!=cell_size
        cfg=Config(**s['cfg'])
        cfg.run=str(out);cfg.grid_cell_size=cell_size
    e = RlEnv({'smb':n}, device='cpu', seed=cfg.seed*7919, sticky=bool(cfg.sticky), idle_max=cfg.idle_max,
              dropout=bool(cfg.dropout), level_cap=64, guide_discount=1)
    e.set_grid_attempts(cfg.attempt_limit, cfg.grid_cell_size)
    layout = e.level_layout(0)
    archive_bytes = 64*(layout['stride']+layout['st']+layout['key_cap']*4+16)
    archive = np.load(source).copy() if scratch else np.asarray(s['env']['env'][-archive_bytes:]).copy()
    assert archive.dtype == np.uint8 and archive.ndim == 1 and len(archive) == archive_bytes
    state = e.state_dict()
    state['env'][-archive_bytes:] = archive
    e.load_state_dict(state)
    assert level in e.saved_levels()['smb']
    weights = np.zeros(64); weights[level] = 1
    e.set_level_plan(0, 1.0, weights)
    e.start()
    assert bool(((e.cells >> (29 if cell_size==8 else 25)) == level).all()), e.cells
    np.save(out/'entrances.npy', archive)
    torch.manual_seed(cfg.seed*1000)
    model = Policy(core=cfg.core, moods=cfg.moods)
    if not scratch:
        model.load_state_dict(s['model'])
    if not regrid:
        torch.nn.init.zeros_(model.value.weight); torch.nn.init.zeros_(model.value.bias)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=cfg.adam_eps, fused=True)
    gen = torch.Generator().manual_seed(cfg.seed*1000+500009)
    grid = GridReward(n,'cpu',history_scale=cfg.grid_history_scale,regional=regional,coverage=coverage,cell_size=cell_size)
    if regional and s is not None and s.get('grid') is not None:
        grid.known = s['grid']['known'].clone()
        grid.count = s['grid']['count'].clone()
        grid.seed_region_history()
    state = dict(cfg=dataclasses.asdict(cfg), model=model.state_dict(), opt=opt.state_dict(), scaler={},
                 ret_mean=torch.zeros(len(GAMES),dtype=torch.float64), ret_sq=torch.ones(len(GAMES),dtype=torch.float64),
                 ret_seen=torch.zeros(len(GAMES),dtype=torch.float64), it=0, decisions=0, elapsed=0,
                 h=torch.zeros(n,cfg.core), env=e.state_dict(), obs=e.obs, emph=e.emph, masks=e.masks,
                 tl=e.timeline, prev=torch.zeros(5,n,dtype=torch.long), start=torch.ones(n,dtype=torch.bool),
                 cells=e.cells, grid=grid.state_dict(),
                 rng_cpu=torch.get_rng_state(), rng_roll=gen.get_state(), rng_cuda=None,
                 div=None, rnd=None, places=None, level_plan2=None)
    if regrid:
        # Preserve all learned parameters and optimizer statistics. The old 32px visits cannot identify
        # which 8px children were visited, so only environment attempts and exploration memory start fresh.
        for key in ('opt','scaler','ret_mean','ret_sq','ret_seen','it','decisions','elapsed','level_plan2',
                    'rng_cpu','rng_roll','rng_cuda'):
            state[key]=s[key]
    # CUDA generators use a different representation. Trainer seeds its fresh CUDA streams on this initial load.
    state['fresh_grid_run'] = not regrid
    atomic_torch_save(state,destination)
    (out/'source.json').write_text(json.dumps(dict(source=str(source), source_iteration=None if scratch else s['it'],
                                                  initialization='continued current run; new grid/entrances' if regrid else
                                                                 'random' if scratch else 'pretrained actor',
                                                  initialization_seed=cfg.seed*1000, level=level,
                                                  archive_layout=layout, cfg=dataclasses.asdict(cfg),
                                                  regional_history_prior='max constituent-cell count' if regional and not scratch else None),indent=2))
    print(json.dumps(dict(checkpoint=str(destination),level=level,envs=n,archive_bytes=archive_bytes)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('source'); p.add_argument('out'); p.add_argument('level',type=int)
    p.add_argument('--envs',type=int,default=4096)
    p.add_argument('--history-scale',type=float)
    p.add_argument('--regions',action='store_true')
    p.add_argument('--scratch',action='store_true',help='random policy and empty histories; SOURCE must be entrance .npy')
    p.add_argument('--coverage',action='store_true',help='persistent direction-neutral coverage reward')
    p.add_argument('--cell-size',type=int,choices=[8,32],default=32)
    p.add_argument('--regrid',action='store_true',help='retain model/optimizer/step count; restart attempts with a different grid')
    a=p.parse_args(); prepare(a.source,a.out,a.level,a.envs,a.history_scale,a.regions,a.scratch,a.coverage,a.cell_size,a.regrid)
