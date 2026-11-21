"""Frozen-policy, entrance-only SMB trials. No restores after the entrance, search, or optimizer updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Policy, pack
from rlenv import RlEnv, END_NAMES

p=argparse.ArgumentParser()
p.add_argument('checkpoint');p.add_argument('level',type=int);p.add_argument('out')
p.add_argument('--n',type=int,default=64);p.add_argument('--seed',type=int,default=7)
p.add_argument('--greedy',action='store_true');p.add_argument('--cap',type=int,default=4000)
a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(a.seed)
path=Path(a.checkpoint);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
s=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
model=Policy(core=s['cfg']['core'],moods=s['cfg']['moods']).cuda().to(memory_format=torch.channels_last)
model.load_state_dict(s['model']);model.eval()
env=RlEnv({'smb':a.n},device='cuda',seed=a.seed,sticky=False,idle_max=0,dropout=True,level_cap=64,guide_discount=1)
cell_size=s['cfg'].get('grid_cell_size',32)
env.set_grid_attempts(a.cap,cell_size)
archive=np.load(path.parent/'entrances.npy')
state=env.state_dict();state['env'][-len(archive):]=archive;env.load_state_dict(state)
w=np.zeros(64);w[a.level]=1;env.set_level_plan(0,1,w);env.start()
assert bool(((env.cells>>(29 if cell_size==8 else 25))==a.level).all())
h=torch.zeros(a.n,model.core,device='cuda')
prev=[torch.zeros(a.n,dtype=torch.long,device='cuda') for _ in range(5)]
mood=torch.arange(a.n,device='cuda')%model.moods
active=torch.ones(a.n,dtype=torch.bool,device='cuda')
last_obs=env.obs[:,-1].clone();last_emph=env.emph[:,-1].clone()
trials=[];first_clear=False
with torch.no_grad():
    for t in range(a.cap):
        idx=active.nonzero().flatten()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model.encode(env.obs[idx],env.emph[idx],env.timeline[idx],[v[idx] for v in prev],mood[idx])
        h[idx]=model.gru(z.float(),h[idx])
        action,_=model.act(h[idx],env.masks[idx],greedy=a.greedy)
        prev=[torch.zeros_like(v).index_copy_(0,idx,part) for v,part in zip(prev,action)]
        valid=active&(env.cells>=0)
        last_obs[valid]=env.obs[valid,-1];last_emph[valid]=env.emph[valid,-1]
        env.actions.copy_(pack(prev));env.step()
        done=active&env.done.bool()
        for i in done.nonzero().flatten().tolist():
            end=END_NAMES[int(env.info[i,2])]
            trials.append(dict(env=i,mood=int(mood[i]),end=end,decisions=t+1))
            if end=='level_clear' and not first_clear:
                from PIL import Image
                from fceux_palette import RGB
                pix=last_obs[i].cpu().numpy().astype(np.int64)
                emph=last_emph[i].cpu().numpy().astype(np.int64)
                Image.fromarray(np.asarray(RGB,dtype=np.uint8)[pix|(emph[:,None]<<6)]).save(out/'before-clear.png')
                first_clear=True
        active &= ~done
        if t%200==0:
            print(json.dumps(dict(step=t+1,remaining=int(active.sum()),ends=dict(Counter(x['end'] for x in trials)))),flush=True)
        if not bool(active.any()):break
for i in active.nonzero().flatten().tolist():
    trials.append(dict(env=i,mood=int(mood[i]),end='cap',decisions=a.cap))
result=dict(level=a.level,checkpoint=str(path),iteration=s['it'],decisions=s['decisions'],seed=a.seed,n=a.n,
            mode='greedy' if a.greedy else 'sample',sticky=False,dropout=True,cap=a.cap,
            ends=dict(Counter(x['end'] for x in trials)),trials=trials)
(out/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='trials'}),flush=True)
