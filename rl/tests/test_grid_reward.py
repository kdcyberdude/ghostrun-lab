"""Grid novelty's reward accounting, reset boundaries, exact resume, and native coordinate/terminal semantics."""
import copy
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rl'))
from grid_reward import GridReward
from rlenv import RlEnv, NESLE_INC


def apply(g, cells, done=None, mood=7):
    cells = torch.tensor(cells, dtype=torch.int64).view(-1, g.n)
    t = len(cells) - 1
    done = torch.zeros(t, g.n, dtype=torch.bool) if done is None else torch.tensor(done, dtype=torch.bool).view(t, g.n)
    info = torch.zeros(t, g.n, 8)
    r, stats = g.apply(cells, done, torch.full((t, g.n), -0.0001), torch.full((t, g.n), mood), 8, info)
    return r, info, stats


def test_first_visit_only_entrance_free_and_memory_crosses_rollouts():
    g = GridReward(1, 'cpu')
    r, _, _ = apply(g, [10, 10, 11, 10, 11, 12])
    assert torch.allclose(r.flatten() + .0001, torch.tensor([0, .005, 0, 0, .005]))
    r, _, _ = apply(g, [12, 11, 13])
    assert torch.allclose(r.flatten() + .0001, torch.tensor([0, .005]))


def test_death_seeds_new_entrance_without_bonus_and_familiarity_decays():
    g = GridReward(1, 'cpu')
    apply(g, [10, 11, 10], [False, True])
    r, _, _ = apply(g, [10, 10, 11])
    assert r[0].item() == pytest.approx(-.0001)
    assert r[1].item() + .0001 == pytest.approx(.005 / 2**.5)


def test_cap_is_reward_not_cell_count_and_resets_after_terminal():
    g = GridReward(1, 'cpu', cap=.007)
    r, info, _ = apply(g, [10, 11, 12, 13, 10, 14], [False, False, False, True, False])
    assert torch.allclose(r.flatten() + .0001, torch.tensor([.005, .002, 0, 0, .005]), atol=1e-8)
    assert info[3, 0, 0].item() == pytest.approx(.007 - .0004)
    assert g.spent.item() == pytest.approx(.005)
    # Capped cells remain visited, rather than paying later if a budget were increased.
    assert (g.known == 13).any()


def test_fast_and_balanced_moods_and_invalid_observations():
    for mood, scale in [(0, 0), (2, .25), (7, 1)]:
        g = GridReward(1, 'cpu')
        g.finished[0]=100;g.total_finished[0]=100;g.wins[0]=50
        r, _, _ = apply(g, [10, -1, 11, 12], mood=mood)
        assert torch.allclose(r.flatten() + .0001, torch.tensor([0, .005*scale, .005*scale]))


def test_all_moods_explore_before_measured_mastery():
    for mood in (0,2,7):
        g=GridReward(1,'cpu')
        g.finished[0]=100;g.total_finished[0]=100;g.wins[0]=49
        r,_,_=apply(g,[10,11],mood=mood)
        assert r.item()+.0001 == pytest.approx(.005)
    g=GridReward(1,'cpu');g.finished[0]=2;g.total_finished[0]=2;g.wins[0]=2
    r,_,_=apply(g,[10,11],mood=0)
    assert r.item()+.0001 == pytest.approx(.005)  # two lucky wins do not enable speed


def test_mastery_is_per_level_uses_native_outcomes_and_can_switch_back():
    g=GridReward(1,'cpu')
    cell=(15<<25)+10
    cells=torch.full((101,1),cell,dtype=torch.long)
    done=torch.ones(100,1,dtype=torch.bool)
    info=torch.zeros(100,1,8);info[:,:,4]=15
    info[:50,:,2]=6;info[50:,:,2]=7
    g.apply(cells,done,torch.zeros(100,1),torch.zeros(100,1,dtype=torch.long),8,info)
    assert g.total_finished[15] == 100 and g.wins[15] == 50
    r,_,_=apply(g,[cell,cell+1],mood=0)
    assert r.item() == pytest.approx(-.0001)
    # Another level remains exploratory.
    other=(27<<25)+10
    r,_,_=apply(g,[other,other+1],mood=0)
    assert r.item()+.0001 == pytest.approx(.005)
    g.wins[15]=49
    r,_,_=apply(g,[cell,cell+2],mood=0)
    assert r.item()+.0001 == pytest.approx(.005)


def test_stationary_grace_charge_cap_and_reset_with_backward_movement():
    g=GridReward(1,'cpu')
    r,_,stats=apply(g,[40]*37)
    assert torch.allclose(r[:30],torch.full((30,1),-.0001))
    assert torch.allclose(r[30:],torch.full((6,1),-.0011))
    assert stats['stationary_penalty'] == pytest.approx(.006)
    # Moving backward is allowed: a changed spatial cell resets the clock.
    r,_,_=apply(g,[40,32,32])
    assert g.idle_steps.item() == 1
    apply(g,[32]*400)
    assert g.idle_spent.item() == pytest.approx(.25,abs=1e-6)
    apply(g,[32,40],[True])
    assert g.idle_steps.item() == 0 and g.idle_spent.item() == 0


def test_jumping_in_place_does_not_evade_stationary_penalty():
    g=GridReward(1,'cpu')
    # Same x column, changing height cells.
    r,_,stats=apply(g,[40]+[40+i%4 for i in range(40)])
    assert stats['stationary_penalty'] == pytest.approx(.010,abs=1e-7)


def test_regional_bonus_prefers_new_regions_and_cannot_pay_repeated_boundary_crossings():
    g=GridReward(1,'cpu',regional=True)
    # Fine x cells 32..35 share one region; 36 is the next 128-pixel region. All have the same y.
    c=lambda x,y=4: x*8+y
    r,_,stats=apply(g,[c(32),c(33),c(34),c(35),c(36),c(35),c(36)])
    assert torch.allclose(r.flatten()+.0001,torch.tensor([.001,.001,.001,.021,0,0]),atol=1e-8)
    assert stats['new_region_visits']==1
    assert stats['regional_reward']==pytest.approx(.02)


def test_regions_preserve_corridors_and_native_phase():
    g=GridReward(1,'cpu',regional=True)
    a=(15<<25)+(32*8)+4  # y 128..159
    b=(15<<25)+(32*8)+5  # y 160..191, including the lower corridor
    c=b+(1<<12)
    assert len(torch.unique(g.regions(torch.tensor([a,b,c]))))==3
    r,_,_=apply(g,[a,b,c])
    assert torch.allclose(r+.0001,torch.full((2,1),.021))


def test_region_familiarity_decays_faster_and_unvisited_region_keeps_full_bonus():
    g=GridReward(1,'cpu',regional=True)
    a,b,c=32*8,36*8,40*8
    apply(g,[a,b,a],[False,True])
    r,_,_=apply(g,[a,b,c])
    assert r[0].item()+.0001==pytest.approx(.001/2**.5+.02/2)
    assert r[1].item()+.0001==pytest.approx(.021)


def test_fine_and_regional_rewards_share_one_cap_and_resume_exactly():
    g=GridReward(1,'cpu',regional=True,cap=.025)
    apply(g,[256,288])
    clone=GridReward(1,'cpu',regional=True,cap=.025)
    clone.load_state_dict(copy.deepcopy(g.state_dict()))
    a=apply(g,[288,320,352]);b=apply(clone,[288,320,352])
    assert torch.equal(a[0],b[0])
    assert torch.allclose(a[0].flatten()+.0001,torch.tensor([.004,0]),atol=1e-8)
    for key in g.state_dict():assert torch.equal(g.state_dict()[key],clone.state_dict()[key])


def test_seeded_region_history_is_a_conservative_count_prior():
    g=GridReward(1,'cpu',regional=True)
    g.known=torch.tensor([256,264,288]);g.count=torch.tensor([5,9,2])
    g.seed_region_history()
    assert g.region_known.tolist()==[64,72]
    assert g.region_count.tolist()==[9,2]


def test_stationary_clock_and_mastery_survive_checkpoint():
    g=GridReward(1,'cpu');apply(g,[10]*31)
    g.total_finished[15]=100;g.finished[15]=100;g.wins[15]=60
    resumed=GridReward(1,'cpu');resumed.load_state_dict(copy.deepcopy(g.state_dict()))
    a=apply(g,[10]*6);b=apply(resumed,[10]*6)
    assert torch.equal(a[0],b[0])
    assert resumed.wins[15] == 60 and resumed.total_finished[15] == 100


def test_checkpoint_resume_preserves_budget_visits_and_returns():
    g = GridReward(1, 'cpu', cap=.009)
    apply(g, [10, 11, 12])
    resumed = GridReward(1, 'cpu', cap=.009)
    resumed.load_state_dict(copy.deepcopy(g.state_dict()))
    a = apply(g, [12, 11, 13, 10], [False, False, True])
    b = apply(resumed, [12, 11, 13, 10], [False, False, True])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for key in g.state_dict():
        assert torch.equal(g.state_dict()[key], resumed.state_dict()[key])


def test_parallel_envs_have_independent_episode_visits_and_shared_familiarity():
    g = GridReward(2, 'cpu')
    r, _, _ = apply(g, [[10,10], [11,11], [10,11]], [[False,False], [True,False]])
    assert torch.allclose(r[0] + .0001, torch.full((2,), .005))
    assert g.count[g.known == 11].item() == 2
    r, _, _ = apply(g, [[10,11], [11,11]])
    assert r[0,0].item() + .0001 == pytest.approx(.005 / 3**.5)
    assert r[0,1].item() == pytest.approx(-.0001)


@pytest.mark.parametrize('regional',[False,True])
def test_familiarity_decay_can_be_invariant_to_parallel_env_count(regional):
    # Duplicating a worker must not make its exploration incentive disappear sooner in training.
    one = GridReward(1, 'cpu', history_scale=1,regional=regional)
    many = GridReward(64, 'cpu', history_scale=64,regional=regional)
    apply(one, [10,11,10], [False,True])
    apply(many, [[10]*64,[11]*64,[10]*64], [[False]*64,[True]*64])
    a,_,_ = apply(one,[10,11])
    b,_,_ = apply(many,[[10]*64,[11]*64])
    assert torch.allclose(b,a.expand_as(b))


def test_native_check_entry_metrics_count_transitions_not_observation_frames():
    g=GridReward(1,'cpu')
    a=(15<<25)+258;b=a+(1<<12);c=a+(2<<12)
    _,_,stats=apply(g,[a,b,b,c,c,c])
    assert stats['maze_check1_entries']==1
    assert stats['maze_check2_entries']==1


def test_coverage_keeps_an_episodic_floor_after_extreme_lifetime_familiarity():
    g=GridReward(1,'cpu',coverage=True)
    g.known=torch.tensor([288]);g.count=torch.tensor([10**12])
    r,_,_=apply(g,[256,288,256,288])
    # The .02 raw floor survives; F(.02) is the amount actually paid at the start of an attempt.
    assert r[0].item()+.0001 == pytest.approx(4*.02/4.02,rel=2e-6)
    assert torch.allclose(r[1:],torch.full((2,1),-.0001))
    fresh=GridReward(1,'cpu',coverage=True)
    s,_,_=apply(fresh,[256,288])
    assert s[0] > r[0]  # unfamiliar cells still get a larger bonus


def test_coverage_treats_all_directions_equally_and_allows_short_backtracking():
    for delta in (-8,8,-1,1):  # left, right, up, down in the packed grid
        g=GridReward(1,'cpu',coverage=True)
        r,_,_=apply(g,[260,260+delta,260,260+2*delta])
        assert r[0].item()+.0001 == pytest.approx(4*.04/4.04)
        assert r[1].item() == pytest.approx(-.0001)
        assert r[2] > 0 and g.no_discovery_steps.item()==0


def test_coverage_jumping_circling_and_invalid_cells_cannot_reset_stagnation():
    g=GridReward(1,'cpu',coverage=True)
    apply(g,[256,257,264,265])
    r,_,stats=apply(g,[265]+[256,257,264,265,-1]*100)
    assert torch.allclose(r[:60],torch.full((60,1),-.0001))
    assert torch.allclose(r[60:],torch.full((440,1),-.0011))
    assert stats['stagnation_penalty']==pytest.approx(.44,abs=3e-6)  # continues beyond the old .25 cap
    assert g.no_discovery_steps.item()==500
    r,_,_=apply(g,[-1,272,265])
    assert r[0] > 0 and r[1].item()==pytest.approx(-.0001)
    assert g.no_discovery_steps.item()==1


def test_coverage_has_no_hard_cutoff_and_total_stays_below_win():
    g=GridReward(1,'cpu',coverage=True)
    r,_,stats=apply(g,list(range(1000)))
    bonuses=r+.0001
    assert bool((bonuses>0).all())
    assert 3.5 < bonuses.sum().item() < 4
    assert stats['stagnation_penalty']==0
    # Even a 4,000-decision win with no novelty and maximum stagnation costs beats any failure's bonuses.
    assert 10 - 4000*(.0001+.001) > 4


def test_coverage_timeout_costs_as_much_as_death_and_terminal_reset_earns_nothing():
    for end,base in [(7,-.5001),(8,-.0001)]:
        g=GridReward(1,'cpu',coverage=True)
        apply(g,[256,264]);apply(g,[264]*100)
        info=torch.zeros(1,1,8);info[...,2]=end
        reward,_=g.apply(torch.tensor([[264],[288]]),torch.ones(1,1,dtype=torch.bool),
                         torch.tensor([[base]]),torch.zeros(1,1,dtype=torch.long),8,info)
        assert reward.item()==pytest.approx(-.5001)
        assert g.no_discovery_steps.item()==0 and g.coverage_mass.item()==0
        r,_,_=apply(g,[288,288,296])
        assert r[0].item()==pytest.approx(-.0001) and r[1]>0


def test_coverage_checkpoint_and_rollout_splits_preserve_accounting():
    g=GridReward(1,'cpu',coverage=True)
    apply(g,[256,264,265]);apply(g,[265]*58)
    resumed=GridReward(1,'cpu',coverage=True)
    resumed.load_state_dict(copy.deepcopy(g.state_dict()))
    a=apply(g,[265]*10+[272]);b=apply(resumed,[265]*10+[272])
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    for k in g.state_dict():assert torch.equal(g.state_dict()[k],resumed.state_dict()[k])
    # Partitioning the same single-attempt path at rollout boundaries must preserve its total return.
    whole=GridReward(1,'cpu',coverage=True)
    total=apply(whole,[256,264,265]+[265]*57+[265]*9+[272])[0].sum()
    assert g.returns.item()==pytest.approx(total.item(),abs=5e-7)


def test_coverage_ranks_exploring_over_repeating_and_faster_success_over_detours():
    def score(path,end=7):
        g=GridReward(1,'cpu',coverage=True)
        cells=torch.tensor(path+[path[0]]).view(-1,1)
        done=torch.zeros(len(path),1,dtype=torch.bool);done[-1]=True
        base=torch.full(done.shape,-.0001);base[-1]+=10 if end==6 else -.5
        info=torch.zeros(len(path),1,8);info[-1,:,2]=end
        r,_=g.apply(cells,done,base,torch.zeros_like(done,dtype=torch.long),8,info)
        return r.sum().item()
    assert score(list(range(256,356))) > score([256,257]*50)
    assert score(list(range(256,356))) > score([256,257])  # exploration+death beats an immediate small farm+death
    assert score([256,257,258],6) > score([256,257]*50+[258],6)
    assert score([256,257,258],6) > score(list(range(256,1000)))


@pytest.mark.parametrize('coverage,cell_size,quadratic,persistent',[
    (False,32,False,False),(True,32,False,False),(True,8,False,False),(True,8,True,False),(True,8,True,True)])
def test_grid_reward_cpu_gpu_parity_with_resets_stagnation_and_resume(coverage,cell_size,quadratic,persistent):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    n,t=4,160
    cells=torch.full((t+1,n),256,dtype=torch.long)
    cells[1:,1]=torch.arange(t)%2+256
    cells[1:,2]=torch.arange(t)+257
    cells[20:25,3]=-1
    done=torch.zeros(t,n,dtype=torch.bool);done[90,3]=True
    base=torch.full((t,n),-.0001);base[90,3]=-.5001
    info=torch.zeros(t,n,8);info[90,3,2]=7
    mood=torch.zeros(t,n,dtype=torch.long)
    if cell_size==8:
        cells=torch.where(cells>=0,cells+(15<<29),cells)
        info[...,4]=15
    grids=[GridReward(n,d,history_scale=n,coverage=coverage,cell_size=cell_size,
                      quadratic_stagnation=quadratic,persistent_visits=persistent) for d in ['cpu','cuda']]
    outputs=[]
    for g in grids:
        args=[a.to(g.device) for a in (cells,done,base,mood)]
        inf=info.to(g.device).clone()
        reward,_=g.apply(*args,8,inf)
        outputs.append((reward.cpu(),inf.cpu()))
    for a,b in zip(*outputs):assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    for k in grids[0].state_dict():
        assert torch.allclose(grids[0].state_dict()[k],grids[1].state_dict()[k].cpu(),atol=1e-6,rtol=1e-6),k


def test_persistent_cells_do_not_regain_bonus_after_death_but_other_level_is_distinct():
    g=GridReward(1,'cpu',coverage=True,cell_size=8,quadratic_stagnation=True,persistent_visits=True)
    a=(15<<29)+256;b=a+32
    r,_,_=apply(g,[a,b,a],[False,True])
    assert r[0]>0 and g.no_discovery_steps.item()==0
    r,_,_=apply(g,[a,b,a,b])
    assert torch.allclose(r,torch.full((3,1),-.0001))
    other=(27<<29)+256
    r,_,_=apply(g,[b,other])
    assert r[0]>0  # identical coordinates on another level are new


def test_persistent_discovery_cannot_pay_again_after_death_inside_one_rollout():
    g=GridReward(1,'cpu',coverage=True,cell_size=8,persistent_visits=True)
    r,_,_=apply(g,[256,288,256,288],[False,True,False])
    assert r[0]>0 and r[2].item()==pytest.approx(-.0001)


def test_persistent_shared_history_resumes_and_seeds_from_existing_checkpoint():
    old=GridReward(2,'cpu',coverage=True,cell_size=8)
    apply(old,[[256,256],[288,256],[256,256]],[[False,False],[True,True]])
    g=GridReward(2,'cpu',coverage=True,cell_size=8,persistent_visits=True)
    g.load_state_dict(copy.deepcopy(old.state_dict()))
    # A cell previously seen by worker zero must not reward worker one after a restart.
    r,_,_=apply(g,[[256,256],[288,288],[320,320]])
    assert bool((r[0]==-.0001).all()) and bool((r[1]>0).all())
    clone=GridReward(2,'cpu',coverage=True,cell_size=8,persistent_visits=True)
    clone.load_state_dict(copy.deepcopy(g.state_dict()))
    a=apply(g,[[320,320],[256,256],[320,320]])
    b=apply(clone,[[320,320],[256,256],[320,320]])
    assert torch.equal(a[0],b[0]) and bool((a[0]==-.0001).all())


def test_quadratic_stagnation_timing_repeats_resume_and_discovery_reset():
    g=GridReward(1,'cpu',coverage=True,cell_size=8,quadratic_stagnation=True)
    r,_,_=apply(g,[256]*17)
    assert torch.allclose(r[:8],torch.full((8,1),-.0001))
    assert r[8].item()==pytest.approx(-.0001-.001/64)
    assert r[11].item()==pytest.approx(-.0001-.00025)
    assert r[15].item()==pytest.approx(-.0011)
    resumed=GridReward(1,'cpu',coverage=True,cell_size=8,quadratic_stagnation=True)
    resumed.load_state_dict(copy.deepcopy(g.state_dict()))
    a=apply(g,[256]*5+[257]);b=apply(resumed,[256]*5+[257])
    assert torch.equal(a[0],b[0]) and bool((a[0][:-1]==-.0011).all())
    assert a[0][-1]>0 and g.no_discovery_steps.item()==0
    r,_,_=apply(g,[257,256]*9)
    assert r[-1].item()==pytest.approx(-.0011)  # crossing familiar cell boundaries cannot evade the ramp
    apply(g,[256,256],[True])
    assert g.no_discovery_steps.item()==0


def test_8px_grid_preserves_reward_per_distance_and_revisit_memory():
    coarse=GridReward(1,'cpu',coverage=True)
    fine=GridReward(1,'cpu',coverage=True,cell_size=8)
    a=(15<<25)+(33<<3)+2
    b=(15<<29)+(132<<5)+8
    r,_,_=apply(coarse,[a,a+8])
    s,_,stats=apply(fine,[b,b+32,b+64,b+96,b+128])
    assert (s+.0001).sum().item()==pytest.approx((r+.0001).sum().item(),abs=1e-7)
    assert stats['cell_size']==8 and stats['max_x']==64
    assert fine.known.numel()==5 and fine.count.tolist()==[1]*5
    clone=GridReward(1,'cpu',coverage=True,cell_size=8)
    clone.load_state_dict(copy.deepcopy(fine.state_dict()))
    repeat,_,_=apply(clone,[b+128,b,b+32,b+64,b+96])
    assert torch.allclose(repeat,torch.full((4,1),-.0001))


def test_8px_keys_preserve_level_mastery_and_native_maze_metrics():
    g=GridReward(1,'cpu',coverage=True,cell_size=8)
    a=(15<<29)+(481<<20)+(132<<5)+8
    b=a+(1<<16);c=a+(2<<16)
    _,_,stats=apply(g,[a,b,c])
    assert stats['maze_check1_entries']==stats['maze_check2_entries']==1
    assert len(stats['height_cells'])==32 and stats['max_x']==32
    g.total_finished[15]=100;g.finished[15]=100;g.wins[15]=50
    r,_,_=apply(g,[c,c+1],mood=0)
    assert r.item()==pytest.approx(-.0001)


def test_scratch_initializer_rejects_model_checkpoint_without_reading_it(tmp_path,monkeypatch):
    sys.path.insert(0,str(ROOT/'rl/scripts'))
    from prepare_grid_run import prepare
    def forbidden(*args,**kwargs):
        raise AssertionError('scratch initializer must not load a model checkpoint')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='entrance archive'):
        prepare('old-policy.pt',tmp_path/'scratch',15,n=1,regional=True,scratch=True)


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    d = tmp_path_factory.mktemp('grid-native')
    src, so = d/'test.cpp', d/'test.so'
    src.write_text('#include "rlenv_core.cuh"\nextern "C" long long cell(const unsigned char* r, int size) { return rle::smb_grid_cell(r,size); }\n')
    subprocess.run(['c++','-O2','-std=c++20','-shared','-fPIC',f'-I{NESLE_INC}',f'-I{ROOT / "rl/csrc"}',str(src),'-o',str(so)],check=True)
    lib = ctypes.CDLL(str(so))
    lib.cell.argtypes = [ctypes.c_void_p,ctypes.c_int]; lib.cell.restype = ctypes.c_longlong
    return lambda r,size=32: lib.cell(r.ctypes.data_as(ctypes.c_void_p),size)


def test_native_world_cells_distinguish_corridors_rooms_and_maze_phase(native):
    r = np.zeros(2048, np.uint8)
    r[0x770],r[0x772],r[0xe],r[0x750],r[0x74e] = 1,3,8,0x61,3
    r[0x75f],r[0x75c],r[0x6d],r[0x86],r[0xce] = 3,3,3,40,128
    original = native(r)
    r[0xce] = 176
    assert native(r) != original
    r[0xce] = 128; r[0x71a],r[0x71c] = 99,99
    assert native(r) == original  # camera is not a cell boundary
    r[0x750] = 0xe1
    assert native(r) == original  # pointer aliases are the same room
    r[0x725] = 6
    assert native(r) != original
    r[0xe] = 6
    assert native(r) == -1


def test_native_8px_grid_distinguishes_all_16_subcells_and_keeps_world_identity(native):
    r=np.zeros(2048,np.uint8)
    r[0x770],r[0x772],r[0xe],r[0x750],r[0x74e]=1,3,8,0x61,3
    r[0x75f],r[0x75c]=3,3
    fine,coarse=set(),set()
    for x in range(32,64,8):
        for y in range(64,96,8):
            r[0x86],r[0xce]=x,y
            k=native(r,8);fine.add(k);coarse.add(native(r,32))
            assert k>>29==15 and (k>>20)&511==481
            assert (k>>5)&2047==(x+1024)//8 and k&31==y//8
    assert len(fine)==16 and len(coarse)==1
    r[0x86],r[0xce]=32,64
    original=native(r,8)
    r[0x71a],r[0x71c],r[0x750]=17,23,0xe1
    assert native(r,8)==original  # camera motion and room aliases do not mint new cells
    r[0x6d]=4
    assert native(r,8)!=original
    r[0x6d]=0
    assert native(r,8)==original  # same maze phase after looping returns to the same cell
    r[0x725]=6
    assert native(r,8)!=original  # native maze progress remains a distinct state
    r[0xe]=6
    assert native(r,8)==-1


def test_native_limit_is_terminal_and_restarts_without_old_reward():
    e = RlEnv({'smb':1}, device='cpu', sticky=False, idle_max=0, dropout=False)
    e.set_grid_attempts(3); e.start()
    assert e.cells.item() >= 0
    for _ in range(2):
        e.step(); assert not e.done.item()
        assert e.reward.item() == pytest.approx(-.0001)
    e.step()
    assert e.done.item() and e.info[0,2].item() == 8
    assert e.info[0,1].item() == 3 and e.cells.item() >= 0
    assert e.parts[0,0].item() == 0  # horizontal progress never enters the new objective


def test_native_first_death_ends_attempt_without_waiting_for_game_over():
    e = RlEnv({'smb':1}, device='cpu', sticky=False, idle_max=0, dropout=False)
    e.set_grid_attempts(4000); e.start()
    e.actions.fill_(3 + 9*3)  # right + B, no jump: first goomba
    for _ in range(500):
        e.step()
        if e.done.item():
            break
    assert e.done.item() and e.info[0,2].item() == 7
    assert e.info[0,3].item() == 1
    assert e.reward.item() == pytest.approx(-.5001)
    assert e.cells.item() >= 0  # new entrance; terminal novelty must not be paid


def test_undiscounted_completion_dominates_maximum_exploration():
    # A 4,000-decision win remains above any failure even without novelty on the winning route.
    assert 10 - .0001*4000 > 1
    from train import Trainer
    from types import SimpleNamespace
    owner = SimpleNamespace(T=3, N=1, dev='cpu', cfg=SimpleNamespace(gamma=1.,lam=1.))
    b = SimpleNamespace(b_done=torch.tensor([[False],[False],[True]]), b_val=torch.zeros(4,1))
    adv, ret = Trainer.gae(owner,b,torch.tensor([[-.0001],[-.0001],[9.9999]]))
    assert ret[0,0].item() == pytest.approx(9.9997, abs=1e-6)
