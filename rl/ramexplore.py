"""Game-agnostic route search: Go-Explore whose cells come from the whole of the console's RAM. No per-game code.

Why (2026-11-15): every stall of the per-game searches (rl/explore.py, rl/goexplore.py) was a hand-picked cell
that missed some part of the game's state (Kage's player state and boss HP, Final Mission's battleship damage,
Mario's maze phase). The game keeps all of its state in its 2 KB of RAM, so here a cell is one (address, value) pair:
a console that shows a value no console has shown at that address is saved, and it owns that pair. Restarts favour
saved states that own rarely-seen pairs, so the search climbs whatever the game counts (boss HP down hit by hit,
a new stage number, a new screen position) without being told what it means. Deaths and game overs lead back to
values seen before (title screen, stage start), so they stop drawing restarts by themselves.

Generic filters only: the stack page ($0100-$01FF) is ignored, and so is any byte that changes on most decisions
(frame counters, random number generators), measured during the run.

The reward tracker (level, progress) is read only to choose which route to save and to log how far the search got;
it does not steer the search. The start is the env's power-on boot (the game's own start sequence).

usage: python rl/ramexplore.py --game smb --out runs/routes/generic-smb [--seconds 3600]
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from rlenv import RlEnv
import slots as slotlib

RAM = 2048
HIST = 1024                    # decisions of buttons kept per console since its last saved state
CHURN_MAX = 0.9                # a byte that changes on more than this share of decisions is noise


def search(env, seconds, explore_steps=64, change=0.1, start_p=0.02, seed=0, log=print, on_save=None, save_every=600):
    N, dev, L0, cap = env.n, env.device, env.level_slots, env.level_cap
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = np.zeros(cap)
    w[0] = 1.0
    env.set_level_plan(0, 1.0, w)
    env.start()
    ar = torch.arange(N, device=dev)
    code_base = (torch.arange(RAM, device=dev) * 256).view(1, RAM)
    cnt = torch.zeros(RAM * 256, dtype=torch.int64, device=dev)          # times each (address, value) was seen
    owner = torch.full((RAM * 256,), -1, dtype=torch.int64, device=dev)  # saved state that first showed it
    churn = torch.zeros(RAM, dtype=torch.float64, device=dev)            # share of decisions a byte changes
    fixed = torch.zeros(RAM, dtype=torch.bool, device=dev)
    fixed[0x100:0x200] = True                                            # stack page
    parent = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    steps = torch.zeros(cap, dtype=torch.int64, device=dev)
    lvl = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    prog = torch.full((cap,), float('nan'), dtype=torch.float32, device=dev)
    chosen = torch.zeros(cap, dtype=torch.float64, device=dev)
    acts = torch.zeros(cap, HIST, dtype=torch.int16, device=dev)
    alen = torch.zeros(cap, dtype=torch.int64, device=dev)
    hist = torch.zeros(N, HIST, dtype=torch.int16, device=dev)
    last = env.cur_start.long().clone()
    since = torch.zeros(N, dtype=torch.int64, device=dev)
    age = torch.zeros(N, dtype=torch.int64, device=dev)
    pad = [torch.randint(0, k, (N,), generator=gen, device=dev) for k in (9, 3, 3)]
    prev = None

    def limits(n):
        return explore_steps * (2 ** torch.randint(0, 5, (n,), generator=gen, device=dev))
    lim = limits(N)
    used, t0, t_log, t_save, it = 0, time.time(), time.time(), time.time(), 0

    def route():
        """Buttons from slot 0 to the saved state the reward tracker rates furthest (level, then progress)."""
        n = L0 + used
        score = lvl[:n].double() * 1e3 + torch.nan_to_num(prog[:n].double(), nan=-1.0) - steps[:n].double() * 1e-9
        score[1:L0] = -1e18
        best = int(score.argmax())
        chain, s = [], best
        while s >= L0:
            chain.append(s)
            s = int(parent[s])
        chain.reverse()
        buttons = torch.cat([acts[s, :int(alen[s])] for s in chain] or [torch.zeros(0, dtype=torch.int16, device=dev)])
        stats = dict(seconds=round(time.time() - t0, 1), decisions=it * N, cells=used, level=int(lvl[best]),
                     won=False, goal_reached=False,
                     progress=float(prog[best]), route_decisions=int(buttons.numel()),
                     active_bytes=int((~fixed & (churn <= CHURN_MAX)).sum()))
        return buttons.cpu().numpy(), chain, steps[chain].cpu().numpy(), lvl[chain].cpu().numpy(), stats

    while time.time() - t0 < seconds and cap - L0 - used > 0:
        for k, size in enumerate((9, 3, 3)):
            new = torch.randint(0, size, (N,), generator=gen, device=dev)
            pad[k] = torch.where(torch.rand(N, generator=gen, device=dev) < change, new, pad[k])
        start = (torch.rand(N, generator=gen, device=dev) < start_p).long()
        select = (torch.rand(N, generator=gen, device=dev) < start_p).long()
        a = (pad[0] + 9 * (pad[1] + 3 * (pad[2] + 3 * (start + 2 * select)))).to(torch.int16)
        env.actions.copy_(a)
        hist[ar, since.clamp(max=HIST - 1)] = a
        env.kill.copy_((age >= lim).to(torch.uint8))
        env.step()
        it += 1
        since += 1
        age += 1
        done = env.done.bool()
        lim = torch.where(done, limits(N), lim)
        ram = env.ram.long()                                           # [N, 2048], after this decision
        live = ~done & (since <= HIST)
        if prev is not None:
            same = live & prev[1]
            if same.any():
                ch = (ram[same] != prev[0][same]).double().mean(0)
                churn = 0.99 * churn + 0.01 * ch if it > 1 else ch
        prev = (ram, live.clone())
        active = ~fixed & (churn <= CHURN_MAX)
        codes = (ram + code_base)[:, active]                           # [N, A]
        cnt += torch.bincount(codes[live].reshape(-1), minlength=RAM * 256)
        fresh = (owner[codes] < 0) & live.view(N, 1)                   # pairs no saved state owns yet
        room = cap - L0 - used
        if fresh.any() and room > 0:
            con = ar.view(N, 1).expand_as(codes)[fresh]
            pc = codes[fresh]
            keys, inv = torch.unique(pc, return_inverse=True)
            first = torch.full((keys.numel(),), N, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, con, 'amin')
            cons = torch.unique(first)[:room]                          # one saved state per console
            sl = L0 + used + torch.arange(cons.numel(), device=dev)
            slot_of = torch.full((N,), -1, dtype=torch.int64, device=dev)
            slot_of[cons] = sl
            got = slot_of[first] >= 0
            owner[keys[got]] = slot_of[first[got]]
            env.place_save(0, cons.to(torch.int32).contiguous(), sl.to(torch.int32).contiguous())
            parent[sl] = last[cons]
            steps[sl] = steps[last[cons]] + since[cons]
            lvl[sl] = env.level_now[cons].long()
            prog[sl] = env.progress_now[cons].float()
            acts[sl] = hist[cons]
            alen[sl] = since[cons]
            last[cons] = sl
            since[cons] = 0
            used += cons.numel()
        st = env.cur_start.long()
        last = torch.where(done, st, last)
        since[done] = 0
        age[done] = 0
        chosen.index_add_(0, st[done], torch.ones(int(done.sum()), dtype=torch.float64, device=dev))
        if it % 4 == 0 and used:
            # a saved state's pull: the rarity of the pairs it owns (on bytes still judged meaningful), less the more
            # often it was already restarted from
            own = owner >= 0
            ok = own & active.repeat_interleave(256)
            nov = torch.zeros(cap, dtype=torch.float64, device=dev).index_add_(
                0, owner[ok], 1.0 / torch.sqrt(1.0 + cnt[ok].double()))
            wt = nov[:L0 + used] / torch.sqrt(1.0 + chosen[:L0 + used])
            wt[0] = max(float(wt[0]), 1e-12)
            wt[1:L0] = 0
            full = np.zeros(cap)
            full[:L0 + used] = (wt / wt.sum()).cpu().numpy()
            env.set_level_plan(0, 1.0, full)
        now = time.time()
        if on_save is not None and now - t_save > save_every:
            t_save = now
            on_save(route())
        if now - t_log > 30:
            t_log = now
            n = L0 + used
            lv = int(lvl[:n].max())
            m = lvl[:n] == lv
            log(f'[ramexplore] {now - t0:.0f}s {it * N / 1e6:.1f}M decisions, {used} saved states, '
                f'{int(active.sum())} active bytes; furthest by the reward tracker: level {lv}, progress '
                f'{float(torch.nan_to_num(prog[:n][m], nan=-1).max()):.3f}')
    return route()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--envs', type=int, default=8192)
    ap.add_argument('--archive', type=int, default=60000)
    ap.add_argument('--seconds', type=float, default=3600)
    ap.add_argument('--explore-steps', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default='', help='go on from the end of this route.npz')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = RlEnv({a.game: a.envs}, device='cuda', seed=a.seed, sticky=False, idle_max=0, dropout=False,
                level_cap=64 + a.archive, level_slots=64)
    env.set_attempts(1 << 30, kill=True)
    env.set_attempt_through(True)
    env.set_only_kill(True)
    env.set_restore_exact(True)
    env.bind_ram(list(range(RAM)))
    env.set_no_render()
    env.drop_obs_buffers()
    slotlib.save_boot_entrance(env, 0)
    boot = slotlib.read(env, 0, [0])
    pre = dict(buttons=np.zeros(0, np.int16), steps=np.zeros(0, np.int64), levels=np.zeros(0, np.int64),
               states={k: v[:0] for k, v in boot.items()})
    if a.resume:
        # go on from the end of a saved route (exact restores): its last state becomes slot 0
        r = np.load(a.resume)
        assert json.loads(str(r['meta'])).get('exact_restore'), 'only routes found with exact restores can be resumed'
        boot = {k: r[f'boot_{k}'] for k in slotlib.FIELDS}
        slotlib.write(env, 0, [0], {k: r[f'state_{k}'][-1:] for k in slotlib.FIELDS})
        pre = dict(buttons=r['buttons'], steps=r['steps'], levels=r['levels'],
                   states={k: r[f'state_{k}'] for k in slotlib.FIELDS})

    def save(route, final=False):
        buttons, chain, st, lv, stats = route
        rec = slotlib.read(env, 0, chain) if len(chain) else {k: v[:0] for k, v in boot.items()}
        stats = dict(stats, game=a.game, final=final, exact_restore=True, search='ramexplore',
                     resumed_from=a.resume or None, route_decisions=len(pre['buttons']) + len(buttons))
        np.savez_compressed(out / 'route.npz', buttons=np.concatenate([pre['buttons'], buttons]),
                            steps=np.concatenate([pre['steps'], st + len(pre['buttons'])]),
                            levels=np.concatenate([pre['levels'], lv]),
                            **{f'state_{k}': np.concatenate([pre['states'][k], v]) for k, v in rec.items()},
                            **{f'boot_{k}': v for k, v in boot.items()},
                            meta=json.dumps(dict(layout=env.level_layout(0), **stats)))
        print('[ramexplore] ' + ('route ' if final else 'saved so far ') + json.dumps(stats), flush=True)

    save(search(env, a.seconds, a.explore_steps, seed=a.seed, log=lambda s: print(s, flush=True), on_save=save),
         final=True)


if __name__ == '__main__':
    main()
