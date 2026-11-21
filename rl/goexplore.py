"""Go-Explore route search with per-game descriptions: one deathless route from power-on through every level, found
by search alone (no network), for games where winning needs fights and resources as well as a path.

The loop is Go-Explore's first phase (Ecoffet et al. 2021, arXiv 1901.10995): consoles restart from saved states of
an archive, play semi-random buttons for a while, and every new cell is saved. What differs from rl/explore.py's
whole-game search:
- the cell comes from a per-game description (GAMES): level, progress (for a boss, the damage dealt counts as
  progress, so a fight is a staircase of hits), and the player's position;
- each cell keeps its best state, not the first: a later arrival with more of the game's quality (life, power)
  takes the cell over in a new slot, so routes through the old state stay valid. Getting strong before a boss then
  happens by itself: the strong states take over the cells near it;
- restarts favour the newest level, the furthest progress, and cells whose restarts still find new cells;
- the random buttons have a per-game bias (a shooter keeps firing);
- each restart plays for a random length, mostly short, sometimes up to 16x longer: some steps need a long stretch
  with no new cell (Final Mission's stage-1 boss: its last hit points take 100-150 decisions of steady fire; with a
  fixed 48 every attempt was cut off just before the kill, 2026-11-14).

Restores are exact (RlEnv.set_restore_exact), so a route is its links' buttons one after another; replay and record
it with rl/scripts/route_video.py.

usage: python rl/goexplore.py --game finalmission --out runs/routes/finalmission-ge [--minutes 240]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlenv import RlEnv  # noqa: E402
import slots as slotlib  # noqa: E402

HIST = 1024          # buttons kept per console since its last saved state
END_WON = 2


def fm_cell(ram, lv, p):
    # level; progress in 4-unit steps (8-px scroll columns; boss damage: 300 units per boss); ship 32-px x, y; HP left
    # in strong objects (>= 8 HP), every hit counting, so destroying the parts of a big enemy is progress even where the
    # reward does not count it (stage 3's battleship stops the screen while the scroll mode still says scrolling; the
    # ship takes 3 hits there within ~100 decisions, so the search must climb hit by hit with the most life kept)
    units = (torch.nan_to_num(p) * 976.0 / 4.0).long().clamp(0, 400)
    hp = ram[:, 3:35].long()
    strong = (hp * (hp >= 8)).sum(1)
    life = ram[:, 2].long().clamp(max=7)        # each life level climbs its own staircase of hits
    return ((((lv * 512 + units) * 16 + ram[:, 0].long() // 32) * 16 + ram[:, 1].long() // 32) * 1024
            + strong.clamp(max=1023)) * 8 + life


# Stage 3's battleship (2026-11-15, ROM + RAM pokes): its core is object 22 (HP $04AE, 192). Collisions subtract the
# attacker's $0498+X from the target's (code at $EF33). Parked against it firing, the ship deals ~1 HP per decision
# and loses a life about every 33 decisions wherever it is; life ($1C) refills by kills only up to a cap of about
# 1 + score / 10000 (3 for this route's 17,860 points). Poking $12 (shot type) did not change that parked rate, but
# rl/explore.py's route (runs/routes/finalmission-fight) took the core from 192 to 0 in ~120 decisions with no death,
# so the right approach deals damage much faster than parking does.


def fm_depth(ram, lv, p):
    # how far: progress, then (at the same progress) less HP left in strong objects
    hp = ram[:, 3:35].long()
    return (torch.nan_to_num(p) * 976.0 / 4.0).floor() * 1024 - (hp * (hp >= 8)).sum(1).clamp(max=1023).float()


def fm_actions(n, state, gen, dev):
    """d-pad held for a while (changes with probability 0.15 per decision); B fires (turbo) 90% of the time."""
    d = state.get('d')
    if d is None:
        d = torch.randint(0, 9, (n,), generator=gen, device=dev)
    d = torch.where(torch.rand(n, generator=gen, device=dev) < 0.15, torch.randint(0, 9, (n,), generator=gen, device=dev), d)
    state['d'] = d
    b = torch.where(torch.rand(n, generator=gen, device=dev) < 0.9, 2, 0)
    return d + 27 * b


GAMES = {
    # ram: bytes read every decision; cell(ram, level, progress) -> int64; quality(ram) -> int64, higher = better
    'finalmission': dict(ram=[0x408, 0x3E4, 0x1C] + list(range(0x4A0, 0x4C0)), cell=fm_cell, depth=fm_depth,
                         quality=lambda ram: ram[:, 2].long(),
                         actions=fm_actions),
}


def search(env, spec, minutes, target, explore_steps, seed, log, on_save, save_every=600, start_level=0):
    N, dev, L0, cap = env.n, env.device, env.level_slots, env.level_cap
    gen = torch.Generator(device=dev).manual_seed(seed)
    ar = torch.arange(N, device=dev)
    # cells: sorted keys and the slot holding each one's best state
    keys = torch.empty(0, dtype=torch.int64, device=dev)
    kslot = torch.empty(0, dtype=torch.int64, device=dev)
    # slots (0 = the start; L0.. = saved states)
    parent = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    steps = torch.zeros(cap, dtype=torch.int64, device=dev)
    lvl = torch.zeros(cap, dtype=torch.int64, device=dev)
    lvl[0] = start_level
    prog = torch.zeros(cap, dtype=torch.float32, device=dev)
    qual = torch.zeros(cap, dtype=torch.int64, device=dev)
    depth = torch.zeros(cap, dtype=torch.float32, device=dev)   # within a level: how far (spec depth, or progress)
    chosen = torch.zeros(cap, dtype=torch.float64, device=dev)
    stale = torch.zeros(cap, dtype=torch.float64, device=dev)       # restarts since it last led to a new cell
    acts = torch.zeros(cap, HIST, dtype=torch.int16, device=dev)
    alen = torch.zeros(cap, dtype=torch.int64, device=dev)
    hist = torch.zeros(N, HIST, dtype=torch.int16, device=dev)
    last = torch.zeros(N, dtype=torch.int64, device=dev)          # slot each console's current link started from
    since = torch.zeros(N, dtype=torch.int64, device=dev)
    age = torch.zeros(N, dtype=torch.int64, device=dev)
    def horizon(n):   # decisions before a restart: explore_steps x 1, 2, 4, 8 or 16 (half of them x 1)
        k = torch.where(torch.rand(n, generator=gen, device=dev) < 0.5, 0,
                        torch.randint(1, 5, (n,), generator=gen, device=dev))
        return explore_steps * (2 ** k)
    hz = horizon(N)
    w = np.zeros(cap)
    w[0] = 1.0
    env.set_level_plan(0, 1.0, w)
    env.start()
    last[:] = env.cur_start.long()
    astate = {}
    used, replaced, it, goal = 0, 0, 0, None
    t0 = t_log = t_save = time.time()
    level_seen = {}

    def observe():
        lv = env.level_now.long()
        p = env.progress_now
        ok = (lv >= 0) & ~torch.isnan(p)
        cell = torch.where(ok, spec['cell'](env.ram, lv.clamp(min=0), p), -1)
        dp = spec['depth'](env.ram, lv.clamp(min=0), p) if 'depth' in spec else torch.nan_to_num(p)
        return cell, lv, torch.nan_to_num(p), spec['quality'](env.ram), dp

    def route(goal):
        if goal is None:
            act = kslot if kslot.numel() else torch.zeros(1, dtype=torch.int64, device=dev)
            score = lvl[act].double() * 1e6 + prog[act].double() * 1e3 + qual[act].double() - steps[act].double() * 1e-7
            goal = dict(console=-1, last=int(act[score.argmax()]), tail=torch.zeros(0, dtype=torch.int16, device=dev),
                        won=False)
        chain, s = [], goal['last']
        while s >= L0:
            chain.append(s)
            s = int(parent[s])
        chain.reverse()
        buttons = torch.cat([acts[s, :int(alen[s])] for s in chain] + [goal['tail']]).cpu().numpy()
        end = chain[-1] if chain else 0
        stats = dict(seconds=round(time.time() - t0, 1), decisions=it * N, cells=int(keys.numel()), slots=used,
                     replaced=replaced, goal_reached=goal['console'] >= 0, won=goal['won'],
                     level=int(lvl[end]) if goal['console'] < 0 else goal['level'], progress=round(float(prog[end]), 3),
                     quality=int(qual[end]), route_decisions=len(buttons), levels_first_reached=level_seen)
        return buttons, chain, steps[chain].cpu().numpy(), lvl[chain].cpu().numpy(), stats

    while goal is None and time.time() - t0 < minutes * 60 and cap - L0 - used > 0:
        a = spec['actions'](N, astate, gen, dev).to(torch.int16)
        env.actions.copy_(a)
        hist[ar, since.clamp(max=HIST - 1)] = a
        cell, clv, cp, cq, cd = observe()
        env.kill.copy_(((age >= hz) & (cell >= 0)).to(torch.uint8))
        env.step()
        it += 1
        since += 1
        age += 1
        done = env.done.bool()
        end = env.info[:, 2]
        cell, clv, cp, cq, cd = observe()
        reached = (~done & (cell >= 0) & (clv >= target)) | (done & (end == END_WON))
        if reached.any():
            i = int(reached.nonzero()[0])
            goal = dict(console=i, last=int(last[i]), tail=hist[i, :int(since[i])].clone(), won=bool(done[i]),
                        level=int(clv[i]) if not done[i] else int(env.info[i, 4]))
            break
        valid = (cell >= 0) & ~done & (since <= HIST)
        if keys.numel():
            pos = torch.searchsorted(keys, cell).clamp(max=keys.numel() - 1)
            found = keys[pos] == cell
            better = found & (cq > qual[kslot[pos]])
            take = valid & (~found | better)
        else:
            pos = torch.zeros_like(cell)
            found = torch.zeros_like(valid)
            take = valid
        idx = take.nonzero().squeeze(1)
        room = cap - L0 - used
        if idx.numel() and room > 0:
            # one console per cell: the best quality, then the shortest route
            score = cq[idx].double() * 1e9 - (steps[last[idx]] + since[idx]).double()
            order = torch.argsort(score, descending=True)
            idx = idx[order]
            ukeys, inv = torch.unique(cell[idx], return_inverse=True)
            firstpos = torch.full((ukeys.numel(),), idx.numel(), dtype=torch.int64, device=dev).scatter_reduce_(
                0, inv, torch.arange(idx.numel(), device=dev), 'amin')
            pick = idx[firstpos][:room]
            ukeys = ukeys[:room]
            sl = L0 + used + torch.arange(pick.numel(), device=dev)
            env.place_save(0, pick.to(torch.int32).contiguous(), sl.to(torch.int32).contiguous())
            parent[sl] = last[pick]
            steps[sl] = steps[last[pick]] + since[pick]
            lvl[sl], prog[sl], qual[sl], depth[sl] = clv[pick], cp[pick], cq[pick], cd[pick]
            acts[sl] = hist[pick]
            alen[sl] = since[pick]
            new = ~found[pick]
            stale[last[pick[new]]] = 0                       # the restart point is still productive
            replaced += int((~new).sum())
            # cell -> slot: replace the better ones in place, insert the new ones
            if (~new).any():
                kslot[pos[pick[~new]]] = sl[~new]
            if new.any():
                allk = torch.cat([keys, ukeys[new]])
                alls = torch.cat([kslot, sl[new]])
                o = torch.argsort(allk)
                keys, kslot = allk[o], alls[o]
            last[pick] = sl
            since[pick] = 0
            used += pick.numel()
            for lv in clv[pick].unique().tolist():
                level_seen.setdefault(int(lv), round(time.time() - t0, 1))
        start = env.cur_start.long()
        nd = int(done.sum())
        if nd:
            chosen.index_add_(0, start[done], torch.ones(nd, dtype=torch.float64, device=dev))
            stale.index_add_(0, start[done], torch.ones(nd, dtype=torch.float64, device=dev))
        last = torch.where(done, start, last)
        since[done] = 0
        age[done] = 0
        if nd:
            hz[done] = horizon(nd)
        if it % 4 == 0 and kslot.numel():
            s = kslot
            top = int(lvl[s].max())
            lf = torch.pow(0.1, (top - lvl[s]).clamp(min=0).double())
            # the frontier, one per quality level (life): the deepest cells of the newest level at each quality, so
            # a strong state that has gone less far and a weak one that has gone further are both pushed on (stage 3's
            # battleship: with one frontier, the states that had traded life for hits crowded out the others, and with
            # life counted into depth, no state would trade a hit for damage)
            top_m = lvl[s] == top
            qv = spec.get('tier', lambda q: q)(qual[s]).clamp(0, 255)
            best = torch.full((256,), -float('inf'), device=dev).scatter_reduce(
                0, qv[top_m], depth[s][top_m], 'amax')
            front = top_m & (depth[s] >= best[qv] - 2.0)
            qmax = qual[s][lvl[s] == top].max().clamp(min=1)
            wt = lf * (1 / torch.sqrt(1 + chosen[s]) + 1 / torch.sqrt(1 + stale[s])) \
                * (1 + 9 * (front & (lvl[s] == top)).double()) * (0.5 + qual[s].double() / qmax.double())
            full = np.zeros(cap)
            full[s.cpu().numpy()] = (wt / wt.sum()).cpu().numpy()
            env.set_level_plan(0, 1.0, full)
        now = time.time()
        if now - t_save > save_every and kslot.numel():
            t_save = now
            on_save(route(None))
        if now - t_log > 30:
            t_log = now
            s = kslot
            top = int(lvl[s].max())
            m = lvl[s] == top
            log(f'[ge] {now - t0:.0f}s {it * N / 1e6:.1f}M dec, {keys.numel()} cells ({replaced} improved), level {top}: '
                f'best progress {float(prog[s][m].max()):.3f}, best quality there {int(qual[s][m].max())}, '
                f'deepest {float(depth[s][m].max()):.0f} (quality {int(qual[s][m][depth[s][m].argmax()])}); '
                f'levels first reached {level_seen}')
    return route(goal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--game', required=True, choices=sorted(GAMES))
    ap.add_argument('--out', required=True)
    ap.add_argument('--envs', type=int, default=8192)
    ap.add_argument('--archive', type=int, default=60000)
    ap.add_argument('--minutes', type=float, default=240)
    ap.add_argument('--target', type=int, default=1 << 30, help='stop on reaching this level')
    ap.add_argument('--explore-steps', type=int, default=48)
    ap.add_argument('--attempt-limit', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default='', help='go on from the end of this route.npz (found with exact restores)')
    a = ap.parse_args()
    spec = GAMES[a.game]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = RlEnv({a.game: a.envs}, device='cuda', seed=a.seed, sticky=False, idle_max=0, dropout=False,
                level_cap=64 + a.archive, level_slots=64)
    env.set_attempts(a.attempt_limit, kill=True)
    env.set_attempt_through(True)
    env.set_restore_exact(True)
    env.bind_ram(spec['ram'])
    env.set_no_render()
    env.drop_obs_buffers()
    slotlib.save_boot_entrance(env, 0)
    boot = slotlib.read(env, 0, [0])
    pre = dict(buttons=np.zeros(0, np.int16), steps=np.zeros(0, np.int64), levels=np.zeros(0, np.int64),
               states={k: v[:0] for k, v in boot.items()})
    start_level = 0
    if a.resume:   # the route's last state becomes slot 0 (exact restore): the replay is its buttons, then the new
        r = np.load(a.resume)
        assert json.loads(str(r['meta'])).get('exact_restore') and len(r['levels'])
        boot = {k: r[f'boot_{k}'] for k in slotlib.FIELDS}
        slotlib.write(env, 0, [0], {k: r[f'state_{k}'][-1:] for k in slotlib.FIELDS})
        start_level = int(r['levels'][-1])
        pre = dict(buttons=r['buttons'], steps=r['steps'], levels=r['levels'],
                   states={k: r[f'state_{k}'] for k in slotlib.FIELDS})

    def save(route, final=False):
        buttons, chain, st, lv, stats = route
        rec = slotlib.read(env, 0, chain) if len(chain) else {k: v[:0] for k, v in boot.items()}
        stats = dict(stats, game=a.game, exact_restore=True, final=final, resumed_from=a.resume or None,
                     route_decisions=len(pre['buttons']) + len(buttons))
        np.savez_compressed(out / 'route.npz', buttons=np.concatenate([pre['buttons'], buttons]),
                            steps=np.concatenate([pre['steps'], st + len(pre['buttons'])]),
                            levels=np.concatenate([pre['levels'], lv]),
                            **{f'state_{k}': np.concatenate([pre['states'][k], v]) for k, v in rec.items()},
                            **{f'boot_{k}': v for k, v in boot.items()},
                            meta=json.dumps(dict(layout=env.level_layout(0), **stats)))
        print('[ge] ' + ('route ' if final else 'saved so far ') + json.dumps(stats), flush=True)

    save(search(env, spec, a.minutes, a.target, a.explore_steps, a.seed, lambda s: print(s, flush=True), save,
                start_level=start_level),
         final=True)


if __name__ == '__main__':
    main()
