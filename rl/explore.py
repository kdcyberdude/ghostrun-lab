"""Route finder: the first phase of Go-Explore (Ecoffet et al. 2021, arXiv 1901.10995) on the GPU consoles, no network.

Every console repeatedly starts from a saved state of an archive (at first only the level entrances), holds semi-random
buttons for a while, and every place ("cell") nobody stood on before is saved whole as a new archive state, linked to
the state its console came from. Within a level, states chosen least often so far are started from most, so the search
keeps pushing where it has been least. When a console clears its level, the links back to the entrance are that
level's route: saved states a second or two apart, from the entrance to the clear. The trainer starts episodes from
them, the last ones first (rl/routes.py).

All levels are searched at once (the consoles are shared equally among the levels still open): one decision takes the
GPU the same time for 4,000 consoles as for 30,000, so searching levels side by side costs no extra time.

Nothing here knows a level's layout. Only the cell is game-specific; SMB uses its world-grid cell (level, room, maze
phase, 32-pixel x and y from RAM: rlenv_core.cuh smb_grid_cell). RAM is search data only, never an actor input.

usage: python rl/explore.py --entrances ENTRANCES.npy --out runs/routes/smb          (every saved level, and 1-1)
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

END_CLEAR, END_WON = 6, 2
LEVEL_SHIFT = 25      # 32-pixel cell key: level in the bits above


def route_of(i, last, parent, steps, L0, spacing):
    """Console i just cleared its level: the archive states it came through, thinned to >= spacing decisions apart."""
    chain, s = [], int(last[i])
    while s >= L0:
        chain.append(s)
        s = int(parent[s])
    chain.reverse()
    keep = chain[-1:]
    for s in reversed(chain[:-1]):
        if steps[keep[-1]] - steps[s] >= spacing:
            keep.append(s)
    if len(keep) > 1 and steps[keep[-1]] < spacing:   # too close to the entrance itself
        keep.pop()
    keep.reverse()
    return s if not chain else int(parent[chain[0]]), chain, keep


def explore(env, levels, seconds, on_route, explore_steps=64, change=0.1, spacing=24, seed=0, log=print):
    """Search `levels` (each one's entrance is saved in its level slot) until all are cleared or time runs out.
    on_route(level, route slots in order, decisions from the entrance to each, stats) is called for every clear."""
    N, dev, L0, cap = env.n, env.device, env.level_slots, env.level_cap
    gen = torch.Generator(device=dev).manual_seed(seed)
    open_ = torch.zeros(L0, dtype=torch.bool, device=dev)
    open_[list(levels)] = True
    w = np.zeros(cap)
    w[list(levels)] = 1.0 / len(levels)
    env.set_level_plan(0, 1.0, w)
    env.start()
    known = torch.empty(0, dtype=torch.int64, device=dev)   # sorted cells of the archive
    parent = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    steps = torch.zeros(cap, dtype=torch.int64, device=dev)  # decisions from the entrance along the links
    level_of = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    level_of[:L0] = torch.arange(L0, device=dev)
    chosen = torch.zeros(cap, dtype=torch.float64, device=dev)
    last = env.cur_start.long().clone()                     # archive state each console came from most recently
    since = torch.zeros(N, dtype=torch.int64, device=dev)   # decisions since then
    age = torch.zeros(N, dtype=torch.int64, device=dev)     # decisions since the console's restart
    pad = [torch.randint(0, k, (N,), generator=gen, device=dev) for k in (9, 3, 3)]   # d-pad, A, B
    one = torch.ones(N, dtype=torch.float64, device=dev)
    used, t0, t_log, it, found = 0, time.time(), time.time(), 0, {}
    while bool(open_.any()):
        for k, size in enumerate((9, 3, 3)):   # each part keeps its choice, changing with probability `change`
            new = torch.randint(0, size, (N,), generator=gen, device=dev)
            pad[k] = torch.where(torch.rand(N, generator=gen, device=dev) < change, new, pad[k])
        env.actions.copy_((pad[0] + 9 * (pad[1] + 3 * pad[2])).to(torch.int16))
        mine = level_of[last]
        # back to the archive after explore_steps decisions (never in the middle of a transition), or at once when
        # the console's level is no longer searched
        env.kill.copy_((((age >= explore_steps) & (env.cells >= 0)) | ~open_[mine]).to(torch.uint8))
        env.step()
        it += 1
        since += 1
        age += 1
        done = env.done.bool()
        end = env.info[:, 2]
        won = done & ((end == END_CLEAR) | (end == END_WON)) & open_[mine]
        if won.any():
            total = (steps[last] + since).masked_fill(~won, torch.iinfo(torch.int64).max)
            st = steps.cpu().numpy()
            for lv in mine[won].unique().tolist():
                i = int(total.masked_fill(mine != lv, torch.iinfo(torch.int64).max).argmin())
                root, chain, keep = route_of(i, last, parent, st, L0, spacing)
                assert root == lv, 'route does not start at its entrance'
                found[lv] = dict(level=lv, seconds=time.time() - t0, decisions=it * N, chain=len(chain),
                                 states=len(keep), clear_decisions=int(total[i]))
                on_route(lv, keep, [int(st[s]) for s in keep], found[lv])
                open_[lv] = False
        cell = env.cells
        valid = (cell >= 0) & ~done & ((cell >> LEVEL_SHIFT) == mine) & open_[mine]
        if known.numel():
            pos = torch.searchsorted(known, cell).clamp(max=known.numel() - 1)
            valid &= known[pos] != cell
        idx = valid.nonzero().squeeze(1)
        room = cap - L0 - used
        if idx.numel() and room > 0:
            keys, inv = torch.unique(cell[idx], return_inverse=True)
            first = torch.full((keys.numel(),), N, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, idx, 'amin')
            keys, first = keys[:room], first[:room]
            sl = L0 + used + torch.arange(keys.numel(), device=dev)
            env.place_save(0, first.to(torch.int32).contiguous(), sl.to(torch.int32).contiguous())
            parent[sl] = last[first]
            steps[sl] = steps[last[first]] + since[first]
            level_of[sl] = mine[first]
            last[first] = sl
            since[first] = 0
            known = torch.cat([known, keys]).sort().values
            used += keys.numel()
        start = env.cur_start.long()
        last = torch.where(done, start, last)
        since[done] = 0
        age[done] = 0
        chosen.index_add_(0, start[done], one[done])
        if it % 4 == 0 and bool(open_.any()):
            # every open level gets the same share of restarts; within a level, 1 / (1 + times chosen)
            live = level_of[:L0 + used].clamp(min=0)
            wt = (1.0 / (1.0 + chosen[:L0 + used])) * open_[live] * (level_of[:L0 + used] >= 0)
            per = torch.zeros(L0, dtype=torch.float64, device=dev).index_add_(0, live, wt)
            wt = wt / per[live].clamp_min(1e-300)
            full = np.zeros(cap)
            full[:L0 + used] = (wt / wt.sum()).cpu().numpy()
            env.set_level_plan(0, 1.0, full)
        now = time.time()
        if now - t_log > 15:
            t_log = now
            lv = level_of[L0:L0 + used]
            cells = torch.zeros(L0, dtype=torch.int64, device=dev).index_add_(0, lv, torch.ones_like(lv))
            deep = torch.zeros(L0, dtype=torch.int64, device=dev).scatter_reduce_(0, lv, steps[L0:L0 + used], 'amax')
            log(f'[explore] {now - t0:.0f}s {it * N / 1e6:.1f}M decisions ({it * N / (now - t0):.0f}/s), {used} cells, '
                f'cleared {sorted(found)}; open level:cells/deepest ' +
                ' '.join(f'{l}:{int(cells[l])}/{int(deep[l])}' for l in open_.nonzero().squeeze(1).tolist()))
        if now - t0 > seconds or room <= 0:
            break
    return found, dict(seconds=time.time() - t0, decisions=it * N, cells=used, archive_full=cap - L0 - used <= 0,
                       unsolved=open_.nonzero().squeeze(1).tolist())


# Whole-game search (--whole-game): one archive from power-on through every level, a single life (attempts go on
# through level events), the buttons of every link kept so the route replays (rl/scripts/route_video.py).
# A cell is (level, game-specific coarse progress); RAM bytes read for it are search data only.
GAME_CELLS = {
    # level from the reward tracker; progress_now * buckets; extra RAM bytes (address, divisor) for diversity
    'pacman': dict(buckets=192, ram=((0x1A, 32), (0x1C, 32))),           # dots eaten; Pac-Man's 32 px maze region
    # Final Mission: 4 units (scroll/boss); health pips; HP left in strong objects (>= 8 HP, $04A0-$04BF, /4).
    # Stage 3's battleship core (192 HP) only loses HP to the ship's body, 1 per decision of contact, while the ship
    # loses a life every ~33 decisions and kills give lives back up to 3; the scroll mode never says halted there, so
    # the reward's progress does not count the damage (2026-11-15). Restarts go to the cells with the most
    # value = progress + 1.56 units per HP dealt (strong HP below the most seen at the same progress) + 52 units per
    # life (a life buys ~33 HP), weight exp((value - best) / 16 units).
    'finalmission': dict(buckets=244, ram=((0x1C, 1),) + tuple((a, 1) for a in range(0x4A0, 0x4C0)),
                         life_value=52 / 976, hp_value=1.56 / 976, value_temp=16 / 976),
    # Kage: a place cell like SMB's (game_cells), not the reward's route progress, which is invalid off the recorded
    # camera path: stage $0040, mode $0002, health $06F0, player world x $0610/$0620 and y $0630/$0640 (fixed /16),
    # and the player's state $06A1: hanging under a ledge is a different cell from jumping at the same spot, so the
    # pull-up that 2-1 needs at x 1779 has a saved state to start from (2026-11-14)
    'kage': dict(buckets=0, ram=((0x40, 1), (0x02, 1), (0x6F0, 1), (0x610, 1), (0x620, 1), (0x630, 1), (0x640, 1),
                                 (0x6A1, 1))),
    'tetris': dict(buckets=10, ram=()),     # lines in the level; + stack height and holes (rl/tetris_actor.py)
}
HIST = 1024     # decisions of buttons kept per console since its last archive state


def fm_strong(r):
    """[N] Final Mission HP left in strong objects (>= 8 HP); r = bound RAM (life, then $04A0-$04BF)."""
    hp = r[:, 1:33]
    return (hp * (hp >= 8)).sum(1)


def game_cells(env, game):
    """[N] int64 cell of every console (-1 = not a valid position), and its level."""
    if game == 'smb':
        c = env.cells
        return c, torch.where(c >= 0, c >> LEVEL_SHIFT, -1)
    spec = GAME_CELLS[game]
    if game == 'kage':
        r = env.ram.long()
        stage, mode, hp = r[:, 0], r[:, 1], r[:, 2]
        x, y = ((r[:, 4] << 8) | r[:, 3]) >> 9, ((r[:, 6] << 8) | r[:, 5]) >> 9     # 32 px
        ok = (mode == 0) & (hp > 0) & (stage < 15)
        key = ((((stage * 256 + x) * 256 + y) * 4 + hp // 4) * 256 + r[:, 7])
        return torch.where(ok, key, -1), torch.where(ok, stage, -1)
    lv = env.level_now.long()
    p = env.progress_now
    ok = (lv >= 0) & ~torch.isnan(p)
    key = lv * 100000 + (torch.nan_to_num(p).clamp(0, 1.5) * spec['buckets']).long()
    if game == 'finalmission':
        r = env.ram.long()
        key = (key * 256 + r[:, 0]) * 256 + (fm_strong(r) // 4).clamp(max=255)
        return torch.where(ok, key, -1), torch.where(ok, lv, -1)
    for k, (_, div) in enumerate(spec['ram']):
        key = key * 256 + env.ram[:, k].long() // div
    if game == 'tetris':
        height, holes = env.tetris.features(env.ram)
        key = (key * 32 + height // 2) * 64 + (holes // 2).clamp(max=15)   # coarse: the archive fills by level 12 otherwise
    return torch.where(ok, key, -1), torch.where(ok, lv, -1)


def explore_game(env, game, seconds, target, explore_steps=64, change=0.1, seed=0, log=print, start_level=0,
                 on_save=None, save_every=600):
    """Search from level slot 0 (power-on, or a resumed route's end at start_level) until the game is won or
    `target` levels are reached (or time runs out). Returns the route: buttons from slot 0, and the archive slots it
    passes with their decisions and levels. on_save(route) gets the furthest route so far every save_every s."""
    N, dev, L0, cap = env.n, env.device, env.level_slots, env.level_cap
    gen = torch.Generator(device=dev).manual_seed(seed)
    w = np.zeros(cap)
    w[0] = 1.0
    env.set_level_plan(0, 1.0, w)
    env.start()
    ar = torch.arange(N, device=dev)
    known = torch.empty(0, dtype=torch.int64, device=dev)
    parent = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    steps = torch.zeros(cap, dtype=torch.int64, device=dev)
    lvl = torch.full((cap,), -1, dtype=torch.int64, device=dev)
    lvl[0] = start_level
    prog = torch.zeros(cap, dtype=torch.float32, device=dev)
    spec = GAME_CELLS.get(game, {})
    life = torch.zeros(cap, dtype=torch.float32, device=dev)    # health byte (first RAM byte) when 'life_value' is set
    strong = torch.zeros(cap, dtype=torch.float32, device=dev)  # Final Mission strong-object HP left
    chosen = torch.zeros(cap, dtype=torch.float64, device=dev)
    acts = torch.zeros(cap, HIST, dtype=torch.int16, device=dev)   # buttons from the parent's state to this one
    alen = torch.zeros(cap, dtype=torch.int64, device=dev)
    hist = torch.zeros(N, HIST, dtype=torch.int16, device=dev)
    last = env.cur_start.long().clone()
    since = torch.zeros(N, dtype=torch.int64, device=dev)
    age = torch.zeros(N, dtype=torch.int64, device=dev)
    pad = [torch.randint(0, k, (N,), generator=gen, device=dev) for k in (9, 3, 3)]
    just = torch.ones(N, dtype=torch.bool, device=dev)          # began from a saved state at the last decision
    # each restart plays explore_steps x 1, 2, 4, 8 or 16 decisions: a boss's last hit points can take 100+ decisions
    # of steady play that reach no new cell, and a fixed short limit cuts every attempt off just before the kill
    # (Final Mission's stage 1 boss, found by the rl/goexplore.py session, 2026-11-14)
    def limits(n):
        return explore_steps * (2 ** torch.randint(0, 5, (n,), generator=gen, device=dev))
    lim = limits(N)
    used, t0, t_log, it, goal = 0, time.time(), time.time(), 0, None
    fm_best = None
    t_save = t0
    level_seen = {}                                                   # level -> seconds when first reached

    def route(goal):
        """The route to the goal, or (goal None: out of time / archive, or a periodic save) to the deepest state."""
        if goal is None:
            n = L0 + used
            score = lvl[:n].double() * 1e6 + prog[:n].double() * 1e3 - steps[:n].double() * 1e-6
            best = int(score.argmax())
            goal = dict(console=-1, last=best, tail=torch.zeros(0, dtype=torch.int16, device=dev), won=False,
                        level=int(lvl[best]))
        chain, s = [], goal['last']
        while s >= L0:
            chain.append(s)
            s = int(parent[s])
        chain.reverse()
        # restores are exact (set_restore_exact), so the route is the links' buttons one after another
        buttons = torch.cat([acts[s, :int(alen[s])] for s in chain] + [goal['tail']]).cpu().numpy()
        stats = dict(game=game, seconds=round(time.time() - t0, 1), decisions=it * N, cells=used,
                     goal_reached=goal['console'] >= 0, won=goal['won'], level=goal['level'],
                     route_decisions=len(buttons), levels_first_reached=level_seen)
        return buttons, chain, steps[chain].cpu().numpy(), lvl[chain].cpu().numpy(), stats

    while goal is None and time.time() - t0 < seconds and cap - L0 - used > 0:
        for k, size in enumerate((9, 3, 3)):
            new = torch.randint(0, size, (N,), generator=gen, device=dev)
            pad[k] = torch.where(torch.rand(N, generator=gen, device=dev) < change, new, pad[k])
        a = (pad[0] + 9 * (pad[1] + 3 * pad[2])).to(torch.int16)
        if game == 'tetris':
            a = env.tetris.act(env.ram, just)
        env.actions.copy_(a)
        hist[ar, since.clamp(max=HIST - 1)] = a
        cell, clv = game_cells(env, game)
        env.kill.copy_(((age >= lim) & (cell >= 0)).to(torch.uint8))
        env.step()
        it += 1
        since += 1
        age += 1
        done = env.done.bool()
        just = done
        lim = torch.where(done, limits(N), lim)
        end = env.info[:, 2]
        cell, clv = game_cells(env, game)
        reached = (~done & (clv >= target)) | (done & (end == END_WON))
        if reached.any():
            i = int(reached.nonzero()[0])
            goal = dict(console=i, last=int(last[i]), tail=hist[i, :int(since[i])].clone(), won=bool(done[i]),
                        level=int(clv[i]) if not done[i] else int(env.info[i, 4]))
            break
        valid = (cell >= 0) & ~done & (since <= HIST)
        if known.numel():
            pos = torch.searchsorted(known, cell).clamp(max=known.numel() - 1)
            valid &= known[pos] != cell
        idx = valid.nonzero().squeeze(1)
        room = cap - L0 - used
        if idx.numel() and room > 0:
            keys, inv = torch.unique(cell[idx], return_inverse=True)
            first = torch.full((keys.numel(),), N, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, idx, 'amin')
            keys, first = keys[:room], first[:room]
            sl = L0 + used + torch.arange(keys.numel(), device=dev)
            env.place_save(0, first.to(torch.int32).contiguous(), sl.to(torch.int32).contiguous())
            parent[sl] = last[first]
            steps[sl] = steps[last[first]] + since[first]
            lvl[sl] = clv[first]
            prog[sl] = torch.nan_to_num(env.progress_now[first]) if game != 'smb' else (keys >> 3 & 0x1FF).float()
            if 'life_value' in spec:
                life[sl] = env.ram[first, 0].float()
                strong[sl] = fm_strong(env.ram[first].long()).float()
            acts[sl] = hist[first]
            alen[sl] = since[first]
            last[first] = sl
            since[first] = 0
            known = torch.cat([known, keys]).sort().values
            used += keys.numel()
            for lv in clv[first].unique().tolist():
                level_seen.setdefault(int(lv), round(time.time() - t0, 1))
        start = env.cur_start.long()
        last = torch.where(done, start, last)
        since[done] = 0
        age[done] = 0
        chosen.index_add_(0, start[done], torch.ones(int(done.sum()), dtype=torch.float64, device=dev))
        if it % 4 == 0:
            # newest levels first (each level further back counts 10x less); within them, least chosen first
            top = int(lvl[:L0 + used].max())
            wt = (1.0 / (1.0 + chosen[:L0 + used])) * torch.pow(0.1, (top - lvl[:L0 + used]).clamp(min=0).double())
            if 'life_value' in spec:
                n = L0 + used
                # HP dealt = strong HP below the most seen in any cell at the same (level, progress bucket)
                pos = lvl[:n] * 100000 + (prog[:n] * spec['buckets']).long()
                keys, inv = torch.unique(pos, return_inverse=True)
                most = torch.zeros(keys.numel(), dtype=torch.float32, device=dev).scatter_reduce_(0, inv, strong[:n], 'amax')
                # only where the screen stops: the scroll is automatic (a bucket passes in ~16 decisions), so cells of
                # one bucket reached more than 48 decisions apart mean a halt (a fight); elsewhere strong HP also drops
                # when enemies leave the screen, which is not damage
                lo = torch.full((keys.numel(),), 1 << 62, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, steps[:n], 'amin')
                hi = torch.zeros(keys.numel(), dtype=torch.int64, device=dev).scatter_reduce_(0, inv, steps[:n], 'amax')
                dealt = ((most[inv] - strong[:n]) * ((hi - lo)[inv] > 48)).double()
                val = (lvl[:n].double() + prog[:n].double() + spec['life_value'] * life[:n].double()
                       + spec['hp_value'] * dealt)
                val[:L0] = -1e9       # level slots (the start) have no RAM record here; restarts go to archive cells
                best = int(val.argmax())
                fm_best = (int(dealt[best]), int(life[best]), float(prog[best]))
                if n > L0:
                    wt = wt * torch.exp(((val - val[L0:].max()) / spec['value_temp']).clamp(min=-60))
            wt[1:L0] = 0
            full = np.zeros(cap)
            full[:L0 + used] = (wt / wt.sum()).cpu().numpy()
            env.set_level_plan(0, 1.0, full)
        now = time.time()
        if on_save is not None and now - t_save > save_every and used:
            t_save = now
            on_save(route(None))
        if now - t_log > 15:
            t_log = now
            top = int(lvl[:L0 + used].max())
            m = lvl[:L0 + used] == top
            log(f'[explore] {now - t0:.0f}s {it * N / 1e6:.1f}M decisions, {used} cells, level {top} '
                f'({int(m.sum())} cells, best progress {float(prog[:L0 + used][m].max()):.3f}); levels first reached at '
                f'{level_seen}' + (f'; best value cell: HP dealt {fm_best[0]}, life {fm_best[1]}, progress '
                                   f'{fm_best[2]:.3f}' if fm_best else ''))
    return route(goal)


def main_whole_game(a):
    """--whole-game: search from power-on, save the route (buttons + states) and the power-on state."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = RlEnv({a.game: a.envs}, device='cuda', seed=a.seed, sticky=False, idle_max=0, dropout=False,
                level_cap=64 + a.archive, level_slots=64)
    env.set_attempts(a.attempt_limit, kill=True)
    env.set_attempt_through(True)
    env.set_restore_exact(True)
    if a.game == 'smb':
        env.bind_cells(32)
    elif a.game == 'tetris':
        from tetris_actor import ADDRS, TetrisActor
        env.bind_ram(ADDRS)
        env.tetris = TetrisActor(env.n, env.device, seed=a.seed)
    else:
        env.bind_ram([addr for addr, _ in GAME_CELLS[a.game]['ram']] or [0])
    env.set_no_render()
    env.drop_obs_buffers()
    slotlib.save_boot_entrance(env, 0)
    boot = slotlib.read(env, 0, [0])
    pre = dict(buttons=np.zeros(0, np.int16), steps=np.zeros(0, np.int64), levels=np.zeros(0, np.int64),
               states={k: v[:0] for k, v in boot.items()})
    start_level = 0
    if a.resume:
        # go on from the end of a saved route: its last state becomes slot 0 (an exact restore), so the replay is its
        # buttons, then the new ones
        r = np.load(a.resume)
        assert len(r['levels']), 'the route to resume has no states'
        boot = {k: r[f'boot_{k}'] for k in slotlib.FIELDS}
        slotlib.write(env, 0, [0], {k: r[f'state_{k}'][-1:] for k in slotlib.FIELDS})
        start_level = int(r['levels'][-1])
        assert json.loads(str(r['meta'])).get('exact_restore'), 'only routes found with exact restores can be resumed'
        pre = dict(buttons=r['buttons'], steps=r['steps'],
                   levels=r['levels'], states={k: r[f'state_{k}'] for k in slotlib.FIELDS})

    def save(route, final=False):
        buttons, chain, st, lv, stats = route
        rec = slotlib.read(env, 0, chain) if len(chain) else {k: v[:0] for k, v in boot.items()}
        stats = dict(stats, resumed_from=a.resume or None, route_decisions=len(pre['buttons']) + len(buttons),
                     final=final, exact_restore=True)
        np.savez_compressed(out / 'route.npz', buttons=np.concatenate([pre['buttons'], buttons]),
                            steps=np.concatenate([pre['steps'], st + len(pre['buttons'])]),
                            levels=np.concatenate([pre['levels'], lv]),
                            **{f'state_{k}': np.concatenate([pre['states'][k], v]) for k, v in rec.items()},
                            **{f'boot_{k}': v for k, v in boot.items()},
                            meta=json.dumps(dict(layout=env.level_layout(0), **stats)))
        print('[explore] ' + ('route ' if final else 'saved so far ') + json.dumps(stats), flush=True)

    save(explore_game(env, a.game, a.seconds, a.target, a.explore_steps, seed=a.seed, start_level=start_level,
                      log=lambda s: print(s, flush=True), on_save=save), final=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whole-game', action='store_true', help='one route from power-on through every level')
    ap.add_argument('--target', type=int, default=1 << 30, help='--whole-game: stop on reaching this level')
    ap.add_argument('--resume', default='', help='--whole-game: go on from the end of this route.npz')
    ap.add_argument('--entrances', default='', help='entrance archive (.npy, level slots of a training env)')
    ap.add_argument('--levels', default='', help='comma-separated level indices (default: every saved entrance)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--game', default='smb')
    ap.add_argument('--envs', type=int, default=32768)
    ap.add_argument('--archive', type=int, default=100000, help='archive states, all levels together')
    ap.add_argument('--seconds', type=float, default=1800, help='search time')
    ap.add_argument('--explore-steps', type=int, default=64, help='decisions per visit to an archive state')
    ap.add_argument('--spacing', type=int, default=24, help='decisions between kept route states')
    ap.add_argument('--attempt-limit', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    if a.whole_game:
        return main_whole_game(a)
    assert a.entrances, '--entrances is needed (or --whole-game)'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = RlEnv({a.game: a.envs}, device='cuda', seed=a.seed, sticky=False, idle_max=0, dropout=False,
                level_cap=64 + a.archive, level_slots=64)
    env.set_attempts(a.attempt_limit, kill=True)
    env.bind_cells(32)
    env.set_no_render()
    env.drop_obs_buffers()
    slotlib.load_entrances(env, 0, a.entrances)
    if 0 not in env.saved_levels()[a.game]:
        slotlib.save_boot_entrance(env, 0)     # the first level's entrance is power-on
    saved = env.saved_levels()[a.game]
    levels = [int(x) for x in a.levels.split(',')] if a.levels else sorted(saved)
    assert all(l in saved for l in levels), f'no saved entrance for {[l for l in levels if l not in saved]}'
    np.save(out / 'entrances.npy', slotlib.tail(env, 0, env.level_slots))
    lay = env.level_layout(0)

    def on_route(level, keep, steps, stats):
        rec = slotlib.read(env, 0, keep)
        np.savez_compressed(out / f'L{level}.npz', steps=np.asarray(steps, np.int64), **rec,
                            meta=json.dumps(dict(game=a.game, layout=lay, **stats)))
        print('[explore] route ' + json.dumps(stats), flush=True)

    found, stats = explore(env, levels, a.seconds, on_route, a.explore_steps, spacing=a.spacing, seed=a.seed,
                           log=lambda s: print(s, flush=True))
    print('[explore] done ' + json.dumps(dict(cleared=sorted(found), **stats)), flush=True)


if __name__ == '__main__':
    main()
