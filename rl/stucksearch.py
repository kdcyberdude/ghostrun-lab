"""Search when stuck (user design 2026-11-21), the search side: a CPU process the trainer starts (rl/stuck.py) when a
level has found nothing new for a long time. It runs beside training (no GPU, low priority) and hands back up to
TARGETS spots the learners have (almost) never reached, each with the states and buttons that lead there.

Search: the first phase of Go-Explore (Ecoffet et al. 2021) as in rl/explore.py, from the level's start and the
trainer's places in that level (where its learners already get): every console starts from an archive state, holds
semi-random buttons for 64-256 decisions, and each 32x32-pixel cell nobody stood on before is saved. A cell keeps up to
VARIANTS states (a reservoir of the different histories that reached it: the same spot after a maze's right or wrong
turn is the same cell, but not the same state). Restarts favour cells chosen least, and rare ones RARE_BONUS times.
Every console carries its whole button sequence from its seed (a restart adds one no-button decision: the restart's
4 neutral frames), so any find can be replayed exactly from its seed.

Rare = the trainer's own counts: an 8x8 block touched by at most rare_max training games (coverage.py bcount;
rare_max = max(rare_floor, rare_frac x the games that touched the level start's block)). A RUN is a console's time on
rare ground without touching common ground; its depth = (screens it entered, rare cells it entered). A cell's depth
is the deepest run that reached it. Targets: a level clear first (the deepest of all), then the deepest rare cells,
one per screen, at least MIN_CELLS deep or on a second screen. A jump above common ground is a shallow run; a way into
the unknown is a deep one.

Stop: no new rare cell for `quiet` decisions (after at least `min` decisions: the reachable unknown is used up), or
`max` decisions, `seconds`, or a full archive. progress.json says which, and how many rare cells the last `quiet`
decisions still found (> 0 at a cap = the budget was too small).

Hand-back (user's path design, 2026-11-21 ~19:00): one target (the clear if found, else the deepest) becomes its
WHOLE WAY FROM THE LEVEL START: the buttons of the learner game the seed came from (rl/stuck.py logs every level-start
learner game's buttons) + the search's buttons from the seed. It is replayed from the level start in IDLE_COPIES
consoles (a level start plays 0..idle_max random idle frames, so each console draws its own; the one whose RAM equals
the seed's at the seed's decision is the learner's game, exactly), then once more to save the states. Its NEW SPOT is
the first block on the way that games from the level start (learners, coverage.py bcount0) have (almost) never
touched and after which the way stays on such ground (at least WIN_RARE of the next WIN blocks); a way that is all
known ground but clears: the clear. Handed back (user, 2026-11-22): the state at EVERY decision before the new spot,
back to the level start, each with a try length of 2 x distance + 64; the trainer practises a sliding window of them
(rl/stuck.py WINDOW). The policy finds its own buttons from them (no replay). Without a matching console (or for the level start seed): the way
from the seed, states back to the seed.

GPU (2026-11-21, user: the search is part of training and must be fast): GSearch is the same search vectorized with
torch on the GPU (no per-console Python), run by the trainer itself between iterations (rl/stuck.py): training
pauses while it runs, it has the whole GPU, and its env is freed before training goes on. Search runs the CPU one.

usage: python rl/stucksearch.py JOB_DIR      (JOB_DIR/job.npz from rl/stuck.py; writes result.npz atomically)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlenv import RlEnv  # noqa: E402
import slots as slotlib  # noqa: E402
from coverage import BLOCK_COUNT_BITS, RAM_OFFSET, smb_block_keys  # noqa: E402

L0 = 64               # level slots of the search envs (seeds and archive come after them)
MAXLEN = 4096         # decisions of a sequence from its seed
VARIANTS = 4          # states kept per cell
EXPLORE_STEPS = 64    # decisions per visit, x 1, 2, 4, 8 or 16 (up to 1024: long stretches with nothing new)
CHANGE = 0.1          # chance per decision that a held button part changes
RARE_BONUS = 4.0      # restart weight of a rare cell
TARGETS = 10
WIN, WIN_RARE = 32, 0.8   # the new spot: a rare block after which >= WIN_RARE of the next WIN blocks are rare
IDLE_COPIES = 32      # consoles replaying a way from the level start (each draws its own idle frames)
MIN_CELLS = 3         # a target's run entered at least this many rare cells (or a second screen)
CLEAR = -2            # target cell of a level clear
END_CLEAR, END_WON = 6, 2
MIX = 0x5851F42D4C957F2D
# budget (user 2026-11-21: some games need many more decisions to find the right path): at least 10M decisions, stop
# when 10M bring no new rare cell, at most 100M or 2.5 h (the trainer gives a search 3 h)
DEFAULTS = dict(envs=2048, threads=12, archive=32768, min=10_000_000, quiet=10_000_000, max=100_000_000,
                seconds=9000, seed=0, rare_floor=20, rare_frac=0.001)
# GPU (2026-11-21 ~16:00, user: use the GPU fully): the emulator runs one thread per console and is latency bound (a
# step takes about as long for 16k consoles as for 131k), so throughput grows with consoles: emulator alone, 8-4 states:
# 16k 78k/s, 32k 138k/s, 64k 233k/s, 131k 298k/s, 262k 363k/s (and half the steps per second). Whole search, 90 s:
# 8-4 89k/s -> 347k/s, 7-4 110k/s -> 446k/s, as many rare cells, 7-4 runs 5 -> 8 screens deep. (Restarting a whole warp
# from one cell, so its 32 consoles run the same code: +50% for the emulator alone but only +8% in the search, deaths
# break the groups up, and shallower finds: not used.) On the GPU min / quiet / max are decisions per REF_ENVS
# consoles: a step at 131k takes ~2.2x one at 16k, so the quiet window stays ~2 minutes as before, with 4x the decisions.
REF_ENVS = 32768
GPU_DEFAULTS = dict(DEFAULTS, envs=131072, seconds=1800)     # training waits for it: at most 30 min


def cell_of(bk):
    """8x8 block keys (coverage.smb_block_keys) -> 32x32 cell keys (level*128 + area) << 14 | x//32 << 4 | y//32;
    -1 stays -1. The cell's level is key >> 21, its screen (level, area, x // 256) key >> 7."""
    return torch.where(bk >= 0, ((bk >> 18) << 14) | (((bk >> 8) & 1023) << 4) | ((bk >> 2) & 15), -1)


# Hidden progress (2026-11-22; no address, level or game is named). A maze sends a run back unless hidden counters are
# right. The same spot after a right and after a wrong check is the same position cell, and the rare right states are
# washed out of its VARIANTS random states, so a half-solved maze is no stepping stone (7-4). GSearch finds the
# counters itself, from the camera and the 2 KB RAM as anonymous bytes:
#   send-back   the camera jumps back between two decisions in the same area (camera_of: the game's camera reader,
#               set_game; a game without a camera x page byte has no send-backs, and so no linked bytes);
#   linked byte a RAM byte the game changes at send-backs (at >= SB_SHARE of them, and >= SB_LIFT x its usual chance to
#               change in a decision) that took <= VMAX values and changes in <= RMAX of decisions: the state the game
#               used, or reset, when it sent the run back. Before SB_MIN send-backs there is none (plain position cells);
#   tag         a console entering a position cell with a linked byte whose value is RARE at that place (below THETA of
#               the entries into this position cell + 32-px camera cell, after NMIN entries; of several the rarest)
#               enters the cell (position, byte, value) instead. A tagged cell keeps TAGV states. What the camera shows
#               is the same for every entry of a place, so page counters and tile bytes among the linked bytes are
#               never rare;
#   restarts    the cells of one position share that position's weight (GSearch.weights): tags decide which states a
#               position keeps, never how often the search goes there, so they cannot thin out the frontier.
TAG_SHIFT = 40       # GSearch archive cell keys: the tag above the position bits
VMAX, RMAX = 16, 1.0 / 64
SB_MIN, SB_SHARE, SB_LIFT = 20, 0.05, 20.0
THETA, NMIN = 1.0 / 32, 128
TAG, TAGV = 1 << 19, 2          # a tag: TAG | byte << 8 | value
TAG_FRAC = 0.5                  # no new tagged cells once the archive is this full
TBITS = 24                      # hashed count tables


def smb_camera(ram):
    """SMB's camera x in level pixels (ScreenLeft page $71A, x $71C: what scripts/romlib/camera.py finds for SMB)."""
    return ram[:, 0x71A].long() * 256 + ram[:, 0x71C].long()


# The game searched and its readers (place keys in coverage.smb_block_keys' layout, camera x in level pixels or -1).
# SMB (the default, what training uses) keeps its hand-read readers; a table game's come from its rl/games_table.json
# row (rl/table_keys.py: every address found per ROM by scripts/romlib, the camera by scripts/romlib/camera.py).
GAME, place_keys, camera_of = 'smb', smb_block_keys, smb_camera


def set_game(slug, row=None):
    """Search `slug` from now on (row: its games_table.json entry; None for SMB)."""
    global GAME, place_keys, camera_of
    if row is None:
        GAME, place_keys, camera_of = slug, smb_block_keys, smb_camera
        return
    from table_keys import TableKeys
    tk = TableKeys(row)
    GAME, place_keys, camera_of = slug, tk.keys, tk.camera


def hash_(x, bits):
    return ((x * MIX) >> (64 - bits)) & ((1 << bits) - 1)


def block_slot(bk):
    """Row of coverage.py's block count table (the same hash)."""
    return ((bk * MIX) >> (64 - BLOCK_COUNT_BITS)) & ((1 << BLOCK_COUNT_BITS) - 1)


def rare_limit(seeds, bcount, p):
    """rare_max from the games that touched the block of the first seed (the level start)."""
    ram = torch.from_numpy(seeds['mem'][:1, RAM_OFFSET:RAM_OFFSET + 2048].copy())
    bk = place_keys(ram)
    b0 = int(bcount[int(block_slot(bk)[0])]) if int(bk[0]) >= 0 else 0
    return max(int(p['rare_floor']), int(p['rare_frac'] * b0)), b0


def make_env(n, cap, seed, device='cpu', idle=0, exact=False):
    """exact: starts from a slot play no restart frames (RlEnv.set_restore_exact), as the trainer's env with the
    hand-off on (rl/stuck.py setup_env): the hand-back replays use it; the search itself does not (its sequences
    carry a restart as one no-button decision)."""
    env = RlEnv({GAME: n}, device=device, seed=seed, sticky=False, idle_max=idle, dropout=False, level_cap=cap,
                level_slots=L0, pictures=False)
    env.set_attempts(1 << 30, kill=True)     # attempts end at a death, a clear or a kill (no decision limit)
    env.set_allow_skip(True)                 # as in training (--coverage)
    env.set_no_render()
    env.drop_obs_buffers()
    env.bind_ram(range(2048))
    if exact:
        env.set_restore_exact(True)
    if env.device.type == 'cuda':
        torch.cuda.empty_cache()             # the picture buffers made at construction are never written
    return env


class Search:
    def __init__(self, env, n_seeds, bcount, rare_max, level, p, log=print):
        self.env, self.S, self.bcount, self.rare_max, self.level, self.p, self.log = \
            env, n_seeds, bcount, rare_max, level, p, log
        N, cap = env.n, env.level_cap
        self.A0 = L0 + n_seeds                      # first archive slot (seeds at L0..A0-1)
        self.free = list(range(cap - 1, self.A0 - 1, -1))
        self.rng = np.random.default_rng(p['seed'])
        self.gen = torch.Generator().manual_seed(p['seed'])
        i64 = lambda n, v=0: np.full(n, v, np.int64)  # noqa: E731
        # archive slots: sequence from the seed, seed, cell index (-1 = a seed), depth of the run (screens, cells),
        # screen of the run's last rare step
        self.seq = np.zeros((cap, MAXLEN), np.uint8)
        self.slen = i64(cap)
        self.sseed = i64(cap, -1)
        self.sseed[L0:self.A0] = np.arange(n_seeds)
        self.scell, self.sscr, self.scel, self.slast = i64(cap, -1), i64(cap), i64(cap), i64(cap, -1)
        self.chosen = np.zeros(cap)                 # restarts from each seed (cells count theirs per cell)
        # cells
        self.cid = {}                               # cell key -> index
        self.ckey, self.crare, self.cvar, self.centries, self.cchosen = [], [], [], [], []
        self.clears = []                            # (sequence, seed) of level clears (first few)
        # consoles
        self.cseq = np.zeros((N, MAXLEN), np.uint8)
        self.clen, self.cseed = i64(N), i64(N)
        self.ccell = i64(N, -1)                     # cell after the last decision
        self.rscr, self.rcel, self.rlast = i64(N), i64(N), i64(N, -1)   # the current run
        self.age = i64(N)
        self.lim = self._limits(N)
        self.pad = [torch.randint(0, k, (N,), generator=self.gen) for k in (9, 3, 3)]
        self.decisions, self.rare_found, self.last_rare = 0, 0, 0
        self.rare_hist = []                         # (decisions, rare cells found so far)

    def _limits(self, n):
        return EXPLORE_STEPS * 2 ** self.rng.integers(0, 5, n)

    def weights(self):
        w = np.zeros(self.env.level_cap)
        w[L0:self.A0] = 1.0 / np.sqrt(1.0 + self.chosen[L0:self.A0])
        for i, slots in enumerate(self.cvar):
            if slots:
                w[slots] = (RARE_BONUS if self.crare[i] else 1.0) / math.sqrt(1.0 + self.cchosen[i]) / len(slots)
        return w / w.sum()

    def restart_from(self, c, s):
        """Console c starts again from slot s: its sequence is the slot's plus one no-button decision (the restart)."""
        n = self.slen[s]
        self.cseq[c, :n] = self.seq[s, :n]
        self.cseq[c, n] = 0
        self.clen[c] = n + 1
        self.cseed[c] = self.sseed[s]
        ci = self.scell[s]
        self.ccell[c] = self.ckey[ci] if ci >= 0 else -1
        self.rscr[c], self.rcel[c], self.rlast[c] = self.sscr[s], self.scel[s], self.slast[s]
        if ci >= 0:
            self.cchosen[ci] += 1
        else:
            self.chosen[s] += 1

    def keep(self, c, slot, ci):
        self.seq[slot] = self.cseq[c]
        self.slen[slot], self.sseed[slot], self.scell[slot] = self.clen[c], self.cseed[c], ci
        self.sscr[slot], self.scel[slot], self.slast[slot] = self.rscr[c], self.rcel[c], self.rlast[c]

    def run(self):
        env, p, N = self.env, self.p, self.env.n
        w = np.zeros(env.level_cap)
        w[L0:self.A0] = 1.0
        env.set_level_plan(0, 1.0, w / w.sum())
        env.start()
        for c, s in enumerate(env.cur_start.numpy()):
            self.restart_from(c, int(s))
        t0 = t_log = time.time()
        stop, it = None, 0
        while stop is None:
            for k, size in enumerate((9, 3, 3)):
                new = torch.randint(0, size, (N,), generator=self.gen)
                self.pad[k] = torch.where(torch.rand(N, generator=self.gen) < CHANGE, new, self.pad[k])
            a = (self.pad[0] + 9 * (self.pad[1] + 3 * self.pad[2])).to(torch.int16)
            env.actions.copy_(a)
            full = self.clen >= MAXLEN - 1
            env.kill.copy_(torch.from_numpy((((self.age >= self.lim) & (self.ccell >= 0)) | full).astype(np.uint8)))
            env.step()
            it += 1
            self.decisions += N
            self.cseq[np.arange(N), np.minimum(self.clen, MAXLEN - 1)] = a.numpy().astype(np.uint8)
            self.clen = np.minimum(self.clen + 1, MAXLEN - 1)
            self.age += 1
            done = env.done.numpy().astype(bool)
            end = env.info[:, 2].numpy()
            for c in np.flatnonzero(done & ((end == END_CLEAR) | (end == END_WON))):
                if len(self.clears) < 4:
                    self.clears.append((self.cseq[c, :self.clen[c]].copy(), int(self.cseed[c])))
                    self.last_rare = self.decisions
            bk = place_keys(env.ram)
            cell = cell_of(bk).numpy()
            ok = ~done & (cell >= 0) & ((cell >> 21) == self.level)
            rare = ok & (self.bcount[block_slot(bk).numpy()] <= self.rare_max)
            common = ok & ~rare
            self.rscr[common], self.rcel[common], self.rlast[common] = 0, 0, -1
            moved = ok & (cell != self.ccell) & (self.clen < MAXLEN - 2)   # (a full sequence is not kept)
            self.rcel[rare & moved] += 1
            scr = cell >> 7
            ns = rare & (scr != self.rlast)
            self.rscr[ns] += 1
            self.rlast[ns] = scr[ns]
            self.ccell = np.where(ok, cell, np.where(done, -1, self.ccell))
            saves = {}
            for c in np.flatnonzero(moved):
                key = int(cell[c])
                ci = self.cid.get(key)
                if ci is None:
                    if not self.free:
                        stop = 'archive full'
                        continue
                    ci = self.cid[key] = len(self.ckey)
                    self.ckey.append(key); self.crare.append(bool(rare[c])); self.cvar.append([])
                    self.centries.append(0); self.cchosen.append(0.0)
                    if rare[c]:
                        self.rare_found += 1
                        self.last_rare = self.decisions
                self.centries[ci] += 1
                var = self.cvar[ci]
                if len(var) < VARIANTS and self.free:
                    slot = self.free.pop()
                    var.append(slot)
                elif var and self.rng.random() < VARIANTS / self.centries[ci]:
                    slot = var[self.rng.integers(len(var))]
                    if slot in saves.values():
                        continue
                else:
                    continue
                saves[int(c)] = slot
                self.keep(c, slot, ci)
            if saves:
                env.place_save(0, torch.tensor(list(saves), dtype=torch.int32),
                               torch.tensor(list(saves.values()), dtype=torch.int32))
            starts = env.cur_start.numpy()
            for c in np.flatnonzero(done):
                self.restart_from(c, int(starts[c]))
            self.age[done] = 0
            self.lim[done] = self._limits(int(done.sum()))
            if it % 4 == 0:
                env.set_level_plan(0, 1.0, self.weights())
            now = time.time()
            self.rare_hist.append((self.decisions, self.rare_found))
            if stop is None and self.decisions >= p['min'] and self.decisions - self.last_rare >= p['quiet']:
                stop = 'used up'
            elif stop is None and self.decisions >= p['max']:
                stop = 'decision cap'
            elif stop is None and now - t0 >= p['seconds']:
                stop = 'time cap'
            if now - t_log > 15 or stop:
                t_log = now
                self.progress(now - t0, stop)
        return stop

    def depth(self, slot):
        return int(self.sscr[slot]) * 100_000 + int(self.scel[slot])

    def targets(self):
        """[(cell key or CLEAR, sequence, seed, screens, cells)] best first (module notes)."""
        out = [(CLEAR, s, sd, -1, -1) for s, sd in self.clears[:1]]
        cand = []
        for i, slots in enumerate(self.cvar):
            if self.crare[i] and slots:
                s = max(slots, key=self.depth)
                if self.scel[s] >= MIN_CELLS or self.sscr[s] >= 2:
                    cand.append((self.depth(s), i, s))
        screens = set()
        for _, i, s in sorted(cand, reverse=True):
            if len(out) >= TARGETS:
                break
            if self.ckey[i] >> 7 in screens:
                continue
            screens.add(self.ckey[i] >> 7)
            out.append((self.ckey[i], self.seq[s, :self.slen[s]].copy(), int(self.sseed[s]), int(self.sscr[s]),
                        int(self.scel[s])))
        return out

    def recent(self):
        """Rare cells found in the last `quiet` decisions."""
        lo = self.decisions - self.p['quiet']
        old = next((r for dd, r in reversed(self.rare_hist) if dd <= lo), 0)
        return self.rare_found - old

    def progress(self, seconds, stop):
        best = sorted((self.depth(s) for slots in self.cvar for s in slots), reverse=True)[:1]
        info = dict(level=self.level, seconds=round(seconds, 1), decisions=self.decisions,
                    rate=round(self.decisions / max(seconds, 1e-9)), cells=len(self.ckey), rare_cells=self.rare_found,
                    rare_recent=self.recent(), clears=len(self.clears),
                    deepest_screens=best[0] // 100_000 if best else 0, deepest_cells=best[0] % 100_000 if best else 0,
                    slots=self.env.level_cap - self.A0 - len(self.free), rare_max=self.rare_max, stop=stop)
        self.info = info
        self.log('[stucksearch] ' + json.dumps(info))


class GSearch:
    """Search, vectorized on the env's device (module notes): the same cells, variants, sequences, runs and stops."""

    def __init__(self, env, n_seeds, bcount, rare_max, level, p, log=print):
        dev, N, cap = env.device, env.n, env.level_cap
        self.env, self.S, self.bcount, self.rare_max, self.level, self.p, self.log, self.dev = \
            env, n_seeds, bcount, rare_max, level, p, log, dev
        self.N, self.cap = N, cap
        self.A0 = self.next = L0 + n_seeds
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(p['seed'])
        i64 = dict(dtype=torch.int64, device=dev)
        full = lambda n, v: torch.full((n,), v, **i64)  # noqa: E731
        self.seq = torch.zeros(cap, MAXLEN, dtype=torch.uint8, device=dev)
        self.slen, self.sseed, self.scell = full(cap, 0), full(cap, -1), full(cap, -1)
        self.sseed[L0:self.A0] = torch.arange(n_seeds, device=dev)
        self.sscr, self.scel, self.slast = full(cap, 0), full(cap, 0), full(cap, -1)
        self.schosen = torch.zeros(cap, device=dev)
        self.ckey, self.cnv, self.cent = full(cap, -1), full(cap, 0), full(cap, 0)
        self.crare = torch.zeros(cap, dtype=torch.bool, device=dev)
        self.cvar = torch.full((cap, VARIANTS), -1, **i64)
        self.cch = torch.zeros(cap, device=dev)
        self.nc = 0
        self.skeys, self.sids = full(0, 0), full(0, 0)       # sorted cell keys -> cell index
        self.cseq = torch.zeros(N, MAXLEN, dtype=torch.uint8, device=dev)
        self.clen, self.cseed, self.ccell = full(N, 0), full(N, 0), full(N, -1)
        self.rscr, self.rcel, self.rlast = full(N, 0), full(N, 0), full(N, -1)
        self.age, self.lim = full(N, 0), self._limits(N)
        self.pad = [torch.randint(0, k, (N,), generator=self.gen, device=dev) for k in (9, 3, 3)]
        self.clears = []
        self.rare_found = torch.zeros((), **i64)
        self.decisions, self.found_host, self.last_rare, self.full = 0, 0, 0, False
        self.rare_hist = []
        # hidden progress (module notes above): the console's last camera x and area, its tag, per-byte statistics
        # (changes, changes at send-backs, values seen), entries per place and per (place, byte, value)
        self.it = 0
        self.pcam, self.parea, self.ctag = full(N, 0), full(N, -1), full(N, 0)
        self.pvalid = torch.zeros(N, dtype=torch.bool, device=dev)
        self.pram = torch.zeros(N, 2048, dtype=torch.uint8, device=dev)
        self.pram_ok = torch.zeros(N, dtype=torch.bool, device=dev)
        self.ref = None                                          # a reference value per byte (the first state seen)
        self.chg, self.sbchg = full(2048, 0), full(2048, 0)
        self.tot, self.nsb = torch.zeros((), **i64), torch.zeros((), **i64)
        self.vseen = torch.zeros(2048, 256, dtype=torch.bool, device=dev)
        self.nonref = torch.zeros(2048, dtype=torch.bool, device=dev)
        self.counted, self.linked = self.nonref.clone(), self.nonref.clone()
        self.b256 = torch.arange(2048, device=dev) * 256
        self.ntab = torch.zeros(1 << 22, dtype=torch.int32, device=dev)      # entries per place
        self.tab = torch.zeros(1 << TBITS, dtype=torch.int32, device=dev)    # ... with (byte, non-reference value)
        self.dtab = torch.zeros(1 << TBITS, dtype=torch.int32, device=dev)   # ... with byte != its reference value

    def _limits(self, n):
        return EXPLORE_STEPS * 2 ** torch.randint(0, 5, (n,), generator=self.gen, device=self.dev)

    def _bytes(self, ram, ok, sent_back):
        """Per-byte statistics and the linked bytes (module notes above)."""
        if self.ref is None:
            if not bool(ok.any()):
                return
            self.ref = ram[int(ok.nonzero()[0])].clone()
        both = ok & self.pram_ok
        self.chg += ((ram != self.pram) & both[:, None]).sum(0)
        self.tot += both.sum()
        sb = both & sent_back
        if bool(sb.any()):
            self.sbchg += (ram[sb] != self.pram[sb]).sum(0)
            self.nsb += sb.sum()
        self.nonref |= ((ram != self.ref[None]) & ok[:, None]).any(0)
        if self.it % 4 == 1:
            vi = ok.nonzero().squeeze(1)
            for c in vi.split(16384):       # in parts: a [consoles, 2048] int64 index is 2 GB at 131k consoles
                self.vseen.view(-1)[(ram[c].long() + self.b256[None]).flatten()] = True
        if self.it % 4 == 3:
            tot, chg, nsb, sbc = self.tot.double(), self.chg.double(), self.nsb.double(), self.sbchg.double()
            self.counted = self.nonref & (self.vseen.sum(1) <= VMAX)
            self.linked = self.counted & (chg <= RMAX * tot) & (nsb >= SB_MIN) & (sbc >= SB_SHARE * nsb) & \
                (sbc * tot >= SB_LIFT * chg * nsb)
        self.pram.copy_(ram)
        self.pram_ok = ok.clone()

    def _tags(self, pos, ok, cam, ram):
        """Every console's tag (0: none), set anew when it enters a position cell (module notes above). Every entry
        counts its place and, per counted byte, its value there (values other than the reference in tab, their sum in
        dtab: the reference value's count is the place's entries minus that)."""
        mi = (ok & (pos != (self.ccell & ((1 << TAG_SHIFT) - 1)))).nonzero().squeeze(1)
        if not mi.numel():
            return self.ctag
        place = pos[mi] * 4096 + ((cam[mi] >> 5) & 4095)
        hn = hash_(place, 22)
        n = self.ntab[hn].long()
        tag = torch.zeros_like(n)
        K = self.counted.nonzero().squeeze(1)
        if K.numel():
            v = ram[mi][:, K].long()
            nr = v != self.ref[K][None]
            pb = place[:, None] * 2048 + K[None]
            hd, ha = hash_(pb, TBITS), hash_(pb * 256 + v, TBITS)
            cnt = torch.where(nr, self.tab[ha].long(), (n[:, None] - self.dtab[hd].long()).clamp_min(0))
            rare = self.linked[K][None] & (n[:, None] >= NMIN) & ((cnt + 1).float() <= THETA * (n[:, None] + 1).float())
            if self.next - self.A0 > TAG_FRAC * (self.cap - self.A0):
                rare = torch.zeros_like(rare)
            best, j = torch.where(rare, cnt, 1 << 40).min(1)
            tag = torch.where(best < (1 << 40), TAG | (K[j] << 8) | v.gather(1, j[:, None]).squeeze(1), 0)
            one = torch.ones(int(nr.sum()), dtype=torch.int32, device=self.dev)
            self.tab.index_add_(0, ha[nr], one)
            self.dtab.index_add_(0, hd[nr], one)
        self.ntab.index_add_(0, hn, torch.ones(mi.numel(), dtype=torch.int32, device=self.dev))
        self.ctag[mi] = tag
        return self.ctag

    def _restart(self, idx, s):
        """Consoles idx start again from slots s (as Search.restart_from)."""
        n = self.slen[s]
        self.cseq[idx] = self.seq[s]
        self.cseq[idx, n.clamp(max=MAXLEN - 1)] = 0
        self.clen[idx] = (n + 1).clamp(max=MAXLEN - 1)
        self.cseed[idx] = self.sseed[s]
        ci = self.scell[s]
        key = torch.where(ci >= 0, self.ckey[ci.clamp_min(0)], -1)
        self.ccell[idx] = key
        self.ctag[idx] = torch.where(key >= 0, key >> TAG_SHIFT, 0)
        self.pvalid[idx] = False
        self.rscr[idx], self.rcel[idx], self.rlast[idx] = self.sscr[s], self.scel[s], self.slast[s]
        one = torch.ones(idx.numel(), device=self.dev)
        self.cch.index_add_(0, ci.clamp_min(0), one * (ci >= 0))
        self.schosen.index_add_(0, s, one * (ci < 0))
        self.age[idx] = 0
        self.lim[idx] = self._limits(idx.numel())

    def _keep(self, c, dst, ci):
        self.seq[dst] = self.cseq[c]
        self.slen[dst], self.sseed[dst], self.scell[dst] = self.clen[c], self.cseed[c], ci
        self.sscr[dst], self.scel[dst], self.slast[dst] = self.rscr[c], self.rcel[c], self.rlast[c]

    def weights(self):
        w = torch.zeros(self.cap, device=self.dev)
        w[L0:self.A0] = (1.0 + self.schosen[L0:self.A0]).rsqrt()
        n = self.nc
        if n:     # the cells of one position (plain and tagged): one weight, from the restarts of all of them, shared
            _, fam = torch.unique(self.ckey[:n] & ((1 << TAG_SHIFT) - 1), return_inverse=True)
            z = torch.zeros(int(fam.max()) + 1, device=self.dev)
            fch, fnv = z.index_add(0, fam, self.cch[:n]), z.index_add(0, fam, self.cnv[:n].float())
            frare = z.index_add(0, fam, self.crare[:n].float()) > 0
            cw = (torch.where(frare, RARE_BONUS, 1.0) * (1.0 + fch).rsqrt() / fnv.clamp_min(1))[fam]
            v = self.cvar[:n]
            ok = v >= 0
            w[v[ok]] = cw[:, None].expand_as(v)[ok]
        w = w.double().cpu().numpy()
        return w / w.sum()

    @torch.no_grad()
    def run(self):
        env, p, N, dev = self.env, self.p, self.N, self.dev
        w = np.zeros(self.cap)
        w[L0:self.A0] = 1.0
        env.set_level_plan(0, 1.0, w / w.sum())
        env.start()
        self._restart(torch.arange(N, device=dev), env.cur_start.long())
        t0 = t_log = time.time()
        stop, it = None, 0
        ar = torch.arange(N, device=dev)
        scale = N / REF_ENVS if dev.type == 'cuda' else 1.0
        d_min, d_quiet, d_max = p['min'] * scale, p['quiet'] * scale, p['max'] * scale
        while stop is None:
            for k, size in enumerate((9, 3, 3)):
                new = torch.randint(0, size, (N,), generator=self.gen, device=dev)
                self.pad[k] = torch.where(torch.rand(N, generator=self.gen, device=dev) < CHANGE, new, self.pad[k])
            a = self.pad[0] + 9 * (self.pad[1] + 3 * self.pad[2])
            env.actions.copy_(a.to(torch.int16))
            env.kill.copy_((((self.age >= self.lim) & (self.ccell >= 0)) | (self.clen >= MAXLEN - 1)).to(torch.uint8))
            env.step()
            it += 1
            self.it = it
            self.decisions += N
            self.cseq[ar, self.clen] = a.to(torch.uint8)
            self.clen = (self.clen + 1).clamp(max=MAXLEN - 1)
            self.age += 1
            done = env.done.bool()
            end = env.info[:, 2]
            if len(self.clears) < 4:
                cl = (done & ((end == END_CLEAR) | (end == END_WON))).nonzero().squeeze(1)
                for c in cl[:4 - len(self.clears)].tolist():
                    self.clears.append((self.cseq[c, :int(self.clen[c])].cpu().numpy(), int(self.cseed[c])))
                    self.last_rare = self.decisions
            bk = place_keys(env.ram)
            pos = cell_of(bk)
            ok = ~done & (pos >= 0) & ((pos >> 21) == self.level)
            cam = camera_of(env.ram)
            sent_back = ok & (cam >= 0) & self.pvalid & ((pos >> 14) == self.parea) & (cam < self.pcam - 64)
            self.pcam, self.parea = torch.where(ok, cam, self.pcam), torch.where(ok, pos >> 14, self.parea)
            self.pvalid |= ok
            self._bytes(env.ram, ok, sent_back)
            cell = torch.where(pos >= 0, pos | (self._tags(pos, ok, cam, env.ram) << TAG_SHIFT), -1)
            rare = ok & (self.bcount[block_slot(bk)] <= self.rare_max)
            common = ok & ~rare
            self.rscr = torch.where(common, 0, self.rscr)
            self.rcel = torch.where(common, 0, self.rcel)
            self.rlast = torch.where(common, -1, self.rlast)
            moved = ok & (cell != self.ccell) & (self.clen < MAXLEN - 2)
            self.rcel += (rare & moved).long()
            scr = pos >> 7
            ns = rare & (scr != self.rlast)
            self.rscr += ns.long()
            self.rlast = torch.where(ns, scr, self.rlast)
            self.ccell = torch.where(ok, cell, torch.where(done, -1, self.ccell))
            self._cells(moved.nonzero().squeeze(1), cell, rare)
            di = done.nonzero().squeeze(1)
            if di.numel():
                self._restart(di, env.cur_start[di].long())
            if it % 4 == 0:
                env.set_level_plan(0, 1.0, self.weights())
            if it % 16 == 0:
                f = int(self.rare_found)
                if f > self.found_host:
                    self.found_host, self.last_rare = f, self.decisions
                self.rare_hist.append((self.decisions, f))
                now = time.time()
                if self.full:
                    stop = 'archive full'
                elif self.decisions >= d_min and self.decisions - self.last_rare >= d_quiet:
                    stop = 'used up'
                elif self.decisions >= d_max:
                    stop = 'decision cap'
                elif now - t0 >= p['seconds']:
                    stop = 'time cap'
                if now - t_log > 15 or stop:
                    t_log = now
                    self.progress(now - t0, stop)
        return stop

    def _cells(self, mi, cell, rare):
        """Consoles mi stepped into a new cell: new cells get a slot; known ones a new variant while they have fewer
        than VARIANTS, else replace one with chance VARIANTS / entries (one console per cell per step)."""
        if not mi.numel():
            return
        dev, N = self.dev, self.N
        keys = cell[mi]
        if self.skeys.numel():
            pos = torch.searchsorted(self.skeys, keys).clamp(max=self.skeys.numel() - 1)
            found = self.skeys[pos] == keys
            cid = torch.where(found, self.sids[pos], -1)
        else:
            found = torch.zeros_like(keys, dtype=torch.bool)
            cid = torch.full_like(keys, -1)
        cs, ss, ids = [], [], []
        nm = ~found
        if bool(nm.any()):
            nk, inv = torch.unique(keys[nm], return_inverse=True)
            first = torch.full((nk.numel(),), N, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, mi[nm], 'amin')
            cnt = torch.bincount(inv, minlength=nk.numel())
            k = min(nk.numel(), self.cap - self.next)
            if k < nk.numel():
                self.full = True
            nk, first, cnt = nk[:k], first[:k], cnt[:k]
            if k:
                cidn = torch.arange(self.nc, self.nc + k, device=dev)
                sl = torch.arange(self.next, self.next + k, device=dev)
                self.ckey[cidn], self.crare[cidn], self.cnv[cidn], self.cent[cidn] = nk, rare[first], 1, cnt
                self.cvar[cidn, 0] = sl
                self.rare_found += rare[first].sum()
                self.nc += k
                self.next += k
                order = torch.cat([self.skeys, nk]).argsort()
                self.skeys = torch.cat([self.skeys, nk])[order]
                self.sids = torch.cat([self.sids, cidn])[order]
                cs.append(first); ss.append(sl); ids.append(cidn)
        if bool(found.any()):
            kc, kid = mi[found], cid[found]
            self.cent.index_add_(0, kid, torch.ones_like(kid))
            uc, inv = torch.unique(kid, return_inverse=True)
            one = torch.full((uc.numel(),), N, dtype=torch.int64, device=dev).scatter_reduce_(0, inv, kc, 'amin')
            nv = self.cnv[uc]
            vcap = torch.where((self.ckey[uc] >> TAG_SHIFT) != 0, TAGV, VARIANTS)
            add = nv < vcap
            u = torch.rand(uc.numel(), generator=self.gen, device=dev)
            rep_ = ~add & (u < vcap / self.cent[uc].clamp_min(1).float())
            na = int(add.sum())
            na = min(na, self.cap - self.next)
            if na:
                ua, oa = uc[add][:na], one[add][:na]
                sl = torch.arange(self.next, self.next + na, device=dev)
                self.cvar[ua, self.cnv[ua]] = sl
                self.cnv[ua] += 1
                self.next += na
                cs.append(oa); ss.append(sl); ids.append(ua)
            if bool(rep_.any()):
                ur, orr = uc[rep_], one[rep_]
                j = torch.randint(0, VARIANTS, (ur.numel(),), generator=self.gen, device=dev) % self.cnv[ur]
                cs.append(orr); ss.append(self.cvar[ur, j]); ids.append(ur)
        if cs:
            c, dst, ci = torch.cat(cs), torch.cat(ss), torch.cat(ids)
            self.env.place_save(0, c.to(torch.int32).contiguous(), dst.to(torch.int32).contiguous())
            self._keep(c, dst, ci)

    def depth_max(self):
        n = self.next
        d = self.sscr[self.A0:n] * 100_000 + self.scel[self.A0:n]
        return int(d.max()) if d.numel() else 0

    def recent(self):
        lo = self.decisions - self.p['quiet']
        old = next((r for dd, r in reversed(self.rare_hist) if dd <= lo), 0)
        return self.found_host - old

    def progress(self, seconds, stop):
        best = self.depth_max()
        self.info = dict(level=self.level, seconds=round(seconds, 1), decisions=self.decisions,
                         rate=round(self.decisions / max(seconds, 1e-9)), cells=self.nc, rare_cells=self.found_host,
                         rare_recent=self.recent(), clears=len(self.clears), deepest_screens=best // 100_000,
                         deepest_cells=best % 100_000, slots=self.next - self.A0, rare_max=self.rare_max, stop=stop,
                         device=str(self.dev), tagged=int(((self.ckey[:self.nc] >> TAG_SHIFT) != 0).sum()),
                         send_backs=int(self.nsb), linked=int(self.linked.sum()))
        self.log('[stucksearch] ' + json.dumps(self.info))

    def targets(self):
        """As Search.targets (module notes), from the device arrays."""
        out = [(CLEAR, s, sd, -1, -1) for s, sd in self.clears[:1]]
        n = self.nc
        if not n:
            return out
        v = self.cvar[:n]
        d = torch.where(v >= 0, self.sscr[v.clamp_min(0)] * 100_000 + self.scel[v.clamp_min(0)], -1)
        best, j = d.max(1)
        s = v.gather(1, j[:, None]).squeeze(1)
        ok = self.crare[:n] & (best >= 0) & ((self.scel[s.clamp_min(0)] >= MIN_CELLS) | (self.sscr[s.clamp_min(0)] >= 2))
        idx = ok.nonzero().squeeze(1)
        idx = idx[best[idx].argsort(descending=True)].tolist()
        keys = self.ckey.cpu().numpy() & ((1 << TAG_SHIFT) - 1)       # handed back as position cells (cell_of)
        slots = s.cpu().numpy()
        screens = set()
        for i in idx:
            if len(out) >= TARGETS:
                break
            if keys[i] >> 7 in screens:
                continue
            screens.add(int(keys[i] >> 7))
            sl = int(slots[i])
            out.append((int(keys[i]), self.seq[sl, :int(self.slen[sl])].cpu().numpy(), int(self.sseed[sl]),
                        int(self.sscr[sl]), int(self.scel[sl])))
        return out


META = ('cell', 'mask', 'target', 'screens', 'cells', 'spot_at', 'length', 'learner', 'matched', 'rare')


def save_result(job_dir, info, targets):
    """result.npz (read by rl/stuck.py), written atomically."""
    job_dir = Path(job_dir)
    arrays = dict(info=json.dumps(info), n=len(targets))
    for i, t in enumerate(targets):
        arrays.update({f't{i}_meta': json.dumps({k: t[k] for k in META}),
                       **{f't{i}_{k}': t[k] for k in ('buttons', 'pos', 'dist', 'wt', 'lim')},
                       **{f't{i}_{k}': v for k, v in t['states'].items()}})
    tmp = job_dir / 'result.tmp.npz'
    with open(tmp, 'wb') as f:
        np.savez(f, **arrays)
    tmp.replace(job_dir / 'result.npz')


def replay(env, s, buttons, save=None, who=0, match=None):
    """Search env (make_env): every console restarts from slot s and plays `buttons`. Returns keys [n, D] (block keys
    after each button), ends [n] (buttons played before the console's game ended, D: none ended), clear [n] (it ended
    in a clear) and same [n] (match = (j, ram): its RAM after button j equals ram; j = 0: right after the restart).
    save {j: slot}: console `who`'s state after button j goes to that slot."""
    n, D = env.n, len(buttons)
    w = np.zeros(env.level_cap)
    w[s] = 1.0
    env.set_level_plan(0, 1.0, w)
    env.kill.fill_(1)
    env.step()                    # the attempts end; every console starts again from slot s
    env.kill.fill_(0)
    assert bool((env.cur_start.long() == s).all())
    keys = np.full((n, D), -1, np.int64)
    ends = np.full(n, D, np.int64)
    alive, clear, same = np.ones(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    saves = dict(save or {})
    one = torch.tensor([who], dtype=torch.int32)

    def look(j):
        nonlocal same
        if match is not None and match[0] == j:
            same = alive & (env.ram.cpu().numpy() == np.asarray(match[1])[None]).all(1)
        if j in saves and alive[who]:
            env.place_save(0, one, torch.tensor([saves[j]], dtype=torch.int32))

    look(0)
    for j, a in enumerate(buttons, 1):
        env.actions.fill_(int(a))
        env.step()
        done = env.done.bool().cpu().numpy()
        end = env.info[:, 2].cpu().numpy()
        now = alive & done
        clear |= now & ((end == END_CLEAR) | (end == END_WON))
        ends[now] = j - 1
        alive &= ~done
        if not alive.any():
            break
        k = place_keys(env.ram).cpu().numpy()
        keys[alive, j - 1] = k[alive]
        look(j)
    return keys, ends, clear, same


def spot_of(rare, clear):
    """1-based button of the new spot on a way whose blocks after each button are rare or not (module notes,
    Hand-back); a way on known ground that clears: its clear (len + 1); None: no new spot."""
    D = len(rare)
    if D:
        c = np.concatenate([[0], np.cumsum(rare)])
        j = np.arange(D)
        hi = np.minimum(j + WIN, D)
        ok = rare & ((c[hi] - c[j]) >= WIN_RARE * (hi - j))
        if ok.any():
            return int(np.argmax(ok)) + 1
    return D + 1 if clear else None


def positions(r, clear):
    """Distances (decisions before the new spot at button r; 0 = its own state) of the handed-back states, nearest
    first: every decision (module notes, Hand-back), and equal weights (rl/stuck.py weighs the window itself)."""
    d = np.arange(1 if clear else 0, r + 1, dtype=np.int64)
    return d, np.ones(len(d))


def cell_blocks(cell):
    """The 16 8x8 block keys of a 32x32 cell, bit order of the trainer's mask: ((x // 8) & 3) * 4 + (y // 8) & 3."""
    la, cx, cy = cell >> 14, (cell >> 4) & 1023, cell & 15
    return np.array([(((la << 12) | (cx * 4 + a)) << 6) | (cy * 4 + b) for a in range(4) for b in range(4)], np.int64)


def hand_back(targets, seeds, hists, seed, bcount0, rare0, idle, log=print, exact=False):
    """The first target with a new spot -> its way from the level start, the new spot on it, the states before it
    (module notes, Hand-back). seeds: slot records ([0] = the level start); hists[i]: the buttons from the level start
    to seed i (empty: unknown); bcount0 / rare0: block counts of level-start learner games and their rare limit."""
    S = len(seeds['mem'])
    bc = torch.as_tensor(np.asarray(bcount0.cpu() if torch.is_tensor(bcount0) else bcount0))

    def rare(keys):
        k = torch.from_numpy(np.ascontiguousarray(keys))
        return ((k >= 0) & (bc[block_slot(k.clamp_min(0))] <= rare0)).numpy()

    def env_for(n, idle_, saves=0):
        """n consoles: slot 0 = the level start as a level slot (its idle frames are drawn), L0 + i = seed i, then
        `saves` slots for the handed-back states."""
        env = make_env(n, L0 + S + saves + 1, seed, idle=idle_, exact=exact)
        slotlib.write(env, 0, [0], {k: v[:1] for k, v in seeds.items()})
        slotlib.write(env, 0, np.arange(L0, L0 + S), seeds)
        w = np.zeros(env.level_cap)
        w[L0] = 1.0
        env.set_level_plan(0, 1.0, w)
        env.start()                   # from a slot: a power-on game is no attempt and cannot be killed
        return env

    for key, seq, sd, scr, cel in targets:
        h = len(hists[sd]) if sd < len(hists) else 0
        ways = []
        if h:       # the learner's game, then the search's way (its restart at the seed = one no-button decision)
            ways.append((np.concatenate([np.asarray(hists[sd], np.int64), np.asarray(seq, np.int64)]), 0,
                         IDLE_COPIES, idle, (h, seeds['mem'][sd, RAM_OFFSET:RAM_OFFSET + 2048])))
        # from the seed: the replay's restart there is the search's (seq[0]); an exact restart plays no frames, so
        # the search's restart is then played as its no-button decision
        ways.append((np.asarray(seq if exact else seq[1:], np.int64), L0 + sd, 1, 0, None))
        for buttons, s0, n, idle_, match in ways:
            keys, ends, clear, same = replay(env_for(n, idle_), s0, buttons, match=match)
            if match is not None and not same.any():
                log(f'[stucksearch] target {key}: no console replays the learner game to its seed (button {h}); '
                    f'the way from the seed instead')
                continue
            who = int(np.argmax(same)) if match is not None else 0
            kw = keys[who, :ends[who]]
            got = bool(clear[who]) if key == CLEAR else bool((cell_of(torch.from_numpy(kw)).numpy() == key).any())
            if not got:
                log(f'[stucksearch] target {key}: its way does not replay to it ({len(buttons)} buttons from slot {s0})')
                continue
            rk = rare(kw)
            r = spot_of(rk, key == CLEAR)
            if r is None:
                log(f'[stucksearch] target {key}: its way stays on known ground; no new spot')
                break
            at_clear = r > len(kw)
            spot = CLEAR if at_clear else int(cell_of(torch.tensor([kw[r - 1]]))[0])
            mask = 0 if at_clear else int(sum(1 << i for i, x in enumerate(rare(cell_blocks(spot))) if x))
            d, wt = positions(r, at_clear)
            pos = r - d
            slots = {int(j): L0 + S + i for i, j in enumerate(pos)}
            env = env_for(n, idle_, len(pos))
            _, _, _, same2 = replay(env, s0, buttons[:max(r, h if match is not None else 0)], save=slots, who=who,
                                    match=match)
            if match is not None and not same2[who]:
                log(f'[stucksearch] target {key}: the second replay differs from the first; left out')
                break
            learner = h if match is not None else 0
            log(f'[stucksearch] target {key} ({scr} screens, {cel} rare cells): way of {len(kw)} buttons from '
                f'{"the level start" if s0 == 0 or sd == 0 else f"seed {sd}"} ({learner} the learner\'s, idle copy '
                f'{who}); new spot {"the clear" if at_clear else spot} at button {r} '
                f'({int(rk[:r].sum())} rare blocks before it); {len(pos)} states (one per decision, distances {int(d[0])}..{int(d[-1])})')
            return [dict(cell=spot, mask=mask, target=int(key), screens=int(scr), cells=int(cel), spot_at=int(r),
                         length=int(len(kw)), learner=int(learner), matched=bool(match is not None), rare=int(rare0),
                         buttons=buttons[:r].astype(np.int16), pos=pos, dist=d, wt=wt, lim=2 * d + 64,
                         states=slotlib.read(env, 0, [slots[int(j)] for j in pos]))]
    return []


def main(job_dir):
    job_dir = Path(job_dir)
    z = np.load(job_dir / 'job.npz', allow_pickle=False)
    p = dict(DEFAULTS, **json.loads(str(z['params'])))
    torch.set_num_threads(p['threads'])
    level = int(z['level'])
    seeds = {k: z[f'seed_{k}'] for k in slotlib.FIELDS}
    S = len(seeds['mem'])
    bcount = z['bcount']
    log_f = open(job_dir / 'log.txt', 'a')

    def log(s):
        print(s, file=log_f, flush=True)

    def write_progress(info):
        tmp = job_dir / 'progress.tmp.json'
        tmp.write_text(json.dumps(info))
        tmp.replace(job_dir / 'progress.json')

    rare_max, b0 = rare_limit(seeds, bcount, p)
    log(f'[stucksearch] level {level}, {S} seeds, the level start block was touched by {b0} games: rare = at most '
        f'{rare_max}; params {json.dumps(p)}')
    env = make_env(p['envs'], L0 + S + p['archive'], p['seed'])
    slotlib.write(env, 0, np.arange(L0, L0 + S), seeds)
    srch = Search(env, S, bcount, rare_max, level, p)
    srch.log = lambda s: (log(s), write_progress(srch.info))
    t0 = time.time()
    stop = srch.run()
    info = dict(srch.info, stop=stop)
    del env
    hists = [z[f'hist_{i}'] if f'hist_{i}' in z.files else np.zeros(0, np.int64) for i in range(S)]
    bcount0 = z['bcount0'] if 'bcount0' in z.files else bcount
    rare0, _ = rare_limit(seeds, bcount0, p)
    targets = hand_back(srch.targets(), seeds, hists, p['seed'] + 1, bcount0, rare0, p.get('idle', 0), log,
                        exact=bool(p.get('exact', False)))
    info.update(handed=len(targets), hand_seconds=round(time.time() - t0 - info['seconds'], 1),
                matched=int(sum(t['matched'] for t in targets)), states=int(sum(len(t['pos']) for t in targets)))
    write_progress(info)
    save_result(job_dir, info, targets)
    log('[stucksearch] done ' + json.dumps(info))


if __name__ == '__main__':
    main(sys.argv[1])
