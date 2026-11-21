"""Stuck -> search -> hand-off (user design 2026-11-21), the trainer side (train.py --stuck-search 1). The same for
every level and game: nothing here knows a level's layout.

STUCK: per level, the learners' decisions spent in the level since an 8x8 block nobody had touched before was found
there (coverage.py `bever`, normal games). Reaching `threshold` (1M, the user's choice) with the level's clear rate
below CLEAR_MAX means the training has, with good confidence, stopped finding new things there (no new block in 1M
decisions: the chance of one is below 3 in 1M, rule of three). After a search that hands back nothing, that level's
threshold doubles (no endless searching of a used-up level).

SEEDS (path design, 2026-11-21): the rollout after the trigger saves states of learner games that began at the
level's start at the blocks such games touched least (coverage.py bcount0; CAPTURE_K per step, on the GPU, no host
sync), each with that game's buttons from the level start (every level-start learner game's buttons are logged, up
to HIST); up to SEEDS of them (one per 32x32 cell, least touched first) and the level start begin the search. No
saved places: the search starts where the model itself gets from the start, so its way can be followed back to it.
A level is searched only after MIN_START_GAMES such games were counted (bcount0 must know what the model reaches).
PLACES OFF (user, 2026-11-21 ~20:00): a level's games may start from its saved places (rl/places.py) until it is first
found stuck; from then on all its normal games start at the level start (the places' start weight goes to it), for
good: the model must get through from the start, the counts above fill fast, and later help comes from the search's
practice states only.

SEARCH (gpu=True, the default since 2026-11-21 at the user's request: the search is part of training): rl/stucksearch.py
GSearch on the GPU inside this process, between rollouts: training waits (minutes), the search has the whole GPU, its
env is freed before training goes on (no second GPU process beside the trainer's peak). gpu=False: the CPU search in its
own low-priority process (training never waits; ~14k decisions/s). Seeds from above, the trainer's block counts as
"known". One search at a time. Its progress (decisions, rare cells, why it stopped) goes to the metrics and TensorBoard.

HAND-OFF (user's path design, 2026-11-21 ~19:00, replacing the gates): the search hands back one way per level, from
the level start to its NEW SPOT (the first ground on the way that level-start learner games never reach and the way
stays on), as the state at every decision before the new spot (rl/stucksearch.py Hand-back), each with a try length
(2 x its distance + 64 decisions). SLIDING WINDOW (user, 2026-11-22: 8-4 practice reached the new spot from every
state on or after the hidden block and never from the ground before it, 7-10 decisions apart): WINDOW states are
practised at a time, the nearest first; a MASTERED state gives its slot to the next earlier state of the way, so the
start moves back one decision at a time to the way's start. Practice starts favour the earliest states of the window
(weight distance + 1). Only start states: the policy finds its own buttons. PRACTICE: about PRACTICE learner consoles at a
time play tries from them (the share of new games sent there adapts to keep that many); a try ends at a death, at the
game's end or at its length (it plays on past the new spot) and the trainer then ends the game (env attempts +
kill), so the console starts a new try or game at once. Reached = the try touched the new spot (a block of its cell
that was rare at the hand-off; a clear for a clear). DONE: when at least DONE_RATE of >= DONE_GAMES level-start
learner games (the recent ones) reach the new spot, it is LEARNED and its slots freed (the level may get stuck and
searched again later, further on); after GIVEUP_ITS iterations without that it is given up. While a level has a new
spot no new search starts for it; the level's normal starts are sampled LEVEL_BOOST times more while it is unsolved
(user, 2026-11-21). Practice games count nowhere else: not in the block counts (coverage.py `skip`), places, seeds,
the stuck counts or the level clear rates.

TEACHER TRIES (user, 2026-11-22: where the model is stuck, show it the search's buttons instead of waiting for it to
stumble on them): a share TEACH of the practice tries (learner consoles) are teacher tries: from their window state
the trainer presses the search's own buttons (the way's buttons from that state on) up to the new spot, then the try
ends. Their decisions carry an imitation loss in the trainer (train.py --imitate: the policy's log-probability of
the pressed buttons; no policy-gradient, value or entropy term there) and count for nothing else: not as tries of the
window (a state is mastered only by the model's own tries), and like every practice game nowhere in the stats.
Nothing else is imitated: only handed-off ways, only from states the model has not mastered, only until the new spot
is learned. The emulator is deterministic, so a teacher try must reach the new spot every time: the report's teacher
reached / tries says so. For that, starts from a slot are EXACT (env.set_restore_exact: no 4 neutral restart frames,
which would take a try off the way, e.g. release a held jump); the hand-back replays mirror it (stucksearch exact).

ENDS: with the env's attempts on and only_kill (a game from a saved slot ends only when killed), the trainer ends a
slot game one decision after reward10 ended it (won, game over, stall, skip) and gives it the same end reason
(info[:, 2]); power-on games end by themselves as before. Starts from a place or hand slot play no random idle frames
(the native attempt rule).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

import slots as slotlib
from stucksearch import CLEAR, block_slot, cell_of

STUCK_DECISIONS = 1_000_000
CLEAR_MAX = 0.1       # only levels cleared less often than this are searched
HAND_SLOTS = 256      # env slots for hand-off states (after every other slot)
MAXT = 32             # new spots at a time (device table rows)
LEVEL_BOOST = 2.0     # an unsolved level (clear < CLEAR_MAX) with a new spot: its normal starts get this many times
                      # their weight (user 2026-11-21: sample the unsolved levels more)
PRACTICE = 2048       # learner consoles in practice tries at a time (target of the adaptive share), teacher tries
                      # included (1024 until 2026-11-22: the model's own tries stay about as many)
TEACH = 0.5           # share of practice tries that are teacher tries (module notes, TEACHER TRIES)
BMAX = 8192           # buttons of a way (HIST of the learner's game + the search's MAXLEN)
SHARE0 = 0.25         # first share of new slot games that start at hand-off states (then adapted, SHARE_MIN..SHARE_MAX)
SHARE_MIN, SHARE_MAX = 0.02, 0.9
DONE_GAMES = 256      # level-start learner games (recent) that decide whether a new spot is learned ...
DONE_RATE = 0.25      # ... when at least this share of them reach it
GIVEUP_ITS = 360      # iterations (~5 h) without that and without the window moving back: given up
WINDOW = 100          # practice states of a new spot at a time (user 2026-11-22): the nearest unmastered ones of the
MASTER_WINS = 20      # way; a state is MASTERED at >= MASTER_WINS tries reaching the new spot and >= MASTER_RATE of
MASTER_RATE = 0.5     # its tries; it then gives its slot to the next earlier state of the way, until the way's start
MIN_START_GAMES = 1000    # level-start learner games counted (bcount0) before a level is searched
HIST = 4096           # buttons logged per level-start learner game (a longer game gives no seed)
JOB_HOURS = 3.0       # a search without a result after this long is given up
CAPTURE_K = 4         # seed candidates saved per step (and rollout unit) in the capture rollout
SEEDS = 64            # captured seeds per search
NICE = 10
THREADS = 12
# reward10 event bits that end a game (rl/csrc/reward10.cuh kEv*) -> its end reason (kEnd*: rlenv END_NAMES)
END_EVENTS = ((8, 2), (16, 1), (32, 4), (1, 3))     # won, game over, stall, skip
END_MASK = 8 | 16 | 32 | 1
SAFETY_LIMIT = 100_000


class StuckSearch:
    def __init__(self, run, N, dev, slot0, levels, learner, seed, threshold=STUCK_DECISIONS, sync=False, params=None,
                 game=0, T=48, units=1, gpu=True, idle=0):
        self.gpu = gpu and dev.type == 'cuda'
        self.job_last = None                             # the GPU search just finished (reported once)
        self.run, self.N, self.dev, self.slot0, self.LV = Path(run), N, dev, slot0, levels
        self.seed0 = slot0 + HAND_SLOTS                 # capture staging slots after the hand slots
        self.stage_n = T * units * CAPTURE_K
        self.U = units
        self.capture = None                              # level whose seeds the current rollout captures
        self.learner, self.threshold, self.sync, self.game = learner, threshold, sync, game
        self.idle = idle                                 # the env's random idle frames at a level start (0..idle)
        self.params = {**dict(threads=THREADS), **(params or {})}
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed * 7907 + 11)
        # host
        self.dry = np.zeros(levels)
        self.no_places = np.zeros(levels, bool)          # levels found stuck: normal games start at the level start
        self.fails = np.zeros(levels, np.int64)
        self.blocks_prev = None
        self.job = None
        self.jobs = 0
        self.targets = [None] * MAXT
        self.slot_row = np.full(HAND_SLOTS, -1, np.int64)
        self.slot_level = np.full(HAND_SLOTS, -1, np.int64)
        self.share = SHARE0
        self.n_prac = 0
        self.events = []                     # finished searches / ended gates since the last report
        self.done_jobs = []                  # summaries of every finished search
        # device
        i64, d = dict(dtype=torch.int64, device=dev), dev
        self.c_slot = torch.full((N,), -1, **i64)       # hand slot the console's practice game started from (-1 none)
        self.c_open = torch.zeros(N, dtype=torch.bool, device=d)   # its try is not counted yet
        self.c_hit = torch.zeros(N, dtype=torch.bool, device=d)    # it touched the gate
        self.c_age = torch.zeros(N, **i64)               # decisions since the try began
        self.c_why = torch.zeros(N, **i64)               # end reason of a game the trainer ends (0: a limit)
        self.s_row = torch.full((HAND_SLOTS,), -1, **i64)
        self.s_lim = torch.full((HAND_SLOTS,), 64, **i64)  # a try's length from that slot (decisions)
        self.t_cell = torch.full((MAXT,), -1, **i64)      # new spot: 32x32 cell (CLEAR: the level's clear)
        self.t_level = torch.full((MAXT,), -1, **i64)
        self.t_mask = torch.zeros(MAXT, **i64)            # the cell's blocks that count (16 bits, rare at hand-off)
        self.lv_row = torch.full((levels,), -1, **i64)    # level -> its new spot's row
        self.acc = torch.zeros(2, HAND_SLOTS + 1, **i64)   # learner tries, tries that reached the new spot
        self.acc0 = torch.zeros(2, MAXT + 1, **i64)        # level-start learner games ended, of them reached it
        self.n0 = torch.zeros(levels, **i64)               # level-start learner games ended per level (counted)
        # teacher tries (module notes): the ways' buttons, each slot's distance, the consoles in a teacher try
        self.t_btn = torch.zeros(MAXT, BMAX, dtype=torch.int16, device=d)
        self.t_n = torch.zeros(MAXT, **i64)                # buttons of the way to the new spot (0: none known)
        self.s_dist = torch.zeros(HAND_SLOTS, **i64)       # decisions from the slot's state to the new spot
        self.c_teach = torch.zeros(N, dtype=torch.bool, device=d)
        self.c_tpos = torch.zeros(N, **i64)                # the way's next button
        self.acc_t = torch.zeros(2, MAXT + 1, **i64)       # teacher tries, of them reached the new spot
        # level-start learner games (the model's own from the start): their buttons and whether they reach the spot
        self.ar = torch.arange(N, device=d)
        self.g_fs = torch.zeros(N, dtype=torch.bool, device=d)
        self.g_lvl = torch.full((N,), -1, **i64)
        self.g_row = torch.full((N,), -1, **i64)
        self.g_hit = torch.zeros(N, dtype=torch.bool, device=d)
        self.h_buf = torch.zeros(N, HIST, dtype=torch.int16, device=d)
        self.h_len = torch.zeros(N, **i64)
        self.h_on = torch.zeros(N, dtype=torch.bool, device=d)
        self.cap_key = torch.full((self.stage_n,), -1, **i64)   # captured seed candidates: cell, block count, saved
        self.cap_cnt = torch.zeros(self.stage_n, **i64)
        self.cap_ok = torch.zeros(self.stage_n, dtype=torch.bool, device=d)
        self.cap_hist = torch.zeros(self.stage_n, HIST, dtype=torch.int16, device=d)   # ... its buttons from the start
        self.cap_hlen = torch.zeros(self.stage_n, **i64)
        self.arange_k = torch.arange(CAPTURE_K, device=d)

    @staticmethod
    def slots_needed(T, units):
        return HAND_SLOTS + T * units * CAPTURE_K

    @staticmethod
    def setup_env(env):
        """Attempts that only the trainer ends (module notes, ENDS): creates env.kill and env.cur_start. A slot game
        also ends after SAFETY_LIMIT decisions (games take ~2,000; a game whose end was somehow missed would
        otherwise hold its console for ever)."""
        env.set_attempts(SAFETY_LIMIT, kill=True)
        env.set_attempt_through(True)
        env.set_only_kill(True)
        env.set_restore_exact(True)     # a try starts on its saved state itself (module notes, TEACHER TRIES)

    # ---------------- per decision (device, no host sync) ----------------
    def practising(self, sl):
        """[n] bool: the consoles sl are in a practice game."""
        return self.c_slot[sl] >= 0

    def from_start(self, sl):
        """[n] bool: the consoles sl play a learner game that began at a level start (coverage.py bcount0)."""
        return self.g_fs[sl]

    @torch.no_grad()
    def fix_end(self, env, sl, done):
        """Right after a step: a game the trainer ended gets its real end reason; its kill is cleared."""
        why = self.c_why[sl]
        env.info[sl, 2] = torch.where(done & (why > 0), why.float(), env.info[sl, 2])
        self.c_why[sl] = torch.where(done, 0, why)
        env.kill[sl] = torch.where(done, 0, env.kill[sl])

    def _count(self, sl, m):
        te = self.c_teach[sl]
        fin = m & self.learner[sl] & ~te                 # the model's own tries: the window's counts
        col = torch.where(fin, self.c_slot[sl], HAND_SLOTS)
        self.acc[0].index_add_(0, col, fin.long())
        self.acc[1].index_add_(0, col, (fin & self.c_hit[sl]).long())
        ft = m & te                                      # teacher tries: counted per new spot, for the report only
        colt = torch.where(ft, self.s_row[self.c_slot[sl].clamp_min(0)].clamp_min(0), MAXT)
        self.acc_t[0].index_add_(0, colt, ft.long())
        self.acc_t[1].index_add_(0, colt, (ft & self.c_hit[sl]).long())
        self.c_open[sl] &= ~m

    @torch.no_grad()
    def teacher(self, sl):
        """Before a decision: ([n] bool: the console is in a teacher try, [n] the way's button to press there)."""
        s = self.c_slot[sl]
        row = self.s_row[s.clamp_min(0)].clamp_min(0)
        pos = self.c_tpos[sl]
        on = self.c_teach[sl] & (s >= 0) & (pos < self.t_n[row])
        return on, self.t_btn[row, pos.clamp(0, BMAX - 1)].long()

    def _at(self, row, bk, level_now, ev):
        """[n] bool: the consoles are at the new spot of `row` (a counted block of its cell, or its level's clear)."""
        tc = self.t_cell[row]
        b = bk.clamp_min(0)
        bit = (((b >> 6) & 3) << 2) | (b & 3)
        inside = (bk >= 0) & (tc >= 0) & (cell_of(b) == tc) & (((self.t_mask[row] >> bit) & 1) == 1)
        return inside | ((tc == CLEAR) & ((level_now.long() > self.t_level[row]) | ((ev & 8) != 0)))

    @torch.no_grad()
    def on_restart(self, sl, done, cur_start):
        """After a step: an open try whose game ended is counted; a new game from a hand slot is a practice try; an
        ended level-start learner game is counted for its level's new spot; a new one starts its button log."""
        self._count(sl, done & self.c_open[sl])
        gr, gl = self.g_row[sl], self.g_lvl[sl]
        fin = done & (gr >= 0)
        col = torch.where(fin, gr, MAXT)
        self.acc0[0].index_add_(0, col, fin.long())
        self.acc0[1].index_add_(0, col, (fin & self.g_hit[sl]).long())
        f0 = done & (gl >= 0)
        self.n0.index_add_(0, gl.clamp_min(0), f0.long())
        cs = cur_start.long()
        ns = cs - self.slot0
        nsc = ns.clamp(0, HAND_SLOTS - 1)
        new = done & (ns >= 0) & (ns < HAND_SLOTS) & (self.s_row[nsc] >= 0)
        self.c_slot[sl] = torch.where(done, torch.where(new, nsc, -1), self.c_slot[sl])
        self.c_open[sl] = torch.where(done, new, self.c_open[sl])
        self.c_hit[sl] &= ~done
        self.c_age[sl] = torch.where(done, 0, self.c_age[sl])
        row = self.s_row[nsc].clamp_min(0)
        te = new & self.learner[sl] & (self.t_n[row] > 0) & \
            (torch.rand(done.shape[0], device=self.dev, generator=self.gen) < TEACH)
        self.c_teach[sl] = torch.where(done, te, self.c_teach[sl])
        self.c_tpos[sl] = torch.where(done, self.t_n[row] - self.s_dist[nsc], self.c_tpos[sl])
        fs = done & (cs >= 0) & (cs < self.LV) & self.learner[sl]
        lvl = cs.clamp(0, self.LV - 1)
        self.g_fs[sl] = torch.where(done, fs, self.g_fs[sl])
        self.g_lvl[sl] = torch.where(done, torch.where(fs, lvl, -1), gl)
        self.g_row[sl] = torch.where(done, torch.where(fs, self.lv_row[lvl], -1), gr)
        self.g_hit[sl] &= ~done
        self.h_on[sl] = torch.where(done, fs, self.h_on[sl])
        self.h_len[sl] = torch.where(done, 0, self.h_len[sl])

    @torch.no_grad()
    def after_step(self, env, sl, bk, level_now, done, death):
        """bk: the consoles' 8x8 block keys after the step. Practice tries: new spot touched, try ended (death, its
        length); level-start learner games: their button, new spot touched; kills for the next decision: ended tries
        and slot games that reward10 ended (module notes)."""
        s = self.c_slot[sl]
        on = (s >= 0) & ~done
        sc = s.clamp_min(0)
        row = self.s_row[sc].clamp_min(0)
        ev = env.events[sl]
        self.c_hit[sl] |= on & self._at(row, bk, level_now, ev)
        age = self.c_age[sl] + on.long()
        self.c_age[sl] = age
        te = on & self.c_teach[sl]
        tpos = self.c_tpos[sl] + te.long()
        self.c_tpos[sl] = tpos
        used = te & (tpos >= self.t_n[row])              # a teacher try ends with the way's last button
        endt = on & self.c_open[sl] & (death | (age >= self.s_lim[sc]) | used)
        self._count(sl, endt)
        gr = self.g_row[sl]
        self.g_hit[sl] |= ~done & (gr >= 0) & self._at(gr.clamp_min(0), bk, level_now, ev)
        rec = ~done & self.h_on[sl]
        n = self.h_len[sl]
        rows, idx = self.ar[sl], n.clamp(max=HIST - 1)
        self.h_buf[rows, idx] = torch.where(rec & (n < HIST), env.actions[sl], self.h_buf[rows, idx])
        self.h_on[sl] = self.h_on[sl] & ~(rec & (n >= HIST))
        self.h_len[sl] = n + rec.long()
        gend = ~done & ((ev & END_MASK) != 0)
        why = torch.zeros_like(self.c_why[sl])
        for bit, code in END_EVENTS:
            why = torch.where((why == 0) & ((ev & bit) != 0), code, why)
        self.c_why[sl] = torch.where(gend, why, self.c_why[sl])
        # (an end event without done is a slot game's: a power-on game ends at its event by itself. env.cur_start is
        # no test: after a resume it is -1 until a console's next start, and a kill on a power-on game is harmless,
        # cleared at its end by fix_end)
        kill = ~done & (endt | gend | used)
        env.kill[sl] = torch.where(kill, 1, env.kill[sl])

    @torch.no_grad()
    def capture_step(self, env, t, unit, a, sl, bk, level_now, done, bcount0):
        """In the capture rollout: learner games that began at the level's start, at the blocks such games touched
        least, are saved (rl/places.py's staging, by bcount0) with their buttons from the start (module notes)."""
        if self.capture is None:
            return
        L = self.capture['level']
        ok = (self.h_on[sl] & ~done & (level_now.long() == L) & (bk >= 0) & (self.c_slot[sl] < 0) &
              (self.g_lvl[sl] == L))
        cnt = bcount0[block_slot(bk)].long()
        n = ok.shape[0]
        k = min(CAPTURE_K, n)
        score = ok.float() * (torch.rand(n, device=self.dev, generator=self.gen) * 0.01 + 1.0) / (1.0 + cnt.float())
        v, i = score.topk(k)
        good = v > 0
        j = (t * self.U + unit) * CAPTURE_K + self.arange_k[:k]
        env.place_save(self.game, (i + a).to(torch.int32), torch.where(good, self.seed0 + j, -1).to(torch.int32))
        self.cap_key[j] = cell_of(bk[i])
        self.cap_cnt[j] = cnt[i]
        self.cap_ok[j] = good
        r = self.ar[sl][i]
        self.cap_hist[j] = self.h_buf[r]
        self.cap_hlen[j] = self.h_len[r]

    # ---------------- per rollout (host) ----------------
    def weights(self, full, level_w, rate, slot_level):
        """full: start weights over the places' slots -> over every slot (module notes)."""
        out = np.zeros(self.slot0 + HAND_SLOTS + self.stage_n)     # every env slot (the env reads level_cap entries)
        out[:len(full)] = full
        head = out[:len(full)]
        for L in np.flatnonzero(self.no_places):          # PLACES OFF: the places' weight goes to the level start
            m = slot_level == L
            m[:self.LV] = False
            head[L] += head[m].sum()
            head[m] = 0.0
        act = [t for t in self.targets if t is not None]
        if not act:
            return out / out.sum()
        for L in sorted({t['level'] for t in act}):
            if rate[L] < CLEAR_MAX:
                out[:len(full)][slot_level == L] *= LEVEL_BOOST
        out *= (1.0 - self.share) / out.sum()
        for t in act:     # the earliest states of the window most (module notes, SLIDING WINDOW)
            w = np.asarray(t['dist'], np.float64) + 1.0
            out[self.slot0 + t['slots']] += self.share / len(act) * w / w.sum()
        return out / out.sum()

    def level_of(self, slots, levels):
        """Start slots of finished games -> their level; practice games (hand slots) -> -1 (not counted)."""
        s = np.asarray(slots, np.int64) - self.slot0
        out = np.array(levels, copy=True)
        out[(s >= 0) & (s < HAND_SLOTS)] = -1
        return out

    def end_rollout(self, env, b, blocks_level, rate, saved, cov, it):
        """Once per rollout, between rollouts: practice and new-spot outcomes, the stuck counts, the search."""
        self._outcomes(env, it)
        self.n_prac = int(((self.c_slot >= 0) & self.learner).sum())
        if any(t is not None for t in self.targets):     # keep about PRACTICE learners practising
            self.share = float(np.clip(self.share * np.sqrt(PRACTICE / max(self.n_prac, 64)), SHARE_MIN, SHARE_MAX))
        lv = b.b_lnow.long()
        lv = lv[self.learner.unsqueeze(0).expand_as(lv) & ~b.b_prac]
        dec = torch.bincount(lv[lv >= 0].clamp(max=self.LV - 1), minlength=self.LV).cpu().numpy()
        bl = np.zeros(self.LV, np.int64)
        bl[:len(blocks_level)] = blocks_level
        new = bl - (self.blocks_prev if self.blocks_prev is not None else bl)
        self.blocks_prev = bl
        self.dry[new > 0] = 0           # a new block this rollout: counting starts again (one rollout's
        self.dry += dec                 # decisions, ~30k at a level, are counted on top: 3% of 1M)
        self.new_blocks = new
        if self.job is not None:
            self._poll(env, it)
        if self.capture is not None and self.gpu:
            self._search_gpu(self.capture['level'], env, cov, it)
        elif self.capture is not None:
            self._launch(self.capture['level'], env, cov, it)
            if self.sync:
                self.proc.wait()
                self._poll(env, it)
        elif self.job is None:
            thr = self.threshold * 2.0 ** self.fails
            busy = {t['level'] for t in self.targets if t is not None}
            n0 = self.n0.cpu().numpy()
            stuck = np.array([self.dry[L] >= thr[L] and rate[L] < CLEAR_MAX and L in saved and L not in busy
                              for L in range(self.LV)])
            for L in np.flatnonzero(stuck & ~self.no_places):
                print(f'[stuck] level {L}: stuck ({self.dry[L] / 1e6:.1f}M learner decisions without a new block): its '
                      f'games start at the level start from now on (places off); search after {MIN_START_GAMES} '
                      f'level-start learner games ({int(n0[L])} so far)', flush=True)
            self.no_places |= stuck
            ok = stuck & (n0 >= MIN_START_GAMES)
            if ok.any():
                L = int(np.argmax(np.where(ok, self.dry / thr, -1)))
                self.capture = dict(level=L, it=it)      # the next rollout captures its seeds, then the search starts
                self.cap_ok.zero_()

    def _outcomes(self, env, it):
        a = self.acc[:, :HAND_SLOTS].cpu().numpy().copy()   # (on the CPU .cpu() is the same memory)
        g = self.acc0[:, :MAXT].cpu().numpy().copy()
        self.last_t = self.acc_t[:, :MAXT].cpu().numpy().copy()
        self.acc.zero_()
        self.acc0.zero_()
        self.acc_t.zero_()
        self.last_acc, self.last_g = a, g
        for row, t in enumerate(self.targets):
            if t is None:
                continue
            sl = t['slots']
            t['tries'] += a[0, sl]
            t['wins'] += a[1, sl]
            t['sg'] += int(g[0, row])
            t['sr'] += int(g[1, row])
            t['its'] += 1
            t['quiet'] += 1
            if t['sg'] >= 4 * DONE_GAMES:              # the recent games count most
                t['sg'] //= 2
                t['sr'] //= 2
            if self._slide(env, t):
                t['quiet'] = 0
            end = ('learned' if t['sg'] >= DONE_GAMES and t['sr'] >= DONE_RATE * t['sg'] else
                   'given up' if t['quiet'] >= GIVEUP_ITS else None)
            if end:
                L = t['level']
                self.events.append(dict(kind='target', level=L, cell=t['cell'], end=end, it=it, born=t['born'],
                                        start_games=t['sg'], start_reached=t['sr'], states=len(sl)))
                print(f"[stuck] level {L}: new spot {t['cell']} {end}: {t['sr']}/{t['sg']} level-start games reach "
                      f"it (born it {t['born']}, now it {it})", flush=True)
                self.fails[L] = 0 if end == 'learned' else self.fails[L] + 1
                self._free(row)

    def _slide(self, env, t):
        """Mastered window states give their slots to the next earlier states of the way (module notes, SLIDING
        WINDOW). Returns whether the window moved."""
        m = np.flatnonzero((t['wins'] >= MASTER_WINS) & (t['wins'] >= MASTER_RATE * t['tries']))
        m = m[:max(0, len(t['all_dist']) - t['next'])]
        if not len(m):
            return False
        nxt = np.arange(t['next'], t['next'] + len(m))
        sl = t['slots'][m]
        slotlib.write(env, self.game, list(self.slot0 + sl), {k: v[nxt] for k, v in t['all_states'].items()})
        t['dist'][m] = t['all_dist'][nxt]
        t['lim'][m] = 2 * t['all_dist'][nxt] + 64
        t['tries'][m] = 0
        t['wins'][m] = 0
        t['next'] += len(m)
        t['mastered'] += len(m)
        f = torch.from_numpy(sl).to(self.dev)
        self.s_lim[f] = torch.from_numpy(np.asarray(t['lim'][m], np.int64)).to(self.dev)
        self.s_dist[f] = torch.from_numpy(np.asarray(t['dist'][m], np.int64)).to(self.dev)
        gone = torch.isin(self.c_slot, f)          # tries already playing from the old states are not counted
        self.c_open.masked_fill_(gone, False)
        return True

    def _free(self, row):
        sl = self.targets[row]['slots']
        f = torch.tensor(sl, dtype=torch.int64, device=self.dev)
        self.s_row[f] = -1
        self.slot_row[sl] = -1
        self.slot_level[sl] = -1
        gone = torch.isin(self.c_slot, f)          # their games play on as normal slot games
        self.c_slot.masked_fill_(gone, -1)
        self.c_open.masked_fill_(gone, False)
        self.c_teach.masked_fill_(gone, False)
        self.t_n[row] = 0
        self.lv_row[self.targets[row]['level']] = -1
        self.g_row.masked_fill_(self.g_row == row, -1)
        self.t_cell[row] = -1
        self.targets[row] = None

    def _seeds(self, L, env):
        """Seed slots of a search at level L (module notes), their records and buttons from the level start."""
        seeds, hists = [L], [np.zeros(0, np.int64)]
        ck, cc, co = self.cap_key.cpu().numpy(), self.cap_cnt.cpu().numpy(), self.cap_ok.cpu().numpy()
        ch, cl = self.cap_hist.cpu().numpy(), self.cap_hlen.cpu().numpy()
        cells = set()
        for j in np.flatnonzero(co)[np.argsort(cc[co], kind='stable')]:
            if len(cells) < SEEDS and ck[j] not in cells:
                cells.add(ck[j])
                seeds.append(self.seed0 + int(j))
                hists.append(ch[j, :cl[j]].astype(np.int64))
        self.capture = None
        self.cap_ok.zero_()
        return seeds, len(cells), slotlib.read(env, self.game, seeds), hists

    def _search_gpu(self, L, env, cov, it):
        """The search on the GPU, now (training waits), then the hand-back (CPU replays, seconds) and the hand-off."""
        import gc
        import stucksearch as ss
        d = self.run / 'search' / f'job{self.jobs:03d}_L{L}'
        d.mkdir(parents=True, exist_ok=True)
        seeds, ncap, rec, hists = self._seeds(L, env)
        p = {**ss.GPU_DEFAULTS, **{k: v for k, v in self.params.items() if k != 'threads'}, 'seed': self.jobs * 1000 + L,
             'idle': self.idle, 'exact': True}
        rare_max, b0 = ss.rare_limit(rec, cov.bcount.cpu().numpy(), p)
        rare0, b00 = ss.rare_limit(rec, cov.bcount0.cpu().numpy(), p)
        log_f = open(d / 'log.txt', 'a')

        def log(m):
            print(m, file=log_f, flush=True)

        def progress(info):
            (d / 'progress.tmp.json').write_text(json.dumps(info))
            (d / 'progress.tmp.json').replace(d / 'progress.json')
        log(f'[stucksearch] level {L}, {len(seeds)} seeds ({ncap} captured from level-start learner games, '
            f'{int(sum(len(x) for x in hists))} logged buttons), level start block touched by {b0} games: rare = at '
            f'most {rare_max} (new spot: by {b00} level-start learner games, rare = at most {rare0}); GPU in the '
            f'trainer; params {json.dumps(p)}')
        print(f'[stuck] level {L}: {self.dry[L] / 1e6:.1f}M learner decisions without a new block -> GPU search {d.name} '
              f'from {len(seeds)} states ({ncap} captured from learners); training waits', flush=True)
        self.job = dict(level=L, dir=str(d), started=time.time(), it=it, dry=float(self.dry[L]), seeds=len(seeds),
                        pid=None, progress=None)
        self.jobs += 1
        self.dry[L] = 0
        S = len(seeds)
        torch.cuda.empty_cache()
        senv = ss.make_env(p['envs'], ss.L0 + S + p['archive'], p['seed'], device=self.dev)
        slotlib.write(senv, 0, np.arange(ss.L0, ss.L0 + S), rec)
        srch = ss.GSearch(senv, S, cov.bcount, rare_max, L, p, log=lambda m: (log(m), progress(srch.info)))
        stop = srch.run()
        info = dict(srch.info, stop=stop)
        targets = srch.targets()
        del srch, senv
        gc.collect()
        torch.cuda.empty_cache()
        t1 = time.time()
        out = ss.hand_back(targets, rec, hists, p['seed'] + 1, cov.bcount0, rare0, self.idle, log, exact=True)
        info.update(handed=len(out), hand_seconds=round(time.time() - t1, 1),
                    matched=int(sum(t['matched'] for t in out)), states=int(sum(len(t['pos']) for t in out)))
        progress(info)
        ss.save_result(d, info, out)
        log('[stucksearch] done ' + json.dumps(info))
        self.job_last = dict(info, level=L, minutes=round((time.time() - self.job['started']) / 60, 1))
        self._apply(env, d, it)

    def _launch(self, L, env, cov, it):
        d = self.run / 'search' / f'job{self.jobs:03d}_L{L}'
        d.mkdir(parents=True, exist_ok=True)
        seeds, ncap, rec, hists = self._seeds(L, env)
        cells = range(ncap)
        p = dict(self.params, seed=self.jobs * 1000 + L, idle=self.idle, exact=True)
        np.savez(d / 'job.npz', level=L, bcount=cov.bcount.cpu().numpy(), bcount0=cov.bcount0.cpu().numpy(),
                 params=json.dumps(p), **{f'seed_{k}': v for k, v in rec.items()},
                 **{f'hist_{i}': h for i, h in enumerate(hists)})
        log = open(d / 'stdout.txt', 'a')
        self.proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve().parent / 'stucksearch.py'), str(d)],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, OMP_NUM_THREADS=str(p['threads'])),
                                     preexec_fn=lambda: os.nice(NICE))
        self.job = dict(level=L, dir=str(d), started=time.time(), it=it, dry=float(self.dry[L]), seeds=len(seeds),
                        pid=self.proc.pid, progress=None)
        self.jobs += 1
        print(f'[stuck] level {L}: {self.dry[L] / 1e6:.1f}M learner decisions without a new block -> search {d.name} '
              f'from {len(seeds)} states ({len(cells)} captured from learners; pid {self.proc.pid})', flush=True)
        self.dry[L] = 0

    def _poll(self, env, it):
        d = Path(self.job['dir'])
        try:
            self.job['progress'] = json.loads((d / 'progress.json').read_text())
        except (OSError, ValueError):
            pass
        if (d / 'result.npz').exists():
            self._apply(env, d, it)
        elif time.time() - self.job['started'] > JOB_HOURS * 3600:
            L = self.job['level']
            self.fails[L] += 1
            self.events.append(dict(kind='search', level=L, it=it, handed=0, stop='no result'))
            print(f'[stuck] level {L}: search {d.name} gave no result in {JOB_HOURS} h; given up', flush=True)
            self.job = None

    def _apply(self, env, d, it):
        z = np.load(d / 'result.npz', allow_pickle=False)
        info = json.loads(str(z['info']))
        L = self.job['level']
        free = list(np.flatnonzero(self.slot_row < 0))
        rows = [r for r in range(MAXT) if self.targets[r] is None]
        first = not any(t is not None for t in self.targets)
        slots, recs, made = [], {k: [] for k in slotlib.FIELDS}, []
        for i in range(int(z['n'])):
            pos = z[f't{i}_pos']
            meta = json.loads(str(z[f't{i}_meta']))
            n = min(WINDOW, len(pos))
            if n > len(free) or not rows or int(self.lv_row[L]) >= 0:
                print(f'[stuck] level {L}: no room for its new spot ({n} states, {len(free)} free slots)', flush=True)
                break
            sl = np.array([free.pop(0) for _ in range(n)], np.int64)
            row = rows.pop(0)
            allst = {k: z[f't{i}_{k}'] for k in slotlib.FIELDS}            # every state of the way, nearest first
            t = dict(level=L, cell=int(meta['cell']), mask=int(meta['mask']), slots=sl, pos=pos,
                     dist=np.array(z[f't{i}_dist'][:n]), all_dist=z[f't{i}_dist'], all_states=allst, next=n,
                     lim=z[f't{i}_lim'][:n], spot_at=int(meta['spot_at']), length=int(meta['length']),
                     learner=int(meta['learner']), matched=bool(meta['matched']), target=int(meta['target']),
                     rare=int(meta['rare']), its=0, quiet=0, sg=0, sr=0, mastered=0, buttons=z[f't{i}_buttons'],
                     tries=np.zeros(n, np.int64), wins=np.zeros(n, np.int64), born=it, job=d.name,
                     screens=meta['screens'], cells=meta['cells'])
            self.targets[row] = t
            self.slot_row[sl] = row
            self.slot_level[sl] = L
            slots += list(self.slot0 + sl)
            for k in slotlib.FIELDS:
                recs[k].append(allst[k][:n])
            self.t_cell[row], self.t_level[row], self.t_mask[row] = t['cell'], L, t['mask']
            st = torch.from_numpy(sl).to(self.dev)
            self.s_row[st] = row
            self.s_lim[st] = torch.from_numpy(np.asarray(t['lim'], np.int64)).to(self.dev)
            self._teach_tables(row, t)
            self.lv_row[L] = row
            made.append(t)
        if slots:
            slotlib.write(env, self.game, slots, {k: np.concatenate(v) for k, v in recs.items()})
            if first:
                self.share = SHARE0
        self.fails[L] = 0 if made else self.fails[L] + 1
        summary = dict(kind='search', level=L, it=it, job=d.name, handed=len(made), **{
            k: info.get(k) for k in ('stop', 'decisions', 'seconds', 'rate', 'cells', 'rare_cells', 'rare_recent',
                                     'clears', 'matched', 'states', 'rare_max')})
        self.events.append(summary)
        self.done_jobs.append(summary)
        spot = ', '.join(f"new spot {'the clear' if t['cell'] == CLEAR else t['cell']} at decision {t['spot_at']} of "
                         f"the way ({t['learner']} the learner's), {len(t['all_dist'])} states, {len(t['slots'])} practised at a time"
                         for t in made)
        print(f"[stuck] level {L}: search {d.name} done ({info.get('stop')}, {info.get('decisions', 0) / 1e6:.1f}M "
              f"decisions in {info.get('seconds', 0) / 60:.0f} min, {info.get('rare_cells')} rare cells, "
              f"{info.get('clears')} clears): {spot or 'nothing handed off'}", flush=True)
        self.job = None

    def _teach_tables(self, row, t):
        """Device tables of a new spot's teacher tries: its way's buttons and its window slots' distances. A target of
        a checkpoint written before the teacher tries gets its buttons from its search's result file."""
        if t.get('buttons') is None:
            p = self.run / 'search' / str(t.get('job')) / 'result.npz'
            if p.exists():
                z = np.load(p, allow_pickle=False)
                for i in range(int(z['n'])):
                    meta = json.loads(str(z[f't{i}_meta']))
                    if int(meta['cell']) == t['cell'] and int(meta['spot_at']) == t['spot_at']:
                        t['buttons'] = z[f't{i}_buttons']
        b = t.get('buttons')
        n = 0 if b is None or len(b) > BMAX else len(b)
        if n:
            self.t_btn[row, :n] = torch.from_numpy(np.asarray(b, np.int16)).to(self.dev)
        self.t_n[row] = n
        st = torch.from_numpy(np.asarray(t['slots'], np.int64)).to(self.dev)
        self.s_dist[st] = torch.from_numpy(np.asarray(t['dist'], np.int64)).to(self.dev)
        if not n:
            print(f"[stuck] level {t['level']}: no buttons for its way: no teacher tries", flush=True)

    # ---------------- report / checkpoint ----------------
    def report(self, rate):
        """Metrics row part (train.py metrics.jsonl and TensorBoard)."""
        thr = self.threshold * 2.0 ** self.fails
        watch = [L for L in range(self.LV) if rate[L] < CLEAR_MAX and (self.blocks_prev is None or self.blocks_prev[L])]
        lv = {}
        a = self.last_acc if hasattr(self, 'last_acc') else np.zeros((2, HAND_SLOTS), np.int64)
        g = self.last_g if hasattr(self, 'last_g') else np.zeros((2, MAXT), np.int64)
        tt = self.last_t if hasattr(self, 'last_t') else np.zeros((2, MAXT), np.int64)
        for row, t in enumerate(self.targets):
            if t is None:
                continue
            at, dist = a[:, t['slots']], np.asarray(t['dist'])
            part = lambda m: [int(at[0, m].sum()), int(at[1, m].sum())]  # noqa: E731
            lv[t['level']] = dict(spot='clear' if t['cell'] == CLEAR else int(t['cell']), states=len(t['slots']),
                                  way=int(t['spot_at']), tries=int(at[0].sum()), reached=int(at[1].sum()),
                                  near=part(dist <= 8), mid=part((dist > 8) & (dist <= 64)), far=part(dist > 64),
                                  start_games=int(t['sg']), start_reached=int(t['sr']),
                                  start_now=[int(g[0, row]), int(g[1, row])], its=int(t['its']),
                                  window=[int(dist.min()), int(dist.max()), len(t['all_dist'])],
                                  mastered=int(t['mastered']), teacher=[int(x) for x in tt[:, row]])
        out = dict(dry_M={L: round(self.dry[L] / 1e6, 3) for L in watch},
                   trigger_M={L: round(thr[L] / 1e6, 3) for L in watch},
                   new_blocks={L: int(self.new_blocks[L]) for L in watch} if hasattr(self, 'new_blocks') else {},
                   job=self.job_last if self.job is None else dict(self.job['progress'] or {}, level=self.job['level'],
                                                          minutes=round((time.time() - self.job['started']) / 60, 1)),
                   handoff=lv, practice=dict(consoles=self.n_prac, share=round(self.share, 4)),
                   start_games={L: int(v) for L, v in enumerate(self.n0.cpu().numpy()) if L in watch},
                   start_only=[int(L) for L in np.flatnonzero(self.no_places)],
                   events=self.events, searches=len(self.done_jobs))
        self.events = []
        self.job_last = None
        return out

    # (with the per-console state: a resume restores every console's game in progress, env.cur_start is -1 for them)
    DEV = ('s_row', 's_lim', 't_cell', 't_level', 't_mask', 'lv_row', 'acc', 'acc0', 'n0', 'c_slot', 'c_open', 'c_hit',
           'c_age', 'c_why', 'g_fs', 'g_lvl', 'g_row', 'g_hit', 'h_buf', 'h_len', 'h_on', 'cap_hist', 'cap_hlen',
           'c_teach', 'c_tpos', 'acc_t')

    def state_dict(self):
        return dict(version=4, dry=self.dry, no_places=self.no_places, fails=self.fails, blocks_prev=self.blocks_prev, job=self.job,
                    jobs=self.jobs, capture=self.capture, cap=(self.cap_key.cpu(), self.cap_cnt.cpu(), self.cap_ok.cpu()),
                    targets=self.targets, slot_row=self.slot_row, slot_level=self.slot_level, share=self.share,
                    events=self.events, done_jobs=self.done_jobs, gen=self.gen.get_state(), slot0=self.slot0,
                    dev={k: getattr(self, k).cpu() for k in self.DEV})

    def load_state_dict(self, s):
        assert s['slot0'] == self.slot0, 'hand slots moved'
        self.dry, self.fails, self.blocks_prev = s['dry'].copy(), s['fails'].copy(), s['blocks_prev']
        if s.get('no_places') is not None:
            self.no_places = s['no_places'].copy()
        self.job, self.jobs = s['job'], s['jobs']
        self.events, self.done_jobs = list(s['events']), list(s['done_jobs'])
        self.gen.set_state(s['gen'].cpu())
        self.capture = s.get('capture')
        if s.get('cap') is not None and len(s['cap'][0]) == self.stage_n:
            for x, v in zip((self.cap_key, self.cap_cnt, self.cap_ok), s['cap']):
                x.copy_(v)
        if s.get('version', 1) < 3:
            # targets of the earlier hand-off designs (replays, then gates) are dropped: their levels are searched
            # again (path design) once MIN_START_GAMES level-start learner games are counted (bcount0 starts empty)
            print(f"[stuck] {sum(t is not None for t in s['targets'])} targets of an earlier hand-off design dropped",
                  flush=True)
            self.capture = None
            return
        for k, v in s['dev'].items():
            getattr(self, k).copy_(v)
        if s.get('version', 1) < 4:
            # sparse path states (version 3): dropped; their levels are searched again (every-decision states, sliding
            # window) once found stuck. The level-start counts and button logs are kept.
            print(f"[stuck] {sum(t is not None for t in s['targets'])} targets with sparse path states dropped",
                  flush=True)
            self.capture = None
            for k in ('s_row', 't_cell', 't_level', 'lv_row', 'c_slot', 'g_row'):
                getattr(self, k).fill_(-1)
            for k in ('t_mask', 'acc', 'acc0', 'c_open', 'c_hit', 'g_hit'):
                getattr(self, k).zero_()
            return
        self.targets = list(s['targets'])
        self.slot_row, self.slot_level = s['slot_row'].copy(), s['slot_level'].copy()
        self.share = s['share']
        for row, t in enumerate(self.targets):
            if t is not None:
                self._teach_tables(row, t)
