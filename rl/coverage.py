"""Camera coverage for SMB (experiment 2026-11-16, train.py --coverage 1): the only reward is +1 for every camera
position (16-pixel step of the screen's left edge, per level and area) seen for the first time in this episode, so
an episode's return = the number of distinct camera positions it reached before game over. No death, level or time
reward. Sum of these per-step rewards = the episode's coverage, so the value baseline compares each play with
the average play from the same situation: "explored more than usual" gets a positive advantage.

Cells come from the 2 KB RAM the env reports after every decision (RlEnv.bind_ram). Per console, the cells seen in
the current episode are bits of a 2^18-bit hashed set (32 KB per console); a hash collision makes a new cell count
as seen (rare: ~1% at 8,000 cells in one game). `ever` keeps every cell any episode found, for the coverage graph.

Rarity bonus (rarity > 0, added 2026-11-16 at the user's request): a cell new in the episode pays
1 + rarity / sqrt(1 + n), n = how many earlier episodes (all consoles, all training) found it - a cell no game has
reached pays 1 + rarity, one found by thousands of games ~1. Counts live in a 2^22-entry hashed table.

Mario blocks (blocks > 0, added 2026-11-17 at the user's request): every 8x8-pixel block of the level Mario's position
(x $6D/$86, y $B5/$CE; per level and area) is in for the first time in this episode also pays. The camera only moves
left-right, so a vine climb or the other path through a maze castle found no new camera position; it does find new
blocks. Own hashed set, 2^19 bits per console.
Pay (2026-11-18, the user's intent: Mario should end up in every reachable block, and go for the untouched ones where
the camera has nothing new): blocks / (1 + n / block_half), n = earlier episodes (all consoles, all training) that
touched the block, from its own count table (2^23 entries). A block no game touched pays `blocks` (1 = one camera
position), one touched by block_half games half of that, one every game touches ~0. No flat pay per block.
Earlier tries, both dropped: a fixed 0.5 per block (it 2520-2660: blocks grew to ~2/3 of the reward, Mario collected
blocks instead of moving on) and a flat pay per block balanced to the camera's total plus a small rarity bonus
(it 2660-3570): ~1,650 common blocks a game paid ~175, a never-touched block ~0.3, so the untouched floor one step
from Mario's usual landing spot in 4-4 (the way to the hole into the bottom corridor) never mattered.
`bever` (added 2026-11-18) keeps every block any episode touched, like `ever` for the camera: only for the graphs
(blocks found per level), never a reward. `bcount0` (2026-11-21, rl/stuck.py path design): the same count over games
the policy itself played from a level start only (learners, not from a saved place or a practice state): what the
model reaches on its own. Never a reward.

RomMap: every camera position the game can show, read from the cartridge before training (the level tables of the
SMB disassembly): the 32 levels' areas, plus every area their pipes / vines lead to in that world (bonus rooms, coin
heavens, water parts, exit areas), each area's width from its object data. An area of 1 screen or a bonus room
(fixed camera) is one position per entrance. A side area entered at page p counts from p to its end.
"""
from pathlib import Path

import numpy as np
import torch

CELL = 16                 # camera pixels per cell
HASH_BITS = 18
BLOCK_BITS = 19
COUNT_BITS = 22           # rarity count table sizes (entries)
BLOCK_COUNT_BITS = 23
RAM_OFFSET = 144          # CPU RAM inside a saved slot's 'mem' record (rl/slots.py)


def make_key(level, area, cell):
    """level 0..31 (world * 4 + level), area = type * 32 + index (0..127), camera cell 0..4095."""
    return ((level * 128 + area) << 12) | cell


def smb_keys(ram):
    """ram [N, 2048] uint8 -> [N] int64 cell key (level, area, camera x // CELL), -1 outside normal play
    (title, dying, pipes, flagpole, falling below the screen). Area = AreaType $074E, AreaAddrsLOffset $074F."""
    r = ram.long()
    play = (r[:, 0x770] == 1) & (r[:, 0x772] == 3) & (r[:, 0x0E] == 8) & (r[:, 0xB5] < 2)
    level = r[:, 0x75F] * 4 + r[:, 0x75C]
    area = (r[:, 0x74E] & 3) * 32 + (r[:, 0x74F] & 31)
    cell = ((r[:, 0x71A] * 256 + r[:, 0x71C]) // CELL).clamp(max=4095)
    return torch.where(play & (level < 32), make_key(level, area, cell), torch.full_like(level, -1))


def smb_block_keys(ram):
    """ram [N, 2048] -> [N] int64 key of Mario's 8x8 block (level, area, x // 8, y // 8), -1 outside normal play."""
    r = ram.long()
    play = (r[:, 0x770] == 1) & (r[:, 0x772] == 3) & (r[:, 0x0E] == 8) & (r[:, 0xB5] < 2)
    level = r[:, 0x75F] * 4 + r[:, 0x75C]
    area = (r[:, 0x74E] & 3) * 32 + (r[:, 0x74F] & 31)
    bx = ((r[:, 0x6D] * 256 + r[:, 0x86]) // 8).clamp(max=4095)
    by = (r[:, 0xB5] * 256 + r[:, 0xCE]) // 8
    return torch.where(play & (level < 32), (((level * 128 + area) << 12 | bx) << 6) | by, torch.full_like(level, -1))


def smb_height(ram):
    """ram [N, 2048] -> [N] int64 Mario's exact height ($B5 * 256 + $CE), -1 outside normal play. For
    rl/situation.py's standing test (the 8-pixel row of the block key stays the same at the top of a jump)."""
    r = ram[:, [0x770, 0x772, 0x0E, 0xB5, 0xCE]].long()
    play = (r[:, 0] == 1) & (r[:, 1] == 3) & (r[:, 2] == 8) & (r[:, 3] < 2)
    return torch.where(play, r[:, 3] * 256 + r[:, 4], torch.full_like(r[:, 0], -1))


def key_level(keys):
    return keys >> 19


class Coverage:
    def __init__(self, n, device, rarity=0.0, blocks=0.0, block_half=100.0):
        self.n, self.device, self.rarity, self.blocks, self.block_half = n, device, rarity, blocks, block_half
        if blocks > 0:
            self.bseen = torch.zeros(n, 1 << (BLOCK_BITS - 6), dtype=torch.int64, device=device)
            self.bfound = torch.zeros(n, dtype=torch.int64, device=device)
            self.ended_blocks = []
            self.bcount = torch.zeros(1 << BLOCK_COUNT_BITS, dtype=torch.int32, device=device)  # episodes per block
            self.bcount0 = torch.zeros(1 << BLOCK_COUNT_BITS, dtype=torch.int32, device=device)  # ... level-start learners
            self.bpay_sum, self.bpay_n, self.cam_sum = 0.0, 0, 0.0   # since the last stats() call
            self.bever = torch.empty(0, dtype=torch.int64, device=device)   # every block any episode touched
        self.count = torch.zeros(1 << 22, dtype=torch.int32, device=device)   # episodes that found each cell
        self.bonus_sum, self.bonus_n = 0.0, 0
        self.seen = torch.zeros(n, 1 << (HASH_BITS - 6), dtype=torch.int64, device=device)
        self.found = torch.zeros(n, dtype=torch.int64, device=device)
        self.ever = torch.empty(0, dtype=torch.int64, device=device)
        self.ended = []           # coverage of the episodes finished since the last stats() call
        self.ended_level = []

    def _visit(self, keys, reset, seen=None, found=None, bits=HASH_BITS):
        """Episode starts (reset) clear their set first. Returns: the key is new in this episode."""
        seen = self.seen if seen is None else seen
        found = self.found if found is None else found
        seen[reset] = 0
        found[reset] = 0
        h = ((keys * -7046029254386353131) >> (64 - bits)) & ((1 << bits) - 1)   # 0x9E3779B97F4A7C15
        w, bit = (h >> 6).unsqueeze(1), torch.ones_like(h) << (h & 63)
        cur = seen.gather(1, w).squeeze(1)
        new = (keys >= 0) & ((cur & bit) == 0)
        seen.scatter_(1, w, torch.where(new, cur | bit, cur).unsqueeze(1))
        found += new
        return new

    def begin(self, ram):
        """Training start: every console's first picture opens an episode (its cell is free)."""
        reset = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self._visit(smb_keys(ram), reset)
        if self.blocks > 0:
            self._visit(smb_block_keys(ram), reset, self.bseen, self.bfound, BLOCK_BITS)

    def step(self, ram_after, done, skip=None, start0=None):
        """One decision for all consoles: ram_after = RAM after it (after an end: the next episode's start).
        skip [n] bool: games that are paid but counted nowhere (rl/stuck.py practice tries: the counts, `ever` and the
        episode stats stay the normal games'). start0 [n] bool: learner games from a level start (bcount0 counts only
        them). Returns the reward [n]."""
        cnt = ~done if skip is None else ~done & ~skip
        if done.any():
            end = done if skip is None else done & ~skip
            self.ended.append(self.found[end].clone())
            if self.blocks > 0:
                self.ended_blocks.append(self.bfound[end].clone())
        keys = smb_keys(ram_after)
        new = self._visit(keys, done)
        rew = (new & ~done)
        k = keys[rew & cnt]
        if k.numel():
            self.ever = torch.unique(torch.cat([self.ever, k]))
        out = rew.float()
        if self.rarity > 0:
            bonus = self._rare(self.count, keys, rew, COUNT_BITS, rew & cnt)
            out = out + bonus
            self.bonus_sum += bonus.sum()
            self.bonus_n += rew.sum()
        if self.blocks > 0:
            bkeys = smb_block_keys(ram_after)
            bnew = self._visit(bkeys, done, self.bseen, self.bfound, BLOCK_BITS) & ~done
            bc = bnew & cnt
            k = bkeys[bc]
            if k.numel():
                self.bever = torch.unique(torch.cat([self.bever, k]))
            slot = ((bkeys * 0x5851F42D4C957F2D) >> (64 - BLOCK_COUNT_BITS)) & ((1 << BLOCK_COUNT_BITS) - 1)
            self.last_bpay = self.blocks / (1.0 + self.bcount[slot].float() / self.block_half) * bnew.float()
            self.bcount.index_add_(0, slot[bc], torch.ones_like(slot[bc], dtype=torch.int32))
            if start0 is not None:
                b0 = bc & start0
                self.bcount0.index_add_(0, slot[b0], torch.ones_like(slot[b0], dtype=torch.int32))
            self.bpay_sum += self.last_bpay.sum()
            self.bpay_n += bnew.sum()
            self.cam_sum += out.sum()
        return out

    def _rare(self, count, keys, new, bits, counted=None):
        """Rarity bonus [n] for keys new in their episode (new), then count them (counted: only these; default new):
        rarity / sqrt(1 + earlier episodes)."""
        counted = new if counted is None else counted
        slot = ((keys * 0x5851F42D4C957F2D) >> (64 - bits)) & ((1 << bits) - 1)
        bonus = self.rarity / (1.0 + count[slot].float()).sqrt() * new.float()
        count.index_add_(0, slot[counted], torch.ones_like(slot[counted], dtype=torch.int32))
        return bonus

    def apply(self, b_ram, b_done, b_skip=None, b_start0=None):
        """A whole rollout: b_ram [T+1, N, 2048], b_done [T, N] (b_skip, b_start0 [T, N]: see step) -> rewards [T, N]
        (camera pay + block pay)."""
        out = []
        for t in range(b_done.shape[0]):
            r = self.step(b_ram[t + 1], b_done[t], None if b_skip is None else b_skip[t],
                          None if b_start0 is None else b_start0[t])
            out.append(r + self.last_bpay if self.blocks > 0 else r)
        return torch.stack(out)

    def stats(self, ref=None):
        e = torch.cat(self.ended).float() if self.ended else torch.zeros(0, device=self.device)
        self.ended = []
        out = dict(episodes=int(e.numel()), ep_mean=float(e.mean()) if e.numel() else None,
                   ep_max=int(e.max()) if e.numel() else None, ever=int(self.ever.numel()),
                   ever_levels=int(key_level(self.ever).unique().numel()))
        if ref is not None:
            out.update(ref.covered(self.ever))
        if self.blocks > 0:
            eb = torch.cat(self.ended_blocks).float() if self.ended_blocks else torch.zeros(0, device=self.device)
            self.ended_blocks = []
            out['blocks_mean'] = float(eb.mean()) if eb.numel() else None
            out['block_pay'] = float(self.bpay_sum / max(float(self.bpay_n), 1.0))     # per new block
            out['block_share'] = float(self.bpay_sum / max(float(self.bpay_sum + self.cam_sum), 1e-9))
            self.bpay_sum, self.bpay_n, self.cam_sum = 0.0, 0, 0.0
            out['blocks_ever'] = int(self.bever.numel())
            out['blocks_level'] = torch.bincount(self.bever >> 25, minlength=32).cpu().tolist()
        if self.rarity > 0:
            out['bonus_per_cell'] = float(self.bonus_sum / max(float(self.bonus_n), 1.0))
            self.bonus_sum, self.bonus_n = 0.0, 0
        return out

    def state_dict(self):
        return dict(ever=self.ever.cpu(), count=self.count.cpu(),
                    **(dict(bcount=self.bcount.cpu(), bever=self.bever.cpu(), bcount0=self.bcount0.cpu())
                      if self.blocks > 0 else {}))

    def load_state_dict(self, s):
        self.ever = s['ever'].to(self.device)
        if s.get('count') is not None:
            self.count.copy_(s['count'])
        if self.blocks > 0 and s.get('bcount') is not None:
            self.bcount.copy_(s['bcount'])
        if self.blocks > 0 and s.get('bcount0') is not None:
            self.bcount0.copy_(s['bcount0'])
        if self.blocks > 0 and s.get('bever') is not None:
            self.bever = s['bever'].to(self.device)


class RomMap:
    """All camera cells of SMB from the ROM (see the module doc). Table addresses are those of the SMB disassembly
    (doppelganger's smbdis), checked against the ROM bytes."""
    BONUS = 2 * 32 + 2          # underground type, index 2: the bonus coin rooms (fixed camera, one per entrance)
    INTRO = 0x29                # the walk-into-the-pipe intro before 1-2, 2-2, 4-2, 7-2

    def __init__(self, rom, device):
        d = open(rom, 'rb').read()
        prg = d[16:16 + 32768]
        b = lambda a: prg[a - 0x8000]   # noqa: E731
        assert prg[0x9cb4 - 0x8000:0x9cbc - 0x8000] == bytes([0, 5, 10, 14, 19, 23, 27, 32]), 'not the SMB tables'
        assert prg[0x9d28 - 0x8000:0x9d2c - 0x8000] == bytes([0, 3, 0x19, 0x1c])
        ah = [b(0x9d28 + i) for i in range(4)]
        eh = [b(0x9ce0 + i) for i in range(4)]
        area_at = lambda t, i: b(0x9d2c + ah[t] + i) | b(0x9d4e + ah[t] + i) << 8   # noqa: E731
        enemy_at = lambda t, i: b(0x9ce4 + eh[t] + i) | b(0x9d06 + eh[t] + i) << 8   # noqa: E731

        def width(t, i):            # last page any of the area's objects is on
            a, page, last = area_at(t, i) + 2, 0, 0
            while b(a) != 0xFD:
                y, v = b(a) & 15, b(a + 1)
                if v & 0x80:
                    page += 1
                if y == 0xD and not v & 0x40:
                    page = v & 0x1F
                last = max(last, page)
                a += 2
            return last

        def exits(t, i):            # (destination area pointer, world it applies to, entrance page)
            a, out = enemy_at(t, i), []
            while b(a) != 0xFF:
                if b(a) & 15 == 0xE:
                    out.append((b(a + 1), b(a + 2) >> 5, b(a + 2) & 31))
                    a += 3
                else:
                    a += 2
            return out

        world_off = [b(0x9cb4 + w) for w in range(8)] + [0x24]
        keys = []
        for w in range(8):
            todo, lv = [], w * 4
            for v in prg[0x9cbc - 0x8000 + world_off[w]:0x9cbc - 0x8000 + world_off[w + 1]]:
                todo.append((lv, v, 0, True))
                if v != self.INTRO:
                    lv += 1
            done = set()
            while todo:
                lv, v, page, main = todo.pop()
                t, i = (v >> 5) & 3, v & 31
                area = t * 32 + i
                if (lv, area, page) in done:
                    continue
                done.add((lv, area, page))
                last = width(t, i)
                if area == self.BONUS or last <= 1:
                    cells = [page * 16]
                else:
                    cells = range(0 if main else page * 16, last * 16 + 1)
                keys += [make_key(lv, area, c) for c in cells]
                if main:   # side areas' exits only lead back (the shared bonus room lists the whole world's returns)
                    todo += [(lv, dv, dp, False) for dv, dw, dp in exits(t, i) if dw == w]
        self.keys = torch.tensor(sorted(set(keys)), dtype=torch.int64, device=device)
        self.per_level = torch.bincount(key_level(self.keys), minlength=32).cpu()

    @property
    def total(self):
        return int(self.keys.numel())

    def covered(self, ever):
        inside = torch.isin(ever, self.keys)
        lv = torch.bincount(key_level(ever[inside]), minlength=32).cpu()
        return dict(map_total=self.total, map_found=int(inside.sum()), off_map=int((~inside).sum()),
                    map_frac=float(inside.sum()) / max(self.total, 1),
                    map_level_frac=[round(float(a) / max(int(b), 1), 3) for a, b in zip(lv, self.per_level)])
