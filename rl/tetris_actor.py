"""Tetris piece placer for the route search (rl/explore.py --whole-game --game tetris). Search only, never trained.

Random buttons almost never complete a line, so for Tetris the search moves are placements: when a piece appears,
every rotation and column is dropped onto the board (from RAM), the resulting boards are scored with El-Tetris-style
weights (Yiyuan Lee 2013: aggregate height, completed lines, holes, bumpiness) plus per-console noise so consoles
try different placements, and the piece is steered there: A taps until the rotation matches, LEFT/RIGHT taps until
the column matches (each tap is one decision held, one released), then DOWN held. The search keeps the boards that
reach new (level, lines, stack height, holes) cells and restarts from them, as for the other games.

Piece layouts per rotation id ($0042) were measured from the game (pieces locked under random play, 2026-11-14):
(row, column) offsets from ($0041, $0040).
"""
from __future__ import annotations

import torch

ADDRS = [0x40, 0x41, 0x42, 0x48, 0x64] + list(range(0x400, 0x4C8))   # x, y, rotation id, playState, level, board
X, Y, ROT, PS, LEVEL, BOARD = 0, 1, 2, 3, 4, 5

SHAPES = {
    0: ((-1, 0), (0, -1), (0, 0), (0, 1)), 1: ((-1, 0), (0, 0), (0, 1), (1, 0)),
    2: ((0, -1), (0, 0), (0, 1), (1, 0)), 3: ((-1, 0), (0, -1), (0, 0), (1, 0)),            # T
    4: ((-1, 0), (0, 0), (1, -1), (1, 0)), 5: ((-1, -1), (0, -1), (0, 0), (0, 1)),
    6: ((-1, 0), (-1, 1), (0, 0), (1, 0)), 7: ((0, -1), (0, 0), (0, 1), (1, 1)),            # J
    8: ((0, -1), (0, 0), (1, 0), (1, 1)), 9: ((-1, 1), (0, 0), (0, 1), (1, 0)),              # Z
    10: ((0, -1), (0, 0), (1, -1), (1, 0)),                                                  # O
    11: ((0, 0), (0, 1), (1, -1), (1, 0)), 12: ((-1, 0), (0, 0), (0, 1), (1, 1)),            # S
    13: ((-1, 0), (0, 0), (1, 0), (1, 1)), 14: ((0, -1), (0, 0), (0, 1), (1, -1)),
    15: ((-1, -1), (-1, 0), (0, 0), (1, 0)), 16: ((-1, 1), (0, -1), (0, 0), (0, 1)),         # L
    17: ((-2, 0), (-1, 0), (0, 0), (1, 0)), 18: ((0, -2), (0, -1), (0, 0), (0, 1)),          # I
}
GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9), (10,), (11, 12), (13, 14, 15, 16), (17, 18))
W_HEIGHT, W_LINES, W_HOLES, W_BUMP = -0.510066, 0.760666, -0.35663, -0.184483
NONE, RIGHT, DOWN, LEFT, A_HOLD = 0, 3, 5, 7, 9          # rl/csrc/controller.cuh move ids


class TetrisActor:
    def __init__(self, n, device, seed=0, noise=(0.0, 0.6)):
        self.n, self.dev = n, device
        g = torch.Generator(device=device).manual_seed(seed)
        self.gen = g
        self.noise = noise[0] + (noise[1] - noise[0]) * torch.rand(n, generator=g, device=device)
        group_of = torch.zeros(19, dtype=torch.long)
        for k, grp in enumerate(GROUPS):
            group_of[list(grp)] = k
        self.group_of = group_of.to(device)
        # candidates: for each group, its rotations x 10 columns (padded to 4 rotations)
        rots = torch.full((7, 4), -1, dtype=torch.long)
        for k, grp in enumerate(GROUPS):
            rots[k, :len(grp)] = torch.tensor(grp)
        self.rots = rots.to(device)
        self.shape = torch.tensor([SHAPES[i] for i in range(19)], dtype=torch.long, device=device)   # [19, 4, 2]
        self.tgt = torch.full((n, 2), -1, dtype=torch.long, device=device)    # target rotation id, column
        self.wait = torch.zeros(n, dtype=torch.long, device=device)            # decisions spent steering
        self.prev_a = torch.zeros(n, dtype=torch.long, device=device)
        self.prev_ps = torch.zeros(n, dtype=torch.long, device=device)

    def board(self, ram):
        return (ram[:, BOARD:BOARD + 200] != 0xEF).view(-1, 20, 10)

    def features(self, ram):
        """[N] max stack height (0..20) and holes of the locked board."""
        b = self.board(ram)
        filled = b.any(1)
        top = torch.where(filled, b.float().argmax(1), torch.full_like(filled, 20, dtype=torch.long).float()).long()
        height = (20 - top).max(1).values
        below = torch.arange(20, device=self.dev).view(1, 20, 1) >= top.view(-1, 1, 10)
        holes = (below & ~b).sum((1, 2))
        return height, holes

    def plan(self, ram, idx):
        """Best placement (rotation id, column) for consoles idx from their board and current piece."""
        b = self.board(ram[idx])                                   # [M, 20, 10]
        M = b.shape[0]
        rot = self.rots[self.group_of[ram[idx, ROT].long()]]       # [M, 4]
        cols = torch.arange(10, device=self.dev)
        R = rot.view(M, 4, 1).expand(M, 4, 10)
        C = cols.view(1, 1, 10).expand(M, 4, 10)
        ok = R >= 0
        off = self.shape[R.clamp(min=0)]                            # [M, 4, 10, 4, 2]
        cc = C.unsqueeze(-1) + off[..., 1]                          # cell columns
        ok &= ((cc >= 0) & (cc <= 9)).all(-1)
        ys = torch.arange(-2, 20, device=self.dev)                 # candidate anchor rows
        rr = ys.view(1, 1, 1, -1, 1) + off[..., 0].unsqueeze(3)    # [M, 4, 10, Y, 4]
        ccy = cc.clamp(0, 9).unsqueeze(3).expand_as(rr)
        inside = rr <= 19
        occ = b[torch.arange(M, device=self.dev).view(M, 1, 1, 1, 1), rr.clamp(0, 19), ccy] & (rr >= 0)
        hit = (~inside | occ).any(-1)                               # [M, 4, 10, Y] anchor row collides
        first = torch.where(hit.any(-1), hit.float().argmax(-1), torch.full_like(hit[..., 0], hit.shape[-1], dtype=torch.long).float()).long()
        land = first - 1                                            # last free anchor index before the first collision
        ok &= land >= 0
        y = ys[land.clamp(min=0)]                                   # [M, 4, 10]
        pr = (y.unsqueeze(-1) + off[..., 0])                        # piece rows
        ok &= (pr >= 0).all(-1)
        nb = b.view(M, 1, 1, 20, 10).expand(M, 4, 10, 20, 10).clone()
        mi = torch.arange(M, device=self.dev).view(M, 1, 1, 1).expand(M, 4, 10, 4)
        ri = torch.arange(4, device=self.dev).view(1, 4, 1, 1).expand(M, 4, 10, 4)
        ci = torch.arange(10, device=self.dev).view(1, 1, 10, 1).expand(M, 4, 10, 4)
        nb[mi, ri, ci, pr.clamp(0, 19), cc.clamp(0, 9)] = True
        full = nb.all(-1)                                           # [M, 4, 10, 20]
        lines = full.sum(-1)
        # clear full rows: keep the others, stacked at the bottom
        keep = ~full
        order = torch.argsort(keep.float() * 100 + torch.arange(20, device=self.dev).float(), dim=-1)  # cleared rows first
        nb = torch.gather(nb & keep.unsqueeze(-1), 3, order.unsqueeze(-1).expand_as(nb))
        filled = nb.any(3)
        top = torch.where(filled, nb.float().argmax(3), torch.full_like(filled, 20, dtype=torch.long).float()).long()
        h = 20 - top                                                 # [M, 4, 10, 10] column heights
        below = torch.arange(20, device=self.dev).view(1, 1, 1, 20, 1) >= top.unsqueeze(3)
        holes = (below & ~nb).sum((3, 4))
        bump = (h[..., 1:] - h[..., :-1]).abs().sum(-1)
        score = W_HEIGHT * h.sum(-1) + W_LINES * lines + W_HOLES * holes + W_BUMP * bump
        score = score + self.noise[idx].view(M, 1, 1) * torch.randn(score.shape, generator=self.gen, device=self.dev)
        score = score.masked_fill(~ok, float('-inf'))
        best = score.view(M, -1).argmax(1)
        any_ok = torch.isfinite(score.view(M, -1).max(1).values)
        r, c = best // 10, best % 10
        tgt = torch.stack([rot[torch.arange(M, device=self.dev), r], c], 1)
        return torch.where(any_ok.unsqueeze(1), tgt, torch.full_like(tgt, -1))

    def act(self, ram, restarted):
        """[N] move ids for this decision. restarted: consoles that just began from a saved state."""
        ram = ram.long()
        ps = ram[:, PS]
        control = ps == 1
        new_piece = control & ((self.prev_ps != 1) | restarted | (self.tgt[:, 0] < 0))
        idx = new_piece.nonzero().squeeze(1)
        if idx.numel():
            self.tgt[idx] = self.plan(ram, idx)
            self.wait[idx] = 0
        rot_ok = ram[:, ROT] == self.tgt[:, 0]
        col_ok = ram[:, X] == self.tgt[:, 1]
        self.wait += control.long()
        give_up = (self.tgt[:, 0] < 0) | (self.wait > 40)
        pressed_a = (self.prev_a // 9) % 3 == 1
        pressed_d = (self.prev_a % 9) != 0
        a_part = torch.where(~rot_ok & ~pressed_a & ~give_up, 1, 0)
        d = torch.where(col_ok, NONE, torch.where(ram[:, X] < self.tgt[:, 1], RIGHT, LEFT))
        d = torch.where(pressed_d | give_up, NONE, d)
        d = torch.where((rot_ok & col_ok) | give_up, DOWN, d)
        a = torch.where(control, d + 9 * a_part, torch.zeros_like(d))
        self.prev_a = a
        self.prev_ps = torch.where(restarted, torch.zeros_like(ps), ps)
        return a.to(torch.int16)
