"""Place keys and camera of a table game (rl/games_table.json), for the search (rl/stucksearch.py) — the same layout as
SMB's coverage.smb_block_keys: (((level * 128 + area) << 12 | x // 8) << 6) | y // 8, -1 outside play.

Every field comes from the game's table row, found per ROM by scripts/romlib (no game is named here):
  in play   table in_play_addr / in_play_values (always in play without one)
  level     table level_addr (0 without one), area: table scene_addr (0 without one)
  camera    row['camera'] (scripts/romlib/camera.py): x / y low byte + page byte; a camera without a page byte gives
            positions on a 256-px ring and no send-backs (camera() = -1)
  player    table x_addr / y_addr; camera['player_x'] says whether the byte is a level coordinate ('world': its page
            is taken from the camera's, the player is on screen) or a screen one ('screen': camera + byte)
Limits: x // 8 has 12 bits (32,768 px), y // 8 has 6 bits (512 px, wraps beyond).
"""
from __future__ import annotations

import torch


def _axis(cam, player):
    """(lo, page, kind) of one axis from the row's camera entry."""
    c = (cam or {}).get(player[0]) or {}
    return c.get('lo', -1), c.get('page', -1), (cam or {}).get(player[1])


class TableKeys:
    def __init__(self, row):
        t, cam = row['table'], row.get('camera') or {}
        self.t = t
        self.cx, self.cxp, self.kx = _axis(cam, ('x', 'player_x'))
        self.cy, self.cyp, self.ky = _axis(cam, ('y', 'player_y'))

    @staticmethod
    def _cam(r, lo, page):
        if lo < 0:
            return None
        return r[:, lo] + (r[:, page] * 256 if page >= 0 else 0)

    @staticmethod
    def _world(r, addr, cam, kind):
        """Player coordinate in level pixels (module notes)."""
        if addr < 0:
            return torch.zeros(r.shape[0], dtype=torch.int64, device=r.device)
        p = r[:, addr]
        if cam is None or kind not in ('world', 'screen'):
            return p
        if kind == 'screen':
            return cam + p
        w = cam - (cam & 255) + p                     # the camera's page; the player is on screen
        w = torch.where(w < cam - 64, w + 256, w)
        return torch.where(w > cam + 320, w - 256, w)

    def keys(self, ram):
        r, t = ram.long(), self.t
        n = r.shape[0]
        play = torch.ones(n, dtype=torch.bool, device=r.device)
        if t['in_play_addr'] >= 0:
            vals = torch.tensor(t['in_play_values'], device=r.device)
            play = (r[:, t['in_play_addr'], None] == vals[None]).any(1)
        zero = torch.zeros(n, dtype=torch.int64, device=r.device)
        level = r[:, t['level_addr']] if t['level_addr'] >= 0 else zero
        area = r[:, t['scene_addr']] & 127 if t['scene_addr'] >= 0 else zero
        x = self._world(r, t['x_addr'], self._cam(r, self.cx, self.cxp), self.kx)
        y = self._world(r, t['y_addr'], self._cam(r, self.cy, self.cyp), self.ky)
        key = (((level * 128 + area) << 12 | ((x // 8) & 4095)) << 6) | ((y // 8) & 63)
        return torch.where(play & (level < 32), key, torch.full_like(key, -1))

    def camera(self, ram):
        """Camera x in level pixels; -1 when the game has no camera x with a page byte (no send-backs then)."""
        r = ram.long()
        if self.cx < 0 or self.cxp < 0:
            return torch.full((r.shape[0],), -1, dtype=torch.int64, device=r.device)
        return r[:, self.cx] + r[:, self.cxp] * 256
