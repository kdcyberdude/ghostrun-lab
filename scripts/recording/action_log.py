"""Shared pieces of the level recordings: controller formats, the action-log JSON and an MP4 writer.

No torch or emulator imports, so --dry-run and the self-test run anywhere.

Controller formats
  frames   one NES controller byte per emulated frame, the byte the game reads from $4016 (FCEUX movie order):
           bit0 A, bit1 B, bit2 Select, bit3 Start, bit4 Up, bit5 Down, bit6 Left, bit7 Right.
           The rl/ environment's `pads` output and rl/evaluate.py's *.pads.bin use exactly this byte.
  actions  the same stream run-length encoded as [[mask, frames], ...] in the native-trace bit order of the
           Three.js reconstruction tools (bit0 Up, 1 Down, 2 Left, 3 Right, 4 A, 5 B, 6 Start, 7 Select), i.e.
           trainerPadToMask() of tools/harness/model-play.mjs, so a log drops into those replay tools unchanged.
  v1 masks the v1 player's 10-bit masks (up, down, left, right, a, b, x, y, start, select; x aliases A, y aliases B).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCHEMA = 'ghostrun-action-log-v1'
FRAME_BITS = 'NES controller byte per frame: bit0 A, bit1 B, bit2 Select, bit3 Start, bit4 Up, bit5 Down, bit6 Left, bit7 Right'
ACTION_BITS = 'run-length [[mask, frames]], native-trace order: bit0 Up, 1 Down, 2 Left, 3 Right, 4 A, 5 B, 6 Start, 7 Select'
LEVELS = 32

# trainer (NES) bit -> native-trace bit, the table in model-play.mjs: [4,5,6,7,0,1,3,2] lists, for native-trace bit i,
# which NES bit feeds it
_NATIVE_FROM_NES = [4, 5, 6, 7, 0, 1, 3, 2]
# v1 10-bit mask bit -> NES bit (x -> A, y -> B)
_NES_FROM_V1 = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 0, 7: 1, 8: 3, 9: 2}


def level_name(index: int) -> str:
    return f'{index // 4 + 1}-{index % 4 + 1}'


def level_index(name: str) -> int:
    w, s = name.split('-')
    return (int(w) - 1) * 4 + int(s) - 1


def parse_levels(spec: str) -> list[int]:
    """'all' | '1-1,4-4' | '0,15' -> sorted level indices."""
    if spec in ('', 'all'):
        return list(range(LEVELS))
    out = {level_index(x) if '-' in x else int(x) for x in spec.split(',') if x}
    bad = [x for x in out if not 0 <= x < LEVELS]
    if bad:
        raise ValueError(f'levels out of range: {bad}')
    return sorted(out)


def nes_to_native(pad: int) -> int:
    return sum(((pad >> nes_bit) & 1) << i for i, nes_bit in enumerate(_NATIVE_FROM_NES))


def v1_to_nes(mask: int) -> int:
    out = 0
    for bit, nes_bit in _NES_FROM_V1.items():
        if mask >> bit & 1:
            out |= 1 << nes_bit
    return out


def run_length(values) -> list[list[int]]:
    out: list[list[int]] = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([int(v), 1])
    return out


def expand(actions) -> list[int]:
    return [m for m, n in actions for _ in range(n)]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def make_log(*, level: int, model: str, mode, frames, start: str, clock: str, result: dict, **extra) -> dict:
    frames = [int(x) for x in frames]
    log = dict(schema=SCHEMA, game='smb', level=level_name(level), level_index=level, model=model, mode=mode,
               start=start, clock=clock, frame_bits=FRAME_BITS, frames=frames,
               actions_bits=ACTION_BITS, actions=run_length(nes_to_native(p) for p in frames), result=result)
    log.update(extra)
    validate_log(log)
    return log


def validate_log(log: dict):
    need = {'schema', 'game', 'level', 'level_index', 'model', 'mode', 'start', 'clock', 'frames', 'actions', 'result'}
    missing = need - log.keys()
    if missing:
        raise ValueError(f'action log misses {sorted(missing)}')
    if log['schema'] != SCHEMA:
        raise ValueError(f'unknown schema {log["schema"]}')
    if level_name(log['level_index']) != log['level']:
        raise ValueError('level and level_index disagree')
    frames = log['frames']
    if any(type(x) is not int or not 0 <= x <= 255 for x in frames):
        raise ValueError('frames must be NES controller bytes 0..255')
    if expand(log['actions']) != [nes_to_native(p) for p in frames]:
        raise ValueError('actions do not encode frames')


def write_log(path, log: dict):
    validate_log(log)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(log, separators=(',', ':')) + '\n')
    tmp.replace(path)
    path.with_suffix('').with_suffix('.pads.bin').write_bytes(bytes(log['frames']))


class Mp4Writer:
    """RGB frames [H, W, 3] uint8 -> H.264 MP4 through ffmpeg (nearest-neighbour upscale keeps NES pixels sharp).
    Same settings as rl/evaluate.py, with the frame rate as a parameter (60 = every emulated frame, real speed)."""

    def __init__(self, path, w, h, fps=60, scale=3):
        self.path = Path(path)
        self.p = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                   '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                                   '-vf', f'scale={w * scale}:{h * scale}:flags=neighbor', '-c:v', 'libx264',
                                   '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                                   str(self.path)], stdin=subprocess.PIPE)
        self.frames = 0

    def write(self, rgb):
        import numpy as np
        self.p.stdin.write(np.ascontiguousarray(rgb).tobytes())
        self.frames += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError(f'ffmpeg failed writing {self.path}')


def self_test(out_dir=None) -> dict:
    """Format checks with no model and no emulator: bit mappings against the Three.js table, run-length round trip,
    a synthetic log through write/validate, and (if out_dir) a one-second synthetic MP4 when ffmpeg exists."""
    import shutil
    # Right+B (NES 0x82) -> native Right(bit3)+B(bit5) = 40, the first action of the v1 route
    assert nes_to_native(0x82) == 40, nes_to_native(0x82)
    assert v1_to_nes(40) == 0x82, v1_to_nes(40)
    assert v1_to_nes(1 << 6) == 0x01 and v1_to_nes(1 << 7) == 0x02   # x -> A, y -> B
    for pad in range(256):   # the mapping is a bijection on the 8 NES buttons
        assert sum(1 << i for i in range(8) if nes_to_native(pad) >> i & 1) == nes_to_native(pad)
    assert len({nes_to_native(p) for p in range(256)}) == 256
    frames = [0] * 10 + [0x82] * 7 + [0x83] * 8 + [0]
    log = make_log(level=15, model='self-test', mode=0, frames=frames, start='level_entrance', clock='ppu_frame',
                   result=dict(cleared=False))
    assert log['level'] == '4-4' and log['actions'] == [[0, 10], [40, 7], [56, 8], [0, 1]]
    report = dict(mappings='ok', run_length='ok')
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_log(out / 'self-test.actions.json', log)
        validate_log(json.loads((out / 'self-test.actions.json').read_text()))
        assert (out / 'self-test.pads.bin').read_bytes() == bytes(frames)
        report['log_file'] = str(out / 'self-test.actions.json')
        if shutil.which('ffmpeg'):
            import numpy as np
            w = Mp4Writer(out / 'self-test.mp4', 32, 30, fps=60, scale=2)
            for i in range(60):
                w.write(np.full((30, 32, 3), i * 4, np.uint8))
            w.close()
            report['mp4'] = f'{out / "self-test.mp4"} ({(out / "self-test.mp4").stat().st_size} bytes)'
        else:
            report['mp4'] = 'skipped: ffmpeg not found'
    return report
