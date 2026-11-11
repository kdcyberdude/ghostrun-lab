"""Every pad byte the controller code can produce (all 324 moves x both turbo phases, plus every dropout-restricted
sticky repeat) reaches FCEUX's controller port unchanged. CPU only (headless FCEUX under xvfb, fresh profile)."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'rl' / 'tests'))
import hostlib  # noqa: E402

FCEUX = ROOT / '.cache/fceux-compat-build-v2/build/src/fceux'
ROM = str(Path(os.environ.get('NES_ROM_DIR', 'roms')) / 'Super Mario Bros (JU) (PRG 0).nes')


def test_pad_bytes_reach_fceux():
    lib = hostlib.load()
    table = np.zeros(2 * 324, np.uint8)
    lib.rlc_pad_table(hostlib.ptr(table))
    pads = np.concatenate([table, np.arange(256, dtype=np.uint8)[np.arange(256) & 0x30 != 0x30]])
    pads = pads[(pads & 0xC0) != 0xC0]  # Left+Right never produced; FCEUX may block it
    work = Path(tempfile.mkdtemp(prefix='fceux-pad-'))
    try:
        (work / 'in.bin').write_bytes(pads.tobytes())
        env = dict(os.environ, HOME=str(work), PAD_IN=str(work / 'in.bin'), PAD_OUT=str(work / 'out.bin'))
        subprocess.run(['xvfb-run', '-a', str(FCEUX), '--sound', '0', '--loadlua',
                        str(ROOT / 'rl/scripts/fceux_pad_echo.lua'), ROM], env=env, check=True, timeout=300,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        got = np.frombuffer((work / 'out.bin').read_bytes(), np.uint8)
    finally:
        shutil.rmtree(work)
    assert len(got) == len(pads)
    bad = np.flatnonzero(got != pads)
    assert len(bad) == 0, [(int(pads[i]), int(got[i])) for i in bad[:10]]


if __name__ == '__main__':
    test_pad_bytes_reach_fceux()
    print('ok: all pad bytes reach FCEUX unchanged')
