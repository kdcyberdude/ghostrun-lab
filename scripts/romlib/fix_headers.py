"""Write fceumm's header corrections into the ROM files, so NeSLE (which trusts the iNES header) sees the board fceumm
emulates.

fceumm keeps a table of known bad headers (src/ines-correct.h: CRC32 of PRG+CHR -> mapper, mirroring, ...) and fixes
them in memory at load. NeSLE has no such table: a ROM like Gyruss (U) with a "DiskDude!" header says mapper 0 with
32 KB CHR, runs the right program (RAM matches fceumm) and draws the wrong CHR bank. This applies the table's mapper
and mirroring (horizontal/vertical/four-screen) to each matching file. Region, battery and RAM sizes are left alone
(NeSLE is NTSC only and sizes its RAM per board); entries that drop CHR ROM ("extra" 0x800) are only reported.

The original header of every changed file is kept: roms_removed/header_backup/<file> (the whole original file) and
reports/romlib/header_fixes.csv (file, old mapper/mirror, new mapper/mirror).

usage: python fix_headers.py --table PATH/TO/ines-correct.h [--only NAMES.txt] [--apply]
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROM_DIR = Path('/home/kd/Desktop/roms')
BACKUP = Path('/home/kd/Desktop/roms_removed/header_backup')


# ines.c's names in the table (mirror: MI_H 0, MI_V 1, MI_4 2 = four-screen; region: NTSC 0, PAL 1, DENDY 3)
CONSTS = dict(MI_H=0, MI_V=1, MI_4=2, MI_0=2, MI_1=3, DFAULT8=8, NTSC=0, PAL=1, MULTI=2, DENDY=3)


def table(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r'\s*\{\s*(0x[0-9a-fA-F]+)\s*,(.*)\}', line)
        if not m:
            continue
        f = [x.strip() for x in m.group(2).split(',')]
        val = lambda x: -1 if x in ('DEFAULT', 'NOEXTRA') else int(eval(x, dict(CONSTS)))  # noqa: S307 - literals
        out[int(m.group(1), 16)] = dict(mapper=val(f[0]), mirror=val(f[2]), extra=val(f[7]) if len(f) > 7 else -1,
                                        name=line.split('/*')[-1].strip(' */') if '/*' in line else '')
    return out


def parse(data):
    h = data[:16]
    trainer = 512 if h[6] & 4 else 0
    prg, chr_ = h[4] * 0x4000, h[5] * 0x2000
    dirty = h[12:16] != b'\0\0\0\0' and (h[7] & 0x0C) != 0x08
    mapper = h[6] >> 4 if dirty else (h[6] >> 4) | (h[7] & 0xF0)
    mirror = 2 if h[6] & 8 else (1 if h[6] & 1 else 0)       # fceumm: 0 horizontal, 1 vertical, 2 four-screen
    body = data[16 + trainer:16 + trainer + prg + chr_]
    crc = zlib.crc32(body[prg:], zlib.crc32(body[:prg])) & 0xFFFFFFFF
    return mapper, mirror, crc, dirty


def fixed_header(data, mapper, mirror, dirty):
    h = bytearray(data[:16])
    if dirty:
        h[7:16] = bytes(9)                     # DiskDude!-style junk: an iNES 1.0 header has zeros there
    h[6] = (h[6] & 0x0F) | ((mapper & 0x0F) << 4)
    h[7] = (h[7] & 0x0F) | (mapper & 0xF0)
    if mirror in (0, 1, 2):
        h[6] = (h[6] & ~0x09) | (1 if mirror == 1 else 0) | (8 if mirror == 2 else 0)
    return bytes(h) + data[16:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', required=True)
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--only', help='file with ROM names: change only these (ROMs NeSLE already runs right stay)')
    a = ap.parse_args()
    only = set(Path(a.only).read_text().splitlines()) if a.only else None
    moo = table(a.table)
    rows = []
    for p in sorted(ROM_DIR.glob('*.nes')):
        data = p.read_bytes()
        if data[:4] != b'NES\x1a':
            continue
        mapper, mirror, crc, dirty = parse(data)
        e = moo.get(crc)
        if not e or (only is not None and p.name not in only):
            continue
        new_mapper = e['mapper'] & 0xFFF if e['mapper'] >= 0 else mapper
        new_mirror = e['mirror'] if e['mirror'] in (0, 1, 2) else mirror
        if e['mirror'] == 8 and mirror == 2:
            new_mirror = 0
        note = 'drops CHR ROM in fceumm (not applied)' if e['extra'] == 0x800 else ''
        if new_mapper > 255:
            note = f'mapper {new_mapper} needs NES 2.0 (not applied)'
        if (new_mapper, new_mirror) == (mapper, mirror) and not note:
            continue
        rows.append(dict(file=p.name, crc=f'{crc:08x}', old_mapper=mapper, new_mapper=new_mapper, old_mirror=mirror,
                         new_mirror=new_mirror, table_name=e['name'], note=note))
        if a.apply and not note:
            BACKUP.mkdir(parents=True, exist_ok=True)
            if not (BACKUP / p.name).exists():
                shutil.copy2(p, BACKUP / p.name)
            p.write_bytes(fixed_header(data, new_mapper, new_mirror, dirty))
    for r in rows:
        print(f"{r['old_mapper']:>3}->{r['new_mapper']:<3} mirror {r['old_mirror']}->{r['new_mirror']}  {r['file'][:60]}  "
              f"{r['note']}")
    print(len(rows), 'files', 'changed' if a.apply else '(dry run)')
    if a.apply:
        out = ROOT / 'reports/romlib/header_fixes.csv'
        old = list(csv.DictReader(open(out))) if out.exists() else []
        names = {r['file'] for r in rows}
        with open(out, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ['file'])
            w.writeheader()
            w.writerows([r for r in old if r['file'] not in names] + rows)


if __name__ == '__main__':
    main()
