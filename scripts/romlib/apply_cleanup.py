"""Apply reports/rom_cleanup_plan.csv (plan_cleanup.py) plus the FCEUX boot check: MOVE (never delete) every removed
ROM to REMOVED_DIR/<reason>/, and write REMOVED_DIR/manifest.csv (file, reason, kept_instead) so any move can be undone:
    python apply_cleanup.py --undo     moves everything in the manifest back.

Removed because FCEUX cannot run it: only ROMs whose screen stayed one blank colour for 30 s of play with Start/A taps
(fceux_boot_check.lua). A static title or demo picture is not proof a ROM cannot run, and VS System dumps wait for a coin.

usage: python apply_cleanup.py [--apply] [--undo]   (default: dry run, prints what would move)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROM_DIR = Path('/home/kd/Desktop/roms')
REMOVED_DIR = Path('/home/kd/Desktop/roms_removed')
PLAN = ROOT / 'reports/rom_cleanup_plan.csv'
FCEUX = ROOT / 'reports/romlib/fceux_check.jsonl'


def folder(reason):
    if reason.startswith('not an NES'):
        return 'not_nes'
    if reason.startswith('bad dump'):
        return 'bad_dump'
    if reason.startswith('identical'):
        return 'duplicate_identical'
    if reason.startswith('fceux'):
        return 'fceux_cannot_run'
    if reason.startswith('no iNES'):
        return 'no_header'
    return 'duplicate_or_translation'


def plan_rows():
    rows = list(csv.DictReader(open(PLAN)))
    protected = {r['file'] for r in rows if r['reason'] == 'used by reward10'}
    out = [(r['file'], r['reason'], r['kept_instead']) for r in rows if r['action'] == 'remove']
    if FCEUX.exists():
        for line in open(FCEUX):
            r = json.loads(line)
            if (r['rom'] not in protected and r.get('ran') and r.get('screens', 2) <= 1 and r.get('colours', 2) <= 1
                    and not re.search(r'\((VS|PC10)\)', r['rom'])):
                out.append((r['rom'], 'fceux: blank screen for 30 s', ''))
    return [x for x in out if x[0] not in protected]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('--undo', action='store_true')
    a = ap.parse_args()
    man = REMOVED_DIR / 'manifest.csv'
    if a.undo:
        for r in csv.DictReader(open(man)):
            src = REMOVED_DIR / folder(r['reason']) / r['file']
            if src.exists():
                shutil.move(str(src), str(ROM_DIR / r['file']))
        print('restored')
        return
    rows = plan_rows()
    print(Counter(folder(r[1]) for r in rows), 'total', len(rows))
    if not a.apply:
        return
    REMOVED_DIR.mkdir(exist_ok=True)
    done = []
    for f, reason, kept in rows:
        src = ROM_DIR / f
        if not src.exists():
            continue
        dst = REMOVED_DIR / folder(reason)
        dst.mkdir(exist_ok=True)
        shutil.move(str(src), str(dst / f))
        done.append((f, reason, kept))
    new = not man.exists()
    with open(man, 'a', newline='') as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(['file', 'reason', 'kept_instead'])
        w.writerows(done)
    print('moved', len(done), '->', REMOVED_DIR)


if __name__ == '__main__':
    main()
