"""Correct the first cleanup (apply_cleanup.py) after an audit of its 1,090 moves.

Three kinds of wrong decision are undone:
  bad_kept   a good dump was removed while a bad dump (Bad CHR/PRG, [b]) of the same game was kept: swap them
  bad_only   as bad_kept, but the removed file is another-language translation: only the bad dump goes
  tag_kept   a clean original was removed while a hacked/alternate/pirate dump ([hM..], [a1], [p1], [f1]) was kept
             (not when the kept file is a verified [!] dump): swap them
  version    an English regional release ((E), (Europe), (Japan) (En), Aladdin, ...) that differs from the kept one
             by 5% or more was removed as a duplicate; it is a different version of the game: restore it
Plus: an NTSC dump is preferred over the PAL one that was kept.

Files are moved, never deleted. Every move goes to roms_removed/manifest.csv (restored rows are dropped from it, swapped
files are added) and to reports/romlib/cleanup_fixes.csv.

usage: python fix_cleanup.py [--apply]
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROM_DIR = Path('/home/kd/Desktop/roms')
REMOVED = Path('/home/kd/Desktop/roms_removed')
MAN = REMOVED / 'manifest.csv'
FOLDER = 'duplicate_or_translation'


def sim(r):
    m = re.search(r'\((\d+)% same\)', r['reason'])
    return int(m.group(1)) if m else 100


def bad(f):
    return bool(re.search(r'Bad (CHR|PRG|Dump)|\[b\d*\]', f))


def dumpvar(f):
    return bool(re.search(r'\[(a|o|p|t|f|h)[^\]]*\]|ZZZ_UNK', f))


def english_release(f):
    if re.search(r'\[T[-+]', f):
        return False
    return bool(re.search(r'\((U|USA|UE|JU|E|Europe|World|UK|A|Australia)\)|\(En\)|\(Aladdin\)', f))


def plan(rows):
    out = []  # (verdict, restore_file, folder_of_restore, remove_file or '', remove_reason)
    seen_removed = set()
    for r in rows:
        if not r['reason'].startswith('worse copy'):
            continue
        f, k = r['file'], r['kept_instead']
        foreign = bool(re.search(r'\[T[-+](?!Eng)', f))  # another-language translation stays removed
        if not bad(f) and bad(k) and foreign:
            out.append(('bad_only', '', k, 'bad dump: a good dump of the same game is in the folder: ' + k))
        elif foreign:
            continue
        elif not bad(f) and bad(k):
            out.append(('bad_kept', f, k, 'bad dump: a good dump of the same game is kept: ' + f))
        elif not bad(f) and not dumpvar(f) and dumpvar(k) and '[!]' not in k:
            out.append(('tag_kept', f, k, 'worse copy of the same version: clean original kept: ' + f))
        elif english_release(f) and not dumpvar(f) and not bad(f) and sim(r) < 95:
            out.append(('version', f, '', ''))
        elif 'NTSC' in f and 'PAL' in k:
            out.append(('ntsc_kept', f, k, 'worse copy of the same version: NTSC dump kept: ' + f))
    # a kept file may be the target of several swaps: move it once, restore one file for it (not a [h] hack if
    # another candidate exists)
    out.sort(key=lambda x: (x[2], bool(re.search(r'\[h', x[1]))))
    final, taken = [], set()
    for v, f, k, why in out:
        if v in ('bad_kept', 'tag_kept') and k in taken:
            continue
        taken.add(k)
        if k and k in seen_removed:
            k = ''
        if k:
            seen_removed.add(k)
        final.append((v, f, k, why))
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(MAN)))
    fixes = plan(rows)
    from collections import Counter
    print(Counter(v for v, *_ in fixes), 'files restored', len(fixes), 'files moved out', sum(bool(k) for _, _, k, _ in fixes))
    if not a.apply:
        for v, f, k, _ in fixes:
            print(f'{v:9s} restore {f}' + (f'   | remove {k}' if k else ''))
        return
    restored = set()
    newrows = []
    for v, f, k, why in fixes:
        src = REMOVED / FOLDER / f
        if f and src.exists() and not (ROM_DIR / f).exists():
            shutil.move(str(src), str(ROM_DIR / f))
            restored.add(f)
        if k and (ROM_DIR / k).exists():
            folder = 'bad_dump' if v == 'bad_kept' else FOLDER
            (REMOVED / folder).mkdir(exist_ok=True)
            shutil.move(str(ROM_DIR / k), str(REMOVED / folder / k))
            newrows.append({'file': k, 'reason': ('bad dump' if v == 'bad_kept' else 'worse copy of the same version')
                            + f' (audit fix: {why.split(": ", 1)[1]})', 'kept_instead': f})
    keep = [r for r in rows if r['file'] not in restored]
    with open(MAN, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['file', 'reason', 'kept_instead'])
        w.writeheader()
        w.writerows(keep + newrows)
    with open(ROOT / 'reports/romlib/cleanup_fixes.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['fix', 'restored_to_roms', 'moved_out_instead'])
        w.writerows([(v, f, k) for v, f, k, _ in fixes])
    print('restored', len(restored), 'moved out', len(newrows))


if __name__ == '__main__':
    main()
