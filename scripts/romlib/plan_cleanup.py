"""Plan the ROM folder cleanup (user request 2026-11-20). Writes a CSV; moves nothing (apply_cleanup.py does that).

Rules, per file:
  remove  not an NES file (.gb/.gba/.gbc/.md/.a26), no iNES header, bad dump ([b], (Bad CHR)), a byte-identical copy of
          another file (PRG+CHR without the header), or a worse copy of the same version of the same game.
  keep    the best copy of each version. A version = base title + the tags that make it a different game: hacks/mods
          ("(... Hack)"), revisions (PRG n, Rev x), VS / PC10 arcade boards, prototypes/betas, version numbers.
Language and dump tags only rank copies of one version (best first):
  official English (U, USA, JU, UE, World, En) > English PAL (E, Europe) > complete English fan translation (T-Eng)
  > original other language (J, Japan, Chinese) > other-language translations (T-Port, T-Span, ...);
  then verified [!] > plain > alternate [a] > hacked header/mapper [hM, hFFE, h] > pirate [p] > trainer [t] > overdump [o].
Safety: a copy is only removed as "same game" when it shares at least SAME_SHARE of its 256-byte PRG chunks with the kept
copy (a translation or region change edits a few banks; a different game under the same name, e.g. Super Mario Bros 2
(J) = The Lost Levels vs (U), shares almost none). The ROMs the reward10 modules use are never removed.
"""
from __future__ import annotations

import csv
import hashlib
import os
import re
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SAME_SHARE = 0.5
NES_EXT = '.nes'


def protected():
    out = set()
    for f in (ROOT / 'scripts/reward10/games').glob('*.py'):
        m = re.search(r"/ ['\"](.+?\.nes)['\"]\)", f.read_text())
        if m:
            out.add(m.group(1))
    return out


def tags(name):
    return re.findall(r'[\(\[]([^\)\]]*)[\)\]]', name)


def base(name):
    b = re.split(r'\s*[\(\[]', name.rsplit('.', 1)[0])[0].strip()
    b = re.sub(r'^ZZZ_UNK_', '', b)
    return re.sub(r'[^a-z0-9]', '', re.sub(r'\bthe\b', '', b.lower()))


LANG_EN = {'U', 'USA', 'JU', 'UE', 'JUE', 'World', 'En', 'USA, Europe', 'Japan, USA', 'U)(PRG0'}
LANG_EUR = {'E', 'Europe', 'EU', 'Europe, USA'}
LANG_ORIG = {'J', 'Japan', 'Chinese', 'Ch', 'K', 'Korea', 'Asia', 'G', 'F', 'S', 'Sw', 'I', 'Japan, Europe'}


def identity(ts):
    """Tags that make a different version (kept side by side)."""
    keep = []
    for t in ts:
        tl = t.lower()
        if ('hack' in tl or tl.startswith(('prg', 'rev', 'v', 'ver', 'beta', 'proto', 'final', 'old', 'pc10', 'vs'))
                or re.fullmatch(r'[\d.]+[a-z]?', tl) or tl in ('pd', 'unl', 'sachen', 'namco', 'hes', 'joy van')):
            keep.append(tl.replace(' ', ''))
    return tuple(sorted(keep))


def rank(name, ts, ok_mapper):
    lang = 3
    for t in ts:
        if t in LANG_EN:
            lang = min(lang, 0)
        elif t in LANG_EUR:
            lang = min(lang, 1)
        elif t.startswith('T-Eng') and 'Partial' not in t:
            lang = min(lang, 2) if lang >= 2 else lang
        elif t in LANG_ORIG:
            lang = min(lang, 3)
    if any(t.startswith('T') and '-' in t and not t.startswith('T-Eng') for t in ts) or any('Partial' in t for t in ts):
        lang = 4
    if any(t.startswith('T-Eng') and 'Partial' not in t for t in ts) and lang > 2:
        lang = 2
    dump = 1
    for t in ts:
        if t == '!':
            dump = min(dump, 0)
    order = [('a', 2), ('h', 3), ('p', 4), ('t', 5), ('o', 6), ('f', 2)]
    for t in ts:
        for p, v in order:
            if re.fullmatch(p + r'[\w+-]*', t) and not t.startswith('T'):
                dump = max(dump, v)
    return (lang, 0 if ok_mapper else 1, dump, len(name))


def header(raw):
    if raw[:4] != b'NES\x1a':
        return None
    m = (raw[6] >> 4) | (raw[7] & 0xF0)
    if raw[12:16] != b'\0\0\0\0' and (raw[7] & 0x0C) != 0x08:
        m = raw[6] >> 4
    return m


def chunks(body):
    """Content-defined fingerprint: hashes of every 16-byte window whose hash is 0 mod 8 (shifted code still matches)."""
    a = np.frombuffer(body, np.uint8).astype(np.uint64)
    if len(a) < 16:
        return set()
    w = sliding_window_view(a, 16)
    h = (w * POW).sum(1, dtype=np.uint64)
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return set(h[(h & np.uint64(7)) == 0].tolist())


GPU_MAPPERS = {0, 1, 2, 3, 4, 9, 34, 94}
POW = np.array([pow(1099511628211, 15 - k, 1 << 64) for k in range(16)], dtype=np.uint64)


def main(rom_dir, out_csv):
    prot = protected()
    files = sorted(os.listdir(rom_dir))
    rows, info = [], {}
    for f in files:
        p = Path(rom_dir) / f
        if not p.is_file():
            continue
        if not f.lower().endswith(NES_EXT):
            rows.append((f, 'remove', 'not an NES file', ''))
            continue
        raw = p.read_bytes()
        m = header(raw)
        ts = tags(f)
        if m is None:
            rows.append((f, 'remove', 'no iNES header', ''))
            continue
        if f not in prot and (any(re.fullmatch(r'b\d*', t) for t in ts) or 'Bad CHR' in ts):
            rows.append((f, 'remove', 'bad dump', ''))
            continue
        trainer = 512 if raw[6] & 4 else 0
        body = raw[16 + trainer:]
        info[f] = dict(mapper=m, ts=ts, body=body, sha=hashlib.sha256(body).hexdigest(), base=base(f))
    # byte-identical copies
    by_sha = defaultdict(list)
    for f, d in info.items():
        by_sha[d['sha']].append(f)
    gone = set()
    for fs in by_sha.values():
        if len(fs) > 1:
            fs = sorted(fs, key=lambda f: (f not in prot, rank(f, info[f]['ts'], info[f]['mapper'] in GPU_MAPPERS)))
            for f in fs[1:]:
                rows.append((f, 'remove', 'identical to another file', fs[0]))
                gone.add(f)
    # one copy per version: walk each title's files best first; a file adds nothing when it is mostly the same bytes
    # as a kept file and has no version tag (hack, PRG/Rev, VS/PC10, prototype, ...) that kept file lacks
    groups = defaultdict(list)
    for f, d in info.items():
        if f not in gone:
            groups[d['base']].append(f)
    fp = {}
    for fs in groups.values():
        fs = sorted(fs, key=lambda f: (f not in prot, rank(f, info[f]['ts'], info[f]['mapper'] in GPU_MAPPERS)))
        kept = []
        for f in fs:
            fp[f] = chunks(info[f]['body'])
            ident = set(identity(info[f]['ts']))
            best, share, cover, cshare = None, 0.0, None, 0.0
            for k in kept:
                sh = len(fp[f] & fp[k]) / max(len(fp[f]), 1)
                if sh > share:
                    best, share = k, sh
                if sh >= SAME_SHARE and ident <= set(identity(info[k]['ts'])) and sh > cshare:
                    cover, cshare = k, sh     # a kept file of (at least) this version, mostly the same bytes
            if f in prot:
                rows.append((f, 'keep', 'used by reward10', ''))
            elif cover is not None:
                rows.append((f, 'remove', f'worse copy of the same version ({cshare:.0%} same)', cover))
                continue
            elif best is None:
                rows.append((f, 'keep', 'best copy of its version', ''))
            elif share >= SAME_SHARE:
                rows.append((f, 'keep', f'other version ({share:.0%} same as {best})', ''))
            else:
                rows.append((f, 'keep', f'same name, different game ({share:.0%} same as {best})', ''))
            kept.append(f)
    rows.sort()
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'action', 'reason', 'kept_instead'])
        w.writerows(rows)
    n = defaultdict(int)
    for r in rows:
        n[(r[1], r[2].split(' (')[0])] += 1
    for k, v in sorted(n.items()):
        print(f'{v:5d}  {k[0]:6s} {k[1]}')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('NES_ROM_DIR', 'roms'),
         sys.argv[2] if len(sys.argv) > 2 else str(ROOT / 'reports/rom_cleanup_plan.csv'))
