"""Check readers across game families: ROMs that share one game's program (its regional versions and revisions, and
hacks of it, which keep the original RAM map) should report the same addresses.

Families: the title before any "(...)" / "[...]" tag; a "(X Hack)" tag puts the ROM in X's family (SMB1 Hack -> Super
Mario Bros). For each field: in families with >= 3 ROMs that report it, the share of reports that equal the family's
most common address (agreement). Disagreement points at a wrong reader or at a hack that moved the variable.

The SMB1 family (Super Mario Bros versions and SMB1 hacks) has a known map (scripts/reward10/games/smb.py): lives
$075A, score digits $07DD-$07E2 (a reported span is right when it covers the changing digits), level $075F (or $075C),
x $0086/$0755. For it the report gives accuracy, not agreement.

usage: python readers_families.py [READERS.jsonl] [--md OUT.md]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ['lives', 'score', 'level', 'death', 'game_over', 'in_play']
ALIAS = {'smb1': 'super mario bros', 'smb': 'super mario bros', 'smb2': 'super mario bros 2', 'smb3': 'super mario bros 3'}
SMB_TRUTH = dict(lives={'$075A'}, level={'$075F', '$075C'}, x={'$0086', '$0755'})


def family(name):
    m = re.search(r'\(([^()]*?)\s+Hack\)', name)
    base = m.group(1) if m else re.split(r'\s*[\(\[]', name[:-4])[0]
    base = re.sub(r'^ZZZ_UNK_', '', base).lower().strip()
    base = re.sub(r'^(.*), the$', r'the \1', base)
    base = re.sub(r'[^a-z0-9 ]', '', base)
    return ALIAS.get(base, base)


def addr(r, k):
    v = r.get(k)
    return v['addr'] if isinstance(v, dict) else None


def smb_score_ok(v):
    if not isinstance(v, dict):
        return None
    a, n = int(v['addr'][1:], 16), v['n']
    span = set(range(a, a + n))
    return 0x7DF in span and 0x7E1 in span     # the digits random play changes (hundreds, tens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('readers', nargs='?', default=str(ROOT / 'reports/romlib/readers.jsonl'))
    ap.add_argument('--md')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.readers) if 'error' not in l]
    fam = collections.defaultdict(list)
    for r in rows:
        fam[family(r['rom'])].append(r)
    lines = ['## Families (hacks and versions of one game should agree)', '',
             '| field | families checked | reports | agree with family majority |', '|---|---|---|---|']
    for k in FIELDS:
        nf = nrep = nagree = 0
        for name, rs in fam.items():
            vals = [addr(r, k) for r in rs if addr(r, k)]
            if len(vals) < 3:
                continue
            top = collections.Counter(vals).most_common(1)[0][1]
            nf += 1
            nrep += len(vals)
            nagree += top
        lines.append(f'| {k} | {nf} | {nrep} | {nagree / max(nrep, 1):.0%} |')
    smb = fam.get('super mario bros', [])
    lines += ['', f'## SMB1 family: {len(smb)} ROMs (versions + SMB1 hacks), against the known map', '',
              '| field | reported | right | wrong |', '|---|---|---|---|']
    versions = [r for r in smb if 'Hack' not in r['rom']]
    for k, rs, label in (('lives', smb, 'lives'), ('level', versions, 'level (versions only)')):
        # hacks may store the world elsewhere (Alice Cooper keeps $075F at $FF in play), so level is scored on
        # Nintendo's own versions only
        rep = [addr(r, k) for r in rs if addr(r, k)]
        ok = sum(v in SMB_TRUTH[k] for v in rep)
        lines.append(f'| {label} | {len(rep)} of {len(rs)} | {ok} | {len(rep) - ok} |')
    sc = [smb_score_ok(r.get('score')) for r in smb if isinstance(r.get('score'), dict)]
    lines.append(f'| score | {len(sc)} of {len(smb)} | {sum(sc)} | {len(sc) - sum(sc)} |')
    xs = [(r['position'] or {}).get('x') for r in smb if isinstance(r.get('position'), dict)]
    xs = [x for x in xs if x]
    lines.append(f"| position x | {len(xs)} of {len(smb)} | {sum(x in SMB_TRUTH['x'] for x in xs)} | "
                 f"{sum(x not in SMB_TRUTH['x'] for x in xs)} |")
    text = '\n'.join(lines)
    print(text)
    if a.md:
        Path(a.md).write_text(text + '\n')


if __name__ == '__main__':
    main()
