"""Write reports/romlib/readers_summary.md from readers.jsonl (all ROMs) and readers_truth.jsonl (truth games).

usage: python readers_summary.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'reports/romlib'
FIELDS = ['position', 'lives', 'score', 'level', 'death', 'in_play', 'game_over']


def main():
    rows = [json.loads(l) for l in open(R / 'readers.jsonl')]
    ok = [r for r in rows if 'error' not in r]
    lines = ['# RAM readers: coverage and truth checks', '',
             f'{len(rows)} controllable ROMs (stage 1, the 40-press retry, the menu fixes and the 80-press deep '
             f'search) processed by '
             f'scripts/romlib/readers.py; {len(rows) - len(ok)} errors.', '',
             '## Coverage (all ROMs)', '', '| field | found | high | medium | low | unknown |', '|---|---|---|---|---|---|']
    for f in FIELDS:
        c = Counter(r[f]['confidence'] if isinstance(r.get(f), dict) else 'unknown' for r in ok)
        found = len(ok) - c['unknown']
        lines.append(f"| {f} | {found} ({found / max(1, len(ok)):.0%}) | {c['high']} | {c['medium']} | {c['low']} "
                     f"| {c['unknown']} |")
    mc = [json.loads(l) for l in open(R / 'menucheck.jsonl')]
    flagged = [m for m in mc if m.get('check', {}).get('menu')]
    why = Counter('+'.join(m['check']['why']) for m in flagged)
    lines += ['', '## Boots that stop on a menu (scripts/romlib/menucheck.py)', '',
              f"{len(flagged)} of {len(mc)} boots flagged as a menu ({', '.join(f'{k}: {v}' for k, v in why.items())}); "
              f"{sum(bool(m.get('fixed')) for m in flagged)} continued to a state the check calls gameplay (boot "
              f"pads in ramfind_fixed.jsonl; their readers re-run), {sum(not m.get('fixed') for m in flagged)} not "
              f"(readers rows carry boot_menu: unfixed). {sum('error' in m for m in mc)} ROM files were gone.",
              'Truth games: Tetris and Punch-Out flagged and fixed, the other 9 not flagged. Eye check of 16 random '
              'flagged boot ends: 12 menus/title/text screens, 3 gameplay (false flags), 1 unclear; of 16 random fixed '
              'boot ends: 13 gameplay (1 of them unclear), 3 still a menu or story text.']
    truth = [json.loads(l) for l in open(R / 'readers_truth.jsonl')]
    lines += ['', '## Truth games (scripts/reward10/games/*.py)', '',
              'Each field is compared with the game\'s own hand-verified Reader on the same recorded play: lives and '
              'level by address, score by value (ours x a constant = truth on 99% of in-play frames), death and game '
              'over by events (ours within -60..+400 / -300..+600 frames of the truth), in_play by balanced frame '
              'accuracy >= 0.9. ok = right; unknown-ok = the game has no such signal and none was reported; '
              'n/a = the stage-1 boot does not reach the game.', '',
              '| game | ' + ' | '.join(FIELDS[1:]) + ' |', '|---' * len(FIELDS) + '|']
    tally = {f: Counter() for f in FIELDS[1:]}
    for t in sorted(truth, key=lambda t: t['truth']['slug']):
        ch = t['truth']['checks']
        cells = []
        for f in FIELDS[1:]:
            v, note = ch[f]
            tally[f][v] += 1
            cells.append(v + (f' ({note})' if note and v not in ('n/a',) else ''))
        lines.append(f"| {t['truth']['slug']} | " + ' | '.join(cells) + ' |')
    lines += ['', '| field | ok | unknown-ok | missed | wrong | false | n/a |', '|---|---|---|---|---|---|---|']
    for f in FIELDS[1:]:
        c = tally[f]
        lines.append(f"| {f} | {c['ok']} | {c['unknown-ok']} | {c['missed']} | {c['wrong']} | {c['false']} | {c['n/a']} |")
    lines += ['', '## Before', '',
              'ramfind2 v1 (lives + score only): lives 4/5 on the lives games (SMB wrong), score 3/5 (SMB, Pac-Man '
              'missed); no level/death/in_play/game_over. Before the menu check, Tetris and Punch-Out never reached '
              'the game (all fields n/a).', '',
              '## Known weak spots', '',
              '- Menu check: false flags on gameplay held by a countdown or drawn by background tiles (Excitebike VS, '
              'Banana, Master Fighter 2 in a 16-sample); a fix can stop on a later menu or story text that the check '
              'does not catch (Baseball Stars, Bo Jackson Baseball, Joy Mecha Fight). 79 flagged boots stay unfixed. '
              'An 80-press search with two seeds (ramfind_deep.jsonl) made 87 of the 261 remaining boots '
              'controllable (Zelda among them); 78 pass the menu check and have readers.',
              '- Lives cannot tell a health counter that ends the game (Final Mission $1C) from lives; death/level '
              'then follow it.',
              '- Level: the respawn test (medium) finds SMB\'s $075F; without a death, the boot test (+1 poke after '
              'the last menu press; always low) also accepts stage-type bytes: it reports SMB\'s area type $074E on '
              '127 SMB1 hacks. Gradius, Pac-Man missed. Game over without lives is low confidence (Tecmo Bowl false).',
              '- in_play comes from d-pad control probes: misses crashed-but-racing (Excitebike), play-call phases '
              '(Tecmo Bowl), boxing and falling pieces where the probed byte is not the player (Punch-Out, Tetris).',
              '- Score byte order is fixed by one carry test; digits past the changing ones are not verified (SMB reads '
              '$07DA-$07E1, proportional to the true $07DD-$07E2). Tetris picked a wrong span.',
              '- Truth checks: 11 games, random play only (stages 1-2, low scores).']
    import subprocess
    import sys
    fam = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/romlib/readers_families.py'), str(R / 'readers.jsonl')],
                         capture_output=True, text=True, check=True).stdout
    lines += ['', fam.replace('## ', '### ').replace('### Families', '## Families', 1)]
    (R / 'readers_summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
