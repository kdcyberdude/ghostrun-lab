"""How a routes run (train.py --routes) is doing, from its metrics.jsonl: learning health, and per level the
curriculum pointer (the level counts as solved from that route state on; 0 = from the entrance), the entrance clear
rate now and earlier, and the clear rate from the route states around the pointer.

usage: python rl/scripts/routes_status.py runs/rl/RUN [--ago 60,240]      (iterations back to compare with)
"""
import argparse
import json
import os
from pathlib import Path
import time

ap = argparse.ArgumentParser()
ap.add_argument('run')
ap.add_argument('--ago', default='60,240')
a = ap.parse_args()
run = Path(a.run)
rows = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines() if x.strip()]
rows = [r for r in rows if 'routes' in r]
last = rows[-1]
pid = (run / 'trainer.pid').read_text().strip() if (run / 'trainer.pid').exists() else ''
print(f"trainer {'running' if pid and os.path.exists(f'/proc/{pid}') else 'NOT running'}; last row "
      f"{(time.time() - last['time']) / 60:.1f} min ago; {len(rows)} iterations in this run, "
      f"{(last['time'] - rows[0]['time']) / 3600:.2f} h")
for name, r in [('first', rows[0])] + [(f'-{k}', rows[-1 - k]) for k in (240, 60, 10) if len(rows) > k] + [('now', last)]:
    print(f"  {name:>5}: it {r['it']} sps {r['sps']:.0f} ent {r['ent']:.2f} kl {r['kl']:.4f} ev "
          f"{r['explained_var']:.2f} v {r['v']:.2f} clip {r['clipfrac']:.3f} reward {r['reward_mean']:+.5f}")
ago = [int(x) for x in a.ago.split(',')]
old = [rows[-1 - k]['routes'] if len(rows) > k else None for k in ago]
now = last['routes']
print('level  pointer/states  entrance' + ''.join(f'   -{k}it' for k in ago) +
      '   (pointer then)   clear % from route states pointer-3 .. pointer+2, [pointer]')
for lv in sorted(now, key=int):
    d = now[lv]
    then = ''.join('      -' if o is None or lv not in o else f"   {o[lv]['entrance'] * 100:4.0f}%" for o in old)
    ptr = ' '.join('-' if o is None or lv not in o else str(o[lv]['pointer']) for o in old)
    p, rates = d['pointer'], d.get('rates', [])
    near = ' '.join(('[%s]' if q == p else '%s') % (rates[q - 1][1] if rates[q - 1][0] >= 3 else '-')
                    for q in range(max(p - 3, 1), min(p + 2, len(rates)) + 1))
    print(f"{int(lv) // 4 + 1}-{int(lv) % 4 + 1} {int(lv):>2}   {p:>3}/{d['states']:<3}      "
          f"{d['entrance'] * 100:4.0f}%{then}   ({ptr})   {near}")
er = [d['entrance'] for d in now.values()]
print(f"entrance >= 70%: {sum(e >= 0.7 for e in er)}/{len(er)} levels; lowest {min(er) * 100:.0f}%; mean "
      f"{sum(er) / len(er) * 100:.0f}%; at the entrance: {sum(d['pointer'] == 0 for d in now.values())}/{len(er)}; "
      f"route states still ahead: {sum(d['pointer'] for d in now.values())}")
