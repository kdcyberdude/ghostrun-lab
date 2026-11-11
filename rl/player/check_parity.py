"""Does the standalone player press the same buttons as the real model in the training emulator?

Plays each mood in the CPU emulator with PyTorch choosing the moves (rl/evaluate.py's loop), shows the same pictures to
the compiled player as 256x240 PNGs, and compares every frame's buttons, including turbo and dropout. Uses no GPU.
usage: CUDA_VISIBLE_DEVICES= python rl/player/check_parity.py PIXEL_POLICY MODEL_FOLDER [--decisions N]
"""
from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fceux_palette import RGB  # noqa: E402
from model import Policy, pack  # noqa: E402
from rlenv import RlEnv  # noqa: E402

PAL = np.array(RGB, np.uint8)
SEED = 12345
# NES pad byte bit -> player mask bit
PAD_TO_MASK = {1: 16, 2: 32, 4: 512, 8: 256, 16: 1, 32: 2, 64: 4, 128: 8}


def to_mask(pad):
    return sum(m for p, m in PAD_TO_MASK.items() if pad & p)


def png(colours, emphasis):
    canvas = np.zeros((240, 256, 3), np.uint8)
    canvas[8:232, 8:248] = PAL[colours.astype(np.int32) | (emphasis.astype(np.int32)[:, None] << 6)]
    out = BytesIO()
    Image.fromarray(canvas).save(out, 'PNG', compress_level=1)
    return base64.b64encode(out.getvalue()).decode()


class Player:
    def __init__(self, binary, model, mood):
        self.p = subprocess.Popen([binary, '--game', 'mario', '--model', model, '--mood', str(mood), '--seed', str(SEED)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def ask(self, **request):
        self.p.stdin.write((json.dumps(request) + '\n').encode())
        self.p.stdin.flush()
        reply = json.loads(self.p.stdout.readline())
        if not reply.get('ok'):
            raise RuntimeError(reply)
        return reply

    def frame(self, reply, picture):
        return self.ask(op='advance', tick=reply['tick'], applied_mask=reply['mask'], frames_advanced=1,
                        image_png_base64=picture)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('binary')
    ap.add_argument('model')
    ap.add_argument('--decisions', type=int, default=100000)
    a = ap.parse_args()
    torch.set_num_threads(4)
    info = json.loads((Path(a.model) / 'model.json').read_text())
    s = torch.load(info['source'], map_location='cpu', weights_only=False)
    model = Policy(core=s['cfg']['core'], moods=s['cfg'].get('moods', 0))
    model.load_state_dict(s['model'])
    model.eval()
    K = max(1, model.moods)
    env = RlEnv({'smb': K}, device='cpu', seed=SEED, sticky=False, idle_max=0, dropout=True)
    env.start()
    h = torch.zeros(K, model.core)
    prev = [torch.zeros(K, dtype=torch.long) for _ in range(5)]
    mood = torch.arange(K) if model.moods else None
    players = [Player(a.binary, a.model, i) for i in range(K)]
    blank = png(np.zeros((224, 240), np.uint8), np.zeros(224, np.uint8))
    replies = [p.ask(op='reset', image_png_base64=blank) for p in players]
    for i, p in enumerate(players):
        for f in range(4):
            assert replies[i]['mask'] == 0, 'start frames must be neutral'
            replies[i] = p.frame(replies[i], png(env.obs[i, f].numpy(), env.emph[i, f].numpy()))
    live = [True] * K
    frames = [0] * K
    decisions = [0] * K
    failures = []
    for _ in range(a.decisions):
        z = model.encode(env.obs, env.emph, env.timeline, prev, mood)
        h = model.gru(z.float(), h)
        parts, _ = model.act(h, env.masks, greedy=True)
        env.actions.copy_(pack(parts))
        env.step()
        prev = parts
        done = env.done.numpy()
        for i, p in enumerate(players):
            if not live[i]:
                continue
            for f in range(4):
                want, got = to_mask(int(env.pads[i, f])), replies[i]['mask']
                if want != got:
                    failures.append(dict(mood=i, decision=decisions[i], frame=f, torch_mask=want, player_mask=got))
                    live[i] = False
                    break
                frames[i] += 1
                # (after an end, env.obs is the next episode's; the last decision's later frames need only the turbo step)
                if not done[i] or f < 3:
                    replies[i] = p.frame(replies[i], png(env.obs[i, f].numpy(), env.emph[i, f].numpy()))
            if live[i]:
                decisions[i] += 1
                live[i] = not done[i]
        if not any(live):
            break
    for p in players:
        p.p.stdin.close()
        p.p.wait()
    print(json.dumps(dict(model=info, decisions=decisions, frames_compared=frames, failures=failures), indent=1))
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
