"""Standalone player for the rl/ policy: reads game screenshots on stdin, writes controller buttons on stdout.

One tick = one game frame. The model decides every 4th frame from the last 4 pictures; this process holds its memory,
turbo timing and button dropout. Needs only the exported model folder: no ROM, emulator, RAM, GPU or training code.
Protocol: PROTOCOL.md. Mirrors rl/csrc/controller.cuh and the episode start in rl/csrc/rlenv_core.cuh.
"""
import argparse
import base64
import binascii
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

BUTTONS = ('up', 'down', 'left', 'right', 'a', 'b', 'x', 'y', 'start', 'select')
UP, DOWN, LEFT, RIGHT, A, B, START, SELECT = 1, 2, 4, 8, 16, 32, 256, 512
DPAD = (0, UP, UP | RIGHT, RIGHT, DOWN | RIGHT, DOWN, DOWN | LEFT, LEFT, UP | LEFT)
SIZES = (9, 3, 3, 2, 2)
FRAMES_PER_DECISION = 4
CROP_X, CROP_Y, OBS_W, OBS_H = 8, 8, 240, 224
MAX_PACKET = 512 * 1024
M64 = (1 << 64) - 1
TIMELINE, RING, DROP_MIN, DROP_MAX, GAP_P32 = 45, 64, 8, 22, 122713351


class RequestError(ValueError):
    pass


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def episode_seed(run_seed, env, episode):
    return splitmix64(splitmix64(run_seed ^ 0xD20F0D20F0) ^ (env << 40) ^ episode)


class Dropout:
    """The button-dropout schedule of controller.cuh: at most one of 8 inputs is off at a time, known 45 decisions ahead."""

    def __init__(self, seed, enabled):
        self.seed, self.counter, self.enabled, self.head = seed, 0, enabled, 0
        self.drop_left, self.drop_input = 0, -1
        self.gap_left = self.draw_gap() if enabled else 0x7FFFFFFF
        self.ring = [-1] * RING
        for j in range(TIMELINE):
            self.ring[j] = self.next_column() if enabled else -1

    def rand_u32(self):
        value = splitmix64(self.seed ^ splitmix64(self.counter)) >> 32
        self.counter += 1
        return value

    def rand_below(self, n):
        return (self.rand_u32() * n) >> 32

    def draw_gap(self):
        k = 1
        while self.rand_u32() >= GAP_P32:
            k += 1
        return k

    def next_column(self):
        if self.drop_left > 0:
            self.drop_left -= 1
            if self.drop_left == 0:
                self.gap_left = self.draw_gap()
            return self.drop_input
        self.gap_left -= 1
        if self.gap_left == 0:
            self.drop_input = self.rand_below(8)
            self.drop_left = DROP_MIN + self.rand_below(DROP_MAX - DROP_MIN + 1)
        return -1

    def advance(self):
        self.ring[self.head % RING] = -1
        self.head += 1
        self.ring[(self.head + TIMELINE - 1) % RING] = self.next_column() if self.enabled else -1

    def timeline(self):
        out = np.zeros((1, 8, TIMELINE), np.float32)
        for j in range(TIMELINE):
            dropped = self.ring[(self.head + j) % RING]
            if dropped >= 0:
                out[0, dropped, j] = 1
        return out

    def masks(self):
        """Legal choices of the 5 move parts (d-pad, A, B, Start, Select); Start/Select stay blocked during play."""
        dropped = self.ring[self.head % RING]
        gone = (UP, DOWN, LEFT, RIGHT)[dropped] if 0 <= dropped < 4 else 0
        dpad = [0 if bits & gone else 1 for bits in DPAD]
        a, b = [1, 1, 1], [1, 1, 1]
        if dropped == 4:
            a[1] = 0
        elif dropped == 6:
            a[2] = 0
        elif dropped == 5:
            b[1] = 0
        elif dropped == 7:
            b[2] = 0
        return np.array([dpad + a + b + [1, 0, 1, 0]], np.float32)


def decode_image(encoded):
    if not isinstance(encoded, str) or len(encoded) > 400000:
        raise RequestError('Expected a base64 PNG of the native game canvas')
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.format != 'PNG' or image.size != (256, 240) or image.mode not in ('RGB', 'RGBA'):
                raise RequestError('Image must be a 256x240 RGB or opaque RGBA PNG')
            if image.mode == 'RGBA' and image.getextrema()[3] != (255, 255):
                raise RequestError('Canvas must be opaque')
            return np.asarray(image.convert('RGB'))[CROP_Y:CROP_Y + OBS_H, CROP_X:CROP_X + OBS_W]
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError) as error:
        if isinstance(error, RequestError):
            raise
        raise RequestError(str(error)) from error


def frame_mask(parts, tick):
    """The buttons of one frame: A/B on 2 are turbo, pressed on even frames counted from reset."""
    dpad, a, b, start, select = parts
    even = tick % 2 == 0
    mask = DPAD[dpad]
    if a == 1 or (a == 2 and even):
        mask |= A
    if b == 1 or (b == 2 and even):
        mask |= B
    if start:
        mask |= START
    if select:
        mask |= SELECT
    return mask


class PolicySession:
    def __init__(self, actor, info, game, mood, seed, dropout):
        if not 0 <= mood < max(1, info['moods']):
            raise ValueError(f"--mood must be 0..{max(1, info['moods']) - 1}")
        self.actor, self.info, self.game, self.mood, self.seed, self.dropout_on = actor, info, game, mood, seed, dropout
        self.tick, self.mask = 0, None

    def clear(self):
        self.pictures, self.parts, self.tick = [], (0, 0, 0, 0, 0), 0
        self.h = np.zeros((1, self.info['core']), np.float32)
        self.dropout = Dropout(episode_seed(self.seed, self.mood, 0), self.dropout_on)

    def decide(self, pictures):
        pixels = np.concatenate(pictures, axis=-1).astype(np.float32) / np.float32(255)
        prev = np.zeros((1, sum(SIZES)), np.float32)
        offset = 0
        for choice, size in zip(self.parts, SIZES):
            prev[0, offset + choice] = 1
            offset += size
        feed = dict(pixels=np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]), timeline=self.dropout.timeline(),
                    prev=prev, h=self.h, masks=self.dropout.masks())
        if self.info['moods']:
            feed['mood'] = np.eye(self.info['moods'], dtype=np.float32)[self.mood:self.mood + 1]
        parts, h = self.actor.run(['parts', 'h_next'], feed)
        if not np.isfinite(h).all():
            raise RequestError('Model inference failed; history was not advanced')
        return tuple(int(p) for p in parts[0]), h

    def response(self):
        return dict(ok=True, tick=self.tick, mask=self.mask, frames=1,
                    buttons=[button for bit, button in enumerate(BUTTONS) if self.mask & (1 << bit)])

    def advance(self, request):
        if self.mask is None:
            raise RequestError('Reset with the current game image first')
        if type(request['tick']) is not int or request['tick'] != self.tick:
            raise RequestError('tick must acknowledge the current pending action')
        applied = request['applied_mask']
        if type(applied) is not int or not 0 <= applied < 1024:
            raise RequestError('applied_mask must be an integer in 0..1023')
        if type(request['frames_advanced']) is not int or request['frames_advanced'] != 1:
            raise RequestError('Game must advance exactly 1 native frame before acknowledgement')
        pictures = (self.pictures + [decode_image(request['image_png_base64'])])[-FRAMES_PER_DECISION:]
        tick = self.tick + 1
        if tick % FRAMES_PER_DECISION == 0:   # the 4 start frames are neutral, then one decision per 4 frames
            self.parts, self.h = self.decide(pictures)
            self.dropout.advance()
        self.pictures, self.tick = pictures, tick
        self.mask = frame_mask(self.parts, tick)

    def handle(self, request):
        try:
            if not isinstance(request, dict) or not isinstance(request.get('op'), str):
                raise RequestError('A JSON object with an op is required')
            op = request['op']
            fields = {'hello': {'op'}, 'suggest': {'op'}, 'reset': {'op', 'image_png_base64'},
                      'advance': {'op', 'tick', 'applied_mask', 'frames_advanced', 'image_png_base64'}}
            if op not in fields or set(request) != fields[op]:
                raise RequestError('Unsupported operation or request fields')
            if op == 'hello':
                return dict(ok=True, protocol='external-pixel-policy-v2', game=self.game, buttons=list(BUTTONS),
                            frames_per_tick=1, frames_per_decision=FRAMES_PER_DECISION, image_size=[256, 240],
                            native_fps_fraction=[1789773, 29781], operations=list(fields), mood=self.mood,
                            moods=self.info['moods'], dropout=self.dropout_on, seed=self.seed,
                            model=dict(iteration=self.info['iteration'], sha256=self.info['onnx_sha256']),
                            input='External PNG only')
            if op == 'reset':
                decode_image(request['image_png_base64'])
                self.clear()
                self.mask = 0
            elif op == 'advance':
                self.advance(request)
            elif self.mask is None:
                raise RequestError('Reset with the current game image first')
            return self.response()
        except RequestError as error:
            return dict(ok=False, error=str(error))


def default_model():
    frozen = getattr(sys, 'frozen', False)
    return (Path(sys.executable) if frozen else Path(__file__)).absolute().parent / 'model'


def load(folder):
    folder = Path(folder).resolve()   # follow the link once, so both files come from the same model
    info = json.loads((folder / 'model.json').read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(folder / 'policy.onnx'), options, providers=['CPUExecutionProvider']), info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game', choices=['mario'], required=True)
    parser.add_argument('--model', type=Path, default=None, help='model folder (default: the "model" link next to this program)')
    parser.add_argument('--mood', type=int, default=0, help="the model's style of play, a number from 0")
    parser.add_argument('--dropout', choices=['on', 'off'], default='on', help='temporarily disabled buttons, as in training')
    parser.add_argument('--seed', type=int, default=12345, help='dropout schedule')
    args = parser.parse_args()
    actor, info = load(args.model or default_model())
    session = PolicySession(actor, info, args.game, args.mood, args.seed, args.dropout == 'on')
    while raw := sys.stdin.buffer.readline(MAX_PACKET + 1):
        if len(raw) > MAX_PACKET:
            print(json.dumps(dict(ok=False, error='Request exceeds 512KiB')), flush=True)
            return 2
        try:
            reply = session.handle(json.loads(raw))
        except (ValueError, UnicodeError):
            reply = dict(ok=False, error='Invalid JSON')
        print(json.dumps(reply, separators=(',', ':'), allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
