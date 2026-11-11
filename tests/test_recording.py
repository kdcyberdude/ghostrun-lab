"""Recording wrapper checks: no model or emulator is loaded."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('options,models,extra', [
    ([], ['v2', 'v1'], []),
    (['--dry-run'], ['v2', 'v1'], ['--dry-run']),
    (['--only', 'v1'], ['v1'], []),
    (['--levels', '1-1,4-4'], ['v2', 'v1'], ['--levels', '1-1,4-4']),
])
def test_recording_wrapper_optional_arguments(tmp_path, options, models, extra):
    root = Path(__file__).resolve().parents[1]
    calls = tmp_path / 'calls.jsonl'
    python = tmp_path / 'fake-python'
    python.write_text(f'#!{sys.executable}\n'
                      'import json, os, sys\n'
                      'with open(os.environ["RECORDING_TEST_CALLS"], "a") as f:\n'
                      '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    python.chmod(0o755)
    out = tmp_path / 'output with spaces'
    subprocess.run(['bash', str(root / 'scripts/record_all_levels.sh'), '--out', str(out), *options],
                   env={**os.environ, 'PYTHON': str(python), 'RECORDING_TEST_CALLS': str(calls)}, check=True)
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        ['-u', f'scripts/recording/record_{model}.py', '--out', str(out), *extra] for model in models
    ]


def test_pixel_player_hud_observer(tmp_path):
    """Synthetic glyphs exercise every level, blank transitions and explicit terminal text."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/recording'))
    from record_v1 import HudObserver
    font = np.zeros((128, 128, 3), dtype=np.uint8)
    for tile in range(256):
        x, y = tile % 16 * 8, tile // 16 * 8
        for bit in range(8):
            if (tile + 1) >> bit & 1:
                font[y + 1, x + bit] = 255
    font_path = tmp_path / 'font.png'
    Image.fromarray(font).save(font_path)
    observer = HudObserver(font_path)
    path = tmp_path / 'frame.png'

    def draw(text, x, y, image):
        for char in text:
            tile = int(char) if char.isdigit() else ord(char) - 55 if char.isupper() else {' ': 36, '-': 40, '.': 175}[char]
            tx, ty = tile % 16 * 8, tile // 16 * 8
            image[y:y + 8, x:x + 8] = font[ty:ty + 8, tx:tx + 8]
            x += 8

    blank = np.zeros((240, 256, 3), dtype=np.uint8)
    Image.fromarray(blank).save(path)
    with pytest.raises(ValueError, match='no recognizable'):
        observer.read(path)
    for level in range(32):
        image = blank.copy()
        draw(f'{level // 4 + 1}-{level % 4 + 1}', 152, 24, image)
        Image.fromarray(image).save(path)
        assert observer.read(path) == dict(level=level, mode=1, x=None)
    Image.fromarray(blank).save(path)
    assert observer.read(path) == dict(level=31, mode=1, x=None)
    for text, x, y, mode in [('GAME OVER', 88, 120, 3), ('YOUR QUEST IS OVER.', 48, 112, 2)]:
        image = blank.copy()
        draw(text, x, y, image)
        Image.fromarray(image).save(path)
        assert observer.read(path) == dict(level=31, mode=mode, x=None)


def test_v1_recording_refuses_existing_capture(tmp_path):
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/recording'))
    from record_v1 import record
    capture = tmp_path / 'v1'
    capture.mkdir()
    marker = capture / 'keep.txt'
    marker.write_text('original capture')
    with pytest.raises(FileExistsError, match='Refusing to overwrite'):
        record(SimpleNamespace(out=tmp_path))
    assert marker.read_text() == 'original capture'
