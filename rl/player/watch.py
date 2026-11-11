"""Keeps runs/rl/policy-export/current pointing at the newest finished model of the training runs.

Training writes RUN/eval/itN/model.pt atomically. When a newer one appears, this exports it to its own folder and only
then switches the link in one atomic step, so a reader never sees a half-written model. If an export fails, the link
stays on the last good model. CPU only; it never touches the training process.
usage: python rl/player/watch.py [--once]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import export  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'runs/rl'
EXPORTS = RUNS / 'policy-export'
LINK = EXPORTS / 'current'
KEEP = 3
POLL_SECONDS = 30


def log(message):
    print(time.strftime('%F %T'), message, flush=True)


def newest():
    found = [(p.stat().st_mtime, p) for p in RUNS.glob('*/eval/it*/model.pt')]
    return max(found)[1] if found else None


def current_source():
    try:
        return json.loads((LINK / 'model.json').read_text())['source']
    except (OSError, ValueError, KeyError):
        return None


def switch(name):
    tmp = EXPORTS / f'.current.tmp{os.getpid()}'
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(name)
    os.replace(tmp, LINK)


def prune(active):
    folders = sorted((p for p in EXPORTS.iterdir() if p.is_dir() and not p.is_symlink() and '.tmp' not in p.name),
                     key=lambda p: p.stat().st_mtime)
    for p in [p for p in folders if p.name != active][:-(KEEP - 1)]:
        shutil.rmtree(p)


def update(failed):
    src = newest()
    if src is None or str(src) == current_source() or src in failed:
        return
    name = f'{src.parents[2].name}-{src.parent.name}'
    try:
        info = export(src, EXPORTS / name)
    except Exception as error:  # keep watching: the next model may be fine, and the link still holds a good one
        failed.add(src)
        log(f'export failed, link unchanged: {src}: {error}')
        return
    switch(name)
    prune(name)
    log(f"current -> {name} (iteration {info['iteration']}, {info['decisions']} decisions)")


def main():
    torch.set_num_threads(2)
    EXPORTS.mkdir(parents=True, exist_ok=True)
    failed = set()
    update(failed)
    while '--once' not in sys.argv:
        time.sleep(POLL_SECONDS)
        update(failed)


if __name__ == '__main__':
    main()
