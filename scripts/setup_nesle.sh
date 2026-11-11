#!/usr/bin/env bash
# Fetch the NES emulator (NeSLE) at the pinned upstream commit, apply this repo's mapper/renderer extensions and build
# the CPU host library. Paths match the defaults of rl/rlenv.py (NESLE_INC, NESLE_ROM_CPP) and
# scripts/nesle_check/neslenes.py (NESLE_HOST_LIB). The rollout environment itself is compiled by rl/rlenv.py on first
# use (CPU build with c++, GPU build with nvcc).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
URL=https://github.com/hbofz/NeSLE.git
REV=0b04d8b9b6a93eea6758b06519206d8f13f6db78
SRC=.cache/nesle-src
if [[ ! -f "$SRC/.patched" ]]; then
    rm -rf "$SRC"
    mkdir -p .cache
    git clone -q "$URL" "$SRC"
    git -C "$SRC" checkout -q "$REV"
    git -C "$SRC" apply "$PWD/third_party/nesle-extensions.patch"
    echo "$REV" > "$SRC/.patched"
fi
mkdir -p .cache/nesle-check
c++ -O2 -std=c++20 -shared -fPIC -I"$SRC/cpp/include" scripts/nesle_check/nesle_capi.cpp "$SRC/cpp/src/rom.cpp" \
    -o .cache/nesle-check/libnesle_host.so
echo "NeSLE ${REV:0:8} + extensions ready: $SRC, .cache/nesle-check/libnesle_host.so"
