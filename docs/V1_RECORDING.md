# Mario v1 native capture

The 2026-11-11 capture used one fresh `auto {ticks:1}` stream from a standard reset of the supplied `pixel-player`. This is the fixed-route-assisted v1 player, not evidence of general exploration or recovery. No route was submitted as input during capture. Its 10,626 applied masks and durations independently matched the bundled route; the run advanced 79,824 legacy reference clock frames and visited 1-1 through 8-4 in order.

The player reported terminal with Mario beside the Princess in 8-4. The final image does **not** yet contain `YOUR QUEST IS OVER.`. The visible ending observer therefore reports `won: false`, which is preserved. All 32 per-level videos exist; this is not an independently verified ending-text win or a claim about the browser remake.

The machine-readable checks are in [v1-capture-audit.json](v1-capture-audit.json).

## PNG-only capture

```sh
python scripts/recording/record_v1.py \
  --pixel-player /path/to/pixel-player \
  --hud-font /path/to/background-tiles-22-30-21-0f.png \
  --rom /path/to/compatible-user-supplied.nes \
  --out recordings --keep-lab-output
```

Use the licensed original player package and a compatible user-supplied game file. Its Linux build needs a compatible runtime; this capture used a private PRoot with unpacked Ubuntu 24 libc on an Ubuntu 22 host, without changing system libraries. Neither ROM nor player internals were inspected. The `--reference-lab` interface remains available when that external launcher is installed.

The HUD observer compares visible white glyphs against the supplied atlas using the same method as the remake's `tools/audit-gameplay.mjs`. It reads the world label, Game Over, and ending text, retaining the last known world through blank transition screens. It does not infer player x; that field is null. `--keep-lab-output` retains every PNG and `reference-lab/ticks.jsonl`. Nonempty capture directories are rejected to protect earlier recordings.

## Interpretation and validation

- Videos are silent and tick-sampled: one unique picture per decision (roughly eight per second), held for the reported 7/8-frame duration in a 60-fps MP4. They are not full-frame native video.
- Level cuts use the world HUD after each decision, so a transition can cross a cut within that decision. Use the full-game action log and each `route_offset_frames` for replay; per-level clips are not independent native resets.
- The native clock is the legacy 29,781 CPU-cycle slice, not a PPU frame. A browser replay must use its documented timing conversion.
- Six lightweight tests passed: four recording-wrapper cases, a synthetic observer test covering all 32 labels/blank transitions/terminal messages, and overwrite protection. The actual initial screenshot independently matched 1-1.
- The final screenshot and action log remain unaltered. The release audit checks every MP4's frame count, ordered level coverage, masks and durations, and action-log/pad consistency.

No v2 outputs or user-owned jobs were modified by this capture.
