-- Does this ROM run in FCEUX? Play BOOT_FRAMES frames from power-on, tapping Start every 120 frames (and A every 60),
-- then write one line to BOOT_OUT: frames, distinct screens seen (sampled every 30 frames), distinct RAM states,
-- distinct colours on the last screen. A dead ROM shows 1 screen / 1 colour; a running one many.
local out = os.getenv('BOOT_OUT')
local n = tonumber(os.getenv('BOOT_FRAMES') or '1800')
local shots, rams, nshot, nram = {}, {}, 0, 0
emu.speedmode('maximum')
local function sig_screen()
  local h, cols, nc = 0, {}, 0
  for y = 8, 231, 4 do
    for x = 0, 255, 4 do
      local _, _, _, p = emu.getscreenpixel(x, y, true)
      h = (h * 31 + p) % 2147483647
      if not cols[p] then cols[p] = true; nc = nc + 1 end
    end
  end
  return h, nc
end
local ncol = 0
for i = 1, n do
  local t = {}
  if i % 120 < 6 then t.start = true end
  if i % 60 >= 30 and i % 60 < 36 then t.A = true end
  joypad.set(1, t)
  emu.frameadvance()
  if i % 30 == 0 then
    local h, nc = sig_screen()
    ncol = nc
    if not shots[h] then shots[h] = true; nshot = nshot + 1 end
    local r = 0
    for a = 0, 2047, 7 do r = (r * 131 + memory.readbyte(a)) % 2147483647 end
    if not rams[r] then rams[r] = true; nram = nram + 1 end
  end
end
local f = assert(io.open(out, 'w'))
f:write(string.format('%d %d %d %d\n', n, nshot, nram, ncol))
f:close()
os.exit(0)
