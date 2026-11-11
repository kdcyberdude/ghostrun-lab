-- Replay a per-frame button file from power-on in FCEUX and dump the 2 KB CPU RAM after every frame.
-- env: TRACE_LEAD (neutral frames first), TRACE_BUTTONS (1 byte/frame, bit0 A .. bit7 Right), TRACE_OUT (2048 bytes/frame), TRACE_FRAMES (max)
local bpath, opath = os.getenv('TRACE_BUTTONS'), os.getenv('TRACE_OUT')
local nmax = tonumber(os.getenv('TRACE_FRAMES') or '0')
local lead = tonumber(os.getenv('TRACE_LEAD') or '0')  -- neutral frames before the button file (not dumped)
local f = assert(io.open(bpath, 'rb')); local buttons = f:read('*a'); f:close()
local out = assert(io.open(opath, 'wb'))
-- optional TRACE_SHOTS = 'f1,f2,..' (0-based frame index after the lead): after that frame, append the full 256x240
-- screen to TRACE_SHOT_OUT as 256*240 palette bytes followed by 256*240*3 RGB bytes (emu.getscreenpixel, emulator buffer)
local shots, shot_out = {}, nil
for v in string.gmatch(os.getenv('TRACE_SHOTS') or '', '%d+') do shots[tonumber(v)] = true end
if os.getenv('TRACE_SHOT_OUT') then shot_out = assert(io.open(os.getenv('TRACE_SHOT_OUT'), 'wb')) end
local function shot()
  local pal, rgb = {}, {}
  for y = 0, 239 do
    for x = 0, 255 do
      local r, g, b, p = emu.getscreenpixel(x, y, true)
      pal[#pal + 1] = string.char(p); rgb[#rgb + 1] = string.char(r, g, b)
    end
  end
  shot_out:write(table.concat(pal)); shot_out:write(table.concat(rgb))
end
local names = {'A', 'B', 'select', 'start', 'up', 'down', 'left', 'right'}
local n = #buttons
if nmax > 0 and nmax < n then n = nmax end
emu.speedmode('maximum')
for i = 1, lead do joypad.set(1, {}); emu.frameadvance() end
for i = 1, n do
  local b = string.byte(buttons, i)
  local t = {}
  for k = 0, 7 do t[names[k + 1]] = (math.floor(b / 2 ^ k) % 2) == 1 end
  joypad.set(1, t)
  emu.frameadvance()
  out:write(memory.readbyterange(0, 2048))
  if shot_out and shots[i - 1] then shot() end
end
out:close()
if shot_out then shot_out:close() end
emu.exit()
