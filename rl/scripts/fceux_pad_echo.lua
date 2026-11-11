-- Controller check: set each pad byte of PAD_IN with joypad.set, advance one frame, write back what FCEUX's
-- controller port 1 held that frame (joypad.get) as a byte (bit0 A, B, Select, Start, Up, Down, Left, bit7 Right).
local f = assert(io.open(os.getenv('PAD_IN'), 'rb')); local pads = f:read('*a'); f:close()
local out = assert(io.open(os.getenv('PAD_OUT'), 'wb'))
local names = {'A', 'B', 'select', 'start', 'up', 'down', 'left', 'right'}
emu.speedmode('maximum')
for i = 1, #pads do
  local b = string.byte(pads, i); local t = {}
  for k = 0, 7 do t[names[k + 1]] = (math.floor(b / 2 ^ k) % 2) == 1 end
  joypad.set(1, t)
  emu.frameadvance()
  local g = joypad.get(1); local v = 0
  for k = 0, 7 do if g[names[k + 1]] then v = v + 2 ^ k end end
  out:write(string.char(v))
end
out:close()
emu.exit()
