// Write NeSLE's CPU RAM after every frame for a per-frame button file (compare.py checks it against fceumm).
//
// usage: nesle_trace ROM BUTTONS.bin OUT_RAM.bin [ram_fill [start_dot [boundary]]]
//   BUTTONS.bin  one byte per frame, NES order (bit0 A, B, Select, Start, Up, Down, Left, bit7 Right)
//   OUT_RAM.bin  2048 bytes per frame
//   ram_fill     power-on CPU RAM: 'ff' (fceumm), 'zero' (NeSLE) or 'fceu' (00 x4, FF x4 repeating)
//   start_dot    power-on PPU position (default 0 = scanline 0 dot 0)
//   boundary     'vblank' (scanline 241 dot 1, where fceumm's frame ends) or 'wrap' (NeSLE: after pre-render)
// Build: c++ -O2 -std=c++20 -I.cache/nesle-src/cpp/include scripts/nesle_check/nesle_trace.cpp
//            .cache/nesle-src/cpp/src/rom.cpp -o .cache/nesle-check/nesle_trace
#include <cstdio>
#include <fstream>
#include <iterator>

#include "nesle_host.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s ROM BUTTONS.bin OUT_RAM.bin [ff|zero|fceu [start_dot [vblank|wrap]]]\n",
                     argv[0]);
        return 2;
    }
    auto read = [](const char* path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error(std::string("cannot open ") + path);
        return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
    };
    const auto buttons = read(argv[2]);
    nh::Console console(read(argv[1]), argc > 4 ? argv[4] : "ff",
                        argc > 5 ? static_cast<std::uint32_t>(std::stoul(argv[5])) : 0u,
                        argc > 6 ? argv[6] : "vblank");
    FILE* out = std::fopen(argv[3], "wb");
    if (!out) throw std::runtime_error("cannot write output");
    std::uint64_t total = 0;
    for (std::size_t f = 0; f < buttons.size(); ++f) {
        const auto r = console.step(buttons[f]);
        total += r.instructions;
        std::fwrite(console.ram(), 1, nh::kCpuRamBytes, out);
        if (r.fault != 0) {
            std::fprintf(stderr, "fault at frame %zu: pc=%04x opcode=%02x\n", f, r.fault >> 8, r.fault & 0xFF);
            std::fclose(out);
            return 4;
        }
        if (r.instructions >= nh::kBudget) std::fprintf(stderr, "instruction budget hit at frame %zu\n", f);
    }
    std::fclose(out);
    std::fprintf(stderr, "ok: %zu frames, %llu instructions\n", buttons.size(),
                 static_cast<unsigned long long>(total));
    return 0;
}
