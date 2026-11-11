// C API over nh::Console for Python (ctypes). Build:
//   c++ -O2 -std=c++20 -shared -fPIC -I.cache/nesle-src/cpp/include scripts/nesle_check/nesle_capi.cpp \
//       .cache/nesle-src/cpp/src/rom.cpp -o .cache/nesle-check/libnesle_host.so
#include <cstdio>
#include <exception>

#include "nesle_host.hpp"

extern "C" {

void* nh_create(const std::uint8_t* rom, std::size_t size) {
    try {
        return new nh::Console(std::vector<std::uint8_t>(rom, rom + size));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nh_create: %s\n", e.what());
        return nullptr;
    }
}

void nh_destroy(void* h) { delete static_cast<nh::Console*>(h); }

// Runs `frames` frames with these buttons; returns the fault code (0 = fine).
std::uint32_t nh_step(void* h, std::uint8_t buttons, int frames) {
    auto* c = static_cast<nh::Console*>(h);
    for (int i = 0; i < frames && c->fault() == 0; ++i) c->step(buttons);
    return c->fault();
}

const std::uint8_t* nh_ram(void* h) { return static_cast<nh::Console*>(h)->ram(); }
std::size_t nh_state_size(void* h) { return static_cast<nh::Console*>(h)->state_size(); }
void nh_save(void* h, std::uint8_t* out) { static_cast<nh::Console*>(h)->save(out); }
void nh_load(void* h, const std::uint8_t* in) { static_cast<nh::Console*>(h)->load(in); }
const std::uint8_t* nh_render(void* h) { return static_cast<nh::Console*>(h)->render(); }

// Resume from an FCEUmm state; returns 0, or 1 if the state cannot be loaded.
int nh_load_fcs(void* h, const std::uint8_t* data, std::size_t size, int dot_offset) {
    try {
        static_cast<nh::Console*>(h)->load_fcs(data, size, dot_offset);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "nh_load_fcs: %s\n", e.what());
        return 1;
    }
}

}  // extern "C"
