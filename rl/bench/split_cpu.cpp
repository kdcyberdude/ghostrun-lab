// Per-env cost split on one CPU thread: emulating a frame vs drawing it (render_obs). Same code as the GPU kernel.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include "rlenv_build.hpp"
int main(int argc, char** argv) {
    for (int a = 1; a + 1 < argc; a += 2) {
        std::ifstream f(argv[a], std::ios::binary), fb(argv[a + 1], std::ios::binary);
        std::vector<std::uint8_t> rom((std::istreambuf_iterator<char>(f)), {}), boot((std::istreambuf_iterator<char>(fb)), {});
        rle::HostMachine hm(rom);
        for (auto p : boot) hm.frame(p);
        std::vector<std::uint8_t> pix(rlm::kObsPixels), em(rlm::kObsH);
        double te = 0, tr = 0; unsigned x = 1;
        const int n = 600;
        for (int i = 0; i < n; ++i) {
            x = x * 1103515245u + 12345u;
            auto t0 = std::chrono::steady_clock::now();
            hm.frame(static_cast<std::uint8_t>((x >> 16) & 0xF3));
            auto t1 = std::chrono::steady_clock::now();
            rlm::render_obs(hm.b, 0, pix.data(), em.data());
            auto t2 = std::chrono::steady_clock::now();
            te += std::chrono::duration<double>(t1 - t0).count();
            tr += std::chrono::duration<double>(t2 - t1).count();
        }
        std::printf("%-60s emulate %.0f us/frame  draw %.0f us/frame  (draw share %.0f%%)\n", argv[a], 1e6 * te / n,
                    1e6 * tr / n, 100 * tr / (te + tr));
    }
}
