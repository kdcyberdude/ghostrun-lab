// Root-cause experiment: our per-env emulation loop alone (4 frames per step, no drawing/reward/reset), compiled
// like NeSLE's console_step_kernel (__launch_bounds__(128, 5), grid-constant buffers). Compare registers/spills and
// per-step time with the fused training kernel. Usage: emu_only ROM BOOTPADS N STEPS [RANDOM=1]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include "rlenv_build.hpp"

struct DevMem {
    void* alloc(std::size_t n) { void* p; cudaMalloc(&p, n ? n : 1); return p; }
    void upload(void* d, const void* s, std::size_t n) { cudaMemcpy(d, s, n, cudaMemcpyHostToDevice); }
    void zero(void* d, std::size_t n) { cudaMemset(d, 0, n); }
};

template <nesle::cuda::BankMode Mode>
__global__ void __launch_bounds__(128, 5) k_emu(const __grid_constant__ rle::GameCtx g, std::uint64_t* target,
                                                int random, std::uint32_t step) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= g.n) return;
    auto& b = const_cast<nesle::cuda::BatchBuffers&>(g.b);
    std::uint64_t t = target[e];
    for (int f = 0; f < 4; ++f) {
        std::uint32_t x = (e * 2654435761u) ^ (step * 40503u);
        b.action_masks[e] = random ? static_cast<std::uint8_t>((x >> 13) & 0xC3) : 0x80;
        rlm::step_until<Mode>(b, e, t);
        t += rlm::kFrameDots;
    }
    target[e] = t;
}

int main(int argc, char** argv) {
    std::ifstream f(argv[1], std::ios::binary), fb(argv[2], std::ios::binary);
    std::vector<std::uint8_t> rom((std::istreambuf_iterator<char>(f)), {}), boot((std::istreambuf_iterator<char>(fb)), {});
    const int n = atoi(argv[3]), steps = atoi(argv[4]), random = argc > 5 ? atoi(argv[5]) : 1;
    DevMem mem;
    auto built = rle::build(mem, {rle::GameSpec{0, rom, boot, n}}, 1, 0, 0, 0);
    auto& g = built.games[0];
    // every env starts from the boot snapshot (copy slot n -> all)
    for (auto& it : built.items_host[0])
        for (int e = 0; e < n; ++e) cudaMemcpy(it.base + (size_t)e * it.bytes, it.base + (size_t)n * it.bytes, it.bytes, cudaMemcpyDeviceToDevice);
    std::vector<std::uint64_t> tgt(n, g.tmpl_target);
    std::uint64_t* target; cudaMalloc(&target, n * 8); cudaMemcpy(target, tgt.data(), n * 8, cudaMemcpyHostToDevice);
    const int blocks = (n + 127) / 128;
    auto launch = [&](std::uint32_t s) {
        if (g.bank_mode == 0) k_emu<nesle::cuda::kBankModeNrom><<<blocks, 128>>>(g, target, random, s);
        else if (g.bank_mode == 2) k_emu<nesle::cuda::kBankModePaged><<<blocks, 128>>>(g, target, random, s);
        else k_emu<nesle::cuda::kBankModeBanked><<<blocks, 128>>>(g, target, random, s);
    };
    for (int s = 0; s < 5; ++s) launch(s);
    cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) launch(100 + s);
    cudaDeviceSynchronize();
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("envs %d random %d: %.1f ms/step, %.0f steps/s (%s)\n", n, random, 1000 * dt / steps, steps * n / dt,
                cudaGetErrorString(cudaGetLastError()));
}
