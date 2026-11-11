// GPU build of the environment: one thread per env, one kernel launch per game per decision (each game's cartridge
// board picks the emulator specialisation, as NeSLE's own launcher does). Same per-env code as the CPU reference.
// build: see rl/rlenv.py (build_cuda); static cudart, shares the CUDA context and stream with PyTorch.
#include <cuda_runtime.h>

#include <cstring>
#include <exception>
#include <string>

#include "rlenv_api.h"
#include "rlenv_build.hpp"

namespace {

thread_local std::string g_error;

struct DevMem {
    std::vector<void*> blocks;
    void* alloc(std::size_t n) {
        void* p = nullptr;
        if (cudaMalloc(&p, n ? n : 1) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
        blocks.push_back(p);
        return p;
    }
    void upload(void* dst, const void* src, std::size_t n) {
        if (cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("upload failed");
    }
    void zero(void* dst, std::size_t n) {
        if (cudaMemset(dst, 0, n) != cudaSuccess) throw std::runtime_error("cudaMemset failed");
    }
    ~DevMem() { for (void* p : blocks) cudaFree(p); }
};

struct Env {
    DevMem mem;
    rle::Built<DevMem> e;
    int device = 0;
    // one stream per game so the 12 games' kernels run at the same time (each alone fills only a few SMs)
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> done_ev;
    cudaEvent_t start_ev = nullptr;
    ~Env() {
        for (auto s : streams) cudaStreamDestroy(s);
        for (auto e : done_ev) cudaEventDestroy(e);
        if (start_ev) cudaEventDestroy(start_ev);
    }
};

// Per-phase kernels (rlenv_core.cuh). The emulation kernels are bounded like NeSLE's console_step_kernel so the
// instruction loop stays in registers; small blocks spread one game's envs over many SMs.
constexpr int kThreads = 32;
#ifndef RLE_EMU_MINB
#define RLE_EMU_MINB 20  // min resident blocks per SM -> register cap (20: 96 registers)
#endif
#define RLE_EMU_BOUNDS __launch_bounds__(kThreads, RLE_EMU_MINB)

#define RLE_IDX                                                   \
    const int e = g.e0 + static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); \
    if (e >= g.e1) return;                                        \
    rle::GameCtx& gg = const_cast<rle::GameCtx&>(g);              \
    rle::Common& cc = const_cast<rle::Common&>(c);

template <nesle::cuda::BankMode Mode>
__global__ void RLE_EMU_BOUNDS k_frame(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c, int f) {
    RLE_IDX rle::ph_frame<Mode>(gg, cc, e, f);
}
__global__ void __launch_bounds__(kThreads) k_render(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c, int f,
                                                     int restart_pass) {
    RLE_IDX rle::ph_render(gg, cc, e, f, restart_pass != 0);
}
__global__ void __launch_bounds__(kThreads) k_finish(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c) {
    RLE_IDX rle::ph_finish(gg, cc, e);
}
__global__ void __launch_bounds__(kThreads) k_reset_state(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c) {
    RLE_IDX rle::reset_state(gg, cc, e);
}
template <nesle::cuda::BankMode Mode>
__global__ void RLE_EMU_BOUNDS k_reset_idle(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c) {
    RLE_IDX rle::ph_reset_idle<Mode>(gg, cc, e);
}
template <nesle::cuda::BankMode Mode>
__global__ void RLE_EMU_BOUNDS k_reset_frame(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c) {
    RLE_IDX rle::ph_reset_frame<Mode>(gg, cc, e);
}
__global__ void __launch_bounds__(kThreads) k_reset_finish(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c) {
    RLE_IDX rle::ph_reset_finish_controls(gg, cc, e);
}

// Level-start saves after a decision: one block per slot of the game.
__global__ void __launch_bounds__(128) k_archive(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c) {
    const int s = static_cast<int>(blockIdx.x);
    if (g.lv_claim[s] == ~0ull) return;   // same for the whole block
    const bool take = rle::archive_take(g, s);
    if (take) rle::archive_copy(g, s, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
    __syncthreads();
    if (threadIdx.x == 0) rle::archive_commit(g, c, s, take);
}

// Place saves (rl/places.py): block j copies env env[j] of the game into slot slot[j] (skipped where slot < 0).
__global__ void __launch_bounds__(128) k_place_save(const __grid_constant__ rle::GameCtx g, const __grid_constant__ rle::Common c,
                                                    const std::int32_t* env, const std::int32_t* slot) {
    const int s = slot[blockIdx.x], e = env[blockIdx.x];
    if (s < g.lv_levels || s >= g.lv_cap || e < 0 || e >= g.n) return;
    rle::archive_copy_env(g, s, e, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x));
    __syncthreads();
    if (threadIdx.x == 0) rle::place_commit(g, c, s, e);
}

__global__ void k_gather_rows(uint4* __restrict__ dst, const uint4* __restrict__ src, const std::int64_t* rows,
                              std::size_t row_vecs) {
    const std::size_t j = blockIdx.x;
    const uint4* s = src + static_cast<std::size_t>(rows[j]) * row_vecs;
    uint4* d = dst + j * row_vecs;
    for (std::size_t i = blockIdx.y * blockDim.x + threadIdx.x; i < row_vecs; i += gridDim.y * blockDim.x) d[i] = s[i];
}

int check(const char* what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        g_error = std::string(what) + ": " + cudaGetErrorString(err);
        return 1;
    }
    return 0;
}

// Four genuine 6-bit palette indices occupy three bytes. Emphasis is stored separately.
__global__ void k_pack6(const std::uint8_t* src, std::uint8_t* dst, std::size_t groups) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    const std::uint32_t v = src[4*i] | (static_cast<std::uint32_t>(src[4*i+1]) << 6) |
                            (static_cast<std::uint32_t>(src[4*i+2]) << 12) |
                            (static_cast<std::uint32_t>(src[4*i+3]) << 18);
    dst[3*i] = v; dst[3*i+1] = v >> 8; dst[3*i+2] = v >> 16;
}
__global__ void k_unpack6(const std::uint8_t* src, std::uint8_t* dst, std::size_t groups) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    const std::uint32_t v = src[3*i] | (static_cast<std::uint32_t>(src[3*i+1]) << 8) |
                            (static_cast<std::uint32_t>(src[3*i+2]) << 16);
    dst[4*i] = v & 63; dst[4*i+1] = (v >> 6) & 63;
    dst[4*i+2] = (v >> 12) & 63; dst[4*i+3] = (v >> 18) & 63;
}

// Phase sequence of one game on its stream.
void game_phases(const rle::GameCtx& g, bool start, const rle::Common& c, cudaStream_t s) {
    const int blocks = (g.e1 - g.e0 + kThreads - 1) / kThreads;
#define RLE_MODE(KERNEL, ...)                                                                              \
    do {                                                                                                   \
        if (g.bank_mode == 0) KERNEL<nesle::cuda::kBankModeNrom><<<blocks, kThreads, 0, s>>>(__VA_ARGS__); \
        else if (g.bank_mode == 2) KERNEL<nesle::cuda::kBankModePaged><<<blocks, kThreads, 0, s>>>(__VA_ARGS__); \
        else KERNEL<nesle::cuda::kBankModeBanked><<<blocks, kThreads, 0, s>>>(__VA_ARGS__);                \
    } while (0)
    if (start) {
        k_reset_state<<<blocks, kThreads, 0, s>>>(g, c);
    } else {
        for (int f = 0; f < rle::kFramesPerDecision; ++f) {
            RLE_MODE(k_frame, g, c, f);
            k_render<<<blocks, kThreads, 0, s>>>(g, c, f, 0);
        }
        k_finish<<<blocks, kThreads, 0, s>>>(g, c);
        if (g.lv_cap > 0) k_archive<<<g.lv_levels, 128, 0, s>>>(g, c);
    }
    RLE_MODE(k_reset_idle, g, c);
    for (int f = 0; f < rle::kFramesPerDecision; ++f) {
        RLE_MODE(k_reset_frame, g, c);
        k_render<<<blocks, kThreads, 0, s>>>(g, c, f, 1);
    }
    k_reset_finish<<<blocks, kThreads, 0, s>>>(g, c);
#undef RLE_MODE
}

// All games at once (one stream each), joined back into the caller's stream.
int launch_all(Env& env, bool start, cudaStream_t s) {
    if (cudaEventRecord(env.start_ev, s) != cudaSuccess) return check("event");
    for (std::size_t gi = 0; gi < env.e.games.size(); ++gi) {
        cudaStreamWaitEvent(env.streams[gi], env.start_ev, 0);
        game_phases(env.e.games[gi], start, env.e.common, env.streams[gi]);
        cudaEventRecord(env.done_ev[gi], env.streams[gi]);
        cudaStreamWaitEvent(s, env.done_ev[gi], 0);
    }
    return check(start ? "start" : "step");
}

}  // namespace

extern "C" {

void* rlenv_create(int ngames, const int* game_ids, const std::uint8_t* const* roms, const std::size_t* rom_sizes,
                   const std::uint8_t* const* boot_pads, const std::size_t* boot_sizes, const int* n_envs,
                   std::uint64_t run_seed, int sticky, int idle_max, int dropout, int device) {
    try {
        if (cudaSetDevice(device) != cudaSuccess) throw std::runtime_error("cudaSetDevice failed");
        // the per-env emulator code keeps its replay state on the stack
        cudaDeviceSetLimit(cudaLimitStackSize, 4096);
        std::vector<rle::GameSpec> specs;
        for (int i = 0; i < ngames; ++i)
            specs.push_back({game_ids[i], std::vector<std::uint8_t>(roms[i], roms[i] + rom_sizes[i]),
                             std::vector<std::uint8_t>(boot_pads[i], boot_pads[i] + boot_sizes[i]), n_envs[i]});
        auto* env = new Env();
        env->device = device;
        env->e = rle::build(env->mem, specs, run_seed, sticky, idle_max, dropout);
        cudaEventCreateWithFlags(&env->start_ev, cudaEventDisableTiming);
        for (std::size_t i = 0; i < env->e.games.size(); ++i) {
            cudaStream_t st;
            cudaEvent_t ev;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            env->streams.push_back(st);
            env->done_ev.push_back(ev);
        }
        if (check("streams")) throw std::runtime_error(g_error);
        return env;
    } catch (const std::exception& ex) {
        g_error = ex.what();
        return nullptr;
    }
}

int rlenv_num_envs(void* h) { return static_cast<Env*>(h)->e.total; }
void rlenv_set_episode_level_end(void* h, int level) { static_cast<Env*>(h)->e.common.episode_level_end = level; }

void rlenv_bind(void* h, const std::int16_t* actions, std::uint8_t* obs, std::uint8_t* emph, float* reward,
                std::uint8_t* done, std::uint8_t* masks, std::uint8_t* timeline, std::uint8_t* pads,
                std::uint32_t* events, float* info) {
    auto& c = static_cast<Env*>(h)->e.common;
    c.actions = actions; c.obs = obs; c.emph = emph; c.reward = reward; c.done = done; c.masks = masks;
    c.timeline = timeline; c.pads = pads; c.events = events; c.info = info;
}

void rlenv_bind_parts(void* h, float* parts) { static_cast<Env*>(h)->e.common.parts = parts; }

void rlenv_bind_obs(void* h, std::uint8_t* obs, std::uint8_t* emph) {
    auto& c = static_cast<Env*>(h)->e.common;
    c.obs = obs;
    c.emph = emph;
}

// Settings (pointers, flags) go to the kernels as launch parameters, so every launch sees the current binding.
int rlenv_start(void* h, void* stream) {
    return launch_all(*static_cast<Env*>(h), true, static_cast<cudaStream_t>(stream));
}
int rlenv_step(void* h, void* stream) {
    return launch_all(*static_cast<Env*>(h), false, static_cast<cudaStream_t>(stream));
}
// One game's decision on the caller's stream, its pictures written into obs/emph (whole-batch tensors; this game
// writes only its own rows). Lets each game's rollout run at its own pace.
int rlenv_num_games(void* h) { return static_cast<int>(static_cast<Env*>(h)->e.games.size()); }
int rlenv_step_game(void* h, int gi, std::uint8_t* obs, std::uint8_t* emph, void* stream) {
    auto& env = *static_cast<Env*>(h);
    rle::Common c = env.e.common;
    c.obs = obs;
    c.emph = emph;
    game_phases(env.e.games[static_cast<std::size_t>(gi)], false, c, static_cast<cudaStream_t>(stream));
    return check("step_game");
}
// Envs [start, start + count) of game gi only (same as rlenv_step_game for that range).
int rlenv_step_range(void* h, int gi, int start, int count, std::uint8_t* obs, std::uint8_t* emph, void* stream) {
    auto& env = *static_cast<Env*>(h);
    rle::Common c = env.e.common;
    c.obs = obs;
    c.emph = emph;
    rle::GameCtx g = env.e.games[static_cast<std::size_t>(gi)];
    g.e0 = start;
    g.e1 = start + count;
    if (start < 0 || count <= 0 || g.e1 > g.n) { g_error = "step_range: bad range"; return 1; }
    game_phases(g, false, c, static_cast<cudaStream_t>(stream));
    return check("step_range");
}

std::size_t rlenv_state_bytes(void* h) {
    std::size_t n = 0;
    rle::for_each_state(static_cast<Env*>(h)->e, [&](void*, std::size_t b) { n += b; });
    return n;
}

int rlenv_save(void* h, std::uint8_t* out, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    int bad = 0;
    rle::for_each_state(static_cast<Env*>(h)->e, [&](void* p, std::size_t b) {
        bad |= cudaMemcpyAsync(out, p, b, cudaMemcpyDeviceToHost, s) != cudaSuccess;
        out += b;
    });
    bad |= cudaStreamSynchronize(s) != cudaSuccess;
    if (bad) g_error = "save failed";
    return bad;
}

int rlenv_load(void* h, const std::uint8_t* in, void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    int bad = 0;
    rle::for_each_state(static_cast<Env*>(h)->e, [&](void* p, std::size_t b) {
        bad |= cudaMemcpyAsync(p, in, b, cudaMemcpyHostToDevice, s) != cudaSuccess;
        in += b;
    });
    bad |= cudaStreamSynchronize(s) != cudaSuccess;
    if (bad) g_error = "load failed";
    return bad;
}

int rlenv_enable_level_starts(void* h, int cap) { return rlenv_enable_places(h, cap, cap); }
int rlenv_enable_places(void* h, int cap, int levels) {
    try {
        auto& env = *static_cast<Env*>(h);
        rle::enable_level_starts(env.mem, env.e, cap, levels);
        return 0;
    } catch (const std::exception& ex) {
        g_error = ex.what();
        return 1;
    }
}
void rlenv_set_level_plan(void* h, int gi, double p, const std::uint32_t* cdf) {
    auto& env = *static_cast<Env*>(h);
    rle::set_level_plan(env.mem, env.e, gi, p, cdf);
}
void rlenv_bind_start(void* h, std::int32_t* start_out) { static_cast<Env*>(h)->e.common.start_out = start_out; }
void rlenv_bind_level(void* h, std::int32_t* level_out) { static_cast<Env*>(h)->e.common.level_out = level_out; }
void rlenv_bind_progress(void* h, float* progress_out) { static_cast<Env*>(h)->e.common.progress_out = progress_out; }
// Deaths of each saved level start of game gi (-1 = none), cap entries, to host memory (synchronous).
int rlenv_level_deaths(void* h, int gi, std::int32_t* out, void* stream) {
    auto& env = *static_cast<Env*>(h);
    const auto& g = env.e.games[static_cast<std::size_t>(gi)];
    if (g.lv_cap == 0) return 0;
    auto s = static_cast<cudaStream_t>(stream);
    if (cudaMemcpyAsync(out, g.lv_deaths, g.lv_cap * 4, cudaMemcpyDeviceToHost, s) != cudaSuccess ||
        cudaStreamSynchronize(s) != cudaSuccess) { g_error = "level_deaths failed"; return 1; }
    return 0;
}

int rlenv_place_save(void* h, int gi, const std::int32_t* env_idx, const std::int32_t* slot_idx, int count, void* stream) {
    auto& env = *static_cast<Env*>(h);
    const auto& g = env.e.games[static_cast<std::size_t>(gi)];
    if (g.lv_cap == 0 || count <= 0) return 0;
    k_place_save<<<count, 128, 0, static_cast<cudaStream_t>(stream)>>>(g, env.e.common, env_idx, slot_idx);
    return check("place_save");
}
int rlenv_place_copy(void* h, int gi, const std::int32_t* src, const std::int32_t* dst, int count, void* stream) {
    auto& env = *static_cast<Env*>(h);
    const auto& g = env.e.games[static_cast<std::size_t>(gi)];
    auto s = static_cast<cudaStream_t>(stream);
    int bad = 0;
    for (int i = 0; i < count; ++i) {
        const std::size_t a = static_cast<std::size_t>(src[i]), b = static_cast<std::size_t>(dst[i]);
        if (src[i] < 0 || dst[i] < 0 || a >= static_cast<std::size_t>(g.lv_cap) || b >= static_cast<std::size_t>(g.lv_cap)) {
            g_error = "place_copy: bad slot"; return 1;
        }
        bad |= cudaMemcpyAsync(g.lv_mem + b * g.lv_stride, g.lv_mem + a * g.lv_stride, g.lv_stride, cudaMemcpyDeviceToDevice, s);
        bad |= cudaMemcpyAsync(g.lv_st + b, g.lv_st + a, sizeof(rl10::EnvState), cudaMemcpyDeviceToDevice, s);
        if (g.key_cap)
            bad |= cudaMemcpyAsync(g.lv_keys + b * g.key_cap, g.lv_keys + a * g.key_cap, g.key_cap * 4, cudaMemcpyDeviceToDevice, s);
        bad |= cudaMemcpyAsync(g.lv_target + b, g.lv_target + a, 8, cudaMemcpyDeviceToDevice, s);
        bad |= cudaMemcpyAsync(g.lv_frame + b, g.lv_frame + a, 4, cudaMemcpyDeviceToDevice, s);
        bad |= cudaMemcpyAsync(g.lv_deaths + b, g.lv_deaths + a, 4, cudaMemcpyDeviceToDevice, s);
    }
    bad |= cudaStreamSynchronize(s);
    if (bad) g_error = "place_copy failed";
    return bad != 0;
}
void rlenv_level_layout(void* h, int gi, std::uint32_t* out) {
    const auto& g = static_cast<Env*>(h)->e.games[static_cast<std::size_t>(gi)];
    out[0] = g.lv_stride; out[1] = g.key_cap; out[2] = sizeof(rl10::EnvState); out[3] = static_cast<std::uint32_t>(g.lv_cap);
    out[4] = static_cast<std::uint32_t>(g.lv_levels);
}

void rlenv_set_no_render(void* h, int v) {
    static_cast<Env*>(h)->e.common.no_render = v;
}


// ---- host-memory rollout storage (pictures of a long rollout do not fit next to the learner in VRAM) ----
// Pinned host memory is device-addressable under UVA, so a kernel can gather learner minibatch rows over PCIe.
void* rl_host_alloc(std::size_t bytes) {
    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) != cudaSuccess) {
        g_error = "cudaHostAlloc failed";
        return nullptr;
    }
    return p;
}
void rl_host_free(void* p) { cudaFreeHost(p); }
int rl_pack6(const void* src, void* dst, std::size_t pixels, void* stream) {
    if (pixels % 4) { g_error = "pack6: pixel count must be divisible by four"; return 1; }
    const auto groups = pixels / 4;
    if (groups) k_pack6<<<(groups + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const std::uint8_t*>(src), static_cast<std::uint8_t*>(dst), groups);
    return check("pack6");
}
int rl_unpack6(const void* src, void* dst, std::size_t pixels, void* stream) {
    if (pixels % 4) { g_error = "unpack6: pixel count must be divisible by four"; return 1; }
    const auto groups = pixels / 4;
    if (groups) k_unpack6<<<(groups + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const std::uint8_t*>(src), static_cast<std::uint8_t*>(dst), groups);
    return check("unpack6");
}
int rl_memcpy_async(void* dst, const void* src, std::size_t bytes, void* stream) {
    if (cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, static_cast<cudaStream_t>(stream)) != cudaSuccess)
        return check("memcpy");
    return 0;
}
// dst row j = src row rows[j]; row_bytes a multiple of 16.
int rl_gather_rows(void* dst, const void* src, const std::int64_t* rows, int nrows, std::size_t row_bytes,
                   void* stream) {
    const int chunks = 32;
    dim3 grid(nrows, chunks);
    k_gather_rows<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<uint4*>(dst),
                                                                      static_cast<const uint4*>(src), rows,
                                                                      row_bytes / 16);
    return check("gather");
}

// Copy-engine gather (no SMs used): dst[t * n + j] = src[t * N + envs[j]] for t < T, one 2-D copy per env.
int rl_gather_envs(void* dst, const void* src, const std::int64_t* envs, int n, int T, int N, std::size_t row_bytes,
                   void* stream) {
    auto s = static_cast<cudaStream_t>(stream);
    for (int j = 0; j < n; ++j) {
        if (cudaMemcpy2DAsync(static_cast<char*>(dst) + j * row_bytes, n * row_bytes,
                              static_cast<const char*>(src) + envs[j] * row_bytes, static_cast<std::size_t>(N) * row_bytes,
                              row_bytes, T, cudaMemcpyDefault, s) != cudaSuccess)
            return check("gather2d");
    }
    return 0;
}

const char* rlenv_error() { return g_error.c_str(); }

void rlenv_destroy(void* h) { delete static_cast<Env*>(h); }

}  // extern "C"
