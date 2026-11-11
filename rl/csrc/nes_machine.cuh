// One NES console per env on NeSLE's batch code (branch llm-gameplay-env 90d0cb3), host + device.
//
// The per-frame loop is the CPU-verified harness scripts/nesle_check/nesle_host.hpp step_until (lazy PPU settlement,
// idle-loop fast-forward, 200k instruction budget, frame ends at vblank = scanline 241 dot 1), without its C++
// exception path (an unimplemented opcode is reported through StepResult::illegal on both host and device).
// Power-on CPU RAM is 0xFF. The picture of a frame is NeSLE's line renderer (FCEUmm's line renderer replayed from the
// frame log) written as NES colour numbers (6 bits) for the model's 240x224 crop (x 8..247, y 8..231) plus each line's
// colour-emphasis bits (PPUMASK >> 5). RGB is made later from FCEUX's own colour table (rl/fceux_palette.py), so the
// model sees the colours FCEUX shows.
#pragma once
#include <cstdint>
#include <type_traits>

#include "nesle/cuda/batch_console.hpp"
#include "nesle/cuda/batch_render.cuh"
#include "nesle/cuda/batch_step.cuh"

#if defined(__CUDACC__)
#define RLM_HD __host__ __device__
#else
#define RLM_HD
#endif

namespace rlm {

using namespace nesle::cuda;

constexpr std::uint64_t kFrameDots = 341ull * 262ull;
constexpr std::uint64_t kVblankDot = 241ull * 341ull + 1ull;
constexpr std::uint64_t kBudget = 200000;  // instructions per frame (NeSLE binding default)
constexpr int kObsW = 240, kObsH = 224, kObsX0 = 8, kObsY0 = 8;
constexpr int kObsPixels = kObsW * kObsH;

// Every per-env array of a BatchBuffers in a fixed order. `alloc_raw(elem_bytes, count) -> void*` gives each array; the same order
// is used for the host machine (1 env) and the device batch (n envs), so a host state can be uploaded slot by slot.
template <class Alloc>
void bind_arrays(BatchBuffers& b, bool chr_ram, bool paged, Alloc&& alloc_raw) {
#define RLM_A(field, count) \
    (field) = static_cast<std::remove_reference_t<decltype(field)>>(alloc_raw(sizeof(*(field)), (count)))
    RLM_A(b.cpu.pc, 1); RLM_A(b.cpu.a, 1); RLM_A(b.cpu.x, 1); RLM_A(b.cpu.y, 1); RLM_A(b.cpu.sp, 1); RLM_A(b.cpu.p, 1);
    RLM_A(b.cpu.cycles, 1); RLM_A(b.cpu.nmi_pending, 1); RLM_A(b.cpu.irq_pending, 1);
    RLM_A(b.cpu.ram, kCpuRamBytes); RLM_A(b.cpu.prg_ram, kPrgRamBytes);
    RLM_A(b.cpu.controller1_shift, 1); RLM_A(b.cpu.controller1_shift_count, 1); RLM_A(b.cpu.controller1_strobe, 1);
    RLM_A(b.cpu.pending_dma_cycles, 1); RLM_A(b.cpu.fault, 1); RLM_A(b.cpu.fault_pending, 1);
    RLM_A(b.ppu.ctrl, 1); RLM_A(b.ppu.mask, 1); RLM_A(b.ppu.status, 1); RLM_A(b.ppu.oam_addr, 1);
    RLM_A(b.ppu.nmi_pending, 1); RLM_A(b.ppu.frame_dot, 1); RLM_A(b.ppu.frame, 1);
    RLM_A(b.ppu.v, 1); RLM_A(b.ppu.t, 1); RLM_A(b.ppu.x, 1); RLM_A(b.ppu.w, 1); RLM_A(b.ppu.open_bus, 1);
    RLM_A(b.ppu.read_buffer, 1); RLM_A(b.ppu.scroll_x, 1); RLM_A(b.ppu.scroll_y, 1);
    RLM_A(b.ppu.nametable_ram, kNametableRamBytes); RLM_A(b.ppu.palette_ram, kPaletteRamBytes);
    RLM_A(b.ppu.oam, kOamBytes);
    if (chr_ram) { RLM_A(b.ppu.chr_ram, kChrRamBytes); } else { b.ppu.chr_ram = nullptr; }
    RLM_A(b.ppu.frame_log, kFrameLogWords); RLM_A(b.ppu.log_count, 1); RLM_A(b.ppu.log_bank, 1);
    RLM_A(b.ppu.sprite0_dot, 1); RLM_A(b.ppu.snap_oam, kOamBytes); RLM_A(b.ppu.snap_nametable, kNametableRamBytes);
    RLM_A(b.mapper.prg_bank, 1);
    if (paged) {
        RLM_A(b.mapper.prg_page, kPrgPages); RLM_A(b.mapper.chr_page, kChrPages); RLM_A(b.mapper.mirroring, 1);
        RLM_A(b.mapper.regs, kMapperRegBytes);
    }
    RLM_A(b.apu.dmc_event, 1); RLM_A(b.apu.dmc, 1);
    RLM_A(b.action_masks, 1); RLM_A(b.done, 1); RLM_A(b.rewards, 1);
    RLM_A(b.previous_mario_x, 1); RLM_A(b.previous_mario_time, 1);
    b.frames_rgb = nullptr;  // pictures go to the env's own observation buffer
#undef RLM_A
}

// Runs env `env` until the PPU reaches absolute dot `target`. Returns the fault code (pc << 8 | opcode), 0 = fine.
template <BankMode Mode>
RLM_HD std::uint32_t step_until(BatchBuffers& buffers, std::uint32_t env, std::uint64_t target) {
    std::uint64_t instructions = 0;
    std::uint32_t fault = 0;
    auto state = load_cpu_state(buffers, env);
    auto hot = load_ppu_hot_state(buffers, env);
    std::uint32_t pending = 0;
    std::uint32_t to_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
    bool done = false;
    while (instructions < kBudget && !done) {
        if (pending >= to_event) {
            batch_ppu_step_env_hot<Mode>(buffers, env, pending, hot);
            pending = 0;
            to_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
            if (hot.frame * kFrameDots + hot.frame_dot >= target) break;
        }
        const auto step = step_batch_console_instruction_lazy<Mode>(buffers, env, state, hot, pending);
        if (step.cpu.illegal) {
            fault = (static_cast<std::uint32_t>(step.cpu.pc) << 8) | step.cpu.opcode;
            break;
        }
        pending += step.ppu_cycles;
        ++instructions;
        if (step.cpu.opcode == 0x4C && state.pc == step.cpu.pc) {
            batch_ppu_step_env_hot<Mode>(buffers, env, pending, hot);
            pending = 0;
            if (hot.frame * kFrameDots + hot.frame_dot >= target) break;
            (void)fast_forward_batch_console_idle_loop_hot<Mode>(buffers, env, state, hot);
            to_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
            done = hot.frame * kFrameDots + hot.frame_dot >= target;
        }
    }
    if (pending != 0) batch_ppu_step_env_hot<Mode>(buffers, env, pending, hot);
    store_cpu_state(buffers, env, state);
    store_ppu_hot_state(buffers, env, hot);
    return fault;
}

// The last finished frame's picture: colour numbers of the 240x224 crop (row-major) and each crop line's emphasis.
// Writes 4 pixels per store (crop width 240 is a multiple of 4).
// NeSLE's frame_resolve_mmc2 (the MMC2 latches the frame's own fetches left), run after the vblank stop instead of at
// it (BatchBuffers::defer_mmc2_resolve = 1): the same latches-only replay of the same frame, read from the frozen log
// and the vblank snapshots of nametable RAM and OAM, which hold what the live copies held at vblank.
RLM_HD inline void resolve_mmc2_frozen(BatchBuffers& b, std::uint32_t env) {
    std::uint32_t scratch[kLogHeaderWords];
    const ReplaySource src = render_source(b, env, scratch);
    ReplayState s;
    replay_load_header(s, src.header);
    ReplayLine line;
    replay_begin_frame(src, s);
    for (int l = 0; l < 240; ++l) {
        replay_line<true>(src, s, line, l, false);
        replay_end_of_line(src, s, l, true);
    }
    while (s.cursor < src.count) {
        replay_apply(src, s, s.cursor++, false);
    }
    std::uint8_t* r = paged::regs(b, env);
    for (std::uint32_t half = 0; half < 2u; ++half) {
        if (r[paged::kMmc2Latch0 + half] != s.latch[half]) {
            r[paged::kMmc2Latch0 + half] = s.latch[half];
            paged::set_chr4(b, env, half, r[2u * half + s.latch[half]]);
        }
    }
}

RLM_HD inline void render_obs(BatchBuffers& buffers, std::uint32_t env, std::uint8_t* pix, std::uint8_t* emph) {
    std::uint32_t scratch[kLogHeaderWords];
    const ReplaySource src = render_source(buffers, env, scratch);
    ReplayState s;
    replay_load_header(s, src.header);
    ReplayLine line;
    replay_begin_frame(src, s);
    for (int l = 0; l < kObsY0 + kObsH; ++l) {
        replay_line(src, s, line, l, true);
        const std::uint32_t e = replay_compose(s, line);
        if (l >= kObsY0) {
            const int y = l - kObsY0;
            emph[y] = static_cast<std::uint8_t>(e & 7u);
            std::uint32_t* row = reinterpret_cast<std::uint32_t*>(pix + y * kObsW);
            for (int x = 0; x < kObsW; x += 4) {
                const int sx = kObsX0 + x;
                row[x >> 2] = (line.px[sx] & 0x3Fu) | ((line.px[sx + 1] & 0x3Fu) << 8) |
                              ((line.px[sx + 2] & 0x3Fu) << 16) | ((line.px[sx + 3] & 0x3Fu) << 24);
            }
        }
        replay_end_of_line(src, s, l, true);
    }
}

}  // namespace rlm
