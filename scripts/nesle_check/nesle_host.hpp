// One NeSLE console on the CPU: NeSLE's CUDA batch code (the code the GPU runs) compiled for the host,
// driven one frame at a time like fceumm. Shared by nesle_trace (frame traces) and nesle_capi (Python).
//
// The frame loop is a host copy of console_step_kernel (NeSLE cpp/src/cuda/kernels.cu): lazy PPU settlement,
// idle-loop fast-forward, the binding's 200k instruction budget. Two choices make it comparable with fceumm:
//   power-on CPU RAM 0xFF (fceumm) instead of NeSLE's zeros, and
//   a frame ends at vblank (scanline 241 dot 1), where fceumm's frame ends, instead of after the pre-render line.
#pragma once

#include <cstdint>
#include <array>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "nesle/cuda/batch_console.hpp"
#include "nesle/cuda/batch_render.cuh"
#include "nesle/cuda/batch_step.cuh"
#include "nesle/fcs.hpp"
#include "nesle/rom.hpp"

namespace nh {

using namespace nesle::cuda;

constexpr std::uint64_t kFrameDots = 341ull * 262ull;
constexpr std::uint64_t kVblankDot = 241ull * 341ull + 1ull;
constexpr std::uint64_t kBudget = 200'000;  // CudaBatchBinding default max_instructions_per_frame_

struct Frame {
    std::uint64_t instructions = 0;
    std::uint32_t fault = 0;  // (pc << 8) | opcode of an unimplemented opcode, 0 = none
};

// Runs one env until the PPU reaches absolute dot `target` (frame * kFrameDots + frame_dot).
template <BankMode Mode>
Frame step_until(BatchBuffers& buffers, std::uint64_t target) {
    const std::uint32_t env = 0;
    Frame out;
    auto state = load_cpu_state(buffers, env);
    auto hot = load_ppu_hot_state(buffers, env);
    std::uint32_t pending = 0;
    std::uint32_t to_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
    auto reached = [&] { return hot.frame * kFrameDots + hot.frame_dot >= target; };
    bool done = false;
    while (out.instructions < kBudget && !done) {
        if (pending >= to_event) {
            batch_ppu_step_env_hot<Mode>(buffers, env, pending, hot);
            pending = 0;
            to_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
            if (reached()) break;
        }
        BatchConsoleStepResult step;
        try {
            step = step_batch_console_instruction_lazy<Mode>(buffers, env, state, hot, pending);
        } catch (const std::exception&) {
            out.fault = (static_cast<std::uint32_t>(state.pc) << 8) | 0xFFu;
            break;
        }
        if (step.cpu.illegal) {
            out.fault = (static_cast<std::uint32_t>(step.cpu.pc) << 8) | step.cpu.opcode;
            break;
        }
        pending += step.ppu_cycles;
        ++out.instructions;
        if (step.cpu.opcode == 0x4C && state.pc == step.cpu.pc) {
            batch_ppu_step_env_hot<Mode>(buffers, env, pending, hot);
            pending = 0;
            if (reached()) break;
            (void)fast_forward_batch_console_idle_loop_hot<Mode>(buffers, env, state, hot);
            to_event = batch_ppu_cycles_until_next_event_hot<Mode>(buffers, env, hot);
            done = reached();
        }
    }
    if (pending != 0) batch_ppu_step_env_hot<Mode>(buffers, env, pending, hot);
    store_cpu_state(buffers, env, state);
    store_ppu_hot_state(buffers, env, hot);
    return out;
}

class Console {
public:
    // fill: power-on CPU RAM 'ff' (fceumm), 'zero' (NeSLE) or 'fceu' (00 x4, FF x4);
    // boundary: 'vblank' (fceumm frame end) or 'wrap' (NeSLE frame end)
    Console(const std::vector<std::uint8_t>& rom_bytes, const std::string& fill = "ff",
            std::uint32_t start_dot = 0, const std::string& boundary = "vblank")
        : rom_(nesle::parse_ines(rom_bytes)) {
        const auto layout = nesle::describe_mapper(rom_.metadata);
        if (!layout.supported) throw std::runtime_error(nesle::unsupported_mapper_reason(rom_.metadata));
        auto& b = b_;
        b.cpu.pc = alloc<std::uint16_t>();
        b.cpu.a = alloc<std::uint8_t>();
        b.cpu.x = alloc<std::uint8_t>();
        b.cpu.y = alloc<std::uint8_t>();
        b.cpu.sp = alloc<std::uint8_t>();
        b.cpu.p = alloc<std::uint8_t>();
        b.cpu.cycles = alloc<std::uint64_t>();
        b.cpu.nmi_pending = alloc<std::uint8_t>();
        b.cpu.irq_pending = alloc<std::uint8_t>();
        b.cpu.ram = alloc<std::uint8_t>(kCpuRamBytes);
        b.cpu.prg_ram = alloc<std::uint8_t>(kPrgRamBytes);
        b.cpu.controller1_shift = alloc<std::uint8_t>();
        b.cpu.controller1_shift_count = alloc<std::uint8_t>();
        b.cpu.controller1_strobe = alloc<std::uint8_t>();
        b.cpu.pending_dma_cycles = alloc<std::uint32_t>();
        b.cpu.fault = alloc<std::uint32_t>();
        b.cpu.fault_pending = alloc<std::uint32_t>();
        b.ppu.ctrl = alloc<std::uint8_t>();
        b.ppu.mask = alloc<std::uint8_t>();
        b.ppu.status = alloc<std::uint8_t>();
        b.ppu.oam_addr = alloc<std::uint8_t>();
        b.ppu.nmi_pending = alloc<std::uint8_t>();
        b.ppu.frame_dot = alloc<std::uint32_t>();
        b.ppu.frame = alloc<std::uint64_t>();
        b.ppu.v = alloc<std::uint16_t>();
        b.ppu.t = alloc<std::uint16_t>();
        b.ppu.x = alloc<std::uint8_t>();
        b.ppu.w = alloc<std::uint8_t>();
        b.ppu.open_bus = alloc<std::uint8_t>();
        b.ppu.read_buffer = alloc<std::uint8_t>();
        b.ppu.scroll_x = alloc<std::uint8_t>();
        b.ppu.scroll_y = alloc<std::uint8_t>();
        b.ppu.nametable_ram = alloc<std::uint8_t>(kNametableRamBytes);
        b.ppu.palette_ram = alloc<std::uint8_t>(kPaletteRamBytes);
        b.ppu.oam = alloc<std::uint8_t>(kOamBytes);
        b.ppu.chr_ram = rom_.chr_rom.empty() ? alloc<std::uint8_t>(kChrRamBytes) : nullptr;
        b.ppu.frame_log = alloc<std::uint32_t>(kFrameLogWords);
        b.ppu.log_count = alloc<std::uint16_t>();
        b.ppu.log_bank = alloc<std::uint8_t>();
        b.ppu.sprite0_dot = alloc<std::uint32_t>();
        b.ppu.snap_oam = alloc<std::uint8_t>(kOamBytes);
        b.ppu.snap_nametable = alloc<std::uint8_t>(kNametableRamBytes);
        b.mapper.prg_bank = alloc<std::uint8_t>();
        if (layout.bank_kind >= kBankingFirstPaged) {  // CNROM/MMC1/MMC2/MMC3 page tables and registers
            b.mapper.prg_page = alloc<std::uint32_t>(kPrgPages);
            b.mapper.chr_page = alloc<std::uint32_t>(kChrPages);
            b.mapper.mirroring = alloc<std::uint8_t>();
            b.mapper.regs = alloc<std::uint8_t>(kMapperRegBytes);
        }
        b.apu.dmc_event = alloc<std::uint64_t>();  // DMC sample channel ($4015 status, fetch cycles)
        b.apu.dmc = alloc<DmcState>();
        b.action_masks = alloc<std::uint8_t>();
        b.done = alloc<std::uint8_t>();
        b.rewards = alloc<float>();
        b.previous_mario_x = alloc<int>();
        b.previous_mario_time = alloc<int>();
        rgb_.assign(kFrameWidth * kFrameHeight * kRgbChannels, 0);  // output only, not machine state
        b.frames_rgb = rgb_.data();

        // Cartridge view, as CudaBatchBinding::upload_rom builds it.
        const std::size_t real_prg = rom_.prg_rom.size();
        std::size_t padded = 1;
        while (padded < real_prg) padded <<= 1u;
        prg_.assign(padded, 0);
        std::memcpy(prg_.data(), rom_.prg_rom.data(), real_prg);
        b.cart.prg_rom = prg_.data();
        b.cart.prg_rom_size = static_cast<std::uint32_t>(padded);
        b.cart.prg_reset_vector_base = nesle::reset_vector_base(layout, real_prg);
        b.cart.prg_rom_mask = static_cast<std::uint32_t>(padded - 1u);
        if (!rom_.chr_rom.empty()) {
            b.cart.chr_rom = rom_.chr_rom.data();
            b.cart.chr_rom_size = static_cast<std::uint32_t>(rom_.chr_rom.size());
        }
        b.cart.mapper = rom_.metadata.mapper;
        switch (rom_.metadata.nametable_arrangement) {
            case nesle::NametableArrangement::Vertical: b.cart.nametable_arrangement = kNametableVertical; break;
            case nesle::NametableArrangement::Horizontal: b.cart.nametable_arrangement = kNametableHorizontal; break;
            case nesle::NametableArrangement::FourScreen: b.cart.nametable_arrangement = kNametableFourScreen; break;
        }
        b.cart.bank_kind = layout.bank_kind;
        b.cart.bus_conflicts = layout.bus_conflicts ? 1 : 0;
        b.cart.prg_window_start = 0x8000u;
        std::uint32_t shift = 0;
        while ((1u << shift) < layout.window_bytes) ++shift;
        b.cart.prg_window_shift = shift;
        b.cart.prg_window_mask = layout.window_bytes == 0 ? 0u : layout.window_bytes - 1u;
        b.cart.prg_bank_mask = layout.bank_mask;
        b.cart.prg_bank_shift = layout.bank_shift;
        b.cart.prg_fixed_base =
            layout.fixed_bytes == 0 ? 0u : static_cast<std::uint32_t>(real_prg - layout.fixed_bytes);
        b.cart.prg_fixed_mask = layout.fixed_bytes == 0 ? 0u : layout.fixed_bytes - 1u;
        b.cart.prg_page_mask = layout.prg_page_mask;
        b.cart.chr_page_mask = layout.chr_page_mask;

        cold_reset_console_env(b, 0);
        if (fill != "zero") {
            for (int i = 0; i < kCpuRamBytes; ++i) {
                b.cpu.ram[i] = fill == "ff" ? 0xFF : ((i & 4) ? 0xFF : 0x00);
            }
        }
        b.ppu.frame_dot[0] = static_cast<std::uint32_t>(start_dot % kFrameDots);
        const std::uint64_t phase = boundary == "vblank" ? kVblankDot : 0;
        target_ = phase > start_dot ? phase : phase + kFrameDots;
    }

    Console(const Console&) = delete;
    Console& operator=(const Console&) = delete;

    // One frame with these buttons (bit0 A, B, Select, Start, Up, Down, Left, bit7 Right).
    Frame step(std::uint8_t buttons) {
        b_.action_masks[0] = buttons;
        const auto r = b_.cart.bank_kind == kBankingNone               ? step_until<kBankModeNrom>(b_, target_)
                       : b_.cart.bank_kind >= kBankingFirstPaged ? step_until<kBankModePaged>(b_, target_)
                                                                 : step_until<kBankModeBanked>(b_, target_);
        target_ += kFrameDots;
        if (r.fault != 0) fault_ = r.fault;
        return r;
    }

    // Resume from an FCEUmm state (stable_retro's em.get_state()), through the same
    // warm_reset_console_env the GPU snapshot reset runs. FCEUmm saves at the end of
    // its frame: line 240 dot 0 of its line grid, plus the dots its last instruction
    // ran past that (ICou). `dot_offset` is where that point sits on NeSLE's grid
    // relative to line 240 dot 0. The next step() runs to the end of the following
    // frame, so its picture is the frame after the saved one.
    void load_fcs(const std::uint8_t* data, std::size_t size, int dot_offset) {
        const auto s = nesle::fcs::parse(std::span<const std::uint8_t>(data, size));
        const auto layout = nesle::describe_mapper(rom_.metadata);
        std::array<std::uint8_t, kMapperRegBytes> regs{};
        std::uint8_t irq = 0;
        if (layout.bank_kind >= kBankingFirstPaged &&
            !nesle::fcs::board_registers(s, layout.bank_kind, regs, irq)) {
            throw std::runtime_error("state has no board registers for this cartridge");
        }
        const std::uint8_t level = 0;
        const std::uint8_t ppu_ctrl = s.ppu_ctrl, ppu_mask = s.ppu_mask, ppu_status = s.ppu_status;
        const std::uint8_t oam_addr = s.ppu_oam_addr, open_bus = s.ppu_open_bus, read_buffer = s.ppu_read_buffer;
        const std::uint8_t ppu_x = s.ppu_x, ppu_w = s.ppu_w, cart_latch = s.cart_latch;
        const std::uint16_t ppu_v = s.ppu_v, ppu_t = s.ppu_t, pc = s.pc;
        const std::uint8_t a = s.a, x = s.x, y = s.y, sp = s.sp, p = s.p;
        const std::uint64_t cycles = 0;
        SnapshotTemplate t{};
        t.cpu_ram = s.cpu_ram.data();
        t.prg_ram = s.prg_ram.data();
        t.nametable_ram = s.nametable_ram.data();
        t.palette_ram = s.palette_ram.data();
        t.oam = s.oam.data();
        t.pc = &pc;
        t.a = &a;
        t.x = &x;
        t.y = &y;
        t.sp = &sp;
        t.p = &p;
        t.cycles = &cycles;
        t.ppu_ctrl = &ppu_ctrl;
        t.ppu_mask = &ppu_mask;
        t.ppu_status = &ppu_status;
        t.ppu_oam_addr = &oam_addr;
        t.ppu_open_bus = &open_bus;
        t.ppu_read_buffer = &read_buffer;
        t.ppu_x = &ppu_x;
        t.ppu_w = &ppu_w;
        t.ppu_v = &ppu_v;
        t.ppu_t = &ppu_t;
        t.env_to_level = &level;
        t.chr_ram = s.has_chr_ram ? s.chr_ram.data() : nullptr;
        t.cart_latch = s.has_cart_latch ? &cart_latch : nullptr;
        t.mapper_regs = layout.bank_kind >= kBankingFirstPaged ? regs.data() : nullptr;
        t.irq_line = layout.bank_kind >= kBankingFirstPaged ? &irq : nullptr;
        t.num_levels = 1;
        warm_reset_console_env(b_, 0, t);
        // NeSLE's $2005 shadow registers, from t and fine x (the state has no such thing).
        b_.ppu.scroll_x[0] = static_cast<std::uint8_t>(((ppu_t & 0x1F) << 3) | ppu_x);
        b_.ppu.scroll_y[0] = static_cast<std::uint8_t>((((ppu_t >> 5) & 0x1F) << 3) | ((ppu_t >> 12) & 7));
        const int overshoot = (-s.cpu_count + 8) / 16;  // 1/48 CPU cycle units -> PPU dots
        b_.ppu.frame_dot[0] = static_cast<std::uint32_t>(240 * 341 + dot_offset + overshoot);
        b_.ppu.frame[0] = 0;
        target_ = kFrameDots + kVblankDot;
        fault_ = 0;
        // FCEUmm's odd-frame line start ("kook") of the next frame, for the frame log.
        b_.ppu.log_bank[0] = static_cast<std::uint8_t>((b_.ppu.log_bank[0] & 1u) | ((s.ppu_kook & 1u) << 1));
    }

    const std::uint8_t* ram() const { return b_.cpu.ram; }
    std::uint32_t fault() const { return fault_; }

    std::size_t state_size() const {
        std::size_t n = sizeof(target_) + sizeof(fault_);
        for (const auto& v : keep_) n += v.size();
        return n;
    }
    void save(std::uint8_t* out) const {
        for (const auto& v : keep_) {
            std::memcpy(out, v.data(), v.size());
            out += v.size();
        }
        std::memcpy(out, &target_, sizeof(target_));
        std::memcpy(out + sizeof(target_), &fault_, sizeof(fault_));
    }
    void load(const std::uint8_t* in) {
        for (auto& v : keep_) {
            std::memcpy(v.data(), in, v.size());
            in += v.size();
        }
        std::memcpy(&target_, in, sizeof(target_));
        std::memcpy(&fault_, in + sizeof(target_), sizeof(fault_));
    }

    // NeSLE's own renderer (vblank snapshot), 256x240 RGB.
    const std::uint8_t* render() {
        render_batch_rgb_frame_env(b_, 0);
        return rgb_.data();
    }

private:
    template <class T>
    T* alloc(std::size_t count = 1) {
        keep_.emplace_back(count * sizeof(T), 0);  // moving the outer vector keeps each buffer in place
        return reinterpret_cast<T*>(keep_.back().data());
    }

    nesle::RomImage rom_;
    std::vector<std::uint8_t> prg_;
    std::vector<std::vector<std::uint8_t>> keep_;  // every per-env buffer = the whole machine state
    std::vector<std::uint8_t> rgb_;
    BatchBuffers b_{};
    std::uint64_t target_ = 0;
    std::uint32_t fault_ = 0;
};

}  // namespace nh
