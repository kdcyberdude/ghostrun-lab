// Builds the environment's memory for either target: the boot snapshot of each game is made on the host by running
// the game's recorded boot pad bytes through the same frame loop, then every per-env array is allocated for n + 1
// envs with `Mem` (host memory for the CPU reference, device memory for the GPU) and the snapshot goes to slot n.
#pragma once
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "nesle/rom.hpp"
#include "rlenv_core.cuh"

namespace rle {

using namespace nesle::cuda;

struct GameSpec {
    int game;                                 // rl10 game id
    std::vector<std::uint8_t> rom;            // iNES file
    std::vector<std::uint8_t> boot_pads;      // pad byte of every frame from power-on to the boot point
    int n;                                    // envs
};

// One console on the host: cartridge view + 1 env of every array.
struct HostMachine {
    BatchBuffers b{};
    std::vector<std::vector<std::uint8_t>> arrays;  // bind_arrays order
    std::vector<std::uint32_t> bytes;
    std::vector<std::uint8_t> prg, chr;
    int mode = 0;
    bool chr_ram = false, paged = false;
    std::uint64_t target = 0;

    HostMachine(const std::vector<std::uint8_t>& rom_bytes) {
        const auto rom = nesle::parse_ines(std::span<const std::uint8_t>(rom_bytes.data(), rom_bytes.size()));
        const auto layout = nesle::describe_mapper(rom.metadata);
        if (!layout.supported) throw std::runtime_error(nesle::unsupported_mapper_reason(rom.metadata));
        chr_ram = rom.chr_rom.empty();
        paged = layout.bank_kind >= kBankingFirstPaged;
        mode = layout.bank_kind == kBankingNone ? 0 : paged ? 2 : 1;
        rlm::bind_arrays(b, chr_ram, paged, [&](std::size_t elem, int count) -> void* {
            const std::uint32_t n = static_cast<std::uint32_t>(elem * count);
            arrays.emplace_back(n, 0);
            bytes.push_back(n);
            return arrays.back().data();
        });
        // re-point: emplace_back may have moved inner vectors' owners, but their data buffers stay in place
        // (std::vector move keeps the heap buffer), so the pointers above remain valid.
        const std::size_t real_prg = rom.prg_rom.size();
        std::size_t padded = 1;
        while (padded < real_prg) padded <<= 1u;
        prg.assign(padded, 0);
        std::memcpy(prg.data(), rom.prg_rom.data(), real_prg);
        chr = rom.chr_rom;
        auto& c = b.cart;
        c.prg_rom = prg.data();
        c.prg_rom_size = static_cast<std::uint32_t>(padded);
        c.prg_reset_vector_base = nesle::reset_vector_base(layout, real_prg);
        c.prg_rom_mask = static_cast<std::uint32_t>(padded - 1u);
        if (!chr.empty()) {
            c.chr_rom = chr.data();
            c.chr_rom_size = static_cast<std::uint32_t>(chr.size());
        }
        c.mapper = rom.metadata.mapper;
        switch (rom.metadata.nametable_arrangement) {
            case nesle::NametableArrangement::Vertical: c.nametable_arrangement = kNametableVertical; break;
            case nesle::NametableArrangement::Horizontal: c.nametable_arrangement = kNametableHorizontal; break;
            case nesle::NametableArrangement::FourScreen: c.nametable_arrangement = kNametableFourScreen; break;
        }
        c.bank_kind = layout.bank_kind;
        c.bus_conflicts = layout.bus_conflicts ? 1 : 0;
        c.prg_window_start = 0x8000u;
        std::uint32_t shift = 0;
        while ((1u << shift) < layout.window_bytes) ++shift;
        c.prg_window_shift = shift;
        c.prg_window_mask = layout.window_bytes == 0 ? 0u : layout.window_bytes - 1u;
        c.prg_bank_mask = layout.bank_mask;
        c.prg_bank_shift = layout.bank_shift;
        c.prg_fixed_base = layout.fixed_bytes == 0 ? 0u : static_cast<std::uint32_t>(real_prg - layout.fixed_bytes);
        c.prg_fixed_mask = layout.fixed_bytes == 0 ? 0u : layout.fixed_bytes - 1u;
        c.prg_page_mask = layout.prg_page_mask;
        c.chr_page_mask = layout.chr_page_mask;
        cold_reset_console_env(b, 0);
        for (int i = 0; i < kCpuRamBytes; ++i) b.cpu.ram[i] = 0xFF;  // power-on RAM 0xFF
        b.ppu.frame_dot[0] = 0;
        target = rlm::kVblankDot;
    }

    // Returns the fault code of the first faulting frame, 0 = fine.
    std::uint32_t frame(std::uint8_t pad) {
        b.action_masks[0] = pad;
        std::uint32_t f = mode == 0 ? rlm::step_until<kBankModeNrom>(b, 0, target)
                        : mode == 2 ? rlm::step_until<kBankModePaged>(b, 0, target)
                                    : rlm::step_until<kBankModeBanked>(b, 0, target);
        target += rlm::kFrameDots;
        return f;
    }
};

// Mem: void* alloc(size_t); void upload(void* dst, const void* src, size_t); void zero(void* dst, size_t)
template <class Mem>
struct Built {
    std::vector<GameCtx> games;
    Common common{};
    int total = 0;
    std::vector<std::vector<CopyItem>> items_host;  // per game, host copy of the CopyItem list
    int lv_cap = 0;                                 // level starts (enable_level_starts)
};

template <class T, class Mem>
T* alloc_n(Mem& mem, std::size_t count) {
    void* p = mem.alloc(sizeof(T) * count);
    mem.zero(p, sizeof(T) * count);
    return static_cast<T*>(p);
}

template <class Mem>
Built<Mem> build(Mem& mem, const std::vector<GameSpec>& specs, std::uint64_t run_seed, int sticky, int idle_max,
                 int dropout) {
    Built<Mem> out;
    for (const auto& s : specs) out.total += s.n;
    int offset = 0;
    for (const auto& s : specs) {
        HostMachine hm(s.rom);
        for (std::size_t i = 0; i < s.boot_pads.size(); ++i) {
            if (hm.frame(s.boot_pads[i]) != 0)
                throw std::runtime_error("emulator fault during boot of game " + std::to_string(s.game));
        }
        GameCtx g{};
        g.game = s.game;
        g.bank_mode = hm.mode;
        g.n = s.n;
        g.e0 = 0;
        g.e1 = s.n;
        g.offset = offset;
        offset += s.n;
        g.start_ok = rl10::kGameConsts[s.game].start_ok;
        g.select_ok = rl10::kGameConsts[s.game].select_ok;
        g.key_cap = rl10::keyset_capacity(s.game);
        g.keys = g.key_cap ? alloc_n<std::uint32_t>(mem, static_cast<std::size_t>(s.n) * g.key_cap) : nullptr;
        g.tmpl_target = hm.target;
        g.idle_tmpl = idle_max > 0 ? idle_max : 0;
        std::vector<CopyItem> items;
        std::size_t idx = 0;
        rlm::bind_arrays(g.b, hm.chr_ram, hm.paged, [&](std::size_t elem, int count) -> void* {
            const std::uint32_t n = static_cast<std::uint32_t>(elem * count);
            if (n != hm.bytes[idx]) throw std::logic_error("array order mismatch");
            auto* base = static_cast<std::uint8_t*>(mem.alloc(static_cast<std::size_t>(n) * (s.n + 1)));
            mem.zero(base, static_cast<std::size_t>(n) * (s.n + 1));
            mem.upload(base + static_cast<std::size_t>(n) * s.n, hm.arrays[idx].data(), n);
            std::uint8_t* idle = g.idle_tmpl ? static_cast<std::uint8_t*>(
                mem.alloc(static_cast<std::size_t>(n) * g.idle_tmpl)) : nullptr;
            items.push_back(CopyItem{base, n, idle});
            ++idx;
            return base;
        });
        // the idle-frame snapshots: the boot snapshot after k = 1..idle_tmpl frames with no buttons
        for (int k = 1; k <= g.idle_tmpl; ++k) {
            if (hm.frame(0) != 0) throw std::runtime_error("emulator fault in idle frames of game " + std::to_string(s.game));
            for (std::size_t i = 0; i < items.size(); ++i)
                mem.upload(items[i].idle + static_cast<std::size_t>(items[i].bytes) * (k - 1), hm.arrays[i].data(),
                           items[i].bytes);
        }
        g.b.cart = hm.b.cart;
        g.b.defer_mmc2_resolve = 1;  // MMC2 latches resolved after each frame by rl/ (rlenv_core.cuh emu_frame)
        auto* prg = mem.alloc(hm.prg.size());
        mem.upload(prg, hm.prg.data(), hm.prg.size());
        g.b.cart.prg_rom = static_cast<std::uint8_t*>(prg);
        if (!hm.chr.empty()) {
            auto* chr = mem.alloc(hm.chr.size());
            mem.upload(chr, hm.chr.data(), hm.chr.size());
            g.b.cart.chr_rom = static_cast<std::uint8_t*>(chr);
        }
        auto* it = static_cast<CopyItem*>(mem.alloc(sizeof(CopyItem) * items.size()));
        mem.upload(it, items.data(), sizeof(CopyItem) * items.size());
        g.items = it;
        out.items_host.push_back(items);
        g.nitems = static_cast<int>(items.size());
        out.games.push_back(g);
    }
    const std::size_t N = static_cast<std::size_t>(out.total);
    Common& c = out.common;
    c.st = alloc_n<rl10::EnvState>(mem, N);
    c.drop = alloc_n<rlc::Dropout>(mem, N);
    c.target = alloc_n<std::uint64_t>(mem, N);
    c.ep_frame = alloc_n<std::uint32_t>(mem, N);
    c.prev_applied = alloc_n<std::int16_t>(mem, N);
    c.sticky_seed = alloc_n<std::uint64_t>(mem, N);
    c.sticky_ctr = alloc_n<std::uint64_t>(mem, N);
    c.episode = alloc_n<std::uint32_t>(mem, N);
    {
        std::vector<std::uint32_t> init(N, 0xFFFFFFFFu);
        mem.upload(c.episode, init.data(), N * 4);
    }
    c.ep_ret = alloc_n<float>(mem, N);
    c.ep_len = alloc_n<std::int32_t>(mem, N);
    c.ep_maxlvl = alloc_n<std::int32_t>(mem, N);
    c.ep_idle = alloc_n<std::int32_t>(mem, N);
    c.fault = alloc_n<std::uint32_t>(mem, N);
    c.resetting = alloc_n<std::uint8_t>(mem, N);
    c.mmc2_pending = alloc_n<std::uint8_t>(mem, N);  // 0 between decisions (not part of the saved state)
    c.run_seed = run_seed;
    c.sticky = sticky;
    c.idle_max = idle_max;
    c.dropout = dropout;
    return out;
}

// Level starts (rlenv_core.cuh): room for cap saved level starts per game, empty. Call before start / load.
template <class Mem>
void enable_level_starts(Mem& mem, Built<Mem>& out, int cap, int levels) {
    if (out.lv_cap != 0 || cap <= 0 || levels <= 0 || levels > cap)
        throw std::logic_error("level starts: enable once, 0 < levels <= cap");
    for (std::size_t gi = 0; gi < out.games.size(); ++gi) {
        GameCtx& g = out.games[gi];
        std::uint32_t stride = 0;
        for (const auto& it : out.items_host[gi]) stride += (it.bytes + 15u) & ~15u;
        g.lv_cap = cap;
        g.lv_levels = levels;
        g.lv_stride = stride;
        g.lv_mem = alloc_n<std::uint8_t>(mem, static_cast<std::size_t>(cap) * stride);
        g.lv_st = alloc_n<rl10::EnvState>(mem, cap);
        g.lv_keys = g.key_cap ? alloc_n<std::uint32_t>(mem, static_cast<std::size_t>(cap) * g.key_cap) : nullptr;
        g.lv_target = alloc_n<std::uint64_t>(mem, cap);
        g.lv_frame = alloc_n<std::uint32_t>(mem, cap);
        g.lv_deaths = alloc_n<std::int32_t>(mem, cap);
        g.lv_claim = alloc_n<unsigned long long>(mem, cap);
        g.lv_cdf = alloc_n<std::uint32_t>(mem, cap);
        std::vector<std::int32_t> none(cap, -1);
        mem.upload(g.lv_deaths, none.data(), cap * 4);
        std::vector<unsigned long long> free_(cap, ~0ull);
        mem.upload(g.lv_claim, free_.data(), cap * 8);
    }
    const std::size_t N = static_cast<std::size_t>(out.total);
    out.common.ep_start = alloc_n<std::int32_t>(mem, N);
    std::vector<std::int32_t> boot(N, -1);
    mem.upload(out.common.ep_start, boot.data(), N * 4);
    out.lv_cap = cap;
}

// Level-start share and weights of game gi (cdf: lv_cap entries, nullptr = uniform). Set between decisions.
template <class Mem>
void set_level_plan(Mem& mem, Built<Mem>& out, int gi, double p, const std::uint32_t* cdf) {
    GameCtx& g = out.games[static_cast<std::size_t>(gi)];
    if (g.lv_cap == 0) return;
    g.lv_thresh = static_cast<std::uint32_t>(p <= 0.0 ? 0.0 : p >= 1.0 ? 4294967295.0 : p * 4294967296.0);
    g.lv_weighted = cdf != nullptr;
    if (cdf != nullptr) mem.upload(g.lv_cdf, cdf, static_cast<std::size_t>(g.lv_cap) * 4);
}

// The state that must be saved for an exact resume: every per-env array (snapshot slot included) and every
// per-env RL state array, in a fixed order. Visits (pointer, bytes).
template <class Mem, class F>
void for_each_state(Built<Mem>& e, F&& f) {
    for (std::size_t gi = 0; gi < e.games.size(); ++gi) {
        auto& g = e.games[gi];
        for (const auto& it : e.items_host[gi]) f(it.base, static_cast<std::size_t>(it.bytes) * (g.n + 1));
        if (g.key_cap) f(g.keys, static_cast<std::size_t>(g.n) * g.key_cap * 4);
    }
    const std::size_t N = static_cast<std::size_t>(e.total);
    Common& c = e.common;
    f(c.st, N * sizeof(rl10::EnvState));
    f(c.drop, N * sizeof(rlc::Dropout));
    f(c.target, N * 8); f(c.ep_frame, N * 4); f(c.prev_applied, N * 2); f(c.sticky_seed, N * 8);
    f(c.sticky_ctr, N * 8); f(c.episode, N * 4); f(c.ep_ret, N * 4); f(c.ep_len, N * 4); f(c.ep_maxlvl, N * 4);
    f(c.ep_idle, N * 4); f(c.fault, N * 4);
    // level starts last, so a state saved without them is a prefix (rl/rlenv.py loads it with an empty archive)
    if (e.lv_cap > 0) {
        f(c.ep_start, N * 4);
        for (auto& g : e.games) {
            const std::size_t L = static_cast<std::size_t>(g.lv_cap);
            f(g.lv_mem, L * g.lv_stride);
            f(g.lv_st, L * sizeof(rl10::EnvState));
            if (g.key_cap) f(g.lv_keys, L * g.key_cap * 4);
            f(g.lv_target, L * 8); f(g.lv_frame, L * 4); f(g.lv_deaths, L * 4);
        }
    }
}

}  // namespace rle
