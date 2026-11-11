// The 12-game training environment, one env per thread: per decision, 4 emulated frames with the move's per-frame
// pad bytes (turbo, sticky repeats, button dropout), each frame's picture, the reward10 update, and an automatic
// restart from the game's boot snapshot when the episode ends. Host + device: the GPU kernel (rlenv.cu) and the CPU
// reference build (rlenv_cpu.cpp) run exactly this code, which is how the GPU result is checked.
//
// Conventions (DECISIONS.md 2026-11-10):
//  * a decision = 4 frames; the observation is those 4 frames (t-3..t), colour numbers + per-line emphasis;
//  * sticky actions (training only): on every frame, with probability 1/4 the previous frame's move is repeated
//    instead of the new one (restricted by the current dropout, so a dropped button stays released);
//  * episode start: boot snapshot, then 0..idle_max random idle frames (training only), then 4 neutral frames whose
//    pictures are the first observation; the reward tracker's first update sees the state after them (no reward);
//  * all reward10 ends (game over, won, skip, stall) are terminal; an emulator fault also ends the episode
//    (end reason 5) and is counted;
//  * after a done, the outputs hold the finished episode's reward/done/info and the NEW episode's first observation;
//  * level starts (optional, DECISIONS.md 2026-11-11): when an env reaches a new level (reward10 'level' event, all
//    games) without its episode ending, its whole state (console, reward tracker, key set) is saved as that level's
//    start; per level the arrival with the fewest deaths so far is kept (the newest on ties). With probability
//    lv_thresh / 2^32 a new episode starts from a saved level start (by the game's level weights, rl/level_plan.py;
//    uniform over the saved levels until weights are set) instead of power-on; dropout / sticky / idle frames are drawn fresh as for any episode. Which arrival is saved does not
//    depend on thread order (atomic minimum of (deaths, env index)), so the GPU and the CPU reference agree.
#pragma once
#include <cstdint>

#include "controller.cuh"
#include "nes_machine.cuh"
#include "reward10.cuh"

namespace rle {

using rlm::kObsPixels;
using rlm::kObsH;

constexpr int kFramesPerDecision = 4;
constexpr int kMaskBytes = 9 + 3 + 3 + 2 + 2;  // legal choices per part: d-pad, A, B, Start, Select
constexpr int kTimelineBytes = rlc::kDropInputs * rlc::kTimeline;
constexpr int kInfo = 8;  // per finished episode: return, length, end reason, deaths, max level, overflow, idle, episode
constexpr int kEndFault = 5;
constexpr int kEndLevelClear = 6;

struct CopyItem {
    std::uint8_t* base;     // array for n + 1 envs; slot n is the boot snapshot
    std::uint32_t bytes;    // per env
    std::uint8_t* idle;     // idle_tmpl snapshots: the boot snapshot after k = 1..idle_tmpl idle frames (slot k - 1)
};

struct GameCtx {
    nesle::cuda::BatchBuffers b;
    int game;            // rl10 game id (rl10::kGame_*)
    int bank_mode;       // 0 NROM, 1 banked, 2 paged
    int n, offset;       // envs of this game and their first global index
    int start_ok, select_ok;
    std::uint32_t* keys;   // n * key_cap
    std::uint32_t key_cap;
    const CopyItem* items;
    int nitems;
    std::uint64_t tmpl_target;  // vblank target of the snapshot's next frame
    int idle_tmpl;              // idle-frame snapshots made at build (a restart copies one instead of emulating)
    int e0, e1;                 // GPU launches cover envs [e0, e1) of this game (a sub-range lets two halves of
                                // one game overlap: one half emulates while the other's moves are inferred)
    // level starts (lv_cap = 0: off). Slot s < lv_levels = the game's level index s (reward10 Signals.level), saved
    // on the level event; slots lv_levels..lv_cap-1 are places (rl/places.py), saved on request (rlenv_place_save).
    int lv_cap;
    int lv_levels;
    std::uint32_t lv_stride;            // bytes of one saved console: every item, each rounded up to 16 bytes
    std::uint8_t* lv_mem;               // [lv_cap, lv_stride]
    rl10::EnvState* lv_st;              // [lv_cap] reward tracker
    std::uint32_t* lv_keys;             // [lv_cap, key_cap]
    std::uint64_t* lv_target;           // [lv_cap]
    std::uint32_t* lv_frame;            // [lv_cap]
    std::int32_t* lv_deaths;            // [lv_cap] deaths of the saved arrival, -1 = no save yet
    unsigned long long* lv_claim;       // [lv_cap] (deaths << 32 | env) of this decision's best arrival, ~0 = none
    std::uint32_t lv_thresh;            // P(start from a saved level start) * 2^32 (0 = power-on only)
    int lv_weighted;                    // 1: pick by lv_cdf; 0: uniform over the saved levels
    std::uint32_t* lv_cdf;              // [lv_cap] cumulative level weights * 2^32 (only saved levels have weight)
};

struct Common {
    // per global env state
    rl10::EnvState* st;
    rlc::Dropout* drop;
    std::uint64_t* target;
    std::uint32_t* ep_frame;
    std::int16_t* prev_applied;
    std::uint64_t* sticky_seed;
    std::uint64_t* sticky_ctr;
    std::uint32_t* episode;
    float* ep_ret;
    std::int32_t* ep_len;
    std::int32_t* ep_maxlvl;
    std::int32_t* ep_idle;
    std::uint32_t* fault;
    std::uint8_t* resetting;   // 1 while an env runs its restart frames (always 0 between decisions)
    std::uint8_t* mmc2_pending;  // 1: the frame just emulated froze a log; its MMC2 latches are resolved next (0 between decisions)
    // input: the policy's move per env (packed id 0..323, legal under the masks given with the last observation)
    const std::int16_t* actions;
    // outputs
    std::uint8_t* obs;       // [N, 4, 224, 240] colour numbers
    std::uint8_t* emph;      // [N, 4, 224]
    float* reward;           // [N]
    std::uint8_t* done;      // [N]
    std::uint8_t* masks;     // [N, 19]
    std::uint8_t* timeline;  // [N, 8, 45]
    std::uint8_t* pads;      // [N, 4] pad byte of each frame of this decision
    std::uint32_t* events;   // [N] reward10 event bits
    float* info;             // [N, 8] valid where done
    float* parts = nullptr;  // [N, rl10::kNumParts] the reward split by source (optional; not an actor input)
    std::uint64_t run_seed;
    int sticky;      // 1: sticky repeats on
    int idle_max;    // random idle frames at episode start: 0..idle_max
    int dropout;     // 1: button dropout schedule on
    int no_render;   // benchmark only: 1 = skip drawing pictures
    int episode_level_end = -1;  // optional curriculum: end after this level's native transition
    std::int32_t* ep_start = nullptr;  // [N] level the current episode started from, -1 = power-on (with lv_cap)
    std::int32_t* start_out = nullptr; // [N] output: ep_start of the episode that just finished (valid where done)
    std::int32_t* level_out = nullptr; // [N] output: level this env is on after the decision's update, -1 = not yet known
    float* progress_out = nullptr;     // [N] output: current within-level progress / PROGRESS_UNIT after the decision, NaN = not valid now
};

RLM_HD inline void atomic_min_u64(unsigned long long* p, unsigned long long v) {
#ifdef __CUDA_ARCH__
    atomicMin(p, v);
#else
    unsigned long long cur = __atomic_load_n(p, __ATOMIC_RELAXED);
    while (v < cur && !__atomic_compare_exchange_n(p, &cur, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
    }
#endif
}

RLM_HD inline void copy_bytes(std::uint8_t* dst, const std::uint8_t* src, std::uint32_t bytes, int tid, int nt) {
    if ((bytes & 3u) == 0) {
        for (std::uint32_t k = tid; k < bytes / 4u; k += nt)
            reinterpret_cast<std::uint32_t*>(dst)[k] = reinterpret_cast<const std::uint32_t*>(src)[k];
    } else {
        for (std::uint32_t k = tid; k < bytes; k += nt) dst[k] = src[k];
    }
}

RLM_HD inline void write_controls(const GameCtx& g, Common& c, int ge) {
    const int d = rlc::dropped_now(c.drop[ge]);
    const rlc::Masks k = rlc::legal(d, g.start_ok != 0, g.select_ok != 0);
    std::uint8_t* m = c.masks + static_cast<std::uint64_t>(ge) * kMaskBytes;
    for (int i = 0; i < 9; ++i) m[i] = (k.dpad >> i) & 1u;
    for (int i = 0; i < 3; ++i) m[9 + i] = (k.a >> i) & 1u;
    for (int i = 0; i < 3; ++i) m[12 + i] = (k.b >> i) & 1u;
    for (int i = 0; i < 2; ++i) m[15 + i] = (k.start >> i) & 1u;
    for (int i = 0; i < 2; ++i) m[17 + i] = (k.select >> i) & 1u;
    rlc::timeline(c.drop[ge], c.timeline + static_cast<std::uint64_t>(ge) * kTimelineBytes);
}

// ---------------------------------------------------------------------------------------------------------------
// One decision is a fixed sequence of per-env PHASES. The GPU runs each phase as its own small kernel (the emulation
// loop alone keeps NeSLE's register budget; one fused kernel spilled ~2 KB per thread inside the instruction loop and
// ran 2-5x slower per env). The CPU reference calls the same phases in the same order, so results are identical.
//   decision:  frame(0) render(0) frame(1) render(1) frame(2) render(2) frame(3) render(3) finish
//   restart (only envs whose episode just ended; also every env at start):  reset_idle,
//              reset_frame(0) render(0) .. reset_frame(3) render(3), reset_finish
//   controls (masks + dropout timeline for the next decision)
// ---------------------------------------------------------------------------------------------------------------

template <nesle::cuda::BankMode Mode>
RLM_HD inline void emu_frame(GameCtx& g, Common& c, int e, int ge, std::uint8_t pad, bool defer_mmc2) {
    g.b.action_masks[e] = pad;
    if (c.fault[ge] == 0) {
        const std::uint8_t bank0 = g.b.ppu.log_bank[e];
        const std::uint32_t f = rlm::step_until<Mode>(g.b, static_cast<std::uint32_t>(e), c.target[ge]);
        if (f != 0) c.fault[ge] = f;
        // MMC2 (g.b.defer_mmc2_resolve = 1): vblank froze this frame's log but left its latches unresolved. Resolve
        // them now, or (defer_mmc2) in the render kernel that follows, before any further instruction runs.
        if (g.b.defer_mmc2_resolve != 0 && g.b.cart.bank_kind == nesle::cuda::kBankingMmc2 &&
            g.b.mapper.regs != nullptr && g.b.ppu.log_bank[e] != bank0) {
            if (defer_mmc2) c.mmc2_pending[ge] = 1;
            else rlm::resolve_mmc2_frozen(g.b, static_cast<std::uint32_t>(e));
        }
    }
    c.target[ge] += rlm::kFrameDots;
    ++c.ep_frame[ge];
}

// Decision frame f: the move's pad byte for this frame (sticky repeat, dropout, turbo phase), then one frame.
template <nesle::cuda::BankMode Mode>
RLM_HD inline void ph_frame(GameCtx& g, Common& c, int e, int f) {
    const int ge = g.offset + e;
    rlc::Move m = rlc::unpack(c.actions[ge]);
    if (c.sticky != 0 && rlc::rand_u32(c.sticky_seed[ge], c.sticky_ctr[ge]) < rlc::kSticky32)
        m = rlc::restrict_move(rlc::unpack(c.prev_applied[ge]), rlc::dropped_now(c.drop[ge]), g.start_ok != 0,
                               g.select_ok != 0);
    c.prev_applied[ge] = static_cast<std::int16_t>(rlc::pack(m));
    const std::uint8_t pad = rlc::pad_byte(m, c.ep_frame[ge]);
    c.pads[static_cast<std::uint64_t>(ge) * kFramesPerDecision + f] = pad;
    emu_frame<Mode>(g, c, e, ge, pad, true);
}

// Picture of the frame just emulated into observation slot f (restart pass: only envs that are restarting).
RLM_HD inline void ph_render(GameCtx& g, Common& c, int e, int f, bool restart_pass) {
    const int ge = g.offset + e;
    if (restart_pass && c.resetting[ge] == 0) return;
    if (c.mmc2_pending[ge] != 0) {
        rlm::resolve_mmc2_frozen(g.b, static_cast<std::uint32_t>(e));
        c.mmc2_pending[ge] = 0;
    }
    if (c.no_render != 0) return;
    rlm::render_obs(g.b, static_cast<std::uint32_t>(e),
                    c.obs + (static_cast<std::uint64_t>(ge) * kFramesPerDecision + f) * kObsPixels,
                    c.emph + (static_cast<std::uint64_t>(ge) * kFramesPerDecision + f) * kObsH);
}

// Restart state: boot snapshot, fresh reward tracker / key set / dropout schedule / sticky stream, idle frame count.
RLM_HD inline void reset_state(GameCtx& g, Common& c, int e) {
    const int ge = g.offset + e;
    // the idle frame count first (it only depends on the episode's seed), so the matching snapshot can be copied:
    // the state after k idle frames (pad 0) from the boot snapshot is the same for every env, made once at build
    const std::uint32_t ep = ++c.episode[ge];
    const std::uint64_t seed = rlc::episode_seed(c.run_seed, static_cast<std::uint32_t>(ge), ep);
    c.sticky_seed[ge] = rlc::splitmix64(seed ^ 0x5717C4A55EEDull);
    c.sticky_ctr[ge] = 0;
    int idle = 0;
    if (c.idle_max > 0)
        idle = static_cast<int>(rlc::rand_below(c.sticky_seed[ge], c.sticky_ctr[ge],
                                                static_cast<std::uint32_t>(c.idle_max + 1)));
    // level start? (random numbers drawn only when level starts are on, so power-on-only runs are unchanged)
    int lv = -1;
    if (g.lv_cap > 0 && g.lv_thresh != 0 && rlc::rand_u32(c.sticky_seed[ge], c.sticky_ctr[ge]) < g.lv_thresh) {
        std::uint32_t saved = 0;
        for (int s = 0; s < g.lv_cap; ++s) saved += g.lv_deaths[s] >= 0 ? 1u : 0u;
        if (g.lv_weighted != 0) {
            const std::uint32_t u = rlc::rand_u32(c.sticky_seed[ge], c.sticky_ctr[ge]);
            for (int s = 0; s < g.lv_cap && lv < 0; ++s)
                if (u < g.lv_cdf[s]) lv = s;
        } else if (saved > 0) {
            std::uint32_t pick = rlc::rand_below(c.sticky_seed[ge], c.sticky_ctr[ge], saved);
            for (int s = 0; s < g.lv_cap && lv < 0; ++s)
                if (g.lv_deaths[s] >= 0 && pick-- == 0) lv = s;
        }
    }
    const int k = lv < 0 && idle <= g.idle_tmpl ? idle : 0;   // snapshot used; frames beyond it are emulated
    std::uint32_t off = 0;
    for (int i = 0; i < g.nitems; ++i) {
        const CopyItem it = g.items[i];
        const std::uint8_t* src = lv >= 0 ? g.lv_mem + static_cast<std::uint64_t>(lv) * g.lv_stride + off
                                : k == 0 ? it.base + static_cast<std::uint64_t>(g.n) * it.bytes
                                         : it.idle + static_cast<std::uint64_t>(k - 1) * it.bytes;
        copy_bytes(it.base + static_cast<std::uint64_t>(e) * it.bytes, src, it.bytes, 0, 1);
        off += (it.bytes + 15u) & ~15u;
    }
    std::uint32_t* keys = g.keys + static_cast<std::uint64_t>(e) * g.key_cap;
    if (lv >= 0) {
        c.target[ge] = g.lv_target[lv];
        c.ep_frame[ge] = g.lv_frame[lv];
        c.st[ge] = g.lv_st[lv];
        for (std::uint32_t q = 0; q < g.key_cap; ++q) keys[q] = g.lv_keys[static_cast<std::uint64_t>(lv) * g.key_cap + q];
    } else {
        c.target[ge] = g.tmpl_target + static_cast<std::uint64_t>(k) * rlm::kFrameDots;
        c.ep_frame[ge] = static_cast<std::uint32_t>(k);
        rl10::reset(c.st[ge]);
        for (std::uint32_t q = 0; q < g.key_cap; ++q) keys[q] = 0;
    }
    if (c.ep_start != nullptr) c.ep_start[ge] = lv;
    c.fault[ge] = 0;
    rlc::dropout_reset(c.drop[ge], seed, c.dropout != 0);
    c.prev_applied[ge] = 0;
    c.ep_idle[ge] = idle;
    c.resetting[ge] = 1;
}

// After the 4 decision frames: reward10 update, outputs, dropout schedule moves on; a finished episode restarts.
RLM_HD inline void ph_finish(GameCtx& g, Common& c, int e) {
    const int ge = g.offset + e;
    std::uint32_t* keys = g.keys + static_cast<std::uint64_t>(e) * g.key_cap;
    const rl10::KeySet set{keys, g.key_cap};
    const rl10::Update u = rl10::update(
        g.game, g.b.cpu.ram + static_cast<std::uint64_t>(e) * nesle::cuda::kCpuRamBytes, kFramesPerDecision,
        c.st[ge], set);
    const bool faulted = c.fault[ge] != 0;
    const bool level_clear = !u.done && !faulted && c.episode_level_end >= 0 &&
        (u.events & rl10::kEvLevel) != 0 && u.event_level == c.episode_level_end + 1;
    const bool done = u.done || faulted || level_clear;
    const float r = faulted ? 0.0f : static_cast<float>(u.reward);
    c.reward[ge] = r;
    if (c.parts != nullptr)
        for (int k = 0; k < rl10::kNumParts; ++k)
            c.parts[static_cast<std::uint64_t>(ge) * rl10::kNumParts + k] = faulted ? 0.0f : static_cast<float>(u.part[k]);
    c.done[ge] = done ? 1 : 0;
    c.events[ge] = u.events;
    c.ep_ret[ge] += r;
    ++c.ep_len[ge];
    if (u.event_level > c.ep_maxlvl[ge]) c.ep_maxlvl[ge] = u.event_level;
    rlc::dropout_advance(c.drop[ge], c.dropout != 0);
    if (g.lv_cap > 0 && !done && (u.events & rl10::kEvLevel) != 0 && u.event_level >= 0 && u.event_level < g.lv_levels) {
        const int d = c.st[ge].deaths < 0 ? 0 : c.st[ge].deaths;
        atomic_min_u64(&g.lv_claim[u.event_level], (static_cast<unsigned long long>(d) << 32) | static_cast<unsigned>(e));
    }
    if (c.level_out != nullptr) c.level_out[ge] = c.st[ge].has_best_level ? c.st[ge].best_level : -1;
    if (c.progress_out != nullptr)
        c.progress_out[ge] = c.st[ge].has_prev && c.st[ge].prev.progress_valid
            ? static_cast<float>(c.st[ge].prev.progress / rl10::RL10_ARR(kGameConsts)[g.game].progress_unit) : NAN;
    if (done) {
        if (c.start_out != nullptr) c.start_out[ge] = c.ep_start != nullptr ? c.ep_start[ge] : -1;
        float* inf = c.info + static_cast<std::uint64_t>(ge) * kInfo;
        inf[0] = c.ep_ret[ge];
        inf[1] = static_cast<float>(c.ep_len[ge]);
        inf[2] = static_cast<float>(faulted ? kEndFault : level_clear ? kEndLevelClear : c.st[ge].end_reason);
        inf[3] = static_cast<float>(c.st[ge].deaths);
        inf[4] = static_cast<float>(c.ep_maxlvl[ge]);
        inf[5] = c.st[ge].overflow ? 1.0f : 0.0f;
        inf[6] = static_cast<float>(c.ep_idle[ge]);
        inf[7] = static_cast<float>(c.episode[ge]);
        reset_state(g, c, e);
    }
}

template <nesle::cuda::BankMode Mode>
RLM_HD inline void ph_reset_idle(GameCtx& g, Common& c, int e) {
    const int ge = g.offset + e;
    if (c.resetting[ge] == 0) return;
    const bool lv = c.ep_start != nullptr && c.ep_start[ge] >= 0;   // a level start: every idle frame is emulated
    const int done = !lv && c.ep_idle[ge] <= g.idle_tmpl ? c.ep_idle[ge] : 0;   // frames already in the copied snapshot
    for (int i = done; i < c.ep_idle[ge]; ++i) emu_frame<Mode>(g, c, e, ge, 0, false);
}

template <nesle::cuda::BankMode Mode>
RLM_HD inline void ph_reset_frame(GameCtx& g, Common& c, int e) {
    const int ge = g.offset + e;
    if (c.resetting[ge] == 0) return;
    emu_frame<Mode>(g, c, e, ge, 0, true);
}

// First tracker update after the restart frames (no reward), episode counters, controls for the next decision.
RLM_HD inline void ph_reset_finish_controls(GameCtx& g, Common& c, int e) {
    const int ge = g.offset + e;
    if (c.resetting[ge] != 0) {
        std::uint32_t* keys = g.keys + static_cast<std::uint64_t>(e) * g.key_cap;
        const rl10::KeySet set{keys, g.key_cap};
        (void)rl10::update(g.game, g.b.cpu.ram + static_cast<std::uint64_t>(e) * nesle::cuda::kCpuRamBytes,
                           c.ep_idle[ge] + kFramesPerDecision, c.st[ge], set);
        c.ep_ret[ge] = 0.0f;
        c.ep_len[ge] = 0;
        const int sl = c.ep_start != nullptr ? c.ep_start[ge] : -1;   // a place: its level is the tracker's
        c.ep_maxlvl[ge] = sl >= g.lv_levels ? (c.st[ge].has_best_level ? c.st[ge].best_level : 0) : sl > 0 ? sl : 0;
        c.resetting[ge] = 0;
    }
    write_controls(g, c, ge);
}

// Level-start save of slot s after a decision's ph_finish (GPU: one block per slot; CPU: tid 0, nt 1). take: the
// claimed arrival replaces the saved one (none saved yet, or no more deaths). Copy first (all threads), then commit.
RLM_HD inline bool archive_take(const GameCtx& g, int s) {
    const unsigned long long cl = g.lv_claim[s];
    return cl != ~0ull && (g.lv_deaths[s] < 0 || static_cast<std::int32_t>(cl >> 32) <= g.lv_deaths[s]);
}

RLM_HD inline void archive_copy_env(const GameCtx& g, int s, int e, int tid, int nt) {
    std::uint8_t* dst = g.lv_mem + static_cast<std::uint64_t>(s) * g.lv_stride;
    for (int i = 0; i < g.nitems; ++i) {
        const CopyItem it = g.items[i];
        copy_bytes(dst, it.base + static_cast<std::uint64_t>(e) * it.bytes, it.bytes, tid, nt);
        dst += (it.bytes + 15u) & ~15u;
    }
    const std::uint32_t* keys = g.keys + static_cast<std::uint64_t>(e) * g.key_cap;
    for (std::uint32_t q = tid; q < g.key_cap; q += nt) g.lv_keys[static_cast<std::uint64_t>(s) * g.key_cap + q] = keys[q];
}

RLM_HD inline void archive_copy(const GameCtx& g, int s, int tid, int nt) {
    archive_copy_env(g, s, static_cast<int>(g.lv_claim[s] & 0xFFFFFFFFull), tid, nt);
}

// Place save (rl/places.py): env e's whole state into slot s (>= lv_levels), after archive_copy_env. The no-progress
// clock restarts, so a place reached after a long wait does not end at once when it is replayed.
RLM_HD inline void place_commit(const GameCtx& g, const Common& c, int s, int e) {
    const int ge = g.offset + e;
    g.lv_st[s] = c.st[ge];
    g.lv_st[s].idle_frames = 0;
    g.lv_target[s] = c.target[ge];
    g.lv_frame[s] = c.ep_frame[ge];
    g.lv_deaths[s] = 0;
}

RLM_HD inline void archive_commit(const GameCtx& g, const Common& c, int s, bool take) {
    const unsigned long long cl = g.lv_claim[s];
    if (take) {
        const int ge = g.offset + static_cast<int>(cl & 0xFFFFFFFFull);
        g.lv_st[s] = c.st[ge];
        g.lv_target[s] = c.target[ge];
        g.lv_frame[s] = c.ep_frame[ge];
        g.lv_deaths[s] = static_cast<std::int32_t>(cl >> 32);
    }
    g.lv_claim[s] = ~0ull;
}

// Whole decision / start for one env in phase order (CPU reference; the GPU launches the phases as kernels).
template <nesle::cuda::BankMode Mode>
RLM_HD inline void restart_phases(GameCtx& g, Common& c, int e) {
    ph_reset_idle<Mode>(g, c, e);
    for (int f = 0; f < kFramesPerDecision; ++f) {
        ph_reset_frame<Mode>(g, c, e);
        ph_render(g, c, e, f, true);
    }
    ph_reset_finish_controls(g, c, e);
}

template <nesle::cuda::BankMode Mode>
RLM_HD inline void env_start(GameCtx& g, Common& c, int e) {
    reset_state(g, c, e);
    restart_phases<Mode>(g, c, e);
}

template <nesle::cuda::BankMode Mode>
RLM_HD inline void env_step(GameCtx& g, Common& c, int e) {
    for (int f = 0; f < kFramesPerDecision; ++f) {
        ph_frame<Mode>(g, c, e, f);
        ph_render(g, c, e, f, false);
    }
    ph_finish(g, c, e);
    restart_phases<Mode>(g, c, e);
}

}  // namespace rle
