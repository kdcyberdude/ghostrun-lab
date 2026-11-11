// Host (CPU) entry points to the shared reward10.cuh / controller.cuh code, for tests against the Python references.
// Build: c++ -O2 -std=c++17 -ffp-contract=off -shared -fPIC rl/csrc/host_api.cpp -o rl/build/librl_host.so
#include <vector>

#include "reward10.cuh"
#include "controller.cuh"

extern "C" {

// Feed `n` RAM snapshots (2048 bytes each) of one episode through RewardTracker semantics.
// frames[i] = emulator frames since the previous snapshot. Outputs per update.
// Returns 0, or 1 if a key set overflowed (results no longer exact).
int rl10_replay(int game, const std::uint8_t* ram, int n, const int* frames, double* reward, std::uint8_t* done,
                std::uint32_t* events, int* event_level, int* deaths, std::uint8_t* end_reason) {
    rl10::EnvState e;
    rl10::reset(e);
    std::vector<std::uint32_t> slots(rl10::keyset_capacity(game), 0u);
    rl10::KeySet set{slots.data(), static_cast<std::uint32_t>(slots.size())};
    for (int i = 0; i < n; ++i) {
        const auto u = rl10::update(game, ram + static_cast<std::size_t>(i) * 2048u, frames[i], e, set);
        reward[i] = u.reward;
        done[i] = u.done ? 1 : 0;
        events[i] = u.events;
        event_level[i] = u.event_level;
        deaths[i] = e.deaths;
        end_reason[i] = e.end_reason;
    }
    return e.overflow ? 1 : 0;
}


// Pad byte of every move id on an even and an odd frame: out[2 * id + parity].
void rlc_pad_table(std::uint8_t* out) {
    for (int id = 0; id < rlc::kMoves; ++id)
        for (int f = 0; f < 2; ++f) out[2 * id + f] = rlc::pad_byte(rlc::unpack(id), static_cast<std::uint64_t>(f));
}

// Legal-move table for dropped input d in -1..7 and the Start/Select flags: out[(d + 1) * kMoves + id] = 1 if legal
// (every part of the move allowed), plus restrict_move of every move: rest[(d + 1) * kMoves + id] = packed id.
void rlc_legal_table(int start_ok, int select_ok, std::uint8_t* out, std::int32_t* rest) {
    for (int d = -1; d < rlc::kDropInputs; ++d) {
        const auto k = rlc::legal(d, start_ok != 0, select_ok != 0);
        for (int id = 0; id < rlc::kMoves; ++id) {
            const auto m = rlc::unpack(id);
            const bool ok = ((k.dpad >> m.dpad) & 1) && ((k.a >> m.a) & 1) && ((k.b >> m.b) & 1) &&
                            ((k.start >> m.start) & 1) && ((k.select >> m.select) & 1);
            out[(d + 1) * rlc::kMoves + id] = ok ? 1 : 0;
            rest[(d + 1) * rlc::kMoves + id] = rlc::pack(rlc::restrict_move(m, d, start_ok != 0, select_ok != 0));
        }
    }
}

// Run one schedule for n decisions: dropped[t] (-1 none) and the timeline the actor sees at t (8 x 45 bytes).
void rlc_schedule(std::uint64_t seed, int enabled, int n, std::int8_t* dropped, std::uint8_t* timelines) {
    rlc::Dropout s;
    rlc::dropout_reset(s, seed, enabled != 0);
    for (int t = 0; t < n; ++t) {
        dropped[t] = static_cast<std::int8_t>(rlc::dropped_now(s));
        if (timelines) rlc::timeline(s, timelines + static_cast<std::size_t>(t) * rlc::kDropInputs * rlc::kTimeline);
        rlc::dropout_advance(s, enabled != 0);
    }
}

std::uint64_t rlc_episode_seed(std::uint64_t run_seed, std::uint32_t env, std::uint64_t episode) {
    return rlc::episode_seed(run_seed, env, episode);
}

}

