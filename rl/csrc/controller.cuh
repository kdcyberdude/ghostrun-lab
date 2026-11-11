// Controller, button dropout and sticky actions, shared by the host tests and the GPU env kernel.
//
// Move = (dpad 0..8, A 0..2, B 0..2, Start 0..1, Select 0..1), chosen in that order by the policy.
//   dpad: 0 none, 1 Up, 2 Up-Right, 3 Right, 4 Down-Right, 5 Down, 6 Down-Left, 7 Left, 8 Up-Left
//         (FCEUX's opposite-direction block: Up+Down / Left+Right cannot be expressed)
//   A, B: 0 off, 1 hold, 2 turbo (FCEUX autofire 1 on / 1 off, pressed on even frame numbers)
//   Start, Select: 0 off, 1 press (held for the decision's frames); only where the game's START_SELECT allows.
// Packed move id = dpad + 9 * (a + 3 * (b + 3 * (start + 2 * select)))  (0..323).
// Pad byte (FCEUX/NES order): bit0 A, 1 B, 2 Select, 3 Start, 4 Up, 5 Down, 6 Left, 7 Right.
//
// Dropout: at most one of 8 inputs is unavailable at a time:
//   0 Up, 1 Down, 2 Left, 3 Right, 4 A-hold, 5 B-hold, 6 TurboA, 7 TurboB  (Start/Select never drop).
// Schedule per env and episode from its own counter-based generator (seed logged, independent of game, RAM and
// actions): healthy gap ~ Geometric(1/35) >= 1 decision, then one input off for 8..22 decisions, repeat.
// The actor is told the next kTimeline decisions (now + 3 s). All randomness is integer-only, so host and GPU
// produce identical schedules.
#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define RLC_HD __host__ __device__ __forceinline__
#else
#define RLC_HD inline
#endif

namespace rlc {

constexpr int kDpad = 9, kAB = 3, kSS = 2;
constexpr int kMoves = kDpad * kAB * kAB * kSS * kSS;  // 324
constexpr int kDropInputs = 8;
constexpr int kTimeline = 45;      // columns: now + next 44 decisions (3 s at 15 decisions/s)
constexpr int kRing = 64;          // ring buffer of generated columns (power of two > kTimeline)
constexpr int kDropMin = 8, kDropMax = 22;
constexpr std::uint32_t kGapP32 = 122713351u;  // floor(2^32 / 35): P(gap ends) per decision, mean gap 35
constexpr std::uint32_t kSticky32 = 0x40000000u;  // 0.25 * 2^32
enum : std::uint8_t { kBitA = 1, kBitB = 2, kBitSelect = 4, kBitStart = 8, kBitUp = 16, kBitDown = 32,
                      kBitLeft = 64, kBitRight = 128 };
enum : int { kDropUp = 0, kDropDown, kDropLeft, kDropRight, kDropAHold, kDropBHold, kDropTurboA, kDropTurboB };

// Direction bits of each dpad choice.
RLC_HD std::uint8_t dpad_bits(int d) {
    switch (d) {
        case 1: return kBitUp;
        case 2: return kBitUp | kBitRight;
        case 3: return kBitRight;
        case 4: return kBitDown | kBitRight;
        case 5: return kBitDown;
        case 6: return kBitDown | kBitLeft;
        case 7: return kBitLeft;
        case 8: return kBitUp | kBitLeft;
        default: return 0;
    }
}

RLC_HD int dpad_from_bits(std::uint8_t bits) {
    for (int d = 0; d < kDpad; ++d) if (dpad_bits(d) == bits) return d;
    return 0;
}

struct Move { std::uint8_t dpad, a, b, start, select; };

RLC_HD int pack(Move m) { return m.dpad + 9 * (m.a + 3 * (m.b + 3 * (m.start + 2 * m.select))); }
RLC_HD Move unpack(int id) {
    Move m;
    m.dpad = static_cast<std::uint8_t>(id % 9); id /= 9;
    m.a = static_cast<std::uint8_t>(id % 3); id /= 3;
    m.b = static_cast<std::uint8_t>(id % 3); id /= 3;
    m.start = static_cast<std::uint8_t>(id % 2); id /= 2;
    m.select = static_cast<std::uint8_t>(id % 2);
    return m;
}

// The pad byte a move gives on emulator frame number `frame` (turbo pressed on even frames).
RLC_HD std::uint8_t pad_byte(Move m, std::uint64_t frame) {
    const bool even = (frame & 1u) == 0;
    std::uint8_t p = dpad_bits(m.dpad);
    if (m.a == 1 || (m.a == 2 && even)) p |= kBitA;
    if (m.b == 1 || (m.b == 2 && even)) p |= kBitB;
    if (m.start) p |= kBitStart;
    if (m.select) p |= kBitSelect;
    return p;
}

// Remove what a dropped input (-1 = none) and the game's Start/Select policy forbid: a held/turbo button or a
// direction is released, the rest of the move stays (used for sticky repeats; the policy's own move is masked).
RLC_HD Move restrict_move(Move m, int dropped, bool start_ok, bool select_ok) {
    std::uint8_t bits = dpad_bits(m.dpad);
    if (dropped == kDropUp) bits &= ~kBitUp;
    if (dropped == kDropDown) bits &= ~kBitDown;
    if (dropped == kDropLeft) bits &= ~kBitLeft;
    if (dropped == kDropRight) bits &= ~kBitRight;
    m.dpad = static_cast<std::uint8_t>(dpad_from_bits(bits));
    if ((dropped == kDropAHold && m.a == 1) || (dropped == kDropTurboA && m.a == 2)) m.a = 0;
    if ((dropped == kDropBHold && m.b == 1) || (dropped == kDropTurboB && m.b == 2)) m.b = 0;
    if (!start_ok) m.start = 0;
    if (!select_ok) m.select = 0;
    return m;
}

// Legal choices for each part of the move given the dropped input: bit i set = choice i allowed.
struct Masks { std::uint16_t dpad; std::uint8_t a, b, start, select; };

RLC_HD Masks legal(int dropped, bool start_ok, bool select_ok) {
    Masks k{0x1FF, 0x7, 0x7, static_cast<std::uint8_t>(start_ok ? 3 : 1), static_cast<std::uint8_t>(select_ok ? 3 : 1)};
    std::uint8_t gone = 0;
    if (dropped == kDropUp) gone = kBitUp;
    if (dropped == kDropDown) gone = kBitDown;
    if (dropped == kDropLeft) gone = kBitLeft;
    if (dropped == kDropRight) gone = kBitRight;
    if (gone) for (int d = 0; d < kDpad; ++d) if (dpad_bits(d) & gone) k.dpad &= static_cast<std::uint16_t>(~(1u << d));
    if (dropped == kDropAHold) k.a &= ~2u;
    if (dropped == kDropTurboA) k.a &= ~4u;
    if (dropped == kDropBHold) k.b &= ~2u;
    if (dropped == kDropTurboB) k.b &= ~4u;
    return k;
}

// ---- counter-based random numbers (splitmix64 of seed and counter): integer only, host == device ----
RLC_HD std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
RLC_HD std::uint32_t rand_u32(std::uint64_t seed, std::uint64_t& counter) {
    return static_cast<std::uint32_t>(splitmix64(seed ^ splitmix64(counter++)) >> 32);
}
RLC_HD std::uint32_t rand_below(std::uint64_t seed, std::uint64_t& counter, std::uint32_t n) {
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(rand_u32(seed, counter)) * n) >> 32);
}
// Seed of one episode's schedule: a pure function of (run seed, env, episode), so it is logged as three numbers.
RLC_HD std::uint64_t episode_seed(std::uint64_t run_seed, std::uint32_t env, std::uint64_t episode) {
    return splitmix64(splitmix64(run_seed ^ 0xD20F0D20F0ull) ^ (static_cast<std::uint64_t>(env) << 40) ^ episode);
}

// ---- dropout schedule ----
struct Dropout {
    std::uint64_t seed, counter;   // the schedule's own generator
    std::int32_t gap_left;         // healthy decisions left before the next dropout (>0 while in a gap)
    std::int32_t drop_left;        // decisions left in the current dropout (>0 while dropping)
    std::int32_t drop_input;       // input of the current dropout
    std::uint32_t head;            // decision index of ring column "now"
    std::int8_t ring[kRing];       // ring[(head + j) % kRing] = input off at decision head + j, -1 none
};

RLC_HD std::int32_t draw_gap(Dropout& s) {  // Geometric(1/35) on {1, 2, ...} by Bernoulli trials (exact)
    std::int32_t k = 1;
    while (rand_u32(s.seed, s.counter) >= kGapP32) ++k;
    return k;
}

RLC_HD std::int8_t next_column(Dropout& s) {
    if (s.drop_left > 0) {
        const std::int8_t out = static_cast<std::int8_t>(s.drop_input);
        if (--s.drop_left == 0) s.gap_left = draw_gap(s);
        return out;
    }
    if (--s.gap_left == 0) {
        s.drop_input = static_cast<std::int32_t>(rand_below(s.seed, s.counter, kDropInputs));
        s.drop_left = kDropMin + static_cast<std::int32_t>(rand_below(s.seed, s.counter, kDropMax - kDropMin + 1));
    }
    return -1;
}

// enabled = false: no dropout at all (all columns -1), e.g. an evaluation without dropout.
RLC_HD void dropout_reset(Dropout& s, std::uint64_t seed, bool enabled) {
    s.seed = seed; s.counter = 0; s.drop_left = 0; s.drop_input = -1; s.head = 0;
    s.gap_left = enabled ? draw_gap(s) : 0x7FFFFFFF;
    for (int j = 0; j < kRing; ++j) s.ring[j] = -1;
    for (int j = 0; j < kTimeline; ++j) s.ring[j] = enabled ? next_column(s) : -1;
}

RLC_HD int dropped_now(const Dropout& s) { return s.ring[s.head % kRing]; }

// After the decision at `head` is taken: move "now" one column on and generate the new last column.
RLC_HD void dropout_advance(Dropout& s, bool enabled) {
    s.ring[s.head % kRing] = -1;
    ++s.head;
    s.ring[(s.head + kTimeline - 1) % kRing] = enabled ? next_column(s) : -1;
}

// Timeline the actor sees: out[i * kTimeline + j] = 1 if input i is off at decision now + j.
RLC_HD void timeline(const Dropout& s, std::uint8_t* out) {
    for (int i = 0; i < kDropInputs * kTimeline; ++i) out[i] = 0;
    for (int j = 0; j < kTimeline; ++j) {
        const int b = s.ring[(s.head + j) % kRing];
        if (b >= 0) out[b * kTimeline + j] = 1;
    }
}

}  // namespace rlc
