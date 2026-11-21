// reward10 on the GPU: a line-by-line port of scripts/reward10/reward.py (RewardTracker) and the 12 game readers
// (scripts/reward10/games/*.py). Host and device share this code, so the CPU replay test checks the same source the
// GPU runs. It must match Python bit for bit on the same RAM:
//   * all arithmetic is double, in Python's evaluation order;
//   * compile with -ffp-contract=off (host) and -fmad=false (nvcc): Python never fuses a*b+c;
//   * Kage route lengths come from Python (reward10_consts.h); hypot is a correctly rounded double hypot.
// RAM is reward/debug data only; nothing here may feed the actor.
#pragma once
#include <cstdint>
#include <cmath>

#include "reward10_consts.h"

#if defined(__CUDACC__)
#define RL10_HD __host__ __device__ __forceinline__
#else
#define RL10_HD inline
#endif

namespace rl10 {

// ---- events (bit flags per update) and end reasons ----
enum : std::uint32_t { kEvSkip = 1u, kEvLevel = 2u, kEvDeath = 4u, kEvWon = 8u, kEvGameOver = 16u, kEvStall = 32u,
                      kEvRollback = 64u };
enum : std::uint8_t { kEndNone = 0, kEndGameOver = 1, kEndWon = 2, kEndSkip = 3, kEndStall = 4 };

// ---- small per-env hash set of uint32 keys (Castlevania cells, Zelda rooms/cells) ----
// Open addressing, linear probing; slot value 0 = empty, so keys are stored as key + 1.
struct KeySet {
    std::uint32_t* slots;  // capacity entries, owned by the caller (per env)
    std::uint32_t capacity;  // power of two; 0 = this game uses no set
};

RL10_HD std::uint32_t mix32(std::uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
    return x;
}

// Returns 1 if inserted, 0 if already present, -1 if the set is too full (the caller flags an overflow).
RL10_HD int keyset_insert(KeySet s, std::uint32_t& count, std::uint32_t key) {
    const std::uint32_t stored = key + 1u;
    if (s.capacity == 0) return -1;
    std::uint32_t i = mix32(key) & (s.capacity - 1u);
    for (std::uint32_t n = 0; n < s.capacity; ++n) {
        const std::uint32_t v = s.slots[i];
        if (v == stored) return 0;
        if (v == 0u) {
            if ((count + 1u) * 4u > s.capacity * 3u) return -1;  // keep load <= 75%
            s.slots[i] = stored;
            ++count;
            return 1;
        }
        i = (i + 1u) & (s.capacity - 1u);
    }
    return -1;
}

RL10_HD void keyset_clear(KeySet s) {
    for (std::uint32_t i = 0; i < s.capacity; ++i) s.slots[i] = 0u;
}

// ---- Signals (reward.Signals), with explicit "has" flags for Python's None ----
struct Signals {
    bool in_play;
    int level;
    double progress;
    bool progress_valid;
    bool dying;
    bool has_lives; int lives;
    bool has_health; double health;
    bool has_score; long long score;
    bool game_over;
    bool won;
};

// ---- per-env state: tracker + reader (all games share one layout; unused fields stay zero) ----
struct EnvState {
    // RewardTracker
    bool has_prev;
    Signals prev;
    bool has_best_level; int best_level;
    bool has_best_progress; double best_progress;
    int deaths;
    long long idle_frames;
    double score_used;
    bool done;
    std::uint8_t end_reason;
    bool overflow;  // a KeySet filled up: the reward of this env is no longer exact (reported, never silent)
    // readers
    bool has_stage; int stage;           // castlevania, kage
    bool has_floor; int floor_y;         // castlevania
    std::uint32_t cells;                 // castlevania cells, zelda cells
    std::uint32_t rooms;                 // zelda rooms
    bool has_base; int base_items, base_containers;  // zelda
    bool has_key; int key_a; int key_b; double key_d;  // excitebike (level, $4F), contra (level, screen), final (level, progress),
                                                       // smb maze guide (has, level, -, last potential): keeps the saved layout
    bool has_cam; int cam; double dist;  // excitebike
    int core_max;                        // contra, final mission (boss_max)
    double best_arc;                     // kage
    bool has_own; int own; int opp;      // tecmo bowl
    bool has_drive; double drive_start, drive_best;  // tecmo bowl
    bool has_week; int week;             // tecmo bowl
    bool has_fight; int fight; int damage; bool has_prev_hp; int prev_hp;  // punch-out
};

RL10_HD int u8(const std::uint8_t* r, int a) { return static_cast<int>(r[a]); }

RL10_HD long long bcd(const std::uint8_t* r, const int* addrs, int n) {
    long long v = 0;
    for (int i = 0; i < n; ++i) { const int x = r[addrs[i]]; v = v * 100 + (x >> 4) * 10 + (x & 15); }
    return v;
}

RL10_HD long long digits(const std::uint8_t* r, int first, int last_excl) {
    long long v = 0;
    for (int a = first; a < last_excl; ++a) v = v * 10 + r[a];
    return v;
}

RL10_HD int popcount8(int x) { int c = 0; while (x) { c += x & 1; x >>= 1; } return c; }

// Correctly rounded sqrt(x*x + y*y) (Python 3.12 math.hypot is correctly rounded for these inputs):
// exact square sum as a double-double, then one Newton correction of the square root.
RL10_HD double hypot_cr(double x, double y) {
    x = fabs(x); y = fabs(y);
    if (x < y) { const double t = x; x = y; y = t; }
    if (y == 0.0) return x;
    const double xx = x * x, yy = y * y;
    const double xx_lo = fma(x, x, -xx), yy_lo = fma(y, y, -yy);
    const double hi = xx + yy;
    const double bv = hi - xx;
    const double lo = (xx - (hi - bv)) + (yy - bv) + xx_lo + yy_lo;  // two-sum error + product errors
    const double s = sqrt(hi);
    const double r = fma(-s, s, hi) + lo;  // residual (hi + lo) - s*s
    return s + r / (2.0 * s);
}

// ---------------------------------------------------------------- readers
RL10_HD Signals blank() {
    Signals s{};
    return s;
}

RL10_HD Signals observe_smb(const std::uint8_t* r) {
    const int mode = u8(r, 0x770), task = u8(r, 0x772), routine = u8(r, 0x0E);
    const int lives = u8(r, 0x75A);
    Signals s = blank();
    s.in_play = mode == 1 && task == 3;
    s.level = u8(r, 0x75F) * 4 + u8(r, 0x75C);
    const int page = u8(r, 0x6D);   // signed: a maze castle's wrong path (4-4) takes it below 0 (255 = -1)
    const int x = (page > 127 ? page - 256 : page) * 256 + u8(r, 0x86);
    int prog = x;   // 8-4 by part (smb.py SMB84_*): castle, water part, castle after the water
    if (s.level == 31) {
        if (u8(r, 0x74E) == 0) prog = kSmb84WaterBase + x;
        else if (x >= kSmb84WaterBase) prog = x + kSmb84FinalShift;
    }
    s.progress = static_cast<double>(prog);
    s.progress_valid = mode == 1 && task == 3 && routine == 8;
    s.dying = mode == 1 && (routine == 6 || routine == 0xB || (routine == 8 && u8(r, 0xB5) >= 2));
    s.has_lives = true; s.lives = lives > 127 ? lives - 256 : lives;
    s.has_score = true; s.score = digits(r, 0x7DD, 0x7E3) * 10;
    s.game_over = mode == 3;
    s.won = mode == 2 && u8(r, 0x75F) == 7;
    return s;
}

// SMB maze guide potential (scripts/reward10/games/smb.py maze_guide, same arithmetic order); has = false outside
// 4-4/7-4/8-4 or when Mario's position is not valid.
RL10_HD double smb_guide(const std::uint8_t* r, bool& has) {
    has = false;
    const int mode = u8(r, 0x770), task = u8(r, 0x772), routine = u8(r, 0x0E);
    if (!(mode == 1 && task == 3 && routine == 8)) return 0.0;
    const int level = u8(r, 0x75F) * 4 + u8(r, 0x75C);
    int li = -1;
    for (int i = 0; i < kSmbGuideNumLevels; ++i) if (RL10_ARR(kSmbGuideLevels)[i].level == level) li = i;
    if (li < 0) return 0.0;
    has = true;
    const SmbGuideLevel L = RL10_ARR(kSmbGuideLevels)[li];
    const int n = L.count;
    if ((u8(r, 0x750) & 0x7F) != L.area || u8(r, 0x74E) != 3)
        return kSmbGuideW * static_cast<double>(n);  // other area/type; bit 7 is not part of the area identity
    const int page = u8(r, 0x6D);
    const int x = (page > 127 ? page - 256 : page) * 256 + u8(r, 0x86);
    const int y = u8(r, 0xCE);
    const bool native = level == 15;
    int k = native ? 0 : -1;
    for (int i = 0; i < n; ++i) {
        const SmbGuideSeg S = RL10_ARR(kSmbGuideSegs)[L.first + i];
        if (native) {
            const int check = S.tx / 256 + 1;
            const int page = u8(r, 0x725);
            if (page > check || (page == check && u8(r, 0x745) == 0)) ++k;
        } else if (x >= S.start) k = i;
    }
    if (k == n) return kSmbGuideW * static_cast<double>(n);
    if (k < 0) return 0.0;
    const SmbGuideSeg S = RL10_ARR(kSmbGuideSegs)[L.first + k];
    const int end = k + 1 < n ? RL10_ARR(kSmbGuideSegs)[L.first + k + 1].start : (S.pipe ? S.start + 1024 : S.tx);
    const int wrong = native ? 0 : u8(r, 0x6DA) - u8(r, 0x6D9);   // 7-4 group checks passed at the wrong height
    if (!native && x >= end) return kSmbGuideW * static_cast<double>(k + 1 - wrong);
    double reach;
    if (native) {
        reach = static_cast<double>(x - S.start) / static_cast<double>(S.tx - S.start);
        if (reach < 0.0) reach = 0.0;
        if (reach > 1.0) reach = 1.0;
    } else if (x <= S.tx) {
        reach = static_cast<double>(x - S.start) / static_cast<double>(S.tx - S.start);
    } else {
        reach = 1.0 - static_cast<double>(x - S.tx) / kSmbGuidePipePast;
        if (reach < 0.0) reach = 0.0;
    }
    double want = static_cast<double>(S.ty);
    if (S.vx > 0) {
        if (x <= S.vx) want = static_cast<double>(S.vy);
        else if (x < S.tx)
            want = static_cast<double>(S.vy) + static_cast<double>(S.ty - S.vy) *
                   (static_cast<double>(x - S.vx) / static_cast<double>(S.tx - S.vx));
    }
    double dy = static_cast<double>(y) - want;
    if (dy < 0) dy = S.pipe ? 0.0 : -dy;   // a pipe is reached from above: higher than the target costs nothing
    double match = 1.0 - dy / kSmbGuideYSpan;
    if (match < 0.0) match = 0.0;
    return kSmbGuideW * (static_cast<double>(k - wrong) + reach * match);
}

RL10_HD Signals observe_pacman(const std::uint8_t* r) {
    const int state = u8(r, 0x3F), remaining = u8(r, 0x6A);
    const bool game = u8(r, 0x48) == 0 && u8(r, 0x47) == 0;
    const bool playing = game && (state == 4 || state == 6 || state == 12) && !(u8(r, 0x4A) & 1);
    Signals s = blank();
    s.in_play = playing;
    s.level = u8(r, 0x68);
    s.progress = remaining <= 192 ? static_cast<double>(192 - remaining) : 0.0;
    s.progress_valid = playing && remaining <= 192;
    s.dying = game && state == 8;
    s.has_lives = true; s.lives = u8(r, 0x67);
    long long sc = 0, p10 = 1;
    for (int i = 0; i < 6; ++i) { sc += u8(r, 0x70 + i) * p10; p10 *= 10; }
    s.has_score = true; s.score = 10 * sc;
    s.game_over = game && state == 10;
    return s;
}

RL10_HD Signals observe_excitebike(const std::uint8_t* r, EnvState& e) {
    const int level = u8(r, 0x43) + u8(r, 0x46);
    const bool racing = u8(r, 0x47) == 1 && u8(r, 0x4F) == 1 && u8(r, 0x52) == 0;
    const int cam = u8(r, 0x12) + 256 * (u8(r, 0x4D) & 1);
    const int k4f = u8(r, 0x4F);
    if (!e.has_key || e.key_a != level || e.key_b != k4f) {
        e.has_key = true; e.key_a = level; e.key_b = k4f; e.has_cam = false; e.dist = 0.0;
    }
    if (racing) {
        if (e.has_cam) {
            const int d = ((cam - e.cam + 256) % 512 + 512) % 512 - 256;  // Python's non-negative modulo
            e.dist += d;
        }
        e.has_cam = true; e.cam = cam;
    } else {
        e.has_cam = false;
    }
    const bool crash = racing && (u8(r, 0xF2) == 1 || (1 <= u8(r, 0x9C) && u8(r, 0x9C) <= 4));
    const int temp = u8(r, 0x3B6);
    Signals s = blank();
    s.in_play = racing && u8(r, 0x3B1) == 0;
    s.level = level;
    s.progress = e.dist;
    s.progress_valid = racing;
    s.dying = crash;
    s.has_health = true; s.health = temp >= 32 ? 0.0 : 1.0;
    s.game_over = u8(r, 0x41) == 0 || u8(r, 0x41) == 1;
    s.won = u8(r, 0x3F3) >= 2;
    return s;
}

RL10_HD Signals observe_contra(const std::uint8_t* r, EnvState& e) {
    const bool game = u8(r, 0x18) == 5 && u8(r, 0x1C) == 0;
    const bool running = game && u8(r, 0x2C) == 4;
    const bool over = game && u8(r, 0x38) == 1;
    const int lives = u8(r, 0x32);
    const int level = u8(r, 0x30), screen = u8(r, 0x64);
    double progress = static_cast<double>(screen * 256 + u8(r, 0x65));
    if (u8(r, 0x40) == 1) {
        progress = screen * kContraRoomUnits;
        int slot = -1;
        for (int i = 0; i < 16; ++i) if (u8(r, 0x528 + i) == 0x14) { slot = i; break; }
        if (slot >= 0) {
            if (!e.has_key || e.key_a != level || e.key_b != screen) {
                e.has_key = true; e.key_a = level; e.key_b = screen; e.core_max = 0;
            }
            const int hp = u8(r, 0x578 + slot);
            if (hp > e.core_max) e.core_max = hp;
            if (e.core_max >= 2) {
                const double boss = 1 - static_cast<double>(hp) / e.core_max;
                progress += 0.5 * kContraRoomUnits * boss;
            }
        }
    } else if (level == 0) {
        int slot = -1;
        for (int i = 0; i < 16; ++i) if (u8(r, 0x528 + i) == 0x11) { slot = i; break; }
        if (slot >= 0) {
            const int hp = u8(r, 0x578 + slot);
            const double boss = static_cast<double>(32 - (hp < 32 ? hp : 32)) / 32;
            progress += kContraBossUnits * boss;
        }
    }
    Signals s = blank();
    s.in_play = running && u8(r, 0x25) == 0 && !over;
    s.level = level;
    s.progress = progress;
    s.progress_valid = running && u8(r, 0x90) == 1;
    s.dying = game && u8(r, 0x90) == 2;
    s.has_lives = true; s.lives = over ? -1 : lives;
    s.has_score = true; s.score = static_cast<long long>(u8(r, 0x7E2) | (u8(r, 0x7E3) << 8)) * 100;
    s.game_over = over;
    return s;
}

RL10_HD Signals observe_castlevania(const std::uint8_t* r, EnvState& e, KeySet set) {
    const int state = u8(r, 0x18);
    const bool play = state == 5 && u8(r, 0x19) == 6;
    const int hp = u8(r, 0x45);
    const int stage = u8(r, 0x28), x = u8(r, 0x41) * 256 + u8(r, 0x40), y = u8(r, 0x3F);
    if (!e.has_stage || stage != e.stage) {
        e.has_stage = true; e.stage = stage; keyset_clear(set); e.cells = 0; e.has_floor = false;
    }
    if (play && hp > 0) {
        if (!e.has_floor || abs(y - e.floor_y) >= 40) { e.has_floor = true; e.floor_y = y; }
        const std::uint32_t key = static_cast<std::uint32_t>((x / 32) * 8 + e.floor_y / 40);
        if (keyset_insert(set, e.cells, key) < 0) e.overflow = true;
    }
    const bool boss_stage = stage == 3 || stage == 6 || stage == 9 || stage == 12 || stage == 15 || stage == 18;
    const int eb = u8(r, 0x1A9);
    const double boss = boss_stage ? static_cast<double>(64 - (eb < 64 ? eb : 64)) / 64.0 : 0.0;
    Signals s = blank();
    s.in_play = play && u8(r, 0x22) == 0;
    s.level = stage;
    s.progress = static_cast<double>(32LL * e.cells) + boss * kCastleBossUnits;
    s.progress_valid = play && hp > 0;
    s.dying = (state == 5 || state == 6 || state == 7) && (hp == 0 || u8(r, 0x46C) == 8);
    s.has_lives = true; s.lives = u8(r, 0x2A);
    s.has_health = true; s.health = hp / 64.0;
    const int a[3] = {0x7FE, 0x7FD, 0x7FC};
    s.has_score = true; s.score = bcd(r, a, 3);
    s.game_over = state == 7 || state == 13;
    return s;
}

RL10_HD Signals observe_gradius(const std::uint8_t* r) {
    const bool game = u8(r, 0x00) == 5 && u8(r, 0x0A) == 1;
    const int state = u8(r, 0x1B), lives = u8(r, 0x20);
    const bool playing = game && (state & 0xE0) == 0x80 && u8(r, 0x15) == 0;
    Signals s = blank();
    s.in_play = playing;
    s.level = u8(r, 0x19);
    s.progress = static_cast<double>(u8(r, 0x3F) * 256 + u8(r, 0x3E));
    s.progress_valid = playing;
    s.dying = game && (state & 0xE0) == 0xA0;
    s.has_lives = true; s.lives = lives > 127 ? lives - 256 : lives;
    const int a[3] = {0x7E6, 0x7E5, 0x7E4};
    s.has_score = true; s.score = bcd(r, a, 3) * 10;
    s.game_over = u8(r, 0x00) == 5 && state == 192;
    return s;
}

RL10_HD Signals observe_zelda(const std::uint8_t* r, EnvState& e, KeySet set) {
    const int mode = u8(r, 0x12);
    const bool play = mode == 5 || mode == 0xB;
    int items = 0;
    for (int i = 0; i < kZeldaNumItems; ++i) {
        const int v = u8(r, RL10_ARR(kZeldaItemAddr)[i]);
        items += v < RL10_ARR(kZeldaItemCap)[i] ? v : RL10_ARR(kZeldaItemCap)[i];
    }
    const int hpb = u8(r, 0x66F), part = u8(r, 0x670);
    const double h = (hpb & 0xF) + (part >= 0x80 ? 1.0 : part > 0 ? 0.5 : 0.0);
    const int containers = (hpb >> 4) + 1;
    if (play) {
        const std::uint32_t d = u8(r, 0x10), room = u8(r, 0xEB);
        if (keyset_insert(set, e.rooms, 0x80000000u | (d << 8) | room) < 0) e.overflow = true;
        const std::uint32_t cell = (d << 16) | (room << 8) | (static_cast<std::uint32_t>(mode == 0xB) << 6) |
                                   (static_cast<std::uint32_t>(u8(r, 0x70) / 32) << 3) |
                                   static_cast<std::uint32_t>(u8(r, 0x84) / 32);
        if (keyset_insert(set, e.cells, cell) < 0) e.overflow = true;
        if (!e.has_base) { e.has_base = true; e.base_items = items; e.base_containers = containers; }
    }
    int gained = 0;
    if (e.has_base) {
        const int gi = items - e.base_items, gc = containers - e.base_containers;
        gained = (gi > 0 ? gi : 0) + (gc > 0 ? gc : 0);
    }
    Signals s = blank();
    s.in_play = play;
    s.level = popcount8(u8(r, 0x671));
    s.progress = (static_cast<double>(e.rooms) + 0.02 * e.cells) + static_cast<double>(10 * gained);
    s.progress_valid = play;
    s.dying = mode == 0x11;
    const double hf = h / containers;
    s.has_health = true; s.health = hf < 1.0 ? hf : 1.0;
    s.has_score = true; s.score = u8(r, 0x66D);
    s.game_over = mode == 0x11;
    return s;
}

RL10_HD Signals observe_tetris(const std::uint8_t* r) {
    const int mode = u8(r, 0xC0), ps = u8(r, 0x48), render = u8(r, 0xBD);
    const int la[2] = {0x51, 0x50};
    const long long lines = bcd(r, la, 2);
    const int level = u8(r, 0x64);
    const bool playing = mode == 4 && 1 <= ps && ps <= 8 && render == 3;
    const bool topout = mode == 4 && ps == 10;
    Signals s = blank();
    s.in_play = playing;
    s.level = level;
    const long long pl = lines - 10LL * level;
    s.progress = static_cast<double>(pl > 0 ? pl : 0);
    s.progress_valid = playing;
    s.dying = topout;
    const int sa[3] = {0x55, 0x54, 0x53};
    s.has_score = true; s.score = bcd(r, sa, 3);
    s.game_over = topout && u8(r, 0x58) == 20;
    return s;
}

RL10_HD Signals observe_tecmobowl(const std::uint8_t* r, EnvState& e) {
    const int oa[1] = {0x43}, pa[1] = {0x48};
    const int own = static_cast<int>(bcd(r, oa, 1)), opp = static_cast<int>(bcd(r, pa, 1));
    const int s0 = u8(r, 0x2C), s1 = u8(r, 0x2D);
    const bool ended = u8(r, 0x8D) >= 4 && u8(r, 0x8B) == 0 && u8(r, 0x8A) == 0;
    const bool on_screen = (s0 == 8 && s1 == 0) || (s0 == 9 && s1 == 9) || (s0 == 12 && s1 == 12);
    const bool playing = on_screen && u8(r, 0x8D) <= 4 && !ended;
    if (playing) { e.has_week = true; e.week = u8(r, 0x8F); }
    const int state = u8(r, 0x49);
    const int los = u8(r, 0xFB) + 256 * u8(r, 0xFC);
    const double yard = (los - kTecmoGoalX) / kTecmoPxPerYard;
    const bool scored = e.has_own && opp - e.opp >= 2;
    if (e.has_own && own != e.own) e.has_drive = false;
    if (playing && state == 1) {
        e.has_drive = false;
    } else if (playing && state == 0 && s0 == 9 && s1 == 9) {
        if (!e.has_drive) { e.has_drive = true; e.drive_start = yard; e.drive_best = yard; }
        e.drive_best = yard > e.drive_best ? yard : e.drive_best;  // Python max(): first arg wins ties
    }
    e.has_own = true; e.own = own; e.opp = opp;
    double gain = 0.0;
    if (e.has_drive) { const double g = e.drive_best - e.drive_start; gain = 0.0 >= g ? 0.0 : g; }
    Signals s = blank();
    s.in_play = playing;
    s.level = u8(r, 0x8F);
    s.progress = 100.0 / 7.0 * own + gain;
    s.progress_valid = playing;
    s.dying = scored;
    s.has_score = true; s.score = own;
    s.game_over = ended && opp > own;
    s.won = ended && own > opp && e.has_week && e.week == 10;
    return s;
}

RL10_HD const KageRoute* kage_route(int stage) {
    for (int i = 0; i < kKageNumRoutes; ++i) if (RL10_ARR(kKageRoutes)[i].stage == stage) return &RL10_ARR(kKageRoutes)[i];
    return nullptr;
}

// Returns false for Python's None.
RL10_HD bool kage_route_frac(EnvState& e, const KageRoute& rt, double cx, double cy, double& frac) {
    bool have = false; double bd = 0.0, barc = 0.0;
    for (int k = 0; k < rt.count; ++k) {
        const KageSeg& g = RL10_ARR(kKageSegs)[rt.first + k];
        if (g.acc > e.best_arc + kKageLookahead) break;
        double t = 0.0;
        if (g.len != 0.0) {
            const double num = (cx - g.x0) * (g.x1 - g.x0) + (cy - g.y0) * (g.y1 - g.y0);
            const double q = num / (g.len * g.len);
            const double m = 1.0 < q ? 1.0 : q;      // min(1.0, q)
            t = 0.0 < m ? m : 0.0;                    // max(0.0, m): Python keeps the first on ties
        }
        const double d = hypot_cr(g.x0 + t * (g.x1 - g.x0) - cx, g.y0 + t * (g.y1 - g.y0) - cy);
        if (!have || d < bd) { have = true; bd = d; barc = g.acc + t * g.len; }
    }
    if (bd > kKageMaxOff) return false;
    e.best_arc = e.best_arc >= barc ? e.best_arc : barc;
    frac = barc / rt.total;
    return true;
}

RL10_HD Signals observe_kage(const std::uint8_t* r, EnvState& e) {
    const int mode = u8(r, 2), stage = u8(r, 0x40), hp = u8(r, 0x6F0);
    if (!e.has_stage || stage != e.stage) { e.has_stage = true; e.stage = stage; e.best_arc = 0.0; }
    const double cx = ((u8(r, 0x23) << 8) | u8(r, 0x22)) / 16.0, cy = ((u8(r, 0x25) << 8) | u8(r, 0x24)) / 16.0;
    const bool play = mode == 0 && stage < 15;
    double progress = 0.0;
    bool valid = play && hp > 0 && u8(r, 0x6A1) != 8;
    const KageRoute* rt = kage_route(stage);
    const KageBoss* boss = nullptr;
    for (int i = 0; i < 6; ++i) if (RL10_ARR(kKageBosses)[i].stage == stage) boss = &RL10_ARR(kKageBosses)[i];
    const double route_w = rt == nullptr ? 0.0 : (boss != nullptr ? 500.0 : 1000.0);
    if (play && rt != nullptr) {
        double f = 0.0;
        if (!kage_route_frac(e, *rt, cx, cy, f)) valid = false;
        else progress = route_w * f;
    }
    if (play && boss != nullptr) {
        if (u8(r, 0x708) == boss->kind && u8(r, 0x798) == boss->sel) {
            const int v = u8(r, 0x7F8), mx = boss->max_hp;
            const double dmg = static_cast<double>(mx - (v < mx ? v : mx)) / mx;
            progress = route_w + (1000.0 - route_w) * dmg;
        }
    }
    const bool won = stage == 0 && mode == 0x6B && u8(r, 0xAE) == 0xFF && u8(r, 0x708) == 0x88 &&
                     u8(r, 0x798) == 0x4F && u8(r, 0x7F8) == 0 && u8(r, 0x758) == 0x10 && hp > 0;
    Signals s = blank();
    s.in_play = play && hp > 0 && u8(r, 0xD1) == 0;
    s.level = stage;
    s.progress = progress;
    s.progress_valid = valid;
    s.dying = hp == 0 && stage < 15;
    s.has_health = true; s.health = hp / 16.0;
    s.game_over = hp == 0 && mode != 0;
    s.won = won;
    return s;
}

RL10_HD Signals observe_finalmission(const std::uint8_t* r, EnvState& e) {
    const bool game = u8(r, 0x02) == 4 && u8(r, 0x03) == 0;
    const int routine = u8(r, 0x38);
    const bool playing = game && routine == 4 && u8(r, 0x4F) >= 1 && u8(r, 0x85) == 0 && u8(r, 0x0E) == 0;
    const int level = u8(r, 0x1E);
    double progress = static_cast<double>(u8(r, 0x4F) * 32 + u8(r, 0x4E));
    if (game && routine == 4 && u8(r, 0x4D) == 0 && u8(r, 0x4F) >= 1) {
        int hp = 0;
        for (int a = 0x4A0; a < 0x4C0; ++a) hp = u8(r, a) > hp ? u8(r, a) : hp;
        if (!e.has_key || e.key_a != level || e.key_d != progress) {
            e.has_key = true; e.key_a = level; e.key_d = progress; e.core_max = 0;
        }
        if (hp > e.core_max) e.core_max = hp;
        if (e.core_max >= 20) {
            const double boss = static_cast<double>(e.core_max - hp) / e.core_max;
            progress += kFinalBossUnits * boss;
        }
    }
    const int pips = u8(r, 0x1C);
    Signals s = blank();
    s.in_play = playing;
    s.level = level;
    s.progress = progress;
    s.progress_valid = playing;
    s.dying = game && u8(r, 0x85) != 0;
    s.has_health = true; s.health = (pips < 4 ? pips : 4) / 4.0;
    s.has_score = true; s.score = 10LL * (u8(r, 0x8A) | (u8(r, 0x8B) << 8) | (u8(r, 0x8C) << 16));
    s.game_over = game && (routine == 7 || routine == 8);
    return s;
}

RL10_HD Signals observe_punchout(const std::uint8_t* r, EnvState& e) {
    const int state = u8(r, 0x00), scene = u8(r, 0x05), fight = u8(r, 0x01);
    const int hp = u8(r, 0x398);
    if (!e.has_fight || fight != e.fight) { e.has_fight = true; e.fight = fight; e.damage = 0; e.has_prev_hp = false; }
    if (state == 1 && e.has_prev_hp && hp < e.prev_hp) e.damage += e.prev_hp - hp;
    e.has_prev_hp = true; e.prev_hp = hp;
    const bool playing = state == 1 && (scene == 0 || scene == 1) && u8(r, 0x300) == 1;
    const bool loss = state == 249 || state == 251 || state == 253;
    const bool winst = state == 250 || state == 252 || state == 254;
    Signals s = blank();
    s.in_play = playing;
    s.level = fight;
    s.progress = e.damage < kPunchProgressUnit ? static_cast<double>(e.damage) : kPunchProgressUnit;
    s.progress_valid = playing;
    s.dying = (state == 1 && scene == 2) || (loss && scene != 2);
    const double hf = u8(r, 0x391) / 96.0;
    s.has_health = true; s.health = hf < 1.0 ? hf : 1.0;
    s.has_score = true; s.score = digits(r, 0x3E8, 0x3EE);
    s.game_over = loss || u8(r, 0x172) * 10 + u8(r, 0x173) > 0;
    s.won = winst && fight == 13;
    return s;
}

// ---- table games (game id >= kNumGames): signals read from per-game RAM addresses (scripts/romlib/readers.py),
// not a hand-written reader. -1 = the game has no such signal. Built by rl/scripts/export_games.py.
constexpr int kGameTable = kNumGames;   // the first table game id; every id >= kNumGames is a table game
enum : int { kScoreDigitsHi = 0, kScoreDigitsLo, kScoreBinLo, kScoreBcdHi, kScoreBcdLo };
constexpr int kTableInPlayValues = 8;
struct TableSpec {
    int in_play_addr;                         // in play when ram[addr] is one of in_play_values (n of them); -1: always
    int in_play_n;
    int in_play_values[kTableInPlayValues];
    int level_addr;                           // stage number (Signals.level)
    int lives_addr;                           // lives: a decrement is a death
    int game_over_addr, game_over_value;      // ram[addr] == value: game over (the episode ends)
    int score_addr, score_n, score_kind;      // first byte, bytes, kScore*
    int x_addr, y_addr, scene_addr;           // position (screen bytes) and an optional scene byte (page/room); -1: none
    int cell_shift;                           // cell = position >> cell_shift (3: 8x8 pixels)
    double progress_unit;                     // new cells that count as one level of progress (reward10 PROGRESS_UNIT)
    double score_unit;                        // points per reward10 score unit
    double stall_frames;                      // frames without progress, level or death before a stall end; < 0: never
    double idle_per_frame;                    // reward10 idle_per_frame (W_IDLE over the stall limit), from Python
};

RL10_HD long long table_score(const std::uint8_t* r, const TableSpec& t) {
    long long v = 0;
    const int a = t.score_addr, n = t.score_n;
    for (int i = 0; i < n; ++i) {
        switch (t.score_kind) {
            case kScoreDigitsHi: v = v * 10 + r[a + i]; break;
            case kScoreDigitsLo: v = v * 10 + r[a + n - 1 - i]; break;
            case kScoreBinLo: v = v * 256 + r[a + n - 1 - i]; break;
            case kScoreBcdHi: { const int x = r[a + i]; v = v * 100 + (x >> 4) * 10 + (x & 15); break; }
            default: { const int x = r[a + n - 1 - i]; v = v * 100 + (x >> 4) * 10 + (x & 15); break; }
        }
    }
    return v;
}

// Progress of a table game = distinct (level, scene, x cell, y cell) the player reached in this episode, so every new
// cell pays W_PROGRESS / 1 and resets the stall clock (the same reward formula as the 12 hand-read games).
RL10_HD Signals observe_table(const std::uint8_t* r, const TableSpec& t, EnvState& e, KeySet set) {
    Signals s = blank();
    s.in_play = t.in_play_addr < 0;
    for (int i = 0; i < t.in_play_n && i < kTableInPlayValues; ++i)
        if (u8(r, t.in_play_addr) == t.in_play_values[i]) s.in_play = true;
    s.level = t.level_addr >= 0 ? u8(r, t.level_addr) : 0;
    if (t.x_addr >= 0) {
        if (s.in_play) {
            const std::uint32_t key = (static_cast<std::uint32_t>(s.level) << 24) |
                (static_cast<std::uint32_t>(t.scene_addr >= 0 ? u8(r, t.scene_addr) : 0) << 16) |
                (static_cast<std::uint32_t>(u8(r, t.x_addr) >> t.cell_shift) << 8) |
                static_cast<std::uint32_t>(t.y_addr >= 0 ? u8(r, t.y_addr) >> t.cell_shift : 0);
            if (keyset_insert(set, e.cells, key) < 0) e.overflow = true;
        }
        s.progress = static_cast<double>(e.cells);
        s.progress_valid = s.in_play;
    }
    if (t.lives_addr >= 0) { const int lv = u8(r, t.lives_addr); s.has_lives = true; s.lives = lv > 127 ? lv - 256 : lv; }
    if (t.score_addr >= 0) { s.has_score = true; s.score = table_score(r, t); }
    s.game_over = t.game_over_addr >= 0 && u8(r, t.game_over_addr) == t.game_over_value;
    return s;
}

RL10_HD Signals observe(int game, const std::uint8_t* r, EnvState& e, KeySet set) {
    switch (game) {
        case kGame_smb: return observe_smb(r);
        case kGame_pacman: return observe_pacman(r);
        case kGame_excitebike: return observe_excitebike(r, e);
        case kGame_contra: return observe_contra(r, e);
        case kGame_castlevania: return observe_castlevania(r, e, set);
        case kGame_gradius: return observe_gradius(r);
        case kGame_zelda: return observe_zelda(r, e, set);
        case kGame_tetris: return observe_tetris(r);
        case kGame_tecmobowl: return observe_tecmobowl(r, e);
        case kGame_kage: return observe_kage(r, e);
        case kGame_finalmission: return observe_finalmission(r, e);
        default: return observe_punchout(r, e);
    }
}

// Capacity (entries) of the per-env key set each game needs; 0 = none.
RL10_HD std::uint32_t keyset_capacity(int game) {
    return game >= kNumGames ? 8192u : game == kGame_zelda ? 16384u : game == kGame_castlevania ? 4096u : 0u;
}

// RewardTracker.reset(): the caller also clears the key set.
RL10_HD void reset(EnvState& e) { e = EnvState{}; }

// Reward parts (same order everywhere): the reward is their sum. Used outside the actor (diversity judge, logging).
enum : int { kPartProgress = 0, kPartLevel, kPartDeath, kPartDamage, kPartScore, kPartWin, kPartStall, kNumParts };

struct Update {
    double reward;
    bool done;
    std::uint32_t events;
    int event_level;  // level of a 'level' or 'skip' event
    double part[kNumParts];  // the reward split by source (sums to reward)
};

// RewardTracker.update(ram, frames): call after every decision (and once right after boot).
RL10_HD Update update(int game, const std::uint8_t* ram, int frames, EnvState& e, KeySet set,
                     double guide_discount = 1.0, bool ignore_stall = false, bool allow_skip = false,
                     const TableSpec* table = nullptr) {
    // a table game (game >= kNumGames) needs its spec; its consts come from it (no idle cost, no START/SELECT rules)
    const bool is_table = game >= kNumGames;
    const GameConsts TG = {is_table ? table->progress_unit : 1.0, is_table ? table->score_unit : 1.0,
                           is_table ? table->stall_frames : -1.0, is_table ? table->idle_per_frame : 0.0,
                           is_table && table->lives_addr >= 0 ? 1 : 0, 0, 0};
    const GameConsts& G = is_table ? TG : RL10_ARR(kGameConsts)[game];
    const Signals s = is_table ? observe_table(ram, *table, e, set) : observe(game, ram, e, set);
    const bool has_p = e.has_prev;
    const Signals p = e.prev;
    double r = 0.0;
    std::uint32_t ev = 0;
    int ev_level = 0;
    bool fresh = false;
    e.has_prev = true;
    e.prev = s;
    Update out{0.0, true, 0u, 0, {}};
    double part[kNumParts] = {};
    if (e.done) return out;
    if (s.in_play) {
        if (!e.has_best_level) {
            e.has_best_level = true; e.best_level = s.level; e.has_best_progress = false; e.score_used = 0.0;
        } else if (s.level > e.best_level + 1 && !allow_skip) {
            ev |= kEvSkip; ev_level = s.level;
            e.done = true; e.end_reason = kEndSkip;
        } else if (s.level > e.best_level + 1) {   // allow_skip: a warp is a plain level change (no reward, no end)
            e.best_level = s.level; e.has_best_progress = false; e.score_used = 0.0;
            fresh = true;
        } else if (s.level > e.best_level) {
            r += kW_LEVEL; part[kPartLevel] += kW_LEVEL;
            ev |= kEvLevel; ev_level = s.level;
            e.best_level = s.level; e.has_best_progress = false; e.score_used = 0.0;
            fresh = true;
        }
        if (s.level == e.best_level && s.progress_valid) {
            if (!e.has_best_progress) {
                e.has_best_progress = true; e.best_progress = s.progress;
            } else if (s.progress > e.best_progress) {
                const double rp = kW_PROGRESS * (s.progress - e.best_progress) / G.progress_unit;
                r += rp; part[kPartProgress] += rp;
                e.best_progress = s.progress;
                fresh = true;
            }
        }
    }
    bool death = false;
    if (has_p) {
        if (!G.death_from_lives) death = s.dying && !p.dying;
        else death = s.has_lives && p.has_lives && s.lives < p.lives;
        if (death && (s.in_play || p.in_play)) {
            r += kW_DEATH; part[kPartDeath] += kW_DEATH;
            e.deaths += 1;
            ev |= kEvDeath;
            fresh = true;
        }
        if (s.in_play && p.in_play && s.has_health && p.has_health && s.health < p.health) {
            const double rd = kW_DAMAGE * (p.health - s.health);
            r += rd; part[kPartDamage] += rd;
        }
        if (s.in_play && p.in_play && s.has_score && p.has_score && s.score > p.score) {
            double sr = kW_SCORE * static_cast<double>(s.score - p.score) / G.score_unit;
            if (kSCORE_CAP < sr) sr = kSCORE_CAP;  // min(SCORE_CAP, x, y): first minimal argument wins
            const double left = kSCORE_LEVEL_BUDGET - e.score_used;
            if (left < sr) sr = left;
            if (sr > 0) { r += sr; part[kPartScore] += sr; e.score_used += sr; }
        }
    }
    if (has_p && !e.done && !death && !s.dying && !p.dying && !s.won && !s.game_over && s.in_play && p.in_play &&
        s.progress_valid && p.progress_valid && s.level == p.level && s.level == e.best_level &&
        p.progress - s.progress >= kROLLBACK_DROP * G.progress_unit) {
        r += kW_ROLLBACK; part[kPartStall] += kW_ROLLBACK;
        ev |= kEvRollback;
    }
    if (!fresh && !e.done && has_p && s.in_play && p.in_play && s.progress_valid && p.progress_valid &&
        s.progress <= p.progress) {   // waiting costs a little (scripts/reward10/reward.py W_IDLE)
        const double ci = G.idle_per_frame * frames;
        r += ci; part[kPartStall] += ci;
    }
    e.idle_frames = (fresh || !has_p) ? 0 : e.idle_frames + frames;
    if (s.won) {
        r += kW_WIN; part[kPartWin] += kW_WIN;
        ev |= kEvWon;
        e.done = true; e.end_reason = kEndWon;
    } else if (s.game_over) {
        ev |= kEvGameOver;
        e.done = true; e.end_reason = kEndGameOver;
    } else if (!ignore_stall && !e.done && G.stall_frames >= 0.0 && static_cast<double>(e.idle_frames) >= G.stall_frames) {
        r += kW_STALL; part[kPartStall] += kW_STALL;
        ev |= kEvStall;
        e.done = true; e.end_reason = kEndStall;
    }
    if (game == kGame_smb) {
        bool has_g = false;
        const double observed = smb_guide(ram, has_g);
        if (has_g || e.has_key) {
            double gv = e.has_key && e.key_a == s.level ? e.key_d : 0.0;
            if (has_g) gv = observed;
            if (e.done) gv = 0.0;
            if (has_p) {
                const double rg = guide_discount * gv - e.key_d;
                r += rg; part[kPartProgress] += rg;
            }
            e.has_key = !e.done && (has_g || (e.has_key && e.key_a == s.level));
            e.key_a = s.level; e.key_d = gv;
        }
    }
    out.reward = r; out.done = e.done; out.events = ev; out.event_level = ev_level;
    for (int k = 0; k < kNumParts; ++k) out.part[k] = part[k];
    return out;
}

}  // namespace rl10
