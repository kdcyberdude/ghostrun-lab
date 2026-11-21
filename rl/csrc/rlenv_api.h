// C API of the environment, identical for the GPU build (rlenv.cu -> librlenv_cuda.so) and the CPU reference build
// (rlenv_cpu.cpp -> librlenv_cpu.so). All buffers passed to rlenv_bind are device pointers for the GPU build and host
// pointers for the CPU build; `stream` is a cudaStream_t (ignored by the CPU build).
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
void* rlenv_create(int ngames, const int* game_ids, const std::uint8_t* const* roms, const std::size_t* rom_sizes,
                   const std::uint8_t* const* boot_pads, const std::size_t* boot_sizes, const int* n_envs,
                   std::uint64_t run_seed, int sticky, int idle_max, int dropout, int device);
int rlenv_num_envs(void* h);
// Optional curriculum terminal at this native level transition; -1 keeps ordinary whole-game episodes.
void rlenv_set_episode_level_end(void* h, int level);
void rlenv_set_guide_discount(void* h, double discount);  // shaping discount; must match the learner's gamma
void rlenv_grid_attempts(void* h, std::int64_t* cells, int decision_limit, int cell_size);
// Attempts (rlenv_core.cuh Common::attempt_limit): decision_limit > 0 turns them on for episodes started from a saved
// slot; kill ([N] uint8 input) and cur_start_out ([N] int32 output) are optional. rlenv_bind_cells: the SMB world-grid
// cell output ([N] int64, reward/search data only) without the grid reward.
void rlenv_set_attempts(void* h, int decision_limit, const std::uint8_t* kill, std::int32_t* cur_start_out);
// Optional output [N] int32: the slot the current episode started from (-1 = power-on), without attempts.
void rlenv_bind_cur_start(void* h, std::int32_t* cur_start_out);
void rlenv_bind_cells(void* h, std::int64_t* cells, int cell_size);
// on != 0: attempts do not end at level events (a single life through the whole game; rl/explore.py --whole-game)
void rlenv_set_attempt_through(void* h, int on);
void rlenv_set_only_kill(void* h, int on);
void rlenv_set_allow_skip(void* h, int on);
// Table game gi: copy its rl10::TableSpec (bytes = sizeof check); 0 = ok, 1 = not a table game / bad size
int rlenv_set_table(void* h, int gi, const void* spec, std::size_t bytes);
// on != 0: a start from a saved slot plays no restart frames (exact restore; search and route replay)
void rlenv_set_restore_exact(void* h, int on);
// Search data: after every decision out[env * n + k] = RAM[addr[k]] (CPU RAM, 0x000-0x7FF; device memory on the GPU build)
void rlenv_bind_ram(void* h, std::uint8_t* out, const std::int32_t* addr, int n);
void rlenv_bind(void* h, const std::int16_t* actions, std::uint8_t* obs, std::uint8_t* emph, float* reward,
                std::uint8_t* done, std::uint8_t* masks, std::uint8_t* timeline, std::uint8_t* pads,
                std::uint32_t* events, float* info);
// Optional output [N, 7] float: the reward split by source (progress, level, death, damage, score, win, stall).
void rlenv_bind_parts(void* h, float* parts);
// Point the picture outputs somewhere else (e.g. the next slot of a rollout buffer); takes effect at the next call.
void rlenv_bind_obs(void* h, std::uint8_t* obs, std::uint8_t* emph);
// Level starts (rlenv_core.cuh): room for `cap` saved level starts per game (call once, before start/load); game
// gi's chance that a new episode starts from one (default 0) and its cumulative level weights * 2^32 (cap entries;
// null = uniform over the saved levels; call between decisions, nothing running); output [N] int32 start level of a finished episode (-1 =
// power-on); deaths of game gi's saves (-1 = empty) into host memory.
int rlenv_enable_level_starts(void* h, int cap);
void rlenv_set_level_plan(void* h, int gi, double p, const std::uint32_t* cdf);
void rlenv_bind_start(void* h, std::int32_t* start_out);
void rlenv_bind_level(void* h, std::int32_t* level_out);   // output [N] int32: level after each decision, -1 unknown
void rlenv_bind_progress(void* h, float* progress_out);  // output [N] float: current within-level progress / PROGRESS_UNIT, NaN = not valid
int rlenv_level_deaths(void* h, int gi, std::int32_t* host_out, void* stream);
// Places (rl/places.py): `cap` slots per game, the first `levels` are level starts (as above), the rest are filled by
// rlenv_place_save: env_idx[j] (index within game gi) -> slot_idx[j] (skipped where < levels), count entries, arrays
// in the env's memory (device for the CUDA build), after a step on the same stream. rlenv_place_copy: slot src[j] ->
// dst[j], host arrays, synchronous. rlenv_level_layout: stride, key_cap, sizeof(EnvState), cap, levels.
int rlenv_enable_places(void* h, int cap, int levels);
int rlenv_place_save(void* h, int gi, const std::int32_t* env_idx, const std::int32_t* slot_idx, int count, void* stream);
int rlenv_place_copy(void* h, int gi, const std::int32_t* src, const std::int32_t* dst, int count, void* stream);
void rlenv_level_layout(void* h, int gi, std::uint32_t* out);
int rlenv_start(void* h, void* stream);   // start every env (first episode); 0 = ok
int rlenv_step(void* h, void* stream);    // one decision for every env; 0 = ok
std::size_t rlenv_state_bytes(void* h);
int rlenv_save(void* h, std::uint8_t* host_out, void* stream);   // exact resume state (synchronous)
int rlenv_load(void* h, const std::uint8_t* host_in, void* stream);
const char* rlenv_error();
void rlenv_destroy(void* h);
}
