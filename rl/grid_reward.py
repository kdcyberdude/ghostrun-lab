"""Bounded episodic world-grid novelty. No cell/position data is given to the actor.

Cells are paid once per attempt, including across rollout/checkpoint boundaries. Familiarity counts first visits
per attempt across training, using the previous rollout's counts for all parallel actors in the next rollout.
The cap is on unweighted reward, not visits. All moods explore until a level's recent clear rate reaches 50%
with at least 100 completed attempts; only then do mood exploration weights differ.
Native code supplies only completion/death/time rewards and stable cell IDs (SMB initially).
"""
import torch

class GridReward:
    def __init__(self, n, device, bonus=0.005, cap=1.0, history_scale=1.0, regional=False, coverage=False, cell_size=32,
                 quadratic_stagnation=False, persistent_visits=False):
        if history_scale <= 0:
            raise ValueError('history_scale must be positive')
        if coverage and regional:
            raise ValueError('coverage uses equal-sized cells in both directions, not horizontal regions')
        if cell_size not in (8,32) or (cell_size == 8 and not coverage):
            raise ValueError('8px cells require coverage mode; supported sizes are 8 and 32')
        self.cell_size = cell_size
        self.y_bits = 5 if cell_size == 8 else 3
        self.phase_shift = 16 if cell_size == 8 else 12
        self.level_shift = self.phase_shift + 13
        self.cell_space = 1 << (self.level_shift + 5)
        self.n, self.device, self.bonus, self.cap = n, device, bonus, cap
        self.history_scale = history_scale
        self.regional = regional
        self.coverage = coverage
        self.quadratic_stagnation = quadratic_stagnation
        self.persistent_visits = persistent_visits
        if coverage:
            self.cap = 4.0
        self.coverage_mass = torch.zeros(n, device=device)
        self.no_discovery_steps = torch.zeros(n, dtype=torch.int64, device=device)
        self.env_id = torch.arange(n, device=device)
        self.episode = torch.zeros(n, dtype=torch.int64, device=device)
        self.spent = torch.zeros(n, device=device)
        self.returns = torch.zeros(n, device=device)
        self.seen = torch.empty(0, dtype=torch.int64, device=device)
        self.known = self.seen.clone()
        self.count = torch.empty(0, dtype=torch.int64, device=device)
        self.finished = torch.zeros(32, device=device)
        self.total_finished = torch.zeros(32, device=device)
        self.wins = torch.zeros(32, device=device)
        self.last_cell = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.idle_steps = torch.zeros(n, dtype=torch.int64, device=device)
        self.idle_spent = torch.zeros(n, device=device)
        self.region_seen = self.seen.clone()
        self.region_known = self.known.clone()
        self.region_count = self.count.clone()

    @staticmethod
    def regions(cells):
        """128x32 world regions: four horizontal cells, preserving corridor height, room and maze phase."""
        x = (cells >> 3) & 511
        key = (cells & ~(511 << 3)) | ((x // 4) << 3)
        return torch.where(cells >= 0, key, -1)

    def seed_region_history(self):
        """Conservative prior: max constituent-cell count, since historical attempt unions are unavailable."""
        keys, inv = torch.unique(self.regions(self.known), return_inverse=True)
        counts = torch.zeros_like(keys)
        counts.scatter_reduce_(0, inv, self.count, reduce='amax')
        self.region_known, self.region_count = keys, counts

    @staticmethod
    def mood_weight(mood, moods):
        # First quarter fast, next quarter balanced, remaining half exploratory.
        group = (mood * 4 // moods).clamp_max(2)
        return torch.where(group == 0, 0.0, torch.where(group == 1, 0.25, 1.0))

    def _novelty(self, cells, done, episodes, seen, known, count, bonus, power, floor=0.):
        t, n = done.shape
        before = self.env_id if self.persistent_visits else self.episode * n + self.env_id
        after = self.env_id.expand_as(episodes) if self.persistent_visits else episodes * n + self.env_id
        seed = before * self.cell_space + cells[0]
        tagged = after * self.cell_space + cells[1:]
        seed = seed[cells[0] >= 0]
        valid = cells[1:] >= 0
        sample_indices = torch.arange(t * n, device=self.device)[valid.flatten()]
        samples = tagged[valid]
        combined = torch.cat([seen, seed, samples])
        unique, inverse = torch.unique(combined, return_inverse=True)
        first = torch.full_like(unique, combined.numel())
        first.scatter_reduce_(0, inverse, torch.arange(combined.numel(), device=self.device), reduce='amin')
        sample_start = seen.numel() + seed.numel()
        new = first >= sample_start
        indices = sample_indices[first[new] - sample_start]
        cell = unique[new] % self.cell_space
        counts = torch.zeros_like(cell)
        if known.numel():
            pos = torch.searchsorted(known, cell).clamp_max(known.numel() - 1)
            counts = torch.where(known[pos] == cell, count[pos], counts)
        raw = torch.zeros(t * n, device=self.device)
        denominator = 1 + counts.float() / self.history_scale
        raw[indices] = floor + bonus / (denominator.sqrt() if power == .5 else denominator)
        if self.persistent_visits:
            # Shared lifetime history already includes the level/room/maze phase in each key.
            # All workers use the previous rollout's history; death never makes a known cell novel again.
            raw[indices] *= (counts == 0)
        raw = raw.view(t, n) * (~done)

        new_attempt_cells = unique[first >= seen.numel()] % self.cell_space
        keys, increments = torch.unique(new_attempt_cells, return_counts=True)
        all_keys, inv = torch.unique(torch.cat([known, keys]), return_inverse=True)
        updated_count = torch.zeros_like(all_keys)
        updated_count.index_add_(0, inv, torch.cat([count, increments]))
        owner = unique // self.cell_space
        current = episodes[-1, owner % n] * n + owner % n
        # In persistent mode known/count are authoritative across rollouts; tags only deduplicate
        # each worker's discoveries within this rollout, including across its deaths.
        live = unique[:0] if self.persistent_visits else unique[owner == current]
        return raw, live, all_keys, updated_count

    @torch.no_grad()
    def apply(self, cells, done, base_reward, mood, moods, info):
        """cells [T+1,N] are observations; terminal rows contain the next entrance, which earns nothing.
        Fine and regional novelty share one reward cap. Counts update once per rollout, after bonus calculation.
        """
        t, n = done.shape
        assert n == self.n and cells.shape == (t + 1, n)
        episodes = self.episode + done.long().cumsum(0)
        coverage_bonus = .02 * self.cell_size / 32  # equal mass for equal straight-line travel at either resolution
        fine, self.seen, self.known, self.count = self._novelty(
            cells, done, episodes, self.seen, self.known, self.count,
            coverage_bonus if self.coverage else .001 if self.regional else self.bonus, .5,
            floor=coverage_bonus if self.coverage else 0.)
        region = torch.zeros_like(fine)
        if self.regional:
            region, self.region_seen, self.region_known, self.region_count = self._novelty(
                self.regions(cells), done, episodes, self.region_seen, self.region_known, self.region_count, .02, 1.)
        raw = fine + region

        # Prefix sums with episode resets: budget carries across rollouts, but never across a death/clear/limit.
        cumulative = raw.cumsum(0) + (self.coverage_mass if self.coverage else self.spent)
        reset = torch.where(done, cumulative, 0).cummax(0).values
        before = (cumulative - raw - reset).clamp_min(0)
        if self.coverage:
            # F(s) = cap*s/(cap+s). Its increments stay positive beyond the old hard cap,
            # but sum to < cap per attempt. Familiarity can remove only the extra rarity bonus.
            paid = self.cap**2 * raw / ((self.cap + before) * (self.cap + before + raw))
            self.coverage_mass = cumulative[-1] - reset[-1]
            self.spent = self.cap * self.coverage_mass / (self.cap + self.coverage_mass)
        else:
            remaining = (self.cap - before).clamp(0, self.cap)
            paid = torch.minimum(raw, remaining)
            self.spent = (cumulative[-1] - reset[-1]).clamp_max(self.cap)
        mastered = (self.total_finished >= 100) & (self.wins >= self.finished * .5)
        level = (cells[1:] >> self.level_shift).clamp(0,31)
        weight = torch.where(mastered[level], self.mood_weight(mood, moods), 1.0)
        weighted = paid * weight
        reward = base_reward + weighted
        if self.coverage:
            # Reaching the attempt limit is a failure too; waiting must not avoid the death cost.
            reward -= .5 * (done & (info[...,2] == 8))
        self.last_cell = torch.where(self.last_cell < 0, cells[0], self.last_cell)
        stationary_total = torch.zeros((),device=self.device)
        stagnant_steps = torch.zeros((),device=self.device)
        for step in range(t):
            current = cells[step+1]
            if self.coverage:
                # New cells in ANY direction reset the clock. Revisits, jumps and invalid
                # observations cannot reset it. The selected grace/ramp carries across rollouts.
                self.no_discovery_steps = torch.where(done[step] | (fine[step] > 0), 0,
                                                       self.no_discovery_steps + 1)
                grace = 8 if self.quadratic_stagnation else 60
                stagnant = (self.no_discovery_steps > grace) & ~done[step]
                ramp = ((self.no_discovery_steps - 8).float() / 8).clamp(0,1).square() if self.quadratic_stagnation else 1.
                charge = stagnant.float() * .001 * ramp
                stagnant_steps += stagnant.sum()
            else:
                valid = (current >= 0) & ~done[step]
                same = valid & ((current >> 3) == (self.last_cell >> 3))
                self.idle_steps = torch.where(same, self.idle_steps + 1, 0)
                # Legacy experiment: same-column cost with an episode cap.
                charge = torch.where(valid & (self.idle_steps > 30),
                                     (.25 - self.idle_spent).clamp(0,.001), 0)
                self.idle_spent = torch.where(done[step], 0, self.idle_spent + charge)
                self.last_cell = torch.where(current >= 0, current, self.last_cell)
            reward[step] -= charge
            stationary_total += charge.sum()
            self.returns += reward[step]
            info[step, :, 0] = torch.where(done[step], self.returns, info[step, :, 0])
            self.returns = torch.where(done[step], 0, self.returns)

        # Actual native outcomes only, with no 50% prior for unseen levels. Used on the NEXT rollout.
        eligible = done & ((info[...,2] == 6) | (info[...,2] == 7) | (info[...,2] == 8) |
                           (info[...,2] == 1) | (info[...,2] == 2))
        outcome_level = info[...,4].long().clamp(0,31)
        cleared = eligible & ((info[...,2] == 6) | (info[...,2] == 2))
        ended = torch.bincount(outcome_level[eligible],minlength=32)
        successes = torch.bincount(outcome_level[cleared],minlength=32)
        self.total_finished += ended
        self.finished = torch.where(ended > 0, .95*self.finished + ended, self.finished)
        self.wins = torch.where(ended > 0, .95*self.wins + successes, self.wins)

        self.episode = episodes[-1]
        stats = dict(cell_size=self.cell_size, new_cells=int((fine > 0).sum()), known_cells=self.known.numel(),
                     discovery_reward=float(weighted.sum()), capped_envs=int((self.spent >= self.cap - 1e-6).sum()),
                     spent_mean=float(self.spent.mean()))
        stats['persistent_visits'] = self.persistent_visits
        stats['stagnation_penalty' if self.coverage else 'stationary_penalty'] = float(stationary_total)
        if self.coverage:
            stats['stagnant_fraction'] = float(stagnant_steps / (t*n))
            stats['no_discovery_steps_mean'] = float(self.no_discovery_steps.float().mean())
        stats['speed_enabled_levels'] = mastered.nonzero().flatten().tolist()
        if self.regional:
            stats['new_region_visits'] = int((region > 0).sum())
            stats['known_regions'] = self.region_known.numel()
            stats['regional_reward'] = float((weighted * region / raw.clamp_min(1e-12)).sum())
        before, after = cells[:-1], cells[1:]
        castle44 = ((before >= 0) & (after >= 0) & (before >> self.level_shift == 15) &
                    (after >> self.level_shift == 15) & ~done)
        p, q = (before >> self.phase_shift) & 15, (after >> self.phase_shift) & 15
        stats['maze_check1_entries'] = int((castle44 & (p == 0) & (q >= 1)).sum())
        stats['maze_check2_entries'] = int((castle44 & (p < 2) & (q == 2)).sum())
        valid_cells = cells[cells >= 0]
        if valid_cells.numel():
            x_mask = (1 << (self.phase_shift-self.y_bits)) - 1
            stats['max_x'] = int((((valid_cells >> self.y_bits) & x_mask) * self.cell_size - 1024).max())
            stats['height_cells'] = torch.bincount(valid_cells & ((1 << self.y_bits)-1), minlength=1 << self.y_bits).tolist()
            stats['maze_phases'] = torch.bincount((valid_cells >> self.phase_shift) & 15, minlength=16).tolist()
        return reward, stats

    def state_dict(self):
        return {k: getattr(self, k) for k in ('episode', 'spent', 'returns', 'seen', 'known', 'count',
                                             'finished','total_finished','wins','last_cell','idle_steps','idle_spent',
                                             'region_seen','region_known','region_count',
                                             'coverage_mass','no_discovery_steps')}

    def load_state_dict(self, state):
        for key in self.state_dict():
            if key in state:
                setattr(self, key, state[key].to(self.device).clone())
