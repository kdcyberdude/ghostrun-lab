"""Unit tests for the shared reward10 tracker (no emulator needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/reward10'))
from reward import RewardTracker, Signals, W_DEATH, W_IDLE, W_LEVEL, W_STALL, W_WIN, W_ROLLBACK, SCORE_CAP, SCORE_LEVEL_BUDGET  # noqa: E402
import contract  # noqa: E402


class FakeGame:
    PROGRESS_UNIT = 100.0
    SCORE_UNIT = 10.0
    DEATH_FROM = 'dying'
    IDLE_SCALE = 0.0   # the waiting cost has its own tests below

    def __init__(self, seq):
        seq_ = seq

        class Reader:
            def reset(self):
                self.i = 0

            def observe(self, ram):
                s = seq_[self.i]
                self.i += 1
                return s
        self.Reader = Reader


def run(seq, death_from='dying'):
    g = FakeGame(seq)
    g.DEATH_FROM = death_from
    tr = RewardTracker(g)
    return [tr.update(None, t) for t in range(len(seq))], tr


def S(**k):
    base = dict(in_play=True, level=0, progress=0.0)
    base.update(k)
    return Signals(**base)


def test_progress_is_high_water_only():
    out, _ = run([S(progress=0), S(progress=50), S(progress=30), S(progress=50), S(progress=60)])
    rs = [r for r, _, _ in out]
    assert [round(r, 9) for r in rs] == [0, 0.5, 0, 0, 0.1]


def test_next_level_bonus_and_stale_progress_ignored():
    out, tr = run([S(progress=10), S(level=1, progress=2900, progress_valid=False),
                   S(level=1, progress=40), S(level=1, progress=90)])
    rs = [r for r, _, _ in out]
    assert rs[1] == W_LEVEL
    assert rs[2] == 0                  # baseline taken at first valid position, not stale 2900
    assert abs(rs[3] - 0.5) < 1e-9


def test_level_skip_ends_episode_without_bonus():
    out, tr = run([S(progress=10), S(level=3, progress=40), S(level=3, progress=90)])
    assert out[1] == (0.0, True, [('skip', 3)])   # e.g. SMB 1-2 warp zone to 2-1
    assert out[2] == (0.0, True, []) and tr.best_level == 0


def test_death_edge_counts_once():
    out, tr = run([S(), S(dying=True), S(dying=True), S(in_play=False, dying=True), S()])
    assert tr.deaths == 1 and out[1][0] == W_DEATH


def test_lives_mode_and_game_over():
    out, tr = run([S(lives=2), S(lives=1), S(lives=1, game_over=True), S(lives=1)], 'lives')
    assert tr.deaths == 1 and out[2][1] is True and out[3] == (0.0, True, [])


def test_score_capped_and_win():
    out, tr = run([S(score=0), S(score=10_000), S(score=10_000, won=True)])
    assert out[1][0] == SCORE_CAP and out[2][0] == W_WIN and tr.done


def test_score_budget_per_level_not_reset_by_death():
    out, tr = run([S(score=0), S(score=1000), S(score=2000), S(score=3000), S(score=4000, dying=True),
                   S(level=1, score=5000)])
    assert [o[0] for o in out[1:4]] == [SCORE_CAP, SCORE_CAP, 0.0] and 2 * SCORE_CAP == SCORE_LEVEL_BUDGET
    assert out[4][0] == W_DEATH                      # dying does not refill the budget
    assert out[5][0] == W_LEVEL + SCORE_CAP          # a new level does


def test_stall_ends_episode_with_penalty_and_waiting_costs():
    g = FakeGame([S(progress=10)] + [S(progress=10)] * 900 + [S(progress=10)])
    g.STALL_SECONDS = 2.0                                       # 120 frames = 15 decisions of 8 frames
    g.IDLE_SCALE = 1.0
    tr = RewardTracker(g)
    out = [tr.update(None, t) for t in range(16)]
    assert [o[1] for o in out[:15]] == [False] * 15
    per = W_IDLE * 8 / 120
    assert out[0][0] == 0 and all(abs(o[0] - per) < 1e-12 for o in out[1:15])
    assert abs(out[15][0] - (per + W_STALL)) < 1e-12 and out[15][1:] == (True, [('stall', 2.0)])
    assert tr.end_reason == 'stall'
    assert abs(sum(o[0] for o in out) - (W_IDLE + W_STALL)) < 1e-9   # waiting a whole limit costs W_IDLE


def test_waiting_cost_only_when_not_moving_forward_in_play():
    # walking forward behind the high-water mark is free; walking back, or standing, costs; off-play is free
    g = FakeGame([S(progress=30), S(progress=10), S(progress=20), S(progress=20), S(progress=20, in_play=False),
                  S(progress=20, in_play=False)])
    g.IDLE_SCALE = 1.0
    tr = RewardTracker(g)
    r = [tr.update(None, t)[0] for t in range(6)]
    per = W_IDLE * 8 / (120 * 60)
    assert r[0] == 0 and abs(r[1] - per) < 1e-12 and r[2] == 0 and abs(r[3] - per) < 1e-12 and r[4] == 0 and r[5] == 0


def test_stall_clock_restarts_on_progress_level_and_death_and_counts_pause():
    seq = [S(progress=0)] + [S(progress=0)] * 10 + [S(progress=5)] + [S(progress=5, in_play=False)] * 14 \
        + [S(progress=5, in_play=False)]
    g = FakeGame(seq)
    g.STALL_SECONDS = 2.0
    tr = RewardTracker(g)
    out = [tr.update(None, t) for t in range(len(seq))]
    assert not any(o[1] for o in out[:26])                                 # progress at step 11 restarted the clock
    assert out[26][2] == [('stall', 2.0)]                                  # paused frames count too
    g = FakeGame([S(), S(dying=True), S(dying=False), S(level=1)])
    g.STALL_SECONDS = 2.0
    tr = RewardTracker(g)
    tr.update(None, 0); tr.idle_frames = 100
    tr.update(None, 1)
    assert tr.idle_frames == 0                                             # a death restarts it
    tr.update(None, 2); tr.idle_frames = 100
    tr.update(None, 3)
    assert tr.idle_frames == 0                                             # so does a new level


def test_rollback_penalty_is_once_per_abrupt_drop_and_keeps_high_water():
    out, tr = run([S(progress=80), S(progress=40), S(progress=60), S(progress=20), S(progress=90)])
    assert [round(o[0], 9) for o in out] == [0, W_ROLLBACK, 0, W_ROLLBACK, 0.1]
    assert out[1][2] == [('rollback', 0.4)] and tr.best_progress == 90
    gradual, _ = run([S(progress=x) for x in (80, 60, 40, 20)])
    assert all(o[0] == 0 and o[2] == [] for o in gradual)


def test_rollback_ignores_deaths_and_invalid_or_changed_level_positions():
    for before, after in [
        (S(progress=80), S(progress=0, dying=True)),
        (S(progress=80, dying=True), S(progress=0)),
        (S(progress=80), S(progress=0, progress_valid=False)),
        (S(progress=80, progress_valid=False), S(progress=0)),
        (S(progress=80), S(progress=0, in_play=False)),
        (S(progress=80), S(progress=0, level=1)),
        (S(progress=80), S(progress=0, won=True)),
        (S(progress=80), S(progress=0, game_over=True)),
    ]:
        out, _ = run([before, after])
        assert not any(e[0] == 'rollback' for e in out[1][2])
    out, _ = run([S(progress=80, lives=2), S(progress=0, lives=1)], 'lives')
    assert out[1][2] == [('death', 1)] and out[1][0] == W_DEATH


def test_smb_page_is_signed_so_a_maze_loop_does_not_pay_progress():
    """4-4's wrong path takes the page counter below 0 (255 = -1); read as unsigned it paid +21.5 in one decision."""
    reader = contract.load('smb').Reader()
    reader.reset()
    ram = bytearray(2048)
    ram[0x770], ram[0x772], ram[0x0E], ram[0x86] = 1, 3, 8, 222
    ram[0x6D] = 3
    before = reader.observe(ram).progress
    ram[0x6D] = 255
    after = reader.observe(ram).progress
    assert before == 3 * 256 + 222 and after == -256 + 222


def test_every_game_follows_the_contract_and_spec_is_current():
    assert len(contract.games()) == 12
    for slug in contract.games():
        assert contract.check(contract.load(slug)) == [], slug
    import json
    assert json.loads(contract.SPEC_PATH.read_text()) == json.loads(json.dumps(contract.all_specs()))


def test_start_and_select_are_blocked_in_play_unless_the_game_needs_them():
    for slug in contract.games():
        pad = contract.play_buttons(contract.load(slug))
        assert pad >= contract.PAD, slug
        if slug == 'punchout':
            assert pad == contract.PAD | {'START', 'SELECT'}   # star punch / next round, rest between rounds
        elif slug == 'zelda':
            assert pad == contract.PAD | {'START'}             # item screen
        else:
            assert pad == contract.PAD, slug                  # Start pauses: pausing before a death dodges the -1


def smb_ram(level, x, y, area=None, routine=8, lives=2, atype=3):
    """SMB RAM in play at level ordinal `level`, Mario at x (signed page) and y; area defaults to the maze castle."""
    smb = contract.load('smb')
    ram = bytearray(2048)
    ram[0x770], ram[0x772], ram[0x0E], ram[0x75A] = 1, 3, routine, lives
    ram[0x75F], ram[0x75C] = level // 4, level % 4
    ram[0x6D], ram[0x86], ram[0xCE] = (x // 256) & 0xFF, x % 256, y
    ram[0x750] = smb.MAZE_GUIDE[level][0] if area is None and level in smb.MAZE_GUIDE else (area or 0)
    ram[0x74E] = atype
    ram[0x725] = max(0, x // 256 + 1) & 0xFF
    return bytes(ram)


def smb_return(rams):
    tr = RewardTracker(contract.load('smb'))
    return sum(tr.update(r, t)[0] for t, r in enumerate(rams))


def test_smb_maze_guide_pays_the_right_height_before_a_loop_check():
    """4-4's page-9 check needs y 176 (lower path): approaching it at 176 must earn more than at 64."""
    walk = lambda y: [smb_ram(15, x, y) for x in range(1100, 2041, 10)]  # noqa: E731
    assert smb_return(walk(176)) > smb_return(walk(64)) + 0.15
    smb = contract.load('smb')
    assert abs(smb.maze_guide(15, 2047, 176, 0x61) - smb.GUIDE_W * 2) < 1e-3  # ready to pass ~ having passed
    assert smb.maze_guide(14, 500, 176, 0x61) is None                # not a maze level



def test_smb_44_pending_first_check_keeps_the_upper_route_target():
    smb = contract.load('smb')
    r = bytearray(smb_ram(15, 1040, 64))
    r[0x725], r[0x745] = 5, 1  # renderer is at the first check, but the ROM has not accepted it
    upper = smb.Reader().observe(r).guide
    r[0xCE] = 176
    lower = smb.Reader().observe(r).guide
    assert upper > lower + 0.2


def test_smb_44_pending_last_check_does_not_pay_full_credit_on_the_wrong_route():
    smb = contract.load('smb')
    r = bytearray(smb_ram(15, 2050, 64))
    r[0x725], r[0x745] = 9, 1
    wrong = smb.Reader().observe(r).guide
    r[0xCE] = 176
    right = smb.Reader().observe(r).guide
    assert right > wrong + 0.2
    r[0x745], r[0xCE] = 0, 64  # once the ROM accepts it, jumping must not lose completed-route credit
    assert smb.Reader().observe(r).guide == right == 2 * smb.GUIDE_W

def test_smb_maze_guide_cannot_be_farmed_by_looping():
    """7-4 loop: walk 260 -> 1280 on the floor, get sent back to 260, repeat. After the first lap set the high-water
    mark, a lap earns only its rollback and waiting costs: the guide gave back exactly what it paid."""
    from reward import idle_per_frame
    smb = contract.load('smb')
    tr = RewardTracker(smb)
    lap = [smb_ram(27, x, 176) for x in range(270, 1281, 10)] + [smb_ram(27, 260, 176)]
    tr.update(smb_ram(27, 260, 176), 0)
    laps = [sum(tr.update(r, t)[0] for t, r in enumerate(lap)) for _ in range(3)]
    want = W_ROLLBACK + idle_per_frame(smb) * 8
    assert abs(laps[1] - want) < 1e-12 and abs(laps[2] - want) < 1e-12


def test_smb_maze_guide_8_4_pipe_and_other_area():
    smb = contract.load('smb')
    on_pipe = smb.maze_guide(31, 288, 96, 0x65)
    assert on_pipe == smb.GUIDE_W                                    # standing on the first pipe
    assert smb.maze_guide(31, 1848, 144, 0x65) > on_pipe             # the pipe's exit is further on
    assert smb.maze_guide(31, 900, 176, 0x65, 0, 0) == 3 * smb.GUIDE_W  # the water part counts as all passed
    assert smb.maze_guide(31, 100, 176, 0x02) == 3 * smb.GUIDE_W    # another area of 8-4 counts as all passed


def test_smb_maze_guide_area_pointer_high_bit_does_not_complete_the_route():
    """8-4 uses $E5 as well as $65 before the floating pipe; both identify the same castle area."""
    smb = contract.load('smb')
    for level, (area, _) in smb.MAZE_GUIDE.items():
        for x, y in ((266, 88), (2404, 112), (2440, 176), (3652, 96)):
            assert smb.maze_guide(level, x, y, area | 0x80) == smb.maze_guide(level, x, y, area)
    assert smb.maze_guide(31, 2440, 64, 0xE5) > smb.maze_guide(31, 2440, 176, 0xE5) + 0.15
    assert smb_return([smb_ram(31, 266, 88, area=a) for a in (0x65, 0xE5)]) <= 0


def test_smb_maze_loop_is_not_profitable_under_the_training_discount():
    tr = RewardTracker(contract.load('smb'))
    tr.guide_discount = 0.995
    tr.update(smb_ram(31, 2608, 176), frames=4)
    tr.update(smb_ram(31, 1536, 176), frames=4)
    xs = list(range(1548, 2557, 12)) + [1536]
    rewards = [tr.update(smb_ram(31, x, 176), frames=4)[0] for x in xs]
    assert sum(tr.guide_discount ** t * r for t, r in enumerate(rewards)) < 0


def test_smb_guide_discount_applies_while_a_pipe_animation_holds_the_position():
    tr = RewardTracker(contract.load('smb'))
    tr.guide_discount = 0.995
    tr.update(smb_ram(31, 2448, 64), frames=4)
    r, done, _ = tr.update(smb_ram(31, 2448, 64, routine=7), frames=4)
    assert abs(r - (0.995 - 1) * 0.5) < 1e-12
    assert not done


def test_smb_terminal_win_refunds_the_remaining_guide_potential():
    tr = RewardTracker(contract.load('smb'))
    tr.update(smb_ram(31, 2448, 64), frames=4)
    won = bytearray(smb_ram(31, 2448, 64))
    won[0x770] = 2
    r, done, _ = tr.update(won, frames=4)
    assert r == W_WIN - 0.5
    assert done and tr.guide_prev == 0


def test_smb_maze_guide_7_4_wrong_group_pass_loses_its_credit_at_once():
    smb = contract.load('smb')
    right = smb.maze_guide(27, 1100, 64, 0x64, 0)
    assert smb.maze_guide(27, 1100, 64, 0x64, 1) == right - smb.GUIDE_W


def test_smb_maze_guide_8_4_block_then_floating_pipe():
    """Section 2 of 8-4: the block at x 2400 (stand y 112) leads up to the floating pipe (y 64); the floor does not."""
    smb = contract.load('smb')
    assert smb.maze_guide(31, 2404, 112, 0x65) > smb.maze_guide(31, 2404, 176, 0x65)
    assert smb.maze_guide(31, 2440, 64, 0x65) > smb.maze_guide(31, 2440, 176, 0x65) + 0.15
    assert smb.maze_guide(31, 2448, 64, 0x65) == 2 * smb.GUIDE_W


def test_smb_8_4_section_3_targets_the_real_pipe_not_the_trap():
    """x 3392 (stand y 80) sends Mario back to x 312; x 3648 (y 96) leads on. Floor there is y 128."""
    smb = contract.load('smb')
    assert smb.maze_guide(31, 3652, 96, 0x65) > smb.maze_guide(31, 3400, 80, 0x65)
    assert smb.maze_guide(31, 312, 144, 0x65) < smb.maze_guide(31, 3400, 80, 0x65) - 2 * smb.GUIDE_W  # taking the trap


def test_smb_8_4_progress_counts_the_water_part_and_after():
    reader = contract.load('smb').Reader()
    reader.reset()
    castle = reader.observe(smb_ram(31, 3650, 96)).progress
    water0 = reader.observe(smb_ram(31, 56, 144, atype=0)).progress
    water1 = reader.observe(smb_ram(31, 1090, 144, atype=0)).progress
    final = reader.observe(smb_ram(31, 4152, 144, area=0xE5)).progress
    assert castle < water0 < water1 < final
    assert reader.observe(smb_ram(15, 900, 176, atype=0)).progress == 900   # only 8-4 is remapped
