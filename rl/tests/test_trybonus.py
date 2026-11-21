import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trybonus import TryBonus, combo  # noqa: E402


def test_bonus_fades_per_place_and_combination_and_skips_explorers_and_no_place():
    tb = TryBonus(0.2, 'cpu')
    key = torch.tensor([5, 5, 7, -1, 5])
    learner = torch.tensor([True, True, True, True, False])
    down = [torch.tensor([5] * 5), torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.long),
            torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.long)]
    b = tb.step(key, down, learner)
    # the two learners at place 5 both see n = 0 (counted after), place 7 fresh, no place and the explorer pay 0
    assert torch.allclose(b, torch.tensor([0.2, 0.2, 0.2, 0.0, 0.0]))
    b = tb.step(key, down, learner)
    assert abs(float(b[0]) - 0.2 / 3 ** 0.5) < 1e-6 and abs(float(b[2]) - 0.2 / 2 ** 0.5) < 1e-6
    # another combination at the same place is fresh; Start/Select do not change the combination
    a_turbo = [p.clone() for p in down]
    a_turbo[1][:] = 2
    a_turbo[3][:] = 1
    assert abs(float(tb.step(key, a_turbo, learner)[0]) - 0.2) < 1e-6
    assert int(combo(a_turbo)[0]) == 5 * 9 + 2 * 3 + 0
    st = tb.stats()
    assert st['pairs'] == 4 and st['fresh'] == 6     # (5, down) (7, down) (5, A turbo) (7, A turbo)
