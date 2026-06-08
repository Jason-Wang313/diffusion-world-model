import numpy as np

from dwm_best_of_n.theory import (
    binary_best_of_n_law,
    finite_best_of_n_law,
    finite_best_of_n_curve,
    monte_carlo_best_of_n,
    real_valued_best_of_n_law,
)


def test_finite_law_without_ties():
    law = finite_best_of_n_law([3, 2, 1], [1, 0, 0], 2)
    assert abs(law.expected_utility - (1 - (2 / 3) ** 2)) < 1e-12


def test_tie_handling_uniform_top_group_mean():
    law = finite_best_of_n_law([1, 1, 0], [0, 1, 0], 3)
    assert abs(law.expected_utility - (13 / 27)) < 1e-12


def test_binary_and_real_law_paths():
    binary = binary_best_of_n_law([0, 1, 2], [0, 0, 1], 4)
    real = real_valued_best_of_n_law([0, 1, 2], [0.2, -0.1, 0.7], 4)
    assert 0.0 <= binary.expected_utility <= 1.0
    assert real.expected_utility > -0.1


def test_constant_utility_edge_case():
    curve = finite_best_of_n_curve([0, 1, 2, 3], [0.4, 0.4, 0.4, 0.4], [1, 8, 64])
    assert np.allclose(curve["expected_utility"], 0.4)


def test_oracle_and_anti_aligned_examples():
    ns = [1, 2, 4, 8]
    oracle = finite_best_of_n_curve([0, 1, 2, 3], [0, 1, 2, 3], ns)
    anti = finite_best_of_n_curve([3, 2, 1, 0], [0, 1, 2, 3], ns)
    assert oracle["expected_utility"].is_monotonic_increasing
    assert anti["expected_utility"].is_monotonic_decreasing


def test_monte_carlo_validation_within_tolerance():
    scores = [3, 3, 2, 1, 0]
    utilities = [0.2, 1.0, -0.1, 0.4, -0.3]
    law = finite_best_of_n_law(scores, utilities, 8)
    mc = monte_carlo_best_of_n(scores, utilities, 8, trials=15000, seed=11)
    assert abs(law.expected_utility - mc["mc_expected_utility"]) < 0.025
