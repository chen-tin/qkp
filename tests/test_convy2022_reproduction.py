import numpy as np
from scipy.linalg import expm

from experiments.reproduce_convy2022 import (MEANS, PaperLSTM, bayes_predict,
                                            fit_bayes, simulate, transition)


def test_transition_is_exact_poisson_generator():
    gamma, dt = .14, .032
    q = np.zeros((8, 8))
    for i in range(8):
        q[i, i] = -3 * gamma
        for mask in (1, 2, 4):
            q[i, i ^ mask] = gamma
    np.testing.assert_allclose(transition(gamma, dt), expm(q * dt), atol=1e-14)
    np.testing.assert_allclose(transition(0, dt), np.eye(8))


def test_simulator_stationary_noise_and_no_error_control():
    x, y, initial = simulate(800, 125, 0, .032, 4.7, 12)
    assert (y == initial[:, None]).all()
    assert np.bincount(initial).tolist() == [100] * 8
    residual = x - MEANS[y]
    assert abs(residual.mean()) < .025
    assert abs(residual.var() - 1 / (4.7 * .032)) < .08
    assert abs(np.mean(residual[:, 1:] * residual[:, :-1])) < .08
    fit = fit_bayes((x, y, initial))
    pred = bayes_predict((x, y, initial), fit)
    assert (pred == y).all()


def test_known_measurements_and_initial_state_are_necessary():
    # Single flips separated by many strong measurements must be identifiable.
    initial = np.arange(8, dtype=np.uint8)
    errors = np.repeat(np.array([0, 1, 3, 7], dtype=np.uint8), 10)
    y = initial[:, None] ^ errors[None, :]
    x = MEANS[y].copy()
    fit = {"transition": transition(.04, .032), "means": MEANS,
           "covs": np.tile(np.eye(2) * .01, (8, 1, 1))}
    pred = bayes_predict((x, y, initial), fit)
    assert (pred == y).all()
    wrong = bayes_predict((x, y, initial ^ 7), fit)
    assert not (wrong == y).any()


def test_architecture_matches_published_parameter_count():
    import torch
    model = PaperLSTM()
    assert sum(p.numel() for p in model.parameters()) == 13448
    torch.manual_seed(9)
    x = torch.randn(2, 10, 3)
    a = model(x)
    x[:, 6:] = torch.randn_like(x[:, 6:])
    b = model(x)
    torch.testing.assert_close(a[:, :6], b[:, :6])
    assert a.shape == (2, 10, 8)
