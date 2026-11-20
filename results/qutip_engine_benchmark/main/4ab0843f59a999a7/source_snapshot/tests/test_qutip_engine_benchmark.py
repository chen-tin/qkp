import numpy as np
import pytest
import qutip as qt
from qutip.distributions import HarmonicOscillatorWaveFunction

from quantum.measurement_benchmark import (
    ExplicitQutip, SmoothMehler, comparison, local_conditional, qutip_wave_basis,
    outcome_grid,
)
from quantum.qutip_reference import Ancilla, L, Oscillator
from simulator.sim import wrap


def weights(y):
    result = np.full(len(y), y[1]-y[0]);result[[0, -1]] *= .5
    return result


def test_nested_grid_resolves_both_sides_of_feedback_jumps():
    y, fine, coarse = outcome_grid(18, .02)
    assert np.all(np.diff(y) > 0)
    assert abs(fine.sum()-36) < 1e-12
    assert abs(coarse.sum()-36) < 1e-12
    left = np.flatnonzero(y < L/2)[-1]
    assert abs(y[left]-L/2) < 1e-10 and abs(y[left+1]-L/2) < 1e-10
    assert wrap(y[left]) > .49*L and wrap(y[left+1]) < -.49*L
    assert fine[left] > 0 and fine[left+1] > 0 and coarse[left] > 0 and coarse[left+1] > 0


def test_qutip_projection_kernel_matches_public_wavefunction():
    y = np.linspace(-7, 7, 301)
    psi = qt.coherent(12, .4+.2j)
    expected = HarmonicOscillatorWaveFunction(psi, extent=[-7, 7], steps=301).data
    np.testing.assert_allclose(psi.full().ravel()@qutip_wave_basis(12, y), expected, atol=1e-14)


@pytest.mark.parametrize("sigma", [.05*L, .16*L, .5])
def test_smooth_resource_analytic_normalization_and_grid_limit(sigma):
    smooth = SmoothMehler(sigma)
    y = np.linspace(-60, 60, 40001)
    assert abs(np.trapezoid(smooth.amplitude(y)**2, y)-1) < 1e-12
    binned = Ancilla(sigma, dx=1e-5)
    np.testing.assert_allclose(binned.amplitude(y[::101]), smooth.amplitude(y[::101]), atol=5e-5)


@pytest.mark.parametrize("axis", ["q", "p"])
def test_explicit_circuit_matches_smooth_reduction_and_norm(axis):
    osc = Oscillator(12, logical_bins=96)
    data = qt.coherent(12, .25+.15j).full().ravel()
    y = np.linspace(-12, 12, 601)
    ref = ExplicitQutip(12, 64, .5, y)
    explicit, _, norm_error, edge = ref.conditional(data, axis)
    local = local_conditional(osc, SmoothMehler(.5), data, y, axis)
    metrics, _ = comparison(explicit, local, weights(y))
    assert norm_error < 1e-8 and edge < 1e-8
    assert abs(metrics["reference_integrated_mass"]-1) < 1e-8
    assert metrics["outcome_tv"] < 1e-6
    assert metrics["weighted_conditional_infidelity"] < 1e-8


def test_wrong_sum_sign_is_detected():
    osc = Oscillator(12, logical_bins=96)
    data = qt.coherent(12, .5+.2j).full().ravel()
    y = np.linspace(-12, 12, 601)
    wrong = ExplicitQutip(12, 64, .5, y, wrong_sign=True).conditional(data, "q")[0]
    local = local_conditional(osc, SmoothMehler(.5), data, y, "q")
    metrics, _ = comparison(wrong, local, weights(y))
    assert metrics["outcome_tv"] > .05
    assert metrics["weighted_conditional_infidelity"] > .01


def test_conditional_comparison_is_global_phase_invariant():
    rng = np.random.default_rng(7)
    state = rng.normal(size=(5, 20))+1j*rng.normal(size=(5, 20))
    metrics, _ = comparison(state, state*np.exp(.7j), np.ones(20))
    assert metrics["outcome_tv"] < 1e-15
    assert metrics["weighted_conditional_infidelity"] < 1e-14
