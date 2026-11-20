import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid
import qutip as qt

from quantum.continuous_ancilla import ContinuousAncilla
from quantum.measurement_benchmark import ExplicitQutip, SmoothMehler, comparison, local_conditional
from quantum.qutip_reference import L, Oscillator


@pytest.mark.parametrize("sigma", [.05*L, .16*L, .5])
def test_fast_amplitude_cdf_and_inverse_match_independent_quadrature(sigma):
    anc = ContinuousAncilla(sigma)
    x = np.linspace(-60, 60, 120001)
    psi = anc.amplitude(x)
    expected = SmoothMehler(sigma).amplitude(x)
    np.testing.assert_allclose(psi, expected, atol=1e-12, rtol=1e-12)
    assert abs(np.trapezoid(psi**2, x)-1) < 1e-12
    # Quadrature error is O(dx^2); this independently verifies the mixture
    # density rather than checking a CDF against its own inverse alone.
    integrated = cumulative_trapezoid(psi**2, x, initial=0.)
    assert np.max(np.abs(integrated-anc.cdf(x))) < 5e-6
    rng = np.random.default_rng(19)
    u = np.r_[np.geomspace(1e-12, .01, 30), rng.uniform(1e-12, 1-1e-12, 4096), 1-np.geomspace(1e-12, .01, 30)]
    samples = anc.sample(u)
    assert np.max(np.abs(anc.cdf(samples)-u)) < 1e-10
    assert np.all(np.diff(anc.sample(np.linspace(1e-10, 1-1e-10, 1000))) >= 0)


@pytest.mark.parametrize("axis", ["q", "p"])
def test_continuous_instrument_accuracy_against_explicit_qutip(axis):
    osc = Oscillator(12, logical_bins=96)
    data = qt.coherent(12, .3-.2j).full().ravel()
    y = np.linspace(-12, 12, 601)
    weights = np.full(len(y), y[1]-y[0]);weights[[0, -1]] *= .5
    ref = ExplicitQutip(12, 64, .5, y).conditional(data, axis)[0]
    actual = local_conditional(osc, ContinuousAncilla(.5), data, y, axis)
    metrics, _ = comparison(ref, actual, weights)
    assert 1-metrics["weighted_conditional_infidelity"] > .99999
    assert metrics["outcome_tv"] < 1e-4


def test_bad_sampling_quantiles_rejected():
    anc = ContinuousAncilla(.2)
    for value in [0., 1., -1., np.nan]:
        with pytest.raises(ValueError):anc.sample(np.array([value]))


def test_stage_logging_reproduces_historical_binned_simulation():
    from experiments.smooth_ancilla_experiment import simulate
    from experiments.qutip_reference import simulation
    from quantum.qutip_reference import Ancilla
    osc = Oscillator(48, logical_bins=192)
    params = (.1, .5/L, .3)
    expected = simulation(osc, Ancilla(.5), params, "nearest", 2, 2, 42, "finite_ancilla")
    actual = simulate(osc, Ancilla(.5), params, 2, 2, 42)
    for key in ("fidelity", "energy", "edge", "syndromes", "corrections"):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(actual["stage_energy"][:, :, -1], actual["energy"][:, 1:], atol=1e-10)
    np.testing.assert_allclose(actual["stage_edge"][:, :, -1], actual["edge"][:, 1:], atol=1e-10)
    assert np.isfinite(actual["outcomes"]).all() and np.isfinite(actual["log_outcome_density"]).all()
