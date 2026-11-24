"""Directed interface checks; tiny bases are not physical accuracy evidence."""
import numpy as np
import pytest

from quantum.continuous_ancilla import ContinuousAncilla
from quantum.qutip_reference import L, Oscillator
from quantum.reference_noise import sample_noise
from quantum.reference_runner import simulate
from simulator.sim import wrap


@pytest.mark.parametrize("arm", ["finite_ancilla", "replayed_corrections"])
def test_controller_receives_only_reported_syndrome(monkeypatch, arm):
    import experiments.qutip_reference as adapter

    controllers = []

    class SpyController:
        def __init__(self):
            self.seen = []

        def decode(self, syndrome):
            assert syndrome.shape == (12,)
            self.seen.append(syndrome.copy())
            return syndrome.copy()

    def controller_factory(name, params, trials):
        assert trials == 12
        controller = SpyController()
        controllers.append(controller)
        return controller

    monkeypatch.setattr(adapter, "decoder", controller_factory)
    raw = simulate(
        Oscillator(24, logical_bins=96), ContinuousAncilla(.5),
        (.08, .5 / L, .3), "spy", 2, 2, 26112301, arm,
        physics={"coupling": 1.07, "detector_efficiency": .7,
                 "loss_rate": .01, "dephasing_rate": .001},
        snapshots_per_state=2, hot_per_state=1,
    )
    assert len(controllers) == 2
    for axis, controller in enumerate(controllers):
        np.testing.assert_array_equal(np.stack(controller.seen, axis=1), raw["syndromes"][axis])
    if arm == "finite_ancilla":
        np.testing.assert_allclose(raw["syndromes"], wrap(raw["reported_outcomes"] / 1.07))
        assert not np.array_equal(raw["reported_outcomes"], raw["outcomes"])
    else:
        assert np.isnan(raw["outcomes"]).all()
        np.testing.assert_allclose(raw["syndromes"], wrap(raw["reported_outcomes"]))


def test_optional_channel_seed_streams_leave_displacements_unchanged():
    osc = Oscillator(24, logical_bins=96)
    kwargs = dict(osc=osc, ancilla=ContinuousAncilla(.5), params=(.08, .5 / L, .3),
                  name="nearest", per_state=2, rounds=1, seed=26112302,
                  arm="finite_ancilla", snapshots_per_state=2, hot_per_state=0)
    baseline = simulate(**kwargs)
    physical = simulate(**kwargs, physics={"detector_efficiency": .7, "loss_rate": .01,
                                         "dephasing_rate": .001, "feedback_delay": .05})
    for key in ("dq", "dp", "nq", "np"):
        np.testing.assert_array_equal(baseline[key], physical[key])


def test_empty_noise_extension_preserves_registered_draws():
    a = sample_noise((.2, .05, .9), 12, 3, 26112303)
    b = sample_noise((.2, .05, .9), 12, 3, 26112303, {})
    for left, right in zip(a[:-1], b[:-1]):
        np.testing.assert_array_equal(left, right)
    assert a[-1] == b[-1] == {}


@pytest.mark.parametrize("extension", [
    {"cross_correlation": float("nan")},
    {"cross_correlation": 1.01},
    {"family": "student_t", "student_df": 2},
    {"schedule": {"family": "step", "target": "rho", "before": .3, "after": .999}},
])
def test_invalid_or_silently_clipped_noise_is_rejected(extension):
    with pytest.raises(ValueError):
        sample_noise((.2, .05, .3), 12, 4, 26112304, extension)
