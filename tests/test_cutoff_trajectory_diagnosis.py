import numpy as np
import pytest

from experiments.diagnose_cutoff_trajectories import diagnose_pair


def fixture():
    return {"fidelity": np.ones((12, 2)), "dq": np.zeros((12, 1)), "dp": np.zeros((12, 1)),
            "energy": np.zeros((12, 2)), "edge": np.zeros((12, 2)),
            "stage_edge": np.zeros((12, 1, 2)), "stage_norm_error": np.zeros((12, 1, 2))}


def test_exact_pair_passes_without_inventing_sampling_uncertainty():
    report = diagnose_pair(fixture(), fixture(), 2)
    assert all(c["classification"] == "within_tolerance" and c["se"] == 0 for c in report["cycles"])
    assert report["low_stage_checks"]["pass"]


def test_inconclusive_interval_is_preserved_and_projection_is_planning_only():
    low, high = fixture(), fixture()
    high["fidelity"][:, 1] += np.tile([-.02, .025], 6)
    report = diagnose_pair(low, high, 2)
    assert report["cycles"][1]["classification"] == "inconclusive"
    assert report["cycles"][1]["same_mean_variance_projected_per_state"] > 2


def test_unmatched_draws_are_rejected():
    low, high = fixture(), fixture()
    high["dq"][0, 0] = .01
    with pytest.raises(AssertionError, match="Unmatched"):
        diagnose_pair(low, high, 2)


def test_failed_stage_is_retained_even_with_identical_scores():
    low, high = fixture(), fixture()
    high["stage_edge"][0, 0, 0] = .02
    report = diagnose_pair(low, high, 2)
    assert not report["high_stage_checks"]["pass"]
    assert all(c["classification"] == "within_tolerance" for c in report["cycles"])
