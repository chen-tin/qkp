"""Directed checks at independent instrument / finite-gate audit boundaries.

These tiny deterministic fixtures are software checks, not production-state
instrument validation or a finite-energy memory benchmark.
"""
import json
import numpy as np
import pytest

from quantum.quality import Physics, stage_gates
from quantum.finite_gate import FiniteGate
from experiments import validate_reference_states as validator


def test_finite_gate_rejects_invalid_quantiles_before_evolution(monkeypatch):
    gate = object.__new__(FiniteGate)
    monkeypatch.setattr(gate, "evolve", lambda *a: pytest.fail("invalid quantile reached evolution"))
    for quantile in (-.1, 0., 1., 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="quantile"):
            gate.measure(np.array([1., 0.]), "q", quantile, 1)


@pytest.fixture
def instrument_report(tmp_path):
    data = np.eye(6, dtype=complex)[:, [0]]
    result = validator.validate(data, .5, [64, 96], 12., .04, output=tmp_path)
    assert result["pass"]
    np.savez_compressed(tmp_path / "inputs.npz", states=data)
    result.update(complete=True, input_sha256=validator.sha(tmp_path / "inputs.npz"),
                  config={"source_hashes": {}})
    (tmp_path / "result.json").write_text(json.dumps(result))
    return tmp_path


@pytest.mark.parametrize("metric", ["ancilla_boundary_mass", "joint_norm_error",
                                    "grid_infidelity_change", "grid_tv_change"])
def test_audit_rejects_forged_numerical_gate_metric(instrument_report, metric):
    path = instrument_report / "result.json"
    result = json.loads(path.read_text())
    result["rows"][-1][metric] = .5
    result["pass"] = True
    path.write_text(json.dumps(result))
    with pytest.raises((ValueError, AssertionError)):
        validator.audit_report(instrument_report)


def test_audit_rejects_forged_pass(instrument_report):
    path = instrument_report / "result.json"
    result = json.loads(path.read_text())
    result["pass"] = False
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="pass"):
        validator.audit_report(instrument_report)


def test_audit_rejects_missing_refinement(instrument_report):
    path = instrument_report / "result.json"
    result = json.loads(path.read_text())
    result["refinements"] = []
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="refinement"):
        validator.audit_report(instrument_report)


def test_audit_rejects_missing_declared_cutoff(instrument_report):
    path = instrument_report / "result.json"
    result = json.loads(path.read_text())
    result["config"]["ancilla_cutoffs"] = [64, 96, 128]
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="declared ancillary"):
        validator.audit_report(instrument_report)


def test_audit_rechecks_valid_report(instrument_report):
    assert validator.audit_report(instrument_report)["pass"]


def test_full_cutoff_and_hot_state_are_preserved_and_preflighted(tmp_path, monkeypatch):
    path = tmp_path / "states.npz"
    np.savez_compressed(path, score_snapshot=np.eye(101, dtype=complex)[:, [100]])
    data, provenance = validator.load_inputs(path, cutoff=6)
    assert data.shape == (101, 1)
    assert provenance["mean_photon_number_per_input"] == [100.]
    monkeypatch.setattr(validator, "Oscillator", lambda *a, **kw: pytest.fail("preflight ran too late"))
    with pytest.raises(ValueError, match="budget exceeded"):
        validator.validate(data, .5, [64, 96], 12., .04, max_joint=100)


def test_invalid_explicit_gate_diagnostics_remain_invalid():
    raw = {"edge": np.zeros((2, 2)), "stage_edge": np.zeros((2, 1, 2)),
           "stage_norm_error": np.zeros((2, 1, 2)), "max_norm_error": 0.,
           "gate_diagnostics": np.array([[0., .02, 1.], [0., .02, 1.]])}
    result = stage_gates(raw)
    assert not result["pass"]
