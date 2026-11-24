"""Memory-bounded instrument implementation controls; no production claim."""
import json
import numpy as np
import pytest
import qutip as qt

from experiments.validate_reference_states import validate, load_inputs, audit_report
from experiments.qutip_reference import sha
from quantum import streaming_measurement as streamed
from quantum.qutip_reference import Oscillator
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.measurement_benchmark import local_conditional


@pytest.mark.parametrize("cutoff", [6, 8])
@pytest.mark.parametrize("chunk", [7, 64])
@pytest.mark.parametrize("backend", ["sesolve","spectral"])
def test_streamed_metrics_match_dense_q_and_p(cutoff, chunk, backend, monkeypatch):
    data = qt.coherent(cutoff, .2+.1j).full()
    dense = validate(data, .5, [64, 96], 12., .04)
    monkeypatch.setattr(streamed.ExplicitQutip, "conditional", lambda *a, **kw: pytest.fail("full outcome matrix allocated"))
    original = streamed.qutip_wave_basis
    def bounded_basis(n, positions, *a, **kw):
        assert len(positions) <= chunk
        return original(n, positions, *a, **kw)
    monkeypatch.setattr(streamed, "qutip_wave_basis", bounded_basis)
    actual = streamed.validate_streamed(data, .5, [64, 96], 12., .04, chunk=chunk,circuit_backend=backend)
    key = lambda r:(r["axis"], r["column"], r["ancilla_cutoff"])
    for expected, observed in zip(sorted(dense["rows"],key=key),sorted(actual["rows"],key=key)):
        for metric in expected:
            if isinstance(expected[metric], (int,float)):
                assert observed[metric] == pytest.approx(expected[metric], abs=1e-10)
            else:
                assert observed[metric] == expected[metric]
    for expected, observed in zip(sorted(dense["refinements"],key=key),sorted(actual["refinements"],key=key)):
        for metric in expected:
            if isinstance(expected[metric], (int,float)):
                assert observed[metric] == pytest.approx(expected[metric], abs=1e-10)
            else:
                assert observed[metric] == expected[metric]
    assert actual["pass"] == dense["pass"]
    assert actual["streaming_capacity"]["chunk"] == chunk


@pytest.mark.parametrize("axis", ["q", "p"])
def test_lightweight_quadrature_matches_oscillator_without_logical_operators(axis):
    cutoff = 8
    state = qt.coherent(cutoff, .2+.1j).full().ravel()
    basis = streamed.QuadratureBasis(cutoff)
    assert not hasattr(basis, "logical_ops")
    assert not hasattr(basis, "code")
    y = np.linspace(-4,4,17)
    resource = ContinuousAncilla(.5)
    actual = basis.conditional(basis.coefficients(state,axis),resource,y,axis,1.13)
    expected = local_conditional(Oscillator(cutoff), resource,state,y,axis,1.13)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)


def test_streamed_full_hot_cutoff_preserved_and_budget_precedes_setup(tmp_path,monkeypatch):
    path=tmp_path/"hot.npz"
    np.savez_compressed(path,score_snapshot=np.eye(101,dtype=complex)[:,[100]])
    data, provenance=load_inputs(path,cutoff=6)
    assert data.shape==(101,1)
    assert provenance["mean_photon_number_per_input"]==[100.]
    monkeypatch.setattr(streamed,"QuadratureBasis",lambda *a:pytest.fail("budget checked after eigensystem"))
    with pytest.raises(ValueError,match="budget exceeded"):
        streamed.validate_streamed(data,.5,[64,96],12.,.04,max_joint=100)
    with pytest.raises(ValueError,match="budget exceeded"):
        streamed.validate_streamed(data,.5,[64,96],12.,.04,max_output_bytes=100)


def test_streamed_invalid_small_ancilla_control_stays_invalid():
    data=np.eye(8,dtype=complex)[:,[7]]
    dense=validate(data,.1,[8,12],8.,.04)
    actual=streamed.validate_streamed(data,.1,[8,12],8.,.04,chunk=7)
    assert not dense["pass"] and not actual["pass"]
    assert any(row["ancilla_boundary_mass"]>=1e-4 for row in actual["rows"])


@pytest.fixture
def streamed_report(tmp_path):
    data=qt.coherent(6,.2+.1j).full()
    result=streamed.validate_streamed(data,.5,[64,96],12.,.04,output=tmp_path,chunk=7)
    np.savez_compressed(tmp_path/"inputs.npz",states=data)
    result.update(complete=True,input_sha256=sha(tmp_path/"inputs.npz"),
                  config={"sigma":.5,"ancilla_cutoffs":[64,96],"radius":12.,"dy":.04,
                          "coupling":1.,"chunk":7,"source_hashes":{}})
    (tmp_path/"result.json").write_text(json.dumps(result))
    return tmp_path


def test_streamed_raw_audit_does_not_repeat_evolution(streamed_report,monkeypatch):
    monkeypatch.setattr(streamed.ExplicitQutip,"evolve",lambda *a:pytest.fail("audit repeated evolution"))
    assert audit_report(streamed_report)["pass"]


def test_streamed_raw_audit_rejects_changed_metric(streamed_report):
    path=streamed_report/"result.json"
    result=json.loads(path.read_text());result["rows"][-1]["grid_tv_change"] = .1
    path.write_text(json.dumps(result))
    with pytest.raises(AssertionError):
        audit_report(streamed_report)


def test_streaming_capacity_reduces_grid_scaling():
    coarse=streamed.capacity(4096,[768,1024],60.,.01,32)
    fine=streamed.capacity(4096,[768,1024],60.,.005,32)
    assert fine["outcome_nodes"]>coarse["outcome_nodes"]
    # Only 1D grid metadata grows: no Ndata*Ny conditional storage.
    assert fine["estimated_scratch_bytes"]-coarse["estimated_scratch_bytes"] == 128*(fine["outcome_nodes"]-coarse["outcome_nodes"])


def test_sealed_joint_resume_avoids_solver(streamed_report,monkeypatch):
    with np.load(streamed_report/'inputs.npz',allow_pickle=False) as raw:
        data=raw['states']
    monkeypatch.setattr(streamed.ExplicitQutip,'evolve',lambda *a,**kw:pytest.fail('sealed joint re-evolved'))
    result=streamed.validate_streamed(data,.5,[64,96],12.,.04,output=streamed_report,chunk=7)
    assert result['pass']


def test_changed_joint_checkpoint_is_rejected(streamed_report):
    path=streamed_report/'joint_Na64_q_input0.npy'
    data=np.load(path,allow_pickle=False);data[0,0]+=.1
    np.save(path,data)
    state=qt.coherent(6,.2+.1j).full()
    with pytest.raises(ValueError,match='checkpoint settings/source/hash changed'):
        streamed.validate_streamed(state,.5,[64,96],12.,.04,output=streamed_report,chunk=7)


def test_streamed_audit_rejects_relabelled_circuit_backend(streamed_report):
    path=streamed_report/'result.json'
    result=json.loads(path.read_text());result['circuit_backend']='spectral'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='backend differs'):
        audit_report(streamed_report)
