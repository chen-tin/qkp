import json
from dataclasses import asdict
import numpy as np
import pytest
import qutip as qt

from quantum.quality import (Physics, Gates, stage_gates, event_masks,
                             binomial_interval, zero_event_budget, stratified_events)
from quantum.reference_runner import simulate
from quantum.reference_noise import sample_noise
from quantum.qutip_reference import Oscillator, L, CARDINAL
from quantum.continuous_ancilla import ContinuousAncilla
from quantum.scoring import logical_fidelity


@pytest.fixture
def osc():
    return Oscillator(24, logical_bins=96)


@pytest.mark.parametrize("arm", ["finite_ancilla", "replayed_corrections"])
def test_instrumented_ideal_preserves_historical_evolution_and_logs_primitives(osc, arm):
    from experiments.qutip_reference import simulation
    params = (.1, .5/L, .3)
    old = simulation(osc, ContinuousAncilla(.5), params, "nearest", 2, 2, 42, arm)
    new = simulate(osc, ContinuousAncilla(.5), params, "nearest", 2, 2, 42, arm)
    for key in ("fidelity", "energy", "edge", "syndromes", "corrections", "dq", "dp"):
        np.testing.assert_allclose(new[key], old[key], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(new["stage_energy"][:, :, -1], new["energy"][:, 1:], atol=1e-11)
    np.testing.assert_allclose(new["stage_edge"][:, :, -1], new["edge"][:, 1:], atol=1e-11)
    assert "q_displacement" in new["stage_names"] and "p_displacement" in new["stage_names"]
    assert new["stage_norm_error"].max() < 1e-11
    assert new["snapshot_cycles"].tolist() == [0, 1, 2]
    assert all(f"snapshot_cycle_{t}" in new for t in range(3))


def test_transient_boundary_and_late_norm_failure_invalidate_correct_horizon():
    raw = {"edge": np.zeros((12, 3)), "stage_edge": np.zeros((12, 2, 4)),
           "stage_norm_error": np.zeros((12, 2, 4)), "max_norm_error": 0.}
    assert stage_gates(raw)["pass"]
    raw["stage_edge"][0, 1, 2] = .02
    assert stage_gates(raw)["valid_through_cycle"] == 1
    raw["stage_edge"][:] = 0
    raw["stage_norm_error"][0, 1, 2] = .01
    raw["max_norm_error"] = .01
    assert stage_gates(raw)["valid_through_cycle"] == 1
    assert not stage_gates({"edge": raw["edge"]})["stage_checks_complete"]


def test_energy_event_is_separate_from_numerical_gate():
    raw = {"energy": np.zeros((12, 3)), "edge": np.zeros((12, 3)),
           "stage_energy": np.zeros((12, 2, 4)), "stage_edge": np.zeros((12, 2, 4)),
           "stage_norm_error": np.zeros((12, 2, 4)), "max_norm_error": 0.}
    raw["stage_energy"][0, 0, 1] = 1200.
    assert event_masks(raw)["energy_jump"][0]
    assert stage_gates(raw)["pass"]


def test_streaming_logical_scores_agree_with_dense_and_independent_qutip(osc):
    rng = np.random.default_rng(3)
    psi = rng.normal(size=(osc.n, 6))+1j*rng.normal(size=(osc.n, 6))
    psi /= np.linalg.norm(psi, axis=0)
    state_ids = np.arange(6)
    result = logical_fidelity(psi, state_ids, bins=96, chunk=37)
    expected = osc.observables(psi)["bloch"]
    np.testing.assert_allclose(result["bloch"], expected, atol=2e-13)
    independent = logical_fidelity(psi, state_ids, bins=96, chunk=49, backend="qutip")
    np.testing.assert_allclose(independent["bloch"], result["bloch"], atol=2e-11)
    refined = logical_fidelity(psi, state_ids, bins=192, chunk=49)
    assert np.max(np.abs(refined["bloch"]-result["bloch"])) < 1e-4
    assert np.max(np.abs(result["trace"]-1)) < 1e-12
    assert result["minimum_eigenvalue"].min() >= -1e-12


def test_scoring_snapshot_refinement_detects_corrupted_original_score(tmp_path, osc):
    from experiments.refine_logical_scores import refine_raw
    raw = simulate(osc, ContinuousAncilla(.5), (.1, .5/L, .3), "nearest", 2, 2, 41, "finite_ancilla")
    path = tmp_path/"raw.npz";np.savez_compressed(path, **raw)
    report = refine_raw(path, 2, bins=[96, 192], chunk=37)
    assert report["all_cycle_snapshots_pass"]
    raw["fidelity"][:, 1] -= .01
    np.savez_compressed(path, **raw)
    report = refine_raw(path, 2, bins=[96, 192], chunk=37)
    assert not report["checks"][1]["pass"]
    # Legacy final-only files explicitly lack all-cycle certification.
    keep = {k:raw[k] for k in ["fidelity","score_snapshot","score_indices"]}
    np.savez_compressed(path, **keep)
    assert refine_raw(path, 2, bins=[96,192])["coverage"] == "selected_cycles_only"


def test_physics_contract_and_detector_variance():
    assert Physics(detector_efficiency=.8).detector_variance == pytest.approx(.125)
    with pytest.raises(ValueError):Physics(gate_duration=.6)
    with pytest.raises(ValueError):Physics(loss_rate=.1, displacement_provenance="loss_equivalent")
    with pytest.raises(ValueError):Physics(ancilla_loss_rate=.1)
    with pytest.raises(ValueError):Physics(detector_efficiency=0)
    with pytest.raises(ValueError):Physics(loss_rate=np.nan)
    assert Physics(gate_backend="qutip",ancilla_loss_rate=.1).record()["duration_model"] == "explicit_coupled_Lindblad_trajectory"


def test_loss_detector_and_feedback_do_not_change_exogenous_noise(osc):
    params = (.1, .5/L, .3)
    plain = simulate(osc, ContinuousAncilla(.5), params, "nearest", 2, 2, 72, "finite_ancilla")
    noisy = simulate(osc, ContinuousAncilla(.5), params, "nearest", 2, 2, 72, "finite_ancilla",
                     physics={"loss_rate":.01,"dephasing_rate":.001,"detector_efficiency":.8,
                              "feedback_delay":.05,"gate_duration":.1,"control_rotation":.002})
    np.testing.assert_array_equal(noisy["dq"], plain["dq"])
    np.testing.assert_array_equal(noisy["dp"], plain["dp"])
    assert not np.array_equal(noisy["reported_outcomes"], noisy["outcomes"])
    expected = (noisy["reported_outcomes"]+L/2) % L-L/2
    np.testing.assert_allclose(noisy["syndromes"], expected, atol=1e-12)
    assert noisy["max_norm_error"] < 1e-10
    assert np.all(np.isfinite(noisy["log_outcome_density"]))


def test_extended_noise_variance_correlation_and_schedule():
    q,p,_,_,_,_,schedule = sample_noise((.2,.05,.8),12000,5,17,{"cross_correlation":.6})
    assert np.std(q[:,0])/L == pytest.approx(.2,abs=.005)
    assert np.corrcoef(q[:,0],p[:,0])[0,1] == pytest.approx(.6,abs=.025)
    assert np.corrcoef(q[:,0],q[:,1])[0,1] == pytest.approx(.8,abs=.025)
    _,_,_,_,_,_,schedule = sample_noise((.2,.05,.8),12,6,18,{"schedule":{"family":"step","target":"rho","before":.8,"after":-.4}})
    assert schedule["rho"].min() < 0
    with pytest.raises(ValueError):sample_noise((.2,.05,.8),2,2,1,{"cross_correlation":1.1})


def test_dynamic_resource_width_runs_with_recorded_schedule(osc):
    raw = simulate(osc,ContinuousAncilla(.5),(.1,.5/L,.3),"nearest",2,3,42,"finite_ancilla",
                   noise_extension={"schedule":{"family":"step","target":"sigma_m","before":.5/L,"after":.4/L}})
    assert np.unique(raw["schedule_sigma_m"]).size == 2
    assert np.isfinite(raw["fidelity"]).all()


def test_rare_event_bounds_and_fixed_budget():
    zero=binomial_interval(0,3072)
    assert zero["one_sided_upper"] == pytest.approx(1-.05**(1/3072))
    assert zero["interval"][1] > 0
    needed=zero_event_budget(1e-4,comparisons=6)
    assert binomial_interval(0,needed,comparisons=6)["one_sided_upper"] <= 1e-4
    assert binomial_interval(0,needed-1,comparisons=6)["one_sided_upper"] > 1e-4
    mixed=stratified_events(np.zeros(120,bool),20,comparisons=3)
    assert len(mixed["states"])==6 and mixed["one_sided_upper"]>zero["one_sided_upper"]
    with pytest.raises(ValueError):binomial_interval(0,0)
    from experiments.quality_benchmark import build_plan
    with pytest.raises(ValueError,match="100,000"):
        build_plan("rare",per_state=2,rounds=1,decoders=["nearest"],target_probability=1e-12)


@pytest.mark.parametrize("axis",["q","p"])
def test_nonunit_coupling_against_independent_circuit(axis):
    from quantum.measurement_benchmark import ExplicitQutip,comparison,local_conditional
    osc=Oscillator(8,logical_bins=96)
    data=qt.coherent(8,.2+.1j).full().ravel()
    y=np.linspace(-12,12,601);weights=np.full(len(y),.04);weights[[0,-1]]/=2
    ref=ExplicitQutip(8,64,.5,y,coupling=1.13)
    result=comparison(ref.conditional(data,axis)[0],
                      local_conditional(osc,ContinuousAncilla(.5),data,y,axis,1.13),weights)[0]
    assert result["weighted_conditional_infidelity"]<1e-8


def test_explicit_finite_gate_noiseless_and_dissipative_paths():
    from quantum.finite_gate import FiniteGate
    from quantum.measurement_benchmark import ExplicitQutip
    data=qt.coherent(6,.1+.1j).full().ravel()
    p=Physics(gate_backend="qutip",gate_ancilla_cutoff=64,homodyne_radius=12,homodyne_dy=.04)
    gate=FiniteGate(6,.5,p)
    for axis in ("q","p"):
        actual=gate.evolve(data,axis,12)
        _,expected,_,_=ExplicitQutip(6,64,.5,np.array([0.])).conditional(data,axis)
        assert abs(np.vdot(actual,expected))**2>1-1e-8
        ket,y,lp,diag=gate.measure(data,axis,.53,12)
        assert np.linalg.norm(ket)==pytest.approx(1)
        assert abs(diag[2]-1)<1e-5
    p=Physics(gate_backend="qutip",gate_duration=.1,loss_rate=.01,ancilla_dephasing_rate=.001,
              gate_ancilla_cutoff=32,homodyne_radius=12,homodyne_dy=.04)
    ket,*_=FiniteGate(6,.5,p).measure(data,"q",.53,12)
    assert np.linalg.norm(ket)==pytest.approx(1)


def test_instrumented_explicit_backend_records_hidden_gate_diagnostics():
    osc=Oscillator(6,logical_bins=96)
    raw=simulate(osc,ContinuousAncilla(.5),(.01,.5/L,.1),"nearest",2,1,14,"finite_ancilla",
                 physics={"gate_backend":"qutip","gate_duration":.1,"loss_rate":.001,
                          "gate_ancilla_cutoff":32,"homodyne_radius":12,"homodyne_dy":.04})
    assert raw["gate_diagnostics"].shape==(24,3)
    assert np.isfinite(raw["reported_outcomes"]).all()
    assert raw["max_norm_error"]<1e-8
    assert np.isnan(raw["max_inverse_cdf_residual"])


def test_cube_coverage_and_negative_rho():
    from experiments.noise_heatmaps import points
    assert len(points())==61
    assert len(points("cube"))==125
    assert len(points("cube",{"sigma_q":[.2],"sigma_m":[.05],"rho":[-.5,0,.9]}))==3


def test_supplied_states_are_never_silently_truncated(tmp_path):
    from experiments.validate_reference_states import load_inputs,validate
    path=tmp_path/"input.npz";np.savez(path,score_snapshot=np.eye(17,dtype=complex)[:,[1,16]])
    data,provenance=load_inputs(path,cutoff=6)
    assert data.shape==(17,2) and provenance["sha256"]
    with pytest.raises(ValueError):
        validate(data,.5,[32,64],12,.04,max_joint=10)


def test_default_supplied_snapshot_inputs_cover_all_six_cardinal_labels(tmp_path):
    from experiments.validate_reference_states import load_inputs
    path=tmp_path/"bank.npz"
    np.savez(path,score_snapshot=np.tile(np.eye(12,dtype=complex)[:,[0]],(1,24)),
             score_indices=np.arange(24),fidelity=np.ones((24,2)))
    data,provenance=load_inputs(path)
    assert data.shape==(12,6)
    assert provenance["cardinal_input_ids"]==list(range(6))
    assert provenance["columns"]==[0,4,8,12,16,20]


def test_planned_coverage_rare_budget_and_hash_tampering(tmp_path):
    from experiments.quality_benchmark import build_plan,write_plan,load_plan
    plan=build_plan("coverage",per_state=2,rounds=1,decoders=["nearest"])
    assert len(plan["jobs"])==250
    rare=build_plan("rare",per_state=50,rounds=2,decoders=["nearest"],target_probability=.01)
    assert rare["fixed_budget"]["optional_stopping"] is False
    assert rare["fixed_budget"]["chunks"]*50 >= zero_event_budget(.01,comparisons=36)
    smoke=build_plan("smoke");write_plan(smoke,tmp_path)
    assert load_plan(tmp_path)==json.loads(json.dumps(smoke))
    saved=json.loads((tmp_path/"plan.json").read_text());saved["expected_jobs"]=0
    (tmp_path/"plan.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError):load_plan(tmp_path)


def test_calibration_recovers_effective_model_and_rejects_split_overlap(tmp_path):
    from experiments.calibrate_reference_noise import fit,evaluate,calibrate
    from simulator.noise import NoiseModel
    rng=np.random.default_rng(77);model=NoiseModel(.2*L,.05*L,.7)
    q,mq=model.sample(rng,500,40);p,mp=model.sample(rng,500,40)
    d,m=np.array([q,p]),np.array([mq,mp])
    fitted=fit(d,m)
    assert fitted["sigma_q_over_L"]==pytest.approx(.2,abs=.006)
    assert fitted["sigma_m_over_L"]==pytest.approx(.05,abs=.002)
    assert fitted["rho"]==pytest.approx(.7,abs=.02)
    assert evaluate(d,m,fitted)["standardized_innovations"]["variance"]==pytest.approx(1,abs=.05)
    path=tmp_path/"calibration.npz"
    np.savez(path,train_displacement=d,train_readout=m,test_displacement=d,test_readout=m,
             train_ids=np.arange(500),test_ids=np.arange(500))
    with pytest.raises(ValueError):calibrate(path)


def test_custom_matrix_is_frozen_and_invalid_paths_rejected():
    from experiments.quality_benchmark import build_plan
    matrix={"points":{"low":[.12,.08,-.3]},"variants":{"eta":{"physics":{"detector_efficiency":.9}}}}
    plan=build_plan("numerical",2,2,decoders=["nearest"],cutoffs=[24,32],matrix=matrix)
    assert len(plan["jobs"])==2
    assert plan["jobs"][0]["config"]["physics"]["detector_efficiency"]==.9
    matrix["points"]={"../escape":[.12,.08,-.3]}
    with pytest.raises(ValueError):
        build_plan("numerical",2,2,decoders=["nearest"],cutoffs=[24],matrix=matrix)


def test_end_to_end_resume_keeps_partial_and_numerical_status_separate(tmp_path):
    from experiments.quality_benchmark import build_plan,write_plan,run_plan
    plan=build_plan("smoke");write_plan(plan,tmp_path)
    first=run_plan(tmp_path,max_jobs=1)
    assert first["completed_jobs"]==1 and not first["all_jobs_executed"]
    second=run_plan(tmp_path)
    assert second["all_jobs_executed"] and second["completed_jobs"]==2
    assert second["classification"]=="infrastructure_smoke"
    assert not second["numerically_certified"]
    assert len(second["cutoff_equivalence"])==1
    assert len(second["scoring_refinement"])==2
    # A completed job is reused, and raw corruption cannot be accepted as cached success.
    assert run_plan(tmp_path)["completed_jobs"]==2
    raw=next((tmp_path/"jobs").glob("*/*/*_finite_ancilla.npz"))
    raw.write_bytes(b"corrupted")
    with pytest.raises(ValueError):run_plan(tmp_path)
