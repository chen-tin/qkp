"""Grid-streamed, full-Fock independent ideal-instrument validation.

Only a real data quadrature eigensystem, current/previous circuit joint kets,
small conditional blocks and one-dimensional outcome curves are retained.
The independent circuit still uses QuTiP evolution; this is a memory change,
not replacement of the circuit by the local Kraus updater.
"""
from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib
import numpy as np
from scipy.linalg import eigh_tridiagonal

from quantum.continuous_ancilla import ContinuousAncilla
from quantum.measurement_benchmark import (ExplicitQutip, outcome_grid, qutip_wave_basis,
                                           REFERENCE_WAVE_BASIS)


class QuadratureBasis:
    """Finite q/p eigensystems without code preparation or logical operators."""
    def __init__(self, cutoff):
        if not isinstance(cutoff, (int, np.integer)) or cutoff < 2:
            raise ValueError("Data cutoff must be an integer >= 2")
        self.n = int(cutoff)
        self.values, self.vectors = eigh_tridiagonal(
            np.zeros(self.n), np.sqrt(np.arange(1, self.n)/2), lapack_driver="stemr")
        self.phase = 1j**np.arange(self.n)

    def coefficients(self, state, axis):
        if axis not in ("q", "p"):
            raise ValueError("Quadrature axis must be q or p")
        return self.vectors.T@(state if axis == "q" else self.phase.conj()*state)

    def conditional(self, coefficients, resource, positions, axis, coupling):
        amplitudes = coefficients[:, None]*resource.amplitude(
            positions[None, :]-coupling*self.values[:, None])
        result = self.vectors@amplitudes
        return result if axis == "q" else self.phase[:, None]*result

    def joint_coefficients(self, joint, axis):
        return self.vectors.T@(joint if axis == "q" else self.phase.conj()[:, None]*joint)


def capacity(cutoff, ancilla_cutoffs, radius, dy, chunk, inputs=1, circuit_backend="sesolve"):
    """Conservative live-array/solver-work estimate; not a runtime/RSS bound."""
    if (not isinstance(chunk, (int, np.integer)) or chunk < 1 or
            len(ancilla_cutoffs) < 2 or sorted(set(ancilla_cutoffs)) != list(ancilla_cutoffs) or
            min(ancilla_cutoffs) < 2):
        raise ValueError("Positive chunk and two increasing ancillary cutoffs >= 2 required")
    y, _, _ = outcome_grid(radius, dy)
    na = max(ancilla_cutoffs)
    # Two real eigensystem work arrays, sparse Hamiltonians and ODE work,
    # current/previous joint states, complex grid chunks and 1D reductions.
    if circuit_backend=="sesolve":
        estimated = (16*cutoff*cutoff + 384*cutoff*na +
                     128*(cutoff+na)*min(chunk, len(y)) + 128*len(y) +
                     16*cutoff*inputs)
    elif circuit_backend=="spectral":
        estimated = (24*cutoff*cutoff + 16*na*na + 64*cutoff*na +
                     128*(cutoff+na)*min(chunk,len(y)) + 128*len(y) + 16*cutoff*inputs)
    else:
        raise ValueError("Circuit backend must be sesolve or spectral")
    return {"data_cutoff":int(cutoff), "max_ancilla_cutoff":int(na),
            "joint_dimension":int(cutoff*na), "outcome_nodes":len(y),
            "chunk":int(chunk), "estimated_scratch_bytes":int(estimated),
            "circuit_backend":circuit_backend,
            "scaling":"O(Ndata^2 + Nancilla^2 for spectral + Ndata*Nancilla + (Ndata+Nancilla)*chunk + Ny); no Ndata*Ny retained"}


def _project(joint, positions, axis):
    phi = qutip_wave_basis(joint.shape[1], positions)
    if axis == "p":
        phi = (1j**np.arange(joint.shape[1])).conj()[:, None]*phi
    return joint@phi


def _infidelity(reference, candidate):
    pr, pc = (np.abs(reference)**2).sum(0), (np.abs(candidate)**2).sum(0)
    overlap = np.abs((reference.conj()*candidate).sum(0))**2
    denominator = pr*pc
    fidelity = np.ones_like(pr)
    np.divide(overlap, denominator, out=fidelity, where=denominator > 1e-280)
    infidelity = np.clip(1-fidelity, 0, 1)
    infidelity[(pc == 0) & (pr > 0)] = 1
    return pr, pc, infidelity


def _metrics(pr, pc, infidelity, weights):
    mr, mc = float(pr@weights), float(pc@weights)
    if not np.isfinite([mr, mc]).all() or min(mr, mc) <= 0:
        raise FloatingPointError("Invalid streamed outcome probability mass")
    relevant = pr*weights/mr >= 1e-8
    return {"outcome_tv":.5*float(np.abs(pr/mr-pc/mc)@weights),
            "weighted_conditional_infidelity":float((pr*infidelity)@weights/mr),
            "worst_conditional_infidelity_relevant":float(infidelity[relevant].max()) if relevant.any() else 0.,
            "relevant_reference_probability_mass":float(pr[relevant]@weights[relevant]/mr),
            "reference_integrated_mass":mr, "candidate_integrated_mass":mc}


def curves(joint, state, basis, resource, y, axis, coupling, chunk, previous=None):
    """Reduce conditional vectors immediately; only 1D curves escape chunks."""
    values = {name:np.empty(len(y)) for name in
              ("reference_probability", "candidate_probability", "infidelity")}
    if previous is not None:
        values.update({name:np.empty(len(y)) for name in
                       ("previous_probability", "refinement_infidelity")})
    coefficients = basis.coefficients(state, axis)
    # Overlaps and probabilities are invariant under a unitary change of the
    # DATA basis. Transform each circuit joint ket once, avoiding an Ndata^2
    # transform for every outcome-grid block. Evolution remains independent.
    joint_basis = basis.joint_coefficients(joint, axis)
    previous_basis = None if previous is None else basis.joint_coefficients(previous, axis)
    for start in range(0, len(y), chunk):
        selection = slice(start, min(start+chunk, len(y)))
        positions = y[selection]
        reference = _project(joint_basis, positions, axis)
        candidate = coefficients[:, None]*resource.amplitude(
            positions[None, :]-coupling*basis.values[:, None])
        pr, pc, infidelity = _infidelity(reference, candidate)
        values["reference_probability"][selection] = pr
        values["candidate_probability"][selection] = pc
        values["infidelity"][selection] = infidelity
        if previous is not None:
            _, pp, delta = _infidelity(reference, _project(previous_basis, positions, axis))
            values["previous_probability"][selection] = pp
            values["refinement_infidelity"][selection] = delta
    return values


def _row(joint, arrays, w, coarse, na, axis, column):
    pr, pc, infidelity = (arrays[k] for k in
                          ("reference_probability", "candidate_probability", "infidelity"))
    metrics = _metrics(pr, pc, infidelity, w)
    rough = _metrics(pr, pc, infidelity, coarse)
    return {"ancilla_cutoff":na, "axis":axis, "column":column, **metrics,
            "joint_norm_error":float(abs(np.vdot(joint, joint).real-1)),
            "ancilla_boundary_mass":float((np.abs(joint[:, -max(4, joint.shape[1]//10):])**2).sum()),
            "grid_infidelity_change":abs(rough["weighted_conditional_infidelity"]-metrics["weighted_conditional_infidelity"]),
            "grid_tv_change":abs(rough["outcome_tv"]-metrics["outcome_tv"])}


def _refinement(arrays, w, na, axis, column):
    metrics = _metrics(arrays["reference_probability"], arrays["previous_probability"],
                       arrays["refinement_infidelity"], w)
    return {"ancilla_cutoff":na, "axis":axis, "column":column, **metrics,
            "pass":metrics["outcome_tv"] <= 1e-4 and metrics["weighted_conditional_infidelity"] <= 1e-5}


def validate_streamed(data, sigma, cutoffs, radius, dy, coupling=1.,
                      max_joint=2_000_000, max_output_bytes=512_000_000,
                      output=None, chunk=64, circuit_backend="sesolve"):
    from experiments.validate_reference_states import _checks_pass
    from experiments.qutip_reference import sha
    from experiments.qutip_reference import save
    data = np.asarray(data, complex)
    if (data.ndim != 2 or min(data.shape) < 1 or not np.isfinite(data).all() or
            not np.allclose((np.abs(data)**2).sum(0), 1, atol=1e-8, rtol=0)):
        raise ValueError("Complete normalized input kets required")
    if not np.isfinite(coupling) or coupling <= 0:
        raise ValueError("Positive finite SUM coupling required")
    resource = ContinuousAncilla(sigma)
    budget = capacity(len(data), cutoffs, radius, dy, chunk, data.shape[1], circuit_backend)
    if budget["joint_dimension"] > max_joint or budget["estimated_scratch_bytes"] > max_output_bytes:
        raise ValueError("Streamed explicit validation budget exceeded: "+json.dumps(budget))
    def progress(status, **detail):
        if output is not None:
            save(Path(output)/"progress.json", {"status":status,
                 "updated_utc":datetime.now(timezone.utc).isoformat(), **detail})
    progress("building_quadrature_basis", capacity=budget)
    basis = QuadratureBasis(len(data))
    progress("quadrature_basis_ready", capacity=budget)
    y, w, coarse = outcome_grid(radius, dy)
    rows, refinements, hashes = [], [], {}
    sources = {p:sha(Path(__file__).resolve().parents[1]/p) for p in
               ("quantum/streaming_measurement.py", "quantum/measurement_benchmark.py")}
    if circuit_backend=="spectral":
        sources["quantum/spectral_circuit.py"] = sha(Path(__file__).with_name("spectral_circuit.py"))
    if output is not None:
        Path(output).mkdir(parents=True, exist_ok=True)
    # Sequential inputs: retaining all prior input joint kets defeats streaming.
    for axis in ("q", "p"):
        for column in range(data.shape[1]):
            previous = None
            for na in cutoffs:
                checkpoint = None if output is None else Path(output)/f"joint_Na{na}_{axis}_input{column}.npy"
                checkpoint_config = {"data_cutoff":len(data), "ancilla_cutoff":na,
                    "axis":axis, "column":column, "sigma":sigma, "coupling":coupling,
                    "circuit_backend":circuit_backend, "reference_wave_basis":REFERENCE_WAVE_BASIS,
                    "input_state_sha256":hashlib.sha256(np.ascontiguousarray(data[:, column]).tobytes()).hexdigest(),
                    "source_sha256":sources}
                if checkpoint is not None and checkpoint.exists():
                    meta = json.loads(checkpoint.with_suffix(".json").read_text())
                    if meta["config"] != checkpoint_config or meta["sha256"] != sha(checkpoint):
                        raise ValueError("Joint checkpoint settings/source/hash changed")
                    joint = np.load(checkpoint,allow_pickle=False)
                    progress("reusing_sealed_circuit_joint", axis=axis, column=column, ancilla_cutoff=na)
                else:
                    progress("evolving_independent_circuit", axis=axis, column=column, ancilla_cutoff=na)
                    if circuit_backend=="sesolve":
                        ref = ExplicitQutip(len(data), na, sigma, np.empty(0), coupling=coupling, axes=(axis,))
                    else:
                        from quantum.spectral_circuit import SpectralCircuit
                        ref = SpectralCircuit(len(data),na,sigma,coupling=coupling,chunk=chunk)
                    joint = ref.evolve(data[:, column], axis)
                    del ref
                    if checkpoint is not None:
                        temporary = checkpoint.with_suffix(".tmp.npy")
                        np.save(temporary,joint);temporary.replace(checkpoint)
                        save(checkpoint.with_suffix(".json"),{"config":checkpoint_config,"sha256":sha(checkpoint)})
                if joint.shape != (len(data),na) or not np.isfinite(joint).all():
                    raise ValueError("Invalid full-dimension joint checkpoint")
                if checkpoint is not None:
                    hashes[checkpoint.name] = sha(checkpoint)
                    hashes[checkpoint.with_suffix(".json").name] = sha(checkpoint.with_suffix(".json"))
                progress("streaming_conditional_grid", axis=axis, column=column, ancilla_cutoff=na)
                arrays = curves(joint, data[:, column], basis, resource, y,
                                axis, coupling, chunk, previous)
                rows.append(_row(joint, arrays, w, coarse, na, axis, column))
                if previous is not None:
                    refinements.append(_refinement(arrays, w, na, axis, column))
                if output is not None:
                    path = Path(output)/f"Na{na}_{axis}_input{column}.npz"
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, y=y, weights=w, coarse_weights=coarse,
                                        joint_file=np.array(checkpoint.name), **arrays)
                    temporary.replace(path)
                    hashes[path.name] = sha(path)
                    save(Path(output)/"partial_report.json", {"complete":False,
                         "validation_backend":"streamed_instrument_v1", "rows":rows,
                         "refinements":refinements, "raw_hashes":hashes,
                         "streaming_capacity":budget,
                         "scope":"Partial grid-streamed single-instrument checks; no completed validation claim"})
                progress("conditional_check_saved", axis=axis, column=column, ancilla_cutoff=na,
                         completed_checks=len(rows))
                previous = joint
    return {"validation_backend":"streamed_instrument_v1", "circuit_backend":circuit_backend,
            "reference_wave_basis":REFERENCE_WAVE_BASIS, "rows":rows,
            "refinements":refinements, "raw_hashes":hashes,
            "pass":_checks_pass(rows, refinements), "streaming_capacity":budget,
            "scope":"single extraction on supplied full finite data operators; streamed outcome grid; common preparation/feedback/scoring"}


def audit_streamed_report(folder, result):
    """Recompute streamed metrics from saved joint states, without evolution."""
    from experiments.qutip_reference import sha
    from experiments.validate_reference_states import _checks_pass
    folder = Path(folder)
    config = result["config"]
    backend=config.get("circuit_backend","sesolve")
    if result.get("circuit_backend","sesolve") != backend:
        raise ValueError("Streamed report circuit backend differs from registered configuration")
    if result.get("reference_wave_basis") != config.get("reference_wave_basis",REFERENCE_WAVE_BASIS):
        raise ValueError("Streamed report reference basis differs from registered configuration")
    for name, digest in result["raw_hashes"].items():
        if sha(folder/name) != digest:
            raise ValueError("Streamed instrument raw hash mismatch")
    with np.load(folder/"inputs.npz", allow_pickle=False) as inputs:
        data = inputs["states"]
    basis, resource = QuadratureBasis(len(data)), ContinuousAncilla(config["sigma"])
    expected = {(na, axis, column) for na in config["ancilla_cutoffs"]
                for axis in ("q", "p") for column in range(data.shape[1])}
    actual = [(r["ancilla_cutoff"], r["axis"], r["column"]) for r in result["rows"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete streamed instrument row/refinement coverage")
    previous, selection, rows, refinements = None, None, [], []
    for row in sorted(result["rows"], key=lambda r:(r["axis"],r["column"],r["ancilla_cutoff"])):
        na, axis, column = row["ancilla_cutoff"], row["axis"], row["column"]
        if selection != (axis, column):
            previous, selection = None, (axis, column)
        name = f"Na{na}_{axis}_input{column}.npz"
        if sha(folder/name) != result["raw_hashes"][name]:
            raise ValueError("Streamed instrument raw hash mismatch")
        with np.load(folder/name, allow_pickle=False) as raw:
            joint_path=folder/str(raw["joint_file"])
            metadata=json.loads(joint_path.with_suffix(".json").read_text())
            expected_checkpoint={"data_cutoff":len(data),"ancilla_cutoff":na,"axis":axis,
                "column":column,"sigma":config["sigma"],"coupling":config["coupling"],
                "circuit_backend":backend,"reference_wave_basis":result["reference_wave_basis"],
                "input_state_sha256":hashlib.sha256(np.ascontiguousarray(data[:,column]).tobytes()).hexdigest()}
            if any(metadata["config"].get(k)!=v for k,v in expected_checkpoint.items()):
                raise ValueError("Streamed joint checkpoint provenance differs from report/input")
            for source,digest in metadata["config"]["source_sha256"].items():
                if source in config["source_hashes"] and config["source_hashes"][source]!=digest:
                    raise ValueError("Streamed joint source differs from report source snapshot")
            joint = np.load(joint_path,allow_pickle=False)
            if joint.shape != (len(data), na):
                raise ValueError("Streamed joint dimensions differ from full input cutoff")
            y, w, coarse = outcome_grid(config["radius"], config["dy"])
            for key, value in (("y",y),("weights",w),("coarse_weights",coarse)):
                np.testing.assert_array_equal(raw[key],value)
            arrays = curves(joint, data[:, column], basis, resource, y, axis,
                            config["coupling"], config["chunk"], previous)
            for key, value in arrays.items():
                np.testing.assert_allclose(raw[key],value,rtol=1e-10,atol=1e-13,err_msg=key)
        checked = _row(joint, arrays, w, coarse, na, axis, column)
        for key, value in checked.items():
            if isinstance(value, (float, int)):
                np.testing.assert_allclose(row[key],value,rtol=1e-10,atol=1e-13,err_msg=key)
            elif row[key] != value:
                raise ValueError("Changed streamed instrument metric "+key)
        rows.append(checked)
        if previous is not None:
            refinements.append(_refinement(arrays, w, na, axis, column))
        previous = joint
    ordered = lambda rs: sorted(rs,key=lambda r:(r["axis"],r["column"],r["ancilla_cutoff"]))
    if len(refinements) != len(result["refinements"]):
        raise ValueError("Incomplete streamed ancillary refinement evidence")
    for actual, expected in zip(ordered(result["refinements"]),ordered(refinements)):
        for key, value in expected.items():
            if isinstance(value, (float, int)) and not isinstance(value,bool):
                np.testing.assert_allclose(actual[key],value,rtol=1e-10,atol=1e-13,err_msg=key)
            elif actual[key] != value:
                raise ValueError("Changed streamed ancillary refinement "+key)
    if result["pass"] != _checks_pass(rows,refinements):
        raise ValueError("Streamed instrument pass flag disagrees with raw checks")
    for path,digest in config["source_hashes"].items():
        if sha(folder/"source_snapshot"/path) != digest:
            raise ValueError("Instrument source snapshot mismatch")
    return result
