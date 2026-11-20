"""Explicit finite-duration SUM with data/ancilla loss and number dephasing.

QuTiP's Monte Carlo wavefunction solver samples hidden environmental records.
Homodyne CDF quadrature has explicit radius/spacing and a probability-mass gate.
This expensive backend is intended for bounded reference/convergence jobs.
"""
import math
import numpy as np
import qutip as qt
from scipy.integrate import cumulative_trapezoid
from quantum.measurement_benchmark import qutip_ancilla, qutip_wave_basis


class FiniteGate:
    def __init__(self, data_cutoff, sigma, physics):
        self.physics = physics
        nd, na = data_cutoff, physics.gate_ancilla_cutoff
        if nd*na > physics.max_joint_dimension:
            raise ValueError("Explicit finite-gate joint dimension exceeds configured budget")
        self.nd, self.na = nd, na
        nodes = math.ceil(2*physics.homodyne_radius/physics.homodyne_dy)+1
        if 8*na*nodes+48*nd*nodes+128*nd*na > physics.max_gate_scratch_mb*1e6:
            raise ValueError("Explicit gate scratch-memory budget exceeded")
        self.ancilla = qutip_ancilla(na, sigma)
        ad, aa = qt.destroy(nd), qt.destroy(na)
        qd, pd = (ad+ad.dag())/math.sqrt(2), -1j*(ad-ad.dag())/math.sqrt(2)
        qa, pa = (aa+aa.dag())/math.sqrt(2), -1j*(aa-aa.dag())/math.sqrt(2)
        # Evolve on [0,1]: Hamiltonian integrates to coupling; rates to rate*time.
        self.hamiltonians = {"q": physics.coupling*qt.tensor(qd, pa),
                             "p": -physics.coupling*qt.tensor(pd, qa)}
        self.collapse = []
        for rate, operator in ((physics.loss_rate, qt.tensor(ad, qt.qeye(na))),
                               (physics.dephasing_rate, qt.tensor(qt.num(nd), qt.qeye(na))),
                               (physics.ancilla_loss_rate, qt.tensor(qt.qeye(nd), aa)),
                               (physics.ancilla_dephasing_rate, qt.tensor(qt.qeye(nd), qt.num(na)))):
            if rate and physics.gate_duration:
                self.collapse.append(math.sqrt(rate*physics.gate_duration)*operator)
        self.y = np.linspace(-physics.homodyne_radius, physics.homodyne_radius,
                             math.ceil(2*physics.homodyne_radius/physics.homodyne_dy)+1)
        self.phi = qutip_wave_basis(na, self.y)
        self.phase = 1j**np.arange(na)

    def evolve(self, data, axis, seed):
        anc = self.ancilla if axis == "q" else self.ancilla*self.phase
        initial = qt.tensor(qt.Qobj(np.asarray(data)[:, None]), qt.Qobj(anc[:, None]))
        options = {"store_final_state": True, "store_states": False, "progress_bar": "",
                   "atol": 1e-11, "rtol": 1e-10, "nsteps": 100000}
        if self.collapse:
            result = qt.mcsolve(self.hamiltonians[axis], initial, [0., 1.],
                                self.collapse, ntraj=1, seeds=int(seed),
                                options={**options, "keep_runs_results": True})
            state = result.runs_final_states[0]
        else:
            state = qt.sesolve(self.hamiltonians[axis], initial, [0., 1.],
                              options={**options,"normalize_output":False}).final_state
        joint = state.full().reshape(self.nd, self.na)
        return joint

    def measure(self, data, axis, quantile, seed):
        joint = self.evolve(data, axis, seed)
        norm_error = float(abs(np.vdot(joint, joint).real-1))
        ancilla_edge = float((np.abs(joint[:, -max(4, self.na//10):])**2).sum())
        weighted = joint if axis == "q" else joint*self.phase.conj()[None, :]
        conditional = weighted@self.phi
        density = (np.abs(conditional)**2).sum(0)
        cdf = cumulative_trapezoid(density, self.y, initial=0.)
        mass = float(cdf[-1])
        if abs(mass-1) > 1e-5:
            raise FloatingPointError(f"Explicit homodyne grid mass {mass:g}: refine radius/dy")
        y = float(np.interp(quantile, cdf/mass, self.y))
        ket = weighted@qutip_wave_basis(self.na, np.array([y]))[:, 0]
        probability = float(np.vdot(ket, ket).real)
        if probability <= 0 or not np.isfinite(probability):
            raise FloatingPointError("Invalid explicit conditional homodyne density")
        return ket/math.sqrt(probability), y, math.log(probability), np.array([norm_error, ancilla_edge, mass])
