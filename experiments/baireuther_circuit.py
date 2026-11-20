"""Batched Pauli-frame implementation of the pinned Baireuther 2018 circuit.

The circuit geometry and operation ordering come from the author's
circuit_model.py (cbdcd64). No phenomenological-noise replacement is used.
"""
from pathlib import Path
import numpy as np

REFERENCE = Path(__file__).resolve().parents[1] / 'reproductions/baireuther2018/reference/circuit_model.py'


def author_class():
    # The original executable's main() has deliberately blank path assignments.
    # Load only the library definitions, leaving the immutable snapshot intact.
    source = REFERENCE.read_text().split('\nif __name__')[0]
    scope = {'__name__': 'baireuther_reference'}
    exec(compile(source, str(REFERENCE), 'exec'), scope)
    return scope['SurfaceCode']


def geometry():
    ref = author_class()(seed=0, distance=3)
    data = {p:i for i,p in enumerate(ref.data_l)}
    anc = {p:i for i,p in enumerate(ref.anc_l)}
    gates = []
    for xd, zd in [('north','north'), ('west','east'), ('east','west'), ('south','south')]:
        pairs = []
        for a, d in getattr(ref, f'x_{xd}_dict').items():
            pairs.append((9 + anc[a], data[d]))
        for a, d in getattr(ref, f'z_{zd}_dict').items():
            pairs.append((data[d], 9 + anc[a]))
        gates.append(pairs)
    final_checks = [[data[d] for d in ref.z_anc_data_conn[a]] for a in sorted(ref.z_anc_l)]
    return gates, final_checks


GATES, FINAL_CHECKS = geometry()


def simulate(n, steps, seed, px=.00048, py=.00048, pz=.00048, pm=.0014,
             tape=None):
    """Return events, final increments, labels, raw ancilla measurements.

    `tape` optionally supplies uniform random numbers for exact reference tests.
    Layout per cycle: six lots of 17*3 random values (8 ancilla then 9 data),
    9 final-data readout draws, 8 ancilla readout draws, 9*3 data-idle draws.
    These draws follow the author's call order, not NumPy's global RNG stream.
    """
    rng = np.random.default_rng(seed)
    state = np.zeros((n, 17, 2), dtype=bool)
    raw = np.empty((n, steps, 8), dtype=bool)
    final = np.empty((n, steps, 4), dtype=bool)
    labels = np.empty((n, steps), dtype=bool)
    probs = np.array([px, py, pz])
    def errors(draws, indices):
        faults = draws < probs
        state[:, indices, 0] ^= faults[:, :, 0] ^ faults[:, :, 1]
        state[:, indices, 1] ^= faults[:, :, 2] ^ faults[:, :, 1]
    order = list(range(9, 17)) + list(range(9))
    for t in range(steps):
        r = tape[:, t] if tape is not None else rng.random((n, 350), dtype=np.float32)
        for step in range(6):
            if step in (0, 5):
                state[:, 9:13, :] = state[:, 9:13, ::-1]
            else:
                for c, target in GATES[step - 1]:
                    state[:, target, 0] ^= state[:, c, 0]
                    state[:, c, 1] ^= state[:, target, 1]
            errors(r[:, step*51:(step+1)*51].reshape(n, 17, 3), order)
        measured = state[:, :9, 0] ^ (r[:, 306:315] < pm)
        labels[:, t] = np.logical_xor.reduce(measured, axis=1)
        for j, qubits in enumerate(FINAL_CHECKS):
            final[:, t, j] = np.logical_xor.reduce(measured[:, qubits], axis=1)
        raw[:, t] = state[:, 9:17, 0] ^ (r[:, 315:323] < pm)
        state[:, 9:17, 1] = False
        errors(r[:, 323:350].reshape(n, 9, 3), list(range(9)))
    events = raw.copy()
    events[:, 2:] ^= raw[:, :-2]
    syndromes = raw.copy()
    syndromes[:, 1:] ^= raw[:, :-1]
    final ^= syndromes[:, :, 4:8]
    return events, final, labels, raw
