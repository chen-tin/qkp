"""Training data for the LSTM decoders.

Inputs are syndrome histories; labels are the nearest-lattice jump parities
y_t, taken from a NearestLattice run of the simulator on the same noise.
These labels use the true errors, which is allowed for training (proposal,
Sec. 5). They never depend on the decoder being trained, because with
corrections c_t = s_t + j_t L the syndromes are the same for every decoder.
"""

import numpy as np

from experiments.common import BROAD_BOX, NOMINAL, ROUNDS, make_rng
from simulator.noise import L, NoiseModel
from simulator.sim import NearestLattice, Simulator
from simulator.switching import SwitchingNoiseModel, sample_varied

# training distributions
DATASETS = {
    "nominal": dict(kind="fixed", **NOMINAL),
    "broad": dict(kind="broad", **{k: list(v) for k, v in BROAD_BOX.items()}),
    "matched_rho0.5": dict(kind="fixed", sigma_q=NOMINAL["sigma_q"],
                           sigma_m=NOMINAL["sigma_m"], rho=0.5),
    "matched_sq0.26": dict(kind="fixed", sigma_q=0.26 * L,
                           sigma_m=NOMINAL["sigma_m"], rho=NOMINAL["rho"]),
    "switching": dict(kind="switching", sigma_q=NOMINAL["sigma_q"],
                      sigma_m=NOMINAL["sigma_m"], rho_lo=0.3, rho_hi=0.97,
                      p_switch=0.05),
}


def sample_noise(spec, rng, trials, rounds=ROUNDS):
    """Noise arrays (and per-trial parameters for the broad set)."""
    kind = spec["kind"]
    if kind == "fixed":
        model = NoiseModel(spec["sigma_q"], spec["sigma_m"], spec["rho"])
        eps, nu = model.sample(rng, trials, rounds)
        params = None
    elif kind == "broad":
        rho = rng.uniform(*spec["rho"], size=trials)
        sq = rng.uniform(*spec["sigma_q"], size=trials)
        sm = rng.uniform(*spec["sigma_m"], size=trials)
        eps, nu = sample_varied(rng, rho, sq, sm, rounds)
        params = np.stack([rho, sq, sm], axis=1)
    elif kind == "switching":
        model = SwitchingNoiseModel(
            spec["sigma_q"], spec["sigma_m"], spec["rho_lo"], spec["rho_hi"],
            spec["p_switch"],
        )
        eps, nu = model.sample(rng, trials, rounds)
        params = None
    else:
        raise ValueError(kind)
    return eps, nu, params


def make_dataset(name, split, trials, seed=0, rounds=ROUNDS):
    """Return dict(syndromes, y, params) for one training distribution."""
    spec = DATASETS[name]
    rng = make_rng(split, f"lstm_data_{name}", point=seed)
    eps, nu, params = sample_noise(spec, rng, trials, rounds)
    result = Simulator(trials=trials, rounds=rounds).run(NearestLattice(), eps, nu)
    y = (result.lattice_jumps() % 2).astype(np.int8)
    return dict(syndromes=result.syndromes.astype(np.float32), y=y, params=params)
