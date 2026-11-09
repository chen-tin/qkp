"""Classical GKP error-correction simulator (q quadrature only).

Each round t of a trial:
    1. a displacement error is added:     x <- x + eps_t
    2. a noisy syndrome is measured:      s_t = wrap(x + nu_t)
    3. the decoder picks a correction:    c_t = decoder.decode(s_t)
    4. the correction is applied:         x <- x - c_t

x is the residual displacement of the oscillator from the code lattice.
The syndrome only reveals x modulo L = sqrt(pi), so the decoder has to
guess which lattice cell the error came from. A shift by L is a logical
X, so the logical state is flipped when the residual is closest to an
odd multiple of L.

The simulation samples classical shifts and never builds the quantum state.
For pure displacement noise on an ideal GKP code this matches the
quantum model.

Run from the repo root:
    python -m simulator.sim
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from simulator.noise import L, NoiseModel


def wrap(x):
    """Map x into [-L/2, L/2). This is the syndrome of a shift x."""
    return (x + L / 2) % L - L / 2


def lattice_index(x):
    """Index k of the nearest lattice point k*L."""
    return np.rint(x / L).astype(np.int64)


def logical_flip(x):
    """True where the residual x is closest to an odd multiple of L."""
    return lattice_index(x) % 2 == 1


# decoders
class Decoder:
    """Interface every decoder implements.

    The simulator calls reset() once per run. It then calls decode() once
    per round with that round's syndromes for all trials, shape (trials,).
    decode() returns the correction for each trial, shape (trials,).

    Decoders that use memory (Bayesian, LSTM, DQN) keep their own history
    of past syndromes and corrections between calls. They never see the
    true errors.
    """

    name = "decoder"

    def reset(self, trials):
        pass

    def decode(self, syndrome):
        raise NotImplementedError


class NoCorrection(Decoder):
    """Never corrects. Reference for 'no error correction'."""

    name = "No correction"

    def decode(self, syndrome):
        return np.zeros_like(syndrome)


class NearestLattice(Decoder):
    """Memoryless baseline: assume the error is the smallest shift
    consistent with the current syndrome."""

    name = "Nearest lattice"

    def decode(self, syndrome):
        return syndrome.copy()


# results
@dataclass
class SimResult:
    """Everything recorded during one run. Arrays are (trials, rounds).

    syndromes and corrections are what a decoder can see. The rest is the
    ground truth, which can be used for training labels and diagnostics
    but must never be given to a decoder at test time.
    """

    decoder_name: str
    syndromes: np.ndarray            # s_t, what the decoder sees
    corrections: np.ndarray          # c_t, what the decoder applied
    residuals: np.ndarray            # x after correction in round t
    displacement_noise: np.ndarray   # eps_t (ground truth)
    measurement_noise: np.ndarray    # nu_t (ground truth)
    logical_errors: np.ndarray = field(init=False)

    def __post_init__(self):
        # logical state if we stopped after round t and read out ideally
        self.logical_errors = logical_flip(self.residuals)

    @property
    def trials(self):
        return self.syndromes.shape[0]

    @property
    def rounds(self):
        return self.syndromes.shape[1]

    def logical_error_rate(self):
        """Fraction of trials in a logical error after each round, (rounds,)."""
        return self.logical_errors.mean(axis=0)

    def final_logical_error_rate(self):
        return self.logical_errors[:, -1].mean()

    def standard_error(self):
        """Binomial standard error of the logical error rate, (rounds,)."""
        p = self.logical_error_rate()
        return np.sqrt(p * (1 - p) / self.trials)

    def lattice_jumps(self):
        """Change in lattice index each round, (trials, rounds).

        An odd jump flips the logical state. The first round is measured
        from the starting point x = 0.
        """
        k = lattice_index(self.residuals)
        return np.diff(k, axis=1, prepend=0)


# simulator
@dataclass
class Simulator:
    """Runs decoders on correlated displacement noise.

    Generate the noise once with sample_noise() and give the same arrays
    to every decoder, so all decoders face the same errors.
    """

    noise_model: NoiseModel = field(default_factory=NoiseModel)
    trials: int = 10000
    rounds: int = 50

    def __post_init__(self):
        for value in (self.trials, self.rounds):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("trials and rounds must be positive integers")

    def sample_noise(self, seed):
        """Return (displacement_noise, measurement_noise), each (trials, rounds)."""
        rng = np.random.default_rng(seed)
        return self.noise_model.sample(rng, self.trials, self.rounds)

    def run(self, decoder, displacement_noise, measurement_noise):
        """Run one decoder on pre-generated noise and record everything."""
        shape = (self.trials, self.rounds)
        if displacement_noise.shape != shape or measurement_noise.shape != shape:
            raise ValueError(f"noise arrays must have shape {shape}")

        syndromes = np.zeros(shape)
        corrections = np.zeros(shape)
        residuals = np.zeros(shape)

        # start in the code space with no displacement
        x = np.zeros(self.trials)
        decoder.reset(self.trials)

        for t in range(self.rounds):
            # displacement error
            x = x + displacement_noise[:, t]

            # noisy syndrome measurement
            syndromes[:, t] = wrap(x + measurement_noise[:, t])

            # decoder sees only the syndrome
            correction = np.asarray(decoder.decode(syndromes[:, t].copy()))
            if correction.shape != (self.trials,):
                raise ValueError(
                    f"{decoder.name} returned shape {correction.shape}, "
                    f"expected ({self.trials},)"
                )
            if not np.all(np.isfinite(correction)):
                raise ValueError(f"{decoder.name} returned nonfinite values")

            corrections[:, t] = correction
            x = x - correction
            residuals[:, t] = x

        return SimResult(
            decoder_name=decoder.name,
            syndromes=syndromes,
            corrections=corrections,
            residuals=residuals,
            displacement_noise=displacement_noise,
            measurement_noise=measurement_noise,
        )

    def run_all(self, decoders, seed):
        """Run several decoders on the same noise. Returns {name: SimResult}."""
        displacement_noise, measurement_noise = self.sample_noise(seed)
        return {
            decoder.name: self.run(decoder, displacement_noise, measurement_noise)
            for decoder in decoders
        }


# checks
def odd_jump_probability(sigma):
    """P(round(z/L) is odd) for z ~ N(0, sigma^2)."""
    if sigma == 0:
        return 0.0
    j = np.arange(-49, 50, 2)  # odd lattice indices
    return float(np.sum(
        norm.cdf((j + 0.5) * L / sigma) - norm.cdf((j - 0.5) * L / sigma)
    ))


def check_nearest_lattice(result, noise_model):
    """Compare the simulated nearest-lattice flip rate with the exact value.

    After a nearest-lattice correction the residual is k*L - nu_t, so the
    jump in round t is round((eps_t + nu_t - nu_{t-1}) / L). Its variance
    is sigma_q^2 + 2 sigma_m^2 for any rho, because each eps_t on its own
    has variance sigma_q^2. In round 1 there is no nu_0, so the variance is
    sigma_q^2 + sigma_m^2.
    """
    jumps = result.lattice_jumps()
    odd = jumps % 2 == 1

    sigma_first = np.hypot(noise_model.sigma_q, noise_model.sigma_m)
    sigma_later = np.sqrt(noise_model.sigma_q**2 + 2 * noise_model.sigma_m**2)

    print(
        f"Round-1 flip rate: {odd[:, 0].mean():.5f}"
        f" | expected: {odd_jump_probability(sigma_first):.5f}"
    )
    if result.rounds > 1:
        print(
            f"Later-round flip rate: {odd[:, 1:].mean():.5f}"
            f" | expected: {odd_jump_probability(sigma_later):.5f}"
        )


def check_syndromes(result):
    """The syndromes should lie in [-L/2, L/2)."""
    s = result.syndromes
    inside = np.all((s >= -L / 2) & (s < L / 2))
    print(
        f"Syndromes inside [-L/2, L/2): {inside}"
        f" | min/L: {s.min()/L:.4f}, max/L: {s.max()/L:.4f}"
    )


# plot
@dataclass
class ResultPlotter:
    """Plots logical error rate against round for several decoders."""

    figsize: tuple = (9, 5.5)
    fontsize: int = 12
    linewidth: float = 1.8
    dpi: int = 300

    def plot(self, results, title=None, show=True, save_path=None):
        if not show and save_path is None:
            return

        plt.rcParams.update({"font.size": self.fontsize})
        fig, ax = plt.subplots(figsize=self.figsize)

        for name, result in results.items():
            t = np.arange(1, result.rounds + 1)
            rate = result.logical_error_rate()
            error = result.standard_error()
            ax.plot(t, rate, lw=self.linewidth, label=name)
            ax.fill_between(t, rate - error, rate + error, alpha=0.25)

        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("Round")
        ax.set_ylabel("Logical error rate")
        ax.set_ylim(0, 0.55)
        ax.legend()
        if title is not None:
            ax.set_title(title)

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


# run this part only when running this file
if __name__ == "__main__":
    trials = 10000
    rounds = 50
    seed = 42

    # controls
    check = True
    plot = True
    save = False

    model = NoiseModel()
    sim = Simulator(model, trials=trials, rounds=rounds)
    results = sim.run_all([NoCorrection(), NearestLattice()], seed=seed)

    for name, result in results.items():
        print(
            f"{name}: logical error rate after {rounds} rounds = "
            f"{result.final_logical_error_rate():.4f}"
            f" +/- {result.standard_error()[-1]:.4f}"
        )

    if check:
        print()
        check_syndromes(results[NearestLattice.name])
        check_nearest_lattice(results[NearestLattice.name], model)

    title = (
        rf"$\sigma_q={model.sigma_q/L:.2f}L$, "
        rf"$\sigma_m={model.sigma_m/L:.2f}L$, $\rho={model.rho}$"
    )
    ResultPlotter().plot(
        results,
        title=title,
        show=plot,
        save_path="gkp_logical_error.png" if save else None,
    )
