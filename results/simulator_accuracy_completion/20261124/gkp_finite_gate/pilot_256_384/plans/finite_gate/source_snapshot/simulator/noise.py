from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

# GKP lattice spacing
L = np.sqrt(np.pi)


@dataclass(frozen=True)
class NoiseModel:
    """Correlated displacement noise + independent measurement noise.

    q is position and p is momentum. All values are in q units.
    """

    sigma_q: float = 0.20 * L   # displacement standard deviation
    sigma_m: float = 0.05 * L   # measurement standard deviation
    rho: float = 0.90           # correlation between consecutive displacements

    def __post_init__(self):
        # check inputs
        if not np.isfinite(self.rho) or not -1 < self.rho < 1:
            raise ValueError("rho must be between -1 and 1")
        if not np.isfinite(self.sigma_q) or self.sigma_q < 0:
            raise ValueError("sigma_q must be finite and nonnegative")
        if not np.isfinite(self.sigma_m) or self.sigma_m < 0:
            raise ValueError("sigma_m must be finite and nonnegative")

    @property
    def innovation_std(self):
        # keep variance fixed when changing rho
        return self.sigma_q * np.sqrt(1 - self.rho**2)

    def sample(self, rng, trials, rounds):
        """Return (displacement_noise, measurement_noise), each (trials, rounds)."""
        for value in (trials, rounds):
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("trials and rounds must be positive integers")

        return (
            self.sample_displacement(rng, trials, rounds),
            self.sample_measurement(rng, trials, rounds),
        )

    def sample_displacement(self, rng, trials, rounds):
        # rows = trials, columns = rounds
        # per-round displacements, not accumulated position
        displacement = np.zeros((trials, rounds))

        # first displacement has the same variance as later rounds
        displacement[:, 0] = rng.normal(0, self.sigma_q, size=trials)

        # previous displacement + fresh noise
        for t in range(1, rounds):
            displacement[:, t] = (
                self.rho * displacement[:, t-1]
                + rng.normal(0, self.innovation_std, size=trials)
            )
        return displacement

    def sample_measurement(self, rng, trials, rounds):
        # independent readout errors
        return rng.normal(0, self.sigma_m, size=(trials, rounds))

    def check(self, displacement_noise, measurement_noise):
        """Print sample mean / std / correlation against the targets."""
        print(f"Displacement mean / L: {np.mean(displacement_noise)/L:.5f}")
        print(
            f"Displacement std / L: {np.std(displacement_noise)/L:.5f}"
            f" | target: {self.sigma_q/L:.5f}"
        )
        print(f"Measurement mean / L: {np.mean(measurement_noise)/L:.5f}")
        print(
            f"Measurement std / L: {np.std(measurement_noise)/L:.5f}"
            f" | target: {self.sigma_m/L:.5f}"
        )

        # compare consecutive displacements within each trial
        if displacement_noise.shape[1] > 1 and self.sigma_q > 0:
            correlation = np.corrcoef(
                displacement_noise[:, :-1].ravel(),
                displacement_noise[:, 1:].ravel()
            )[0, 1]
            print(
                f"Displacement correlation: {correlation:.5f}"
                f" | target: {self.rho:.5f}"
            )


@dataclass
class NoisePlotter:
    """Plots the first trial of displacement and measurement noise."""

    figsize: tuple = (10, 6)
    fontsize: int = 12
    wave_color: str = "darkblue"
    wave_linewidth: float = 1.8
    dpi: int = 300

    def plot(self, displacement_noise, measurement_noise,
             show=True, save_path=None):
        # create figure only if displaying or saving
        if not show and save_path is None:
            return

        # divide by L for display only
        plt.rcParams.update({"font.size": self.fontsize})
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        t = np.arange(1, displacement_noise.shape[1] + 1)

        panels = (
            (displacement_noise, r"$\epsilon_t/L$", "Displacement noise"),
            (measurement_noise, r"$\nu_t/L$", "Measurement noise"),
        )
        for ax, (data, ylabel, title) in zip(axes, panels):
            ax.plot(
                t, data[0]/L, color=self.wave_color, lw=self.wave_linewidth
            )
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        axes[-1].set_xlabel("Round")

        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")

        # display or close
        if show:
            plt.show()
        else:
            plt.close(fig)


def noise(rng, trials, rounds, sigma_q, sigma_m, rho):
    # keep this output order for the main simulator
    return NoiseModel(sigma_q, sigma_m, rho).sample(rng, trials, rounds)


# run this part only when running this file
if __name__ == "__main__":
    rounds = 50
    trials = 10000
    seed = 42

    # controls
    check = False
    plot = True
    save = False

    model = NoiseModel()
    rng = np.random.default_rng(seed)
    displacement_noise, measurement_noise = model.sample(rng, trials, rounds)

    if check:
        model.check(displacement_noise, measurement_noise)

    NoisePlotter().plot(
        displacement_noise,
        measurement_noise,
        show=plot,
        save_path="gkp_noise.png" if save else None,
    )
