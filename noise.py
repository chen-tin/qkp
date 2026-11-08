import numpy as np
import matplotlib.pyplot as plt

# GKP
L = np.sqrt(np.pi)
rounds = 50
trials = 10000

# q is position and p is momentum
# noise
sigma_q = 0.20 * L       # displacement standard deviation
sigma_m = 0.05 * L       # measurement standard deviation
rho = 0.90              # correlation between consecutive displacements
seed = 42

# controls
check = False
plot = True
save = False

# plot
figsize = (10, 6)
fontsize = 12
wave_color = "darkblue"
wave_linewidth = 1.8
dpi = 300


def noise(rng, trials, rounds, sigma_q, sigma_m, rho):
    # check inputs
    for value in (trials, rounds):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("trials and rounds must be positive integers")

    if not np.isfinite(rho) or not -1 < rho < 1:
        raise ValueError("rho must be between -1 and 1")
    if not np.isfinite(sigma_q) or sigma_q < 0:
        raise ValueError("sigma_q must be finite and nonnegative")
    if not np.isfinite(sigma_m) or sigma_m < 0:
        raise ValueError("sigma_m must be finite and nonnegative")

    # rows = trials, columns = rounds
    # per-round displacements in q units, not accumulated position
    displacement_noise = np.zeros((trials, rounds))

    # first displacement has the same variance as later rounds
    displacement_noise[:, 0] = rng.normal(0, sigma_q, size=trials)

    # keep variance fixed when changing rho
    innovation_std = sigma_q * np.sqrt(1 - rho**2)

    # previous displacement + fresh noise
    for t in range(1, rounds):
        displacement_noise[:, t] = (
            rho * displacement_noise[:, t-1]
            + rng.normal(0, innovation_std, size=trials)
        )

    # independent readout errors, also in q units
    measurement_noise = rng.normal(0, sigma_m, size=(trials, rounds))

    # keep this output order for the main simulator
    return displacement_noise, measurement_noise


# run this part only when running this file
if __name__ == "__main__":
    rng = np.random.default_rng(seed)
    displacement_noise, measurement_noise = noise(
        rng, trials, rounds, sigma_q, sigma_m, rho
    )

    # check mean and standard deviation
    if check:
        print(f"Displacement mean / L: {np.mean(displacement_noise)/L:.5f}")
        print(
            f"Displacement std / L: {np.std(displacement_noise)/L:.5f}"
            f" | target: {sigma_q/L:.5f}"
        )
        print(f"Measurement mean / L: {np.mean(measurement_noise)/L:.5f}")
        print(
            f"Measurement std / L: {np.std(measurement_noise)/L:.5f}"
            f" | target: {sigma_m/L:.5f}"
        )

        # compare consecutive displacements within each trial
        if rounds > 1 and sigma_q > 0:
            correlation = np.corrcoef(
                displacement_noise[:, :-1].ravel(),
                displacement_noise[:, 1:].ravel()
            )[0, 1]
            print(
                f"Displacement correlation: {correlation:.5f}"
                f" | target: {rho:.5f}"
            )

    # create figure only if displaying or saving
    if plot or save:
        # divide by L for display only
        plt.rcParams.update({"font.size": fontsize})
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        t = np.arange(1, rounds + 1)

        # displacement noise
        axes[0].plot(
            t, displacement_noise[0]/L, color=wave_color, lw=wave_linewidth
        )
        axes[0].axhline(0, color="gray", lw=0.8)
        axes[0].set_ylabel(r"$\epsilon_t/L$")
        axes[0].set_title("Displacement noise")

        # measurement noise
        axes[1].plot(
            t, measurement_noise[0]/L, color=wave_color, lw=wave_linewidth
        )
        axes[1].axhline(0, color="gray", lw=0.8)
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel(r"$\nu_t/L$")
        axes[1].set_title("Measurement noise")

        fig.tight_layout()

        # save figure only
        if save:
            fig.savefig("gkp_noise.png", dpi=dpi, bbox_inches="tight")

        # display or close
        if plot:
            plt.show()
        else:
            plt.close(fig)