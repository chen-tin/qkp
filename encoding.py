
import numpy as np
import matplotlib.pyplot as plt
import qutip as qt
from scipy.integrate import trapezoid
from mpl_toolkits.axes_grid1 import make_axes_locatable


# GKP
# q is dimensionless position and p is dimensionless momentum, with [q, p] = i
delta = 0.30
n_cutoff = 100
k_max = 8

# controls
check = True
plot = True
save = False

# figure
figsize = (9, 9)
plot_range = 5
resolution = 601
dpi = 600


# plot
cmap = "RdBu_r"
wave_color = "darkblue"
wave_linewidth = 2
contrast = 1.0
fontsize = 13
show_title = True

# side plots
top_height = "30%"
right_width = "30%"
spacing = 0.10

# colorbar
colorbar_height = "4%"
colorbar_spacing = 0.75


# prepare a normalized finite-energy GKP logical-zero state
def encode_gkp(delta, n_cutoff, k_max):
    # check inputs
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    if not isinstance(n_cutoff, (int, np.integer)) or n_cutoff < 2:
        raise ValueError("n_cutoff must be an integer of at least 2")
    if not isinstance(k_max, (int, np.integer)) or k_max < 1:
        raise ValueError("k_max must be a positive integer")

    # square lattice
    alpha = np.sqrt(np.pi / 2)
    beta = 1j * alpha
    coeffs = np.zeros(n_cutoff, dtype=complex)

    # coherent-state sum for logical zero
    for k in range(-k_max, k_max + 1):
        for ell in range(-k_max, k_max + 1):
            z = 2 * k * alpha + ell * beta

            # coherent state in the energy basis
            fock = np.zeros(n_cutoff, dtype=complex)
            fock[0] = np.exp(-abs(z)**2 / 2)

            for n in range(1, n_cutoff):
                fock[n] = fock[n - 1] * z / np.sqrt(n)

            coeffs += (-1)**(k * ell) * fock

    # finite-energy envelope
    coeffs *= np.exp(-delta**2 * np.arange(n_cutoff))

    # normalize
    norm = np.linalg.norm(coeffs)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not normalize the state")

    coeffs /= norm

    # oscillator ket in the energy basis
    return qt.Qobj(coeffs.reshape(-1, 1))


# check that the input is a normalized oscillator state
def validate_state(psi):
    if not isinstance(psi, qt.Qobj):
        raise ValueError("psi must be a QuTiP object")
    if len(psi.dims[0]) != 1:
        raise ValueError("psi must describe one oscillator")
    if not np.all(np.isfinite(psi.full())):
        raise ValueError("psi contains nonfinite values")

    tolerance = 1e-10

    if psi.isket:
        if abs(psi.norm() - 1) > tolerance:
            raise ValueError("ket must be normalized")

    elif psi.isoper:
        if not psi.isherm:
            raise ValueError("density matrix must be Hermitian")
        if abs(psi.tr() - 1) > tolerance:
            raise ValueError("density matrix must have trace 1")
        if np.min(psi.eigenenergies()) < -tolerance:
            raise ValueError("density matrix must be positive semidefinite")

    else:
        raise ValueError("psi must be a ket or density matrix")


# plot an oscillator ket or density matrix
# returns the figure without displaying or saving it
def plot_gkp(psi, title=None):
    validate_state(psi)

    if not np.isfinite(contrast) or contrast <= 0:
        raise ValueError("contrast must be finite and positive")

    # use the input state's basis size
    N = psi.shape[0]

    # phase-space coordinates: zeta = (q + i*p)/sqrt(2)
    zeta = np.linspace(-plot_range, plot_range, resolution)
    qp = np.sqrt(2) * zeta

    # Wigner function in q,p normalization
    wigner = qt.wigner(psi, qp, qp, method="clenshaw")

    # harmonic oscillator wavefunctions
    phi = np.zeros((N, resolution))
    phi[0] = np.pi**(-0.25) * np.exp(-qp**2 / 2)

    if N > 1:
        phi[1] = np.sqrt(2) * qp * phi[0]

    for n in range(1, N - 1):
        phi[n + 1] = (
            np.sqrt(2 / (n + 1)) * qp * phi[n]
            - np.sqrt(n / (n + 1)) * phi[n - 1]
        )

    # momentum basis wavefunctions
    phi_p = (-1j)**np.arange(N)[:, None] * phi

    # probability densities for a pure state
    if psi.isket:
        coeffs = psi.full().ravel()
        probability_q = np.abs(coeffs @ phi)**2
        probability_p = np.abs(coeffs @ phi_p)**2

    # probability densities for a mixed state
    else:
        density = psi.full()

        probability_q = np.sum(
            phi * (density @ phi.conj()), axis=0
        ).real

        probability_p = np.sum(
            phi_p * (density @ phi_p.conj()), axis=0
        ).real

        # remove tiny negative values from numerical rounding
        probability_q = np.maximum(probability_q, 0)
        probability_p = np.maximum(probability_p, 0)

    # probability captured by the displayed range
    if check:
        mass_q = trapezoid(probability_q, qp)
        mass_p = trapezoid(probability_p, qp)

        print(f"Displayed position probability: {mass_q:.8f}")
        print(f"Displayed momentum probability: {mass_p:.8f}")

    # create figure
    with plt.rc_context({"font.size": fontsize}):
        fig, ax = plt.subplots(figsize=figsize)
        divider = make_axes_locatable(ax)

        ax_top = divider.append_axes(
            "top", size=top_height, pad=spacing, sharex=ax
        )

        ax_right = divider.append_axes(
            "right", size=right_width, pad=spacing, sharey=ax
        )

        # Wigner function
        limit = max(np.max(np.abs(wigner)) * contrast, 1e-15)

        im = ax.imshow(
            wigner,
            origin="lower",
            extent=[-plot_range, plot_range, -plot_range, plot_range],
            cmap=cmap,
            vmin=-limit,
            vmax=limit,
            interpolation="bicubic",
            aspect="equal"
        )

        ax.set_xlabel(r"$\mathrm{Re}\,\zeta$")
        ax.set_ylabel(r"$\mathrm{Im}\,\zeta$")
        ax.set_xlim(-plot_range, plot_range)
        ax.set_ylim(-plot_range, plot_range)

        ticks = np.linspace(-plot_range, plot_range, 5)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        # position probability density
        ax_top.plot(
            zeta, probability_q, color=wave_color, lw=wave_linewidth
        )

        ax_top.set_ylim(
            0, 1.15 * max(probability_q.max(), 1e-10)
        )

        ax_top.text(
            0.02, 0.85, r"$P(q)$", transform=ax_top.transAxes
        )

        ax_top.axis("off")

        # momentum probability density
        ax_right.plot(
            probability_p, zeta, color=wave_color, lw=wave_linewidth
        )

        ax_right.set_xlim(
            0, 1.15 * max(probability_p.max(), 1e-10)
        )

        ax_right.text(
            0.1, 1.02, r"$P(p)$", transform=ax_right.transAxes
        )

        ax_right.axis("off")

        # colorbar below the Re zeta label
        ax_colorbar = divider.append_axes(
            "bottom", size=colorbar_height, pad=colorbar_spacing
        )

        colorbar = fig.colorbar(
            im, cax=ax_colorbar, orientation="horizontal"
        )

        colorbar.set_label(r"$W(q,p)$")
        colorbar.ax.tick_params(labelsize=fontsize - 2)

        if show_title and title is not None:
            fig.suptitle(title, fontsize=16)

        fig.subplots_adjust(
            left=0.15, right=0.88, bottom=0.18, top=0.88
        )

    return fig


# run checks and plots only when running this file
if __name__ == "__main__":
    psi = encode_gkp(delta, n_cutoff, k_max)

    if check:
        # normalization and mean excitation number
        coeffs = psi.full().ravel()
        probabilities = np.abs(coeffs)**2
        mean_n = np.sum(np.arange(n_cutoff) * probabilities)

        print("State: approximate GKP logical zero")
        print(f"State norm: {psi.norm():.10f}")
        print(f"Mean excitation number: {mean_n:.6f}")
        print(
            f"Population in highest 10 levels: "
            f"{probabilities[-10:].sum():.3e}"
        )

        # compare with a larger calculation
        reference_cutoff = n_cutoff + 50
        reference = encode_gkp(
            delta, reference_cutoff, k_max + 4
        )
        reference_coeffs = reference.full().ravel()

        # match basis sizes
        padded_coeffs = np.pad(
            coeffs, (0, reference_cutoff - n_cutoff)
        )

        fidelity = abs(
            np.vdot(reference_coeffs, padded_coeffs)
        )**2

        print(f"Fidelity with larger calculation: {fidelity:.10f}")

    if plot or save:
        title = (
            rf"Approximate square GKP $|\widetilde{{0}}_L\rangle$"
            rf" ($\Delta={delta}$)"
        )

        fig = plot_gkp(psi, title=title)

        # save figure only
        if save:
            fig.savefig(
                filename, dpi=dpi, bbox_inches="tight"
            )

        if plot:
            plt.show()
        else:
            plt.close(fig)
