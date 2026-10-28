"""
Plots of the tests of NNTesting (NN_testing.py): ANN compared to the reference (CVODE).
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 14


def new_folder(folder):
    """Create folder, removing it first if it exists."""

    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def plot_0D_case(result, folder, spec_to_plot, species_names, remove_N2):
    """Plots of a 0D reactor case (result of NNTesting._run_0D_case) in folder.

    Temperature [K] and mass fractions [-] of spec_to_plot versus time [ms] for CVODE and ANN,
    sum of the ANN mass fractions, and ANN element mass fractions (C, H, O, N).
    species_names: names of the columns 1: of the states (all species of the mechanism).
    """

    time_vect = result["time_vect"]
    states_cvode = result["states_ref"]
    states_ann = result["states_ann"]
    sumYs = result["sumYs"]
    atomic_cons = result["atomic_cons"]

    new_folder(folder)

    # Temperature
    fig, ax = plt.subplots(1, 1)
    ax.plot(time_vect, states_cvode[:,0], ls="--", color = "k", lw=2, label="CVODE")
    ax.plot(time_vect, states_ann[:,0], ls="-", color = "b", lw=2, marker='x', label="NN")
    ax.set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
    ax.set_ylabel('$T$ $[K]$', fontsize=FONTSIZE)
    ax.legend()
    fig.tight_layout()

    fig.savefig(folder +  "/Temperature.png", dpi=500)

    # Mass fractions
    for spec in spec_to_plot:
        j = species_names.index(spec) + 1

        fig, ax = plt.subplots(1, 1)
        ax.plot(time_vect, states_cvode[:,j], ls="--", color = "k", lw=2, label="CVODE")
        ax.plot(time_vect, states_ann[:,j], ls="-", color = "b", lw=2, marker='x', label="NN")
        ax.set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
        ax.set_ylabel(f'{spec} mass fraction $[-]$', fontsize=FONTSIZE)
        if spec=="N2" and remove_N2: #because N2 is often constant
            ax.set_ylim([states_cvode[0,j]*0.95, states_cvode[0,j]*1.05])
        ax.legend()
        fig.tight_layout()

        fig.savefig(folder + f"/Y{spec}.png", dpi=500)

    # Mass conservation
    fig2, ax2 = plt.subplots(1, 1)
    ax2.plot(time_vect, sumYs, ls="-", color = "b", lw=2)
    ax2.plot(time_vect, np.ones(len(time_vect)), ls="--", color = "k", lw=2)
    ax2.plot(time_vect, np.ones(len(time_vect))+0.01, ls="--", color = "k", lw=2, alpha=0.5)
    ax2.plot(time_vect, np.ones(len(time_vect))-0.01, ls="--", color = "k", lw=2, alpha=0.5)
    ax2.set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
    ax2.set_ylabel(r'$\sum_{k} Y_k$ $[-]$', fontsize=FONTSIZE)
    ax2.set_ylim([0.95,1.05])
    fig2.tight_layout()

    fig2.savefig(folder + "/SumYs.png", dpi=500)

    # Atomic conservation
    fig3, axs3 = plt.subplots(2, 2)

    ratio_scale_down = 0.9
    ratio_scale_up = 1.1

    axs3[0,0].plot(time_vect, atomic_cons[:,0], lw=2, color="purple")
    axs3[0,0].set_ylabel('$Y_C$ $[-]$', fontsize=FONTSIZE)
    axs3[0,0].set_ylim([atomic_cons[0,0]*ratio_scale_down,atomic_cons[0,0]*ratio_scale_up])
    axs3[0,0].xaxis.set_major_formatter(plt.NullFormatter())

    axs3[0,1].plot(time_vect, atomic_cons[:,1], lw=2, color="purple")
    axs3[0,1].set_ylabel('$Y_H$ $[-]$', fontsize=FONTSIZE)
    axs3[0,1].set_ylim([atomic_cons[0,1]*ratio_scale_down,atomic_cons[0,1]*ratio_scale_up])
    axs3[0,1].xaxis.set_major_formatter(plt.NullFormatter())

    axs3[1,0].plot(time_vect, atomic_cons[:,2], lw=2, color="purple")
    axs3[1,0].set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
    axs3[1,0].set_ylabel('$Y_O$ $[-]$', fontsize=FONTSIZE)
    axs3[1,0].set_ylim([atomic_cons[0,2]*ratio_scale_down,atomic_cons[0,2]*ratio_scale_up])

    axs3[1,1].plot(time_vect, atomic_cons[:,3], lw=2, color="purple")
    axs3[1,1].set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
    axs3[1,1].set_ylabel('$Y_N$ $[-]$', fontsize=FONTSIZE)
    axs3[1,1].set_ylim([atomic_cons[0,3]*ratio_scale_down,atomic_cons[0,3]*ratio_scale_up])

    fig3.tight_layout()
    fig3.savefig(folder + "/AtomCons.png", dpi=500)


def plot_1D_reaction_rates(grid, Omega_exact, Omega_ann, spec_to_plot, species_names, folder, x_window=None):
    """Reaction rates [1/s] of spec_to_plot along the 1D flame grid [m], exact (CVODE) and ANN, in folder.

    Omega_exact, Omega_ann: (n_species, n_points), rows in the order of species_names.
    x_window: optional [x_min, x_max] [m] of the plots, full grid if None.
    """

    new_folder(folder)

    for spec in spec_to_plot:
        k = species_names.index(spec)

        fig, ax = plt.subplots(1, 1)
        ax.plot(grid, Omega_exact[k,:], ls="-", color = "b", lw=2, label="Exact")
        ax.plot(grid, Omega_ann[k,:], ls="--", color = "purple", lw=2, label="Neural network")
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel(f'{spec} Reaction rate')
        if x_window is not None:
            ax.set_xlim(x_window)
        ax.legend()
        fig.tight_layout()

        fig.savefig(folder + f"/Y{spec}.png", dpi=700)


def plot_validation_errors(diff, i_cluster, folder, max_points_scatter=50000):
    """Scatter plot of the validation errors diff (n_points, n_outputs) of cluster i_cluster, in folder.

    Points are randomly subsampled to max_points_scatter if larger.
    """

    # Subsample if needed
    n_pts = diff.shape[0]
    if n_pts > max_points_scatter:
        idx = np.random.choice(n_pts, size=max_points_scatter, replace=False)
        diff_plot = diff[idx, :]
    else:
        diff_plot = diff

    fig, ax = plt.subplots(1, 1)
    # Flatten over species/output dimensions
    ax.scatter(np.arange(diff_plot.size), diff_plot.ravel(), s=1, alpha=0.5)
    ax.set_xlabel("Sample index (flattened)")
    ax.set_ylabel("Prediction error (physical units)")
    ax.set_title(f"Cluster {i_cluster} validation errors")
    fig.tight_layout()

    fig.savefig(os.path.join(folder, f"cluster_{i_cluster}_errors.png"), dpi=400)
    plt.close(fig)
