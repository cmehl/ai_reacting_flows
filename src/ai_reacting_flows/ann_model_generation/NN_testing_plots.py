"""
Plots of the tests of NNTesting (NN_testing.py): ANN compared to the reference (CVODE).
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

FONTSIZE = 14

# Floor of T or Y_k before the logarithm of the changes per step (zero or negative ANN mass fractions)
LOG_FLOOR = 1.0e-30

# Changes per step of ln x plotted on a linear scale below this value, on a log scale above [-]
LOG_CHANGE_LINTHRESH = 1.0e-8


def new_folder(folder):
    """Create folder, removing it first if it exists."""

    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def plot_0D_case(result, folder, spec_to_plot, species_names, remove_N2):
    """Plots of a 0D reactor case (result of NNTesting._run_0D_case) in folder.

    Temperature [K] and mass fractions [-] of spec_to_plot versus time [ms] for CVODE and ANN,
    sum of the ANN mass fractions, and ANN element mass fractions (C, H, O, N).
    If the stability steps were run (stability_factor > 0), temperature [K] until their end, with the
    HP equilibrium temperature of the initial mixture.
    Changes per step ln T(t+dt) - ln T(t) and ln Y_k(t+dt) - ln Y_k(t) of spec_to_plot [-], until the end
    of the stability steps for the ANN, in the subfolder log_changes (see plot_0D_log_changes).
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

    # ANN states and times [ms] until the end of the stability steps (ANN alone after equilibrium)
    time_ann = time_vect
    states_ann_all = states_ann
    if result["states_stability"] is not None:
        dt_ms = time_vect[1] - time_vect[0]
        time_stability = time_vect[-1] + dt_ms * np.arange(1, len(result["states_stability"]) + 1)
        time_ann = np.concatenate([time_vect, time_stability])
        states_ann_all = np.concatenate([states_ann, result["states_stability"]])

    # Temperature until the end of the stability steps
    if result["states_stability"] is not None:
        fig, ax = plt.subplots(1, 1)
        ax.plot(time_vect, states_cvode[:,0], ls="--", color = "k", lw=2, label="CVODE")
        ax.plot(time_ann, states_ann_all[:,0], ls="-", color = "b", lw=2, label="NN")
        ax.axhline(result["T_eq"], ls=":", color="r", lw=1.5, label="Equilibrium (HP)")
        ax.axvline(time_vect[-1], ls=":", color="grey", lw=1)
        ax.set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
        ax.set_ylabel('$T$ $[K]$', fontsize=FONTSIZE)
        ax.legend()
        fig.tight_layout()

        fig.savefig(folder +  "/Temperature_stability.png", dpi=500)

    # Changes per step of ln T and ln Y_k, in the subfolder log_changes
    log_changes_folder = folder + "/log_changes"
    os.makedirs(log_changes_folder)
    plot_0D_log_changes(time_vect, states_cvode[:,0], time_ann, states_ann_all[:,0], '$T$', log_changes_folder + "/dlnT.png")
    for spec in spec_to_plot:
        j = species_names.index(spec) + 1
        plot_0D_log_changes(time_vect, states_cvode[:,j], time_ann, states_ann_all[:,j], f'$Y_{{{spec}}}$', log_changes_folder + f"/dlnY{spec}.png")

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


def plot_0D_log_changes(time_cvode, x_cvode, time_ann, x_ann, label, file_name):
    """Change per step ln x(t+dt) - ln x(t) [-] of a variable x (T [K] or Y_k [-]) versus time [ms], CVODE and ANN.

    Plotted at t+dt. x is floored at LOG_FLOOR before the logarithm (zero or negative mass fractions).
    Symmetric log scale: the changes span orders of magnitude between the ignition and the burnt gases,
    linear within +-LOG_CHANGE_LINTHRESH. A vertical line marks the end of the CVODE simulation (start
    of the stability steps if the ANN is longer).
    """

    dlnx_cvode = np.diff(np.log(np.maximum(x_cvode, LOG_FLOOR)))
    dlnx_ann = np.diff(np.log(np.maximum(x_ann, LOG_FLOOR)))

    fig, ax = plt.subplots(1, 1)
    ax.plot(time_cvode[1:], dlnx_cvode, ls="--", color = "k", lw=2, label="CVODE")
    ax.plot(time_ann[1:], dlnx_ann, ls="-", color = "b", lw=1.5, label="NN")
    ax.axhline(0.0, ls="-", color="grey", lw=0.5)
    ax.axvline(time_cvode[-1], ls=":", color="grey", lw=1)
    ax.set_yscale("symlog", linthresh=LOG_CHANGE_LINTHRESH)
    ax.set_xlabel('$t$ $[ms]$', fontsize=FONTSIZE)
    ax.set_ylabel(r'$\Delta \ln$ ' + label + ' per step $[-]$', fontsize=FONTSIZE)
    ax.legend()
    fig.tight_layout()

    fig.savefig(file_name, dpi=500)


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


def plot_hybrid_criterion(E, proxy, T, tol, proxy_label, file_name, folder, max_points_scatter=50000):
    """Scatter plot of a conservation error proxy of the hybrid ANN/CVODE model against the error E of the
    ANN step (as Fig. 12 of Mehl & Aubagnac-Karkar 2023), colored by the temperature T [K] at t.

    tol: threshold of the proxy (dashed line, with the fraction of states above it, which the hybrid model
    would compute with CVODE), None for no line. Points are randomly subsampled to max_points_scatter.
    """

    # Subsample if needed
    idx = np.arange(len(E))
    if len(E) > max_points_scatter:
        idx = np.random.default_rng(0).choice(len(E), size=max_points_scatter, replace=False)

    fig, ax = plt.subplots()
    sc = ax.scatter(E[idx], proxy[idx], c=T[idx], s=2, cmap="plasma")
    fig.colorbar(sc, ax=ax, label="T [K]")

    if tol is not None:
        ax.axhline(tol, color="r", linestyle="--")
        ax.text(0.98, 0.98, f"tol = {tol:.1e}: {100.0 * np.mean(proxy > tol):.1f} % of states with CVODE",
                transform=ax.transAxes, ha="right", va="top", color="r")

    ax.set_xlabel(r"Mean normalized error of the step $E$ [-]")
    ax.set_ylabel(proxy_label)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, file_name), dpi=200)
    plt.close(fig)


def plot_simulation_errors_by_dt(table, by_dt, case_type, folder):
    """Simulation error E_sim [-] of a series (0D or 1D) versus the time step dt [s], in
    folder/simulation_errors_<case_type>_by_dt.png.

    Left: E(dt), mean over the simulations of each dt, with its standard deviation as error bars.
    Right: box plot of the E_sim of the simulations of each dt.
    table: one row per simulation (columns dt and E_sim); by_dt: E_mean and E_std indexed by dt.
    """

    dts = by_dt.index.to_numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Mean and standard deviation per dt
    axs[0].errorbar(dts, by_dt["E_mean"], yerr=by_dt["E_std"], color="b", lw=2, marker="o", capsize=4)
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r'$\Delta t$ $[s]$', fontsize=FONTSIZE)
    axs[0].set_ylabel(r'$E(\Delta t)$ $[-]$', fontsize=FONTSIZE)

    # Distribution of the simulation errors per dt
    errors_per_dt = [table.loc[table["dt"] == dt, "E_sim"].to_numpy() for dt in dts]
    axs[1].boxplot(errors_per_dt, tick_labels=[f"{dt:.1e}" for dt in dts])
    axs[1].set_xlabel(r'$\Delta t$ $[s]$', fontsize=FONTSIZE)
    axs[1].set_ylabel(r'$E_i$ $[-]$', fontsize=FONTSIZE)

    fig.suptitle(case_type)
    fig.tight_layout()

    fig.savefig(os.path.join(folder, f"simulation_errors_{case_type}_by_dt.png"), dpi=400)
    plt.close(fig)


def plot_errors_per_variable(by_dt, variables, case_type, folder):
    """Mean contribution E_<variable> [-] of each variable (T and network species) to the simulation
    error of a series, one bar per dt [s], in folder/simulation_errors_<case_type>_by_variable.png.

    by_dt: indexed by dt, with the columns of variables (E_T, E_<species>).
    """

    dts = by_dt.index.to_numpy()
    positions = np.arange(len(variables))
    width = 0.8 / len(dts)

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 0.5 * len(variables) * len(dts)), 4))

    for i, dt in enumerate(dts):
        ax.bar(positions + i * width, by_dt.loc[dt, variables], width, label=rf'$\Delta t$ = {dt:.1e} s')

    ax.set_xticks(positions + 0.5 * (len(dts) - 1) * width, [var.removeprefix("E_") for var in variables], rotation=90)
    ax.set_ylabel(r'$E_{variable}$ $[-]$', fontsize=FONTSIZE)
    ax.set_title(case_type)
    ax.legend()
    fig.tight_layout()

    fig.savefig(os.path.join(folder, f"simulation_errors_{case_type}_by_variable.png"), dpi=400)
    plt.close(fig)
