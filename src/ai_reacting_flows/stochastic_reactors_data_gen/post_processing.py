import os

import numpy as np
import pandas as pd
from scipy.interpolate import interpn
import cantera as ct
import h5py

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 
from matplotlib import cm
from matplotlib import animation

import seaborn as sns

import ai_reacting_flows.tools.utilities as utils

sns.set_style("darkgrid")

class StochDatabase(object):
    
    def __init__(self, stoch_dtb_folder, save_folder, with_traj=True):
        
        self.stoch_dtb_folder = stoch_dtb_folder

        # By default, no 
        self.add_0D_ignition_archetype = False
        self.add_1D_premixed_archetype = False

        # Loading database: concatenation of each data in h5 file
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        names = h5file_r.keys()
        self.nb_solutions = len(names)
        h5file_r.close()
        self.get_all_states()

        if with_traj:
            # Loading trajectories
            traj_data = stoch_dtb_folder + "/mean_trajectories.h5"
            f = h5py.File(traj_data,"r")
            #
            traj_dataset = f['TRAJECTORIES']
            nb_inlets = 0 # number of inlets
            self.inlets_data_list = {}  # dictionary for storing data
            for inlet in traj_dataset.keys():
                i = int(inlet[-1])
                self.inlets_data_list[i] = np.asarray(traj_dataset[inlet])
                nb_inlets += 1
            #
            # Getting number of states variable
            self.nb_state_vars = self.inlets_data_list[1].shape[1] - 3  # all except Z, phi and time
            # 
            f.close()

        # Additional post_processing
        self.compute_additional_postpros()

        # Saving folder
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
        self.save_folder = save_folder


    #--------------------------------------------------------
    # READING H5 SOLUTION FILES
    #--------------------------------------------------------

    def get_all_states(self):

        # Reading file
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')

        # Solution 0 read to get columns names
        col_names = h5file_r["ITERATION_00000/all_states"].attrs["cols"]

        # Loop on solutions
        list_df = []
        for i in range(self.nb_solutions):

            if i%100==0:
                print(f"Opening solution: {i} / {self.nb_solutions}")

            data = h5file_r.get(f"ITERATION_{i:05d}/all_states")[()]

            list_df.append(pd.DataFrame(data=data, columns=col_names))
        
        self.df = pd.concat(list_df, ignore_index=True)

        h5file_r.close()
        
    #--------------------------------------------------------
    # CANONICAL FLAMES CALCULATION
    #--------------------------------------------------------

    def compute_0D_ignition(self, phi, p, T0, fuel, mech_file):

        self.add_0D_ignition_archetype = True
        
        # Compute flame
        self.T_cano_0D, self.Y_cano_dict_0D = utils.compute_0D_reactor(fuel, mech_file, phi, T0, p)

    def compute_1D_premixed(self, phi, p, T0, fuel, mech_file, diffusion_model):

        self.add_1D_premixed_archetype = True

        # Compute flame
        self.T_cano_1D, self.Y_cano_dict_1D = utils.compute_adiabatic(fuel, mech_file, phi, T0, p, diffusion_model)

    #--------------------------------------------------------
    # ADDITIONAL CALCULATIONS
    #--------------------------------------------------------

    def compute_additional_postpros(self):

        self.df["abs_HRR"] = np.abs(self.df["HRR"])
        self.df["log_abs_HRR"] = np.log(self.df["abs_HRR"])

    #--------------------------------------------------------
    # SCATTER PLOTS: ALL STATES
    #--------------------------------------------------------

    def plot_T_Z(self):

        # Creating axis
        fig, ax = plt.subplots()

        self.df.plot.scatter(x='Mix_frac', y='Temperature', ax=ax, c='Time', colormap='viridis')
        ax.set_xlabel(r"$Z$ $[-]$")
        ax.set_ylabel(r"$T$ $[K]$")

        ax.set_xlim([0.9*self.df['Mix_frac'].min(), 1.1*self.df['Mix_frac'].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + "/dtb_TZ_plot.png", dpi=300)

    def plot_Z_Yk(self, species_to_plot):

        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            self.df.plot.scatter(x='Mix_frac', y=spec, ax=ax, c='Time', colormap='viridis')
            ax.set_xlabel(r"$Z$ $[-]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            
            ax.set_xlim([0.9*self.df['Mix_frac'].min(), 1.1*self.df['Mix_frac'].max()])
            
            fig.tight_layout()
            
            # Save
            fig.savefig(self.save_folder + f"/dtb_{spec}_Z_plot.png", dpi=300)

    def plot_T_Yk(self, species_to_plot):

        # Creating axis
        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            self.df.plot.scatter(x='Temperature', y=spec, ax=ax, c='Time', colormap='viridis')
            
            # Canonical flame structures
            if self.add_1D_premixed_archetype:
                ax.plot(self.T_cano_1D, self.Y_cano_dict_1D[spec], color='r', lw=3, ls='--', label="Laminar")
                
            if self.add_0D_ignition_archetype:
                ax.plot(self.T_cano_0D, self.Y_cano_dict_0D[spec], color='b', lw=3, ls='--', label="Ignition")
            
            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            ax.legend()
            
            ax.set_xlim([0.9*self.df['Temperature'].min(), 1.1*self.df['Temperature'].max()])
            
            fig.tight_layout()
            
            plt.show()

            # Save
            fig.savefig(self.save_folder + f"/dtb_T_{spec}_plot.png", dpi=300)

    # Generic plotting function

    def plot_generic(self, var_x, var_y, var_c):

        # Creating axis
        fig, ax = plt.subplots()

        self.df.plot.scatter(x=var_x, y=var_y, ax=ax, c=var_c, colormap='viridis')
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)

        ax.set_xlim([0.9*self.df[var_x].min(), 1.1*self.df[var_x].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/dtb_x{var_x}_y{var_y}_c{var_c}_plot.png", dpi=300)

    #--------------------------------------------------------
    # COMPARISON AGAINST A REFERENCE CFD SNAPSHOT
    #--------------------------------------------------------

    def load_cfd_snapshot(self, cfd_h5_path, n_sample=50000, T_threshold=None, seed=0):
        """
        Load a converged CFD field snapshot (CONVERGE-style h5, group
        STREAM_00/CELL_CENTER_DATA with TEMPERATURE/PRESSURE/MASSFRAC_<species>
        datasets, one row per cell) as a reference to compare against this
        stochastic database. Stores the (optionally T-thresholded and
        randomly subsampled) result in self.df_cfd.

        Parameters
        ----------
        cfd_h5_path : str   Path to the CFD snapshot h5 file.
        n_sample    : int   Max number of CFD cells to keep (files can hold
                             millions of cells, too many to scatter-plot).
        T_threshold : float or None   If set, only keep cells with T above it.
        seed        : int   RNG seed for the subsampling.
        """

        # Species columns sit between 'Pressure' and 'Mix_frac' in the
        # all_states column layout, so this stays consistent with self.df.
        cols = list(self.df.columns)
        species_names = cols[cols.index("Pressure") + 1 : cols.index("Mix_frac")]

        with h5py.File(cfd_h5_path, "r") as f:
            cell_data = f["STREAM_00/CELL_CENTER_DATA"]

            T = cell_data["TEMPERATURE"][()].astype(np.float64)
            P = cell_data["PRESSURE"][()].astype(np.float64)

            Y = np.empty((T.shape[0], len(species_names)), dtype=np.float64)
            for i_sp, name in enumerate(species_names):
                key = f"MASSFRAC_{name}"
                if key not in cell_data:
                    raise KeyError(f"Species '{name}' (dataset '{key}') not found in {cfd_h5_path}")
                Y[:, i_sp] = cell_data[key][()].astype(np.float64)

        if T_threshold is not None:
            mask = T > T_threshold
            T, P, Y = T[mask], P[mask], Y[mask]

        rng = np.random.default_rng(seed)
        n = min(n_sample, T.shape[0])
        idx = rng.choice(T.shape[0], size=n, replace=False)
        T, P, Y = T[idx], P[idx], Y[idx]

        # CFD post-processing mass fractions rarely sum exactly to 1.
        Y = Y / Y.sum(axis=1, keepdims=True)

        self.df_cfd = pd.DataFrame(
            data=np.column_stack([T, P, Y]),
            columns=["Temperature", "Pressure"] + species_names,
        )

    def plot_T_Yk_vs_cfd(self, species_to_plot=None):

        if not hasattr(self, "df_cfd"):
            raise RuntimeError("load_cfd_snapshot() must be called before plot_T_Yk_vs_cfd()")

        if species_to_plot is None:
            cols = list(self.df.columns)
            species_to_plot = cols[cols.index("Pressure") + 1 : cols.index("Mix_frac")]

        scatter_folder = self.save_folder + "/scatter"
        os.makedirs(scatter_folder, exist_ok=True)

        for spec in species_to_plot:

            # Two flat, high-contrast colors with a proper legend, so the two
            # datasets stay distinguishable even where the clouds overlap
            # almost exactly (which is the "good" outcome to look for).
            fig, ax = plt.subplots()

            ax.scatter(
                self.df_cfd["Temperature"], self.df_cfd[spec],
                s=6, alpha=0.35, color="tab:blue", edgecolors="none", label="CFD",
            )
            ax.scatter(
                self.df["Temperature"], self.df[spec],
                s=6, alpha=0.35, color="tab:red", edgecolors="none", marker="x", label="Stochastic reactor",
            )

            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            ax.legend(markerscale=3)

            fig.tight_layout()

            # Save
            fig.savefig(scatter_folder + f"/dtb_T_{spec}_vs_cfd_plot.png", dpi=300)
            plt.close(fig)

        # Side-by-side panels: same axis limits, one dataset per panel, for
        # cases where the overlaid version is still hard to read due to
        # near-total overlap or very different point densities.
        for spec in species_to_plot:

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)

            axes[0].scatter(self.df_cfd["Temperature"], self.df_cfd[spec], s=4, alpha=0.3, color="tab:blue")
            axes[0].set_title("CFD")

            axes[1].scatter(self.df["Temperature"], self.df[spec], s=4, alpha=0.3, color="tab:red")
            axes[1].set_title("Stochastic reactor")

            for ax in axes:
                ax.set_xlabel(r"$T$ $[K]$")
            axes[0].set_ylabel(f"${spec}$ mass fraction $[-]$")

            fig.tight_layout()

            fig.savefig(scatter_folder + f"/dtb_T_{spec}_vs_cfd_sidebyside_plot.png", dpi=300)
            plt.close(fig)

        # Grid of all species in one figure, for a quick overview. Subsampled
        # (independently of the full-resolution per-species plots above) so a
        # 32-species grid renders quickly.
        rng = np.random.default_rng(0)
        stoch_idx = rng.choice(len(self.df), size=min(20000, len(self.df)), replace=False)
        cfd_idx = rng.choice(len(self.df_cfd), size=min(20000, len(self.df_cfd)), replace=False)

        cfd_T = self.df_cfd["Temperature"].to_numpy()[cfd_idx]
        stoch_T = self.df["Temperature"].to_numpy()[stoch_idx]

        n = len(species_to_plot)
        ncols = 6
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, spec in enumerate(species_to_plot):
            ax = axes[i]
            ax.scatter(cfd_T, self.df_cfd[spec].to_numpy()[cfd_idx], s=3, alpha=0.25, color="tab:blue")
            ax.scatter(stoch_T, self.df[spec].to_numpy()[stoch_idx], s=3, alpha=0.25, color="tab:red", marker="x")
            ax.set_title(spec, fontsize=9)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        axes[0].legend(["CFD", "Stochastic reactor"], fontsize=7, markerscale=2)
        fig.tight_layout()
        fig.savefig(self.save_folder + "/dtb_T_Yk_all_species_vs_cfd_grid_plot.png", dpi=200)
        plt.close(fig)

    def plot_T_binned_mean_vs_cfd(self, species_to_plot=None, n_bins=50):
        """
        Mean species mass fraction per temperature bin, CFD vs stochastic
        reactor, on shared bins so the two conditional means are directly
        comparable point-by-point (rather than raw scatter clouds).
        """

        if not hasattr(self, "df_cfd"):
            raise RuntimeError("load_cfd_snapshot() must be called before plot_T_binned_mean_vs_cfd()")

        if species_to_plot is None:
            cols = list(self.df.columns)
            species_to_plot = cols[cols.index("Pressure") + 1 : cols.index("Mix_frac")]

        T_min = min(self.df["Temperature"].min(), self.df_cfd["Temperature"].min())
        T_max = max(self.df["Temperature"].max(), self.df_cfd["Temperature"].max())
        bins = np.linspace(T_min, T_max, n_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        stoch_groups = self.df.groupby(pd.cut(self.df["Temperature"], bins=bins), observed=False)
        cfd_groups = self.df_cfd.groupby(pd.cut(self.df_cfd["Temperature"], bins=bins), observed=False)
        stoch_means = stoch_groups[species_to_plot].mean()
        cfd_means = cfd_groups[species_to_plot].mean()

        mean_folder = self.save_folder + "/mean"
        os.makedirs(mean_folder, exist_ok=True)

        # One plot per species
        for spec in species_to_plot:

            fig, ax = plt.subplots()

            ax.plot(bin_centers, cfd_means[spec].values, color="tab:blue", marker="o", ms=3, lw=1.5, label="CFD")
            ax.plot(bin_centers, stoch_means[spec].values, color="tab:red", marker="x", ms=3, lw=1.5, label="Stochastic reactor")

            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"mean ${spec}$ mass fraction $[-]$")
            ax.legend()

            fig.tight_layout()

            fig.savefig(mean_folder + f"/dtb_T_binned_{spec}_vs_cfd_plot.png", dpi=300)
            plt.close(fig)

        # Grid of all species in one figure, for a quick overview
        n = len(species_to_plot)
        ncols = 6
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.0 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, spec in enumerate(species_to_plot):
            ax = axes[i]
            ax.plot(bin_centers, cfd_means[spec].values, color="tab:blue", lw=1.3, label="CFD")
            ax.plot(bin_centers, stoch_means[spec].values, color="tab:red", lw=1.3, label="Stochastic reactor")
            ax.set_title(spec, fontsize=9)
            ax.tick_params(labelsize=6)

        for j in range(n, len(axes)):
            axes[j].axis("off")

        axes[0].legend(fontsize=7)
        fig.tight_layout()

        fig.savefig(self.save_folder + "/dtb_T_binned_all_species_vs_cfd_grid_plot.png", dpi=200)
        plt.close(fig)

    #--------------------------------------------------------
    # SCATTER PLOTS: ONE SOLUTION
    #--------------------------------------------------------

    def plot_T_Z_indiv(self, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Creating axis
        fig, ax = plt.subplots()

        df.plot.scatter(x='Mix_frac', y='Temperature', ax=ax, c='Time', colormap='viridis')
        ax.set_xlabel(r"$Z$ $[-]$")
        ax.set_ylabel(r"$T$ $[K]$")

        ax.set_xlim([0.9*df['Mix_frac'].min(), 1.1*df['Mix_frac'].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/dtb_TZ_plot_iteration{iteration:05d}.png", dpi=300)

    def plot_Z_Yk_indiv(self, species_to_plot, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            df.plot.scatter(x='Mix_frac', y=spec, ax=ax, c='Time', colormap='viridis')
            ax.set_xlabel(r"$Z$ $[-]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            
            ax.set_xlim([0.9*df['Mix_frac'].min(), 1.1*df['Mix_frac'].max()])
            
            fig.tight_layout()
            
            # Save
            fig.savefig(self.save_folder + f"/dtb_{spec}_Z_plot_iteration{iteration:05d}.png", dpi=300)

    def plot_T_Yk_indiv(self, species_to_plot, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Creating axis
        for spec in species_to_plot:
            
            fig, ax = plt.subplots()
            
            df.plot.scatter(x='Temperature', y=spec, ax=ax, c='Time', colormap='viridis')
            
            # Canonical flame structures
            if self.add_1D_premixed_archetype:
                ax.plot(self.T_cano, self.Y_cano_dict[spec], color='r', lw=3, ls='--', label="Laminar")
                
            if self.add_0D_ignition_archetype:
                ax.plot(self.T_cano_0D, self.Y_cano_dict_0D[spec], color='b', lw=3, ls='--', label="Ignition")
            
            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            ax.legend()
            
            ax.set_xlim([0.9*df['Temperature'].min(), 1.1*df['Temperature'].max()])
            
            fig.tight_layout()
            
            plt.show()

            # Save
            fig.savefig(self.save_folder + f"/dtb_T_{spec}_plot_{iteration:05d}.png", dpi=300)

    # Generic plotting function

    def plot_generic_indiv(self, var_x, var_y, var_c, iteration):

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Creating axis
        fig, ax = plt.subplots()

        df.plot.scatter(x=var_x, y=var_y, ax=ax, c=var_c, colormap='viridis')
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)

        ax.set_xlim([0.9*df[var_x].min(), 1.1*df[var_x].max()])

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + f"/dtb_x{var_x}_y{var_y}_c{var_c}_plot_{iteration:05d}.png", dpi=300)

    #--------------------------------------------------------
    # ANIMATIONS
    #--------------------------------------------------------

    def plot_animation(self, var1, var2, var_color, iterations=None, step=100, interval=300, save_path=None, fps=5):
        """
        Animate the evolution of the T-Z scatter plot across iterations.

        Parameters
        ----------
        var1: str
            Variable to plot on x.
        var2: str
            Variable to plot on y.
        var_color: str
            Variable to use to color plot.
        iterations : list[int] or None
            Iterations to include. If None, all iterations found in the H5 file are used.
        step : int
            Keep only every `step`-th iteration (e.g. step=100 keeps iterations 0, 100, 200, ...).
            Ignored if `iterations` is explicitly provided.
        interval : int
            Delay between frames in milliseconds (for on-screen display).
        save_path : str or None
            If given, path to save the animation (.mp4 or .gif). If None, defaults to
            self.save_folder + "/dtb_TZ_animation.mp4".
        fps : int
            Frames per second when saving.
        """

        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')

        # Discover iterations if not provided
        if iterations is None:
            iter_keys = sorted(
                [k for k in h5file_r.keys() if k.startswith("ITERATION_")],
                key=lambda k: int(k.split("_")[1])
            )
            all_iterations = [int(k.split("_")[1]) for k in iter_keys]
            iterations = all_iterations[::step]

        # First pass: load all dataframes and compute global limits
        dfs = []
        var1_min, var1_max = np.inf, -np.inf
        var2_min, var2_max = np.inf, -np.inf
        var_color_min, var_color_max = np.inf, -np.inf

        for it in iterations:
            data = h5file_r.get(f"ITERATION_{it:05d}/all_states")[()]
            col_names = h5file_r[f"ITERATION_{it:05d}/all_states"].attrs["cols"]
            df = pd.DataFrame(data=data, columns=col_names)
            dfs.append(df)

            var1_min = min(var1_min, df[var1].min())
            var1_max = max(var1_max, df[var1].max())
            var2_min = min(var2_min, df[var2].min())
            var2_max = max(var2_max, df[var2].max())
            var_color_min = min(var_color_min, df[var_color].min())
            var_color_max = max(var_color_max, df[var_color].max())

        h5file_r.close()

        # Creating axis
        fig, ax = plt.subplots()

        scat = ax.scatter([], [], c=[], cmap='viridis', vmin=var_color_min, vmax=var_color_max)
        cbar = fig.colorbar(scat, ax=ax)
        cbar.set_label(var_color)

        ax.set_xlabel(fr"{var1}")
        ax.set_ylabel(fr"{var2}")
        ax.set_xlim([0.9 * var1_min, 1.1 * var1_max])
        ax.set_ylim([0.9 * var2_min, 1.1 * var2_max])

        title = ax.set_title("")

        fig.tight_layout()

        def update(frame_idx):
            df = dfs[frame_idx]
            it = iterations[frame_idx]

            scat.set_offsets(np.column_stack([df[var1], df[var2]]))
            scat.set_array(df[var_color])
            title.set_text(f"Iteration {it:05d}")

            return scat, title

        anim = animation.FuncAnimation(
            fig, update, frames=len(iterations), interval=interval, blit=False
        )

        save_path = self.save_folder + "/dtb_animation.mp4"

        anim.save(save_path, fps=fps, dpi=300)
        plt.close(fig)

        return anim

    def plot_species_evolution_animation(self, species_to_plot=None, n_frames=60, interval=150, fps=8):
        """
        One (T, Y_k) animation per species: particles at the current frame are
        colored (by a colormap over physical time, so the hue itself advances
        with the animation), particles from all previous frames are kept on
        screen as a grey trail so the manifold visibly builds up over time.
        """

        video_folder = self.save_folder + "/video"
        os.makedirs(video_folder, exist_ok=True)

        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        iter_keys = sorted(
            [k for k in h5file_r.keys() if k.startswith("ITERATION_")],
            key=lambda k: int(k.split("_")[1])
        )
        all_iterations = [int(k.split("_")[1]) for k in iter_keys]

        frame_idx = np.unique(np.linspace(0, len(all_iterations) - 1, n_frames).astype(int))
        iterations = [all_iterations[i] for i in frame_idx]

        if species_to_plot is None:
            col_names = list(h5file_r[f"ITERATION_{iterations[0]:05d}/all_states"].attrs["cols"])
            species_to_plot = col_names[col_names.index("Pressure") + 1 : col_names.index("Mix_frac")]

        dfs = []
        T_min, T_max = np.inf, -np.inf
        for it in iterations:
            data = h5file_r.get(f"ITERATION_{it:05d}/all_states")[()]
            col_names = h5file_r[f"ITERATION_{it:05d}/all_states"].attrs["cols"]
            df = pd.DataFrame(data=data, columns=col_names)
            dfs.append(df)
            T_min = min(T_min, df["Temperature"].min())
            T_max = max(T_max, df["Temperature"].max())

        h5file_r.close()

        time_min = dfs[0]["Time"].iloc[0]
        time_max = dfs[-1]["Time"].iloc[0]
        norm = Normalize(vmin=time_min, vmax=max(time_max, time_min + 1e-30))
        cmap = cm.get_cmap("viridis")

        for spec in species_to_plot:

            y_min = min(df[spec].min() for df in dfs)
            y_max = max(df[spec].max() for df in dfs)
            if y_max <= y_min:
                y_max = y_min + 1e-30

            fig, ax = plt.subplots()

            trail_scat = ax.scatter([], [], s=4, alpha=0.25, color="lightgrey", label="Historique")
            current_scat = ax.scatter([], [], s=10, alpha=0.9, label="Instant courant")

            ax.set_xlim(0.95 * T_min, 1.05 * T_max)
            pad = 0.1 * (y_max - y_min)
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlabel(r"$T$ $[K]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            ax.legend(loc="upper right")
            title = ax.set_title("")

            fig.tight_layout()

            trail = {"T": np.array([]), "Y": np.array([])}

            def update(i, spec=spec, trail=trail):
                df = dfs[i]
                it = iterations[i]

                if i > 0:
                    prev = dfs[i - 1]
                    trail["T"] = np.concatenate([trail["T"], prev["Temperature"].to_numpy()])
                    trail["Y"] = np.concatenate([trail["Y"], prev[spec].to_numpy()])
                    trail_scat.set_offsets(np.column_stack([trail["T"], trail["Y"]]))

                current_scat.set_offsets(np.column_stack([df["Temperature"], df[spec]]))
                current_scat.set_color(cmap(norm(df["Time"].iloc[0])))
                title.set_text(f"t = {df['Time'].iloc[0]:.4e} s (iteration {it:05d})")

                return trail_scat, current_scat, title

            anim = animation.FuncAnimation(
                fig, update, frames=len(iterations), interval=interval, blit=False
            )

            anim.save(video_folder + f"/dtb_T_{spec}_evolution_animation.mp4", fps=fps, dpi=150)
            plt.close(fig)

    #--------------------------------------------------------
    # TRAJECTORIES PLOTS
    #--------------------------------------------------------

    def plot_traj_T_Z(self):

        # styles
        linestyles = ["--", "-", "-.", ":"]

        fig, ax = plt.subplots()

        j = 0   
        for i in self.inlets_data_list.keys():
            ax.plot(self.inlets_data_list[i][:,self.nb_state_vars+1], self.inlets_data_list[i][:, 1], color="k", linestyle=linestyles[j], lw=2, label=f"Inlet {i:d}")
            j += 1    
            
        ax.set_xlabel(r"$Z$ $[-]$")
        ax.set_ylabel(r"$T$ $[K]$")

        ax.set_xlim([0.9*np.min(self.inlets_data_list[i][:,self.nb_state_vars+1]), 1.1*np.max(self.inlets_data_list[i][:,self.nb_state_vars+1])])

        ax.legend()

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + "traj_TZ_plot.png", dpi=300)

    def plot_traj_T_time(self):

        # styles
        linestyles = ["--", "-", "-.", ":"]

        fig, ax = plt.subplots()

        j = 0   
        for i in self.inlets_data_list.keys():
            ax.plot(self.inlets_data_list[i][:,0], self.inlets_data_list[i][:, 1], color="k", linestyle=linestyles[j], lw=2, label=f"Inlet {i:d}")
            j += 1    
            
        ax.set_xlabel(r"$t$ $[s]$")
        ax.set_ylabel(r"$T$ $[K]$")

        fig.tight_layout()

        ax.legend()

    def plot_traj_Yk_time(self, species_to_plot, mech_file):

        # To get species index, maybe to be put in __init__ at some point
        gas = ct.Solution(mech_file)

        for spec in species_to_plot:
    
            fig, ax = plt.subplots()
            
            j = 0   
            for i in self.inlets_data_list.keys():
                ax.plot(self.inlets_data_list[i][:,0], self.inlets_data_list[i][:, 3+gas.species_index(spec)], color="k", lw=2, linestyle=self.linestyles[j], label=f"Inlet {i:d}")
                j += 1
                
            ax.set_xlabel(r"$t$ $[s]$")
            ax.set_ylabel(f"${spec}$ mass fraction $[-]$")
            
            ax.legend()
            
            fig.tight_layout()
            
            # Save
            fig.savefig(self.save_folder + f"traj_{spec}_time_plot.png", dpi=300)

    #--------------------------------------------------------
    # INDIVIDUAL PARTICLES TRACKING
    #--------------------------------------------------------

    def plot_indiv_traj(self, inlet_nb, var):

        # Filter only desired inlet 
        inlet_df = self.df[self.df["Inlet_number"]==inlet_nb]

        # Get list of dataframes for each particle
        df_part_list = list(inlet_df.groupby('Particle_number'))

        # Create figure
        fig, ax = plt.subplots()

        for item in df_part_list:
            
            df_part = item[1]
            
            ax.plot(df_part["Time"], df_part[var], alpha=0.2, color="purple")

        fig.tight_layout()

        # Save
        fig.savefig(self.save_folder + "/indiv_rajectory.png", dpi=300)

    #--------------------------------------------------------
    # DISTRIBUTIONS
    #--------------------------------------------------------


    def plot_pdf_inst(self, var, iteration):    

        # Loading solution at given iteration
        h5file_r = h5py.File(self.stoch_dtb_folder + "/solutions.h5", 'r')
        data = h5file_r.get(f"ITERATION_{iteration:05d}/all_states")[()]
        col_names = h5file_r[f"ITERATION_{iteration:05d}/all_states"].attrs["cols"]
        h5file_r.close()
        df = pd.DataFrame(data=data, columns=col_names)

        # Smart binwidth via Freedman-Diaconis rule
        binwidth = utils.smart_binwidth(df[var])
            
        # Temperature histogram
        fig, ax = plt.subplots()
        
        sns.histplot(data=df, x=var, ax=ax, stat="probability",
                     binwidth=binwidth, kde=True)

        fig.tight_layout()
            
        fig.savefig(self.save_folder + f"/PDF_{var}_plot_iteration{iteration:05d}.png")
            

    def plot_pdf_all(self, var): 
            
        # Temperature histogram
        fig, ax = plt.subplots()

        # Smart binwidth via Freedman-Diaconis rule
        binwidth = utils.smart_binwidth(self.df[var])
        
        sns.histplot(data=self.df, x=var, ax=ax, stat="probability",
                     binwidth=binwidth, kde=True)

        fig.tight_layout()
            
        fig.savefig(self.save_folder + f"/PDF_{var}_plot.png")
    

    #--------------------------------------------------------
    # Points density
    #--------------------------------------------------------

    def density_scatter(self, var_x , var_y, sort = True, bins = 100):
        # Functions from https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

        x = self.df[var_x]
        y = self.df[var_y]

        fig , ax = plt.subplots()
        data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
        z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

        #To be sure to plot all data
        z[np.where(np.isnan(z))] = 0.0

        # Sort the points by density, so that the densest points are plotted last
        if sort :
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]

        ax.scatter(x, y, c=z)

        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Density')

        fig.tight_layout()

        fig.savefig(self.save_folder + f"/pnts_density_x{var_x}_y{var_y}_plot.png")