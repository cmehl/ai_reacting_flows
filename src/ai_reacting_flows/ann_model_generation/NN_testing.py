import glob
import os
import sys
import joblib
import pickle
import shutil
import oyaml as yaml
import h5py

import numpy as np
import matplotlib.pyplot as plt

import cantera as ct

import torch
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, DeepONet, DeepONet_shift

import ai_reacting_flows.tools.utilities as utils


class NNTesting():

    def __init__(self, testing_parameters):
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.verbose = True

        self.run_folder = os.getcwd()
        # Model folder name
        self.models_folder = f"{self.run_folder}/MODELS/{testing_parameters['models_folder']}"
        with open(os.path.join(self.models_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        self.dataset_path = os.path.join(self.run_folder, networks_parameters["database_path"])
        
        with open(f"{self.dataset_path}/dtb_processing.yaml", "r") as file:
            dtb_processing_params = yaml.safe_load(file)

        data_processing = dtb_processing_params["data_processing"]
        self.log_transform_X = data_processing["log_transform_X"]
        self.log_transform_Y = data_processing["log_transform_Y"]
        # Box-Cox parameter (kept consistent with NN_manager)
        self.lambda_bct = data_processing.get("lambda_bct", 0.1)
        self.threshold = data_processing["threshold"]
        self.remove_N2 = not data_processing["with_N_chemistry"]
        self.output_omegas = data_processing["output_omegas"]

        data_clustering = dtb_processing_params["data_clustering"]
        self.clustering_method  = data_clustering["clustering_method"]

        database_params = dtb_processing_params["database_params"]
        self.dt_var = database_params["dt_var"]
        self.fuel = database_params["fuel"][0] # fuel is a list, testing only takes 1 component fuel so far
        self.mech = database_params["mech_file"]

        # Getting parameters from ANN model
        with open(os.path.join(self.run_folder, f"STOCH_DTB_{database_params['dtb_folder_suffix']}","dtb_params.yaml"), "r") as file:
            dtb_parameters = yaml.safe_load(file)

        # Chemistry (maybe an information to get from model folder ?)
        self.spec_to_plot = testing_parameters["spec_to_plot"]

        # Hybrid computation
        self.hybrid_ann_cvode = testing_parameters["hybrid_ann_cvode"]
        self.hybrid_ann_cvode_tol = testing_parameters["hybrid_ann_cvode_tol"]

        # Renormalization of Yk to satisfy elements conservation
        self.yk_renormalization = testing_parameters["yk_renormalization"]

        self.hard_constraints_model = 0 # DA: no longer of use ?

        # CANTERA solution object needed to access species indices
        self.gas = ct.Solution(self.mech)
        self.spec_list_ref = self.gas.species_names
        self.nb_species_ref = len(self.spec_list_ref)

        # Getting species and number of species
        self.spec_list_ANN = self.spec_list_ref
        self.nb_species_ANN_tot = self.nb_species_ref
        if self.remove_N2:
            self.nb_species_ANN = self.nb_species_ref - 1

        # Find number of clusters -> equal for instance to number of json files in model folder
        self.nb_clusters = len(glob.glob1(self.models_folder,"*.pth"))
        print(f">> {self.nb_clusters} models were found")

        # If more than one cluster, we need to check the clustering method
        if self.nb_clusters>1:
            if os.path.exists(self.models_folder + "/kmeans_model.pkl"):
                self.clustering_method = "kmeans"
            elif os.path.exists(self.models_folder + "/c_bounds.pkl"):
                self.clustering_method = "progvar"
            else:
                sys.exit("ERROR: There are more than one clusters and no clustering method file in the model folder")
            print(f"   >> Clustering method {self.clustering_method} is used \n")
        else:
            self.clustering_method = None

        # Load models
        self.load_models()

        # Load scalers
        self.load_scalers()

        # Load clustering algorithm if relevant
        if self.nb_clusters>1:
           self.load_clustering_algorithm()

        # Progress variable definition (we do it in any case, even if no clustering based on progress variable)
        self.pv_species = dtb_parameters["pv_species"]
        self.pv_ind = []
        for i in range(len(self.pv_species)):
            self.pv_ind.append(self.gas.species_index(self.pv_species[i]))

        # Some constants used in the code
        self.W_atoms = np.array([12.011, 1.008, 15.999, 14.007]) # Order: C, H, O, N

        # Cached validation data (optional, used by evaluate_on_validation_set)
        self._val_data_loaded = False
        self._X_val = []
        self._Y_val = []

#-----------------------------------------------------------------------
#   MAIN TESTING FUNCTIONS
#-----------------------------------------------------------------------
    def _run_0D_case(self, T0, pressure, dt, nb_ite, gas_init_compo, label_prefix="ignition"):
        """Internal helper to run one 0D constant-pressure reactor case.

        Parameters
        ----------
        T0 : float
            Initial temperature [K].
        pressure : float
            Initial pressure [Pa].
        dt : float
            Time step used by both CVODE and ANN loops.
        nb_ite : int
            Number of time steps.
        gas_init_compo : str
            Cantera composition string used for the initial state (e.g. fuel+air
            for ignition, or pure fuel for pyrolysis).
        label_prefix : str
            Label used in the returned dictionary keys for ignition-delay
            quantities (e.g. "ignition" or "pyrolysis").

        Returns
        -------
        dict
            Structured results containing CVODE and ANN states, ignition-like
            delays, equilibrium temperatures, error metrics, and conservation
            diagnostics.
        """
        tau_ign_CVODE = 0.0
        tau_ign_ANN = 0.0

        #-------------------- INITIALISATION---------------------------
        # Setting composition
        self.gas.TPX = T0, pressure, gas_init_compo

        # Defining reactor
        r = ct.IdealGasConstPressureReactor(self.gas)
        sim = ct.ReactorNet([r])
        simtime = 0.0
        states = ct.SolutionArray(self.gas, extra=['t'])

        # Initial state (saved for later use in NN computation)
        states.append(r.thermo.state, t=0.0)
        T_ini = states.T
        Y_ini = states.Y

        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.parse_species_names(self.spec_list_ANN)

        #-------------------- CVODE COMPUTATION---------------------------
        for _ in range(nb_ite):
            simtime += dt
            sim.advance(simtime)
            states.append(r.thermo.state, t=simtime*1.0e3)

            if states.T[-1] > T0+200.0 and tau_ign_CVODE==0.0:
                tau_ign_CVODE = simtime

        # Equilibrium temperature
        Teq_ref = states.T[len(states.T)-1]

        # Convert in a big array (m,NS+1) for later normalization
        state_ref = states.T.reshape(nb_ite+1, 1)
        for spec in self.spec_list_ref:
            Y = states.Y[:,self.gas.species_index(spec)].reshape(nb_ite+1, 1)
            state_ref = np.concatenate((state_ref, Y), axis=-1)

        #-------------------- ANN SOLVER ---------------------------
        ann_calls = 0
        cvode_calls = 0

        # Initial solution
        Y_k_ann = Y_ini
        T_ann = T_ini

        # Vectors to store time data
        state_save = np.append(T_ann, Y_k_ann)
        sumYs = np.array([1.0])
        progvar_vect = [0.0]

        for model in self.models_list:
            model.to(self.device)
            model.eval()

        # atomic composition of species
        molecular_weights = utils.get_molecular_weights(self.gas.species_names)
        atomic_cons = np.dot(atomic_array,np.reshape(Y_k_ann,-1)/molecular_weights)
        atomic_cons = np.multiply(self.W_atoms, atomic_cons)

        # NEURAL NETWORK COMPUTATION
        time = 0.0
        state = np.append(T_ann, Y_k_ann)
        for i in range(nb_ite):
            # Old state
            T_old = state[0]
            Y_old = state[1:]

            # Computing current progress variable
            progvar = self.compute_progvar(state, pressure)

            T_new, Y_new = self.advance_state_NN(T_old, Y_old, pressure, dt)

            # If hybrid, we check conservation and use CVODE if not satisfied
            if self.hybrid_ann_cvode:
                if self.ia_success is False:
                    # CVODE advance
                    T_new, Y_new = self.advance_state_CVODE(T_old, Y_old, pressure, dt)
                    cvode_calls += 1
                else:
                    ann_calls +=1
            else:
                ann_calls += 1

            # Vector with all variable (T and Yks)
            state = np.append(T_new, Y_new)

            # Saving values
            state_save = np.vstack([state_save,state])
            progvar_vect.append(progvar)

            # Mass conservation
            sumYs = np.append(sumYs, np.sum(Y_new,axis=0))

            # atomic composition of species
            atomic_cons_current = np.dot(atomic_array,Y_new.reshape(self.nb_species_ANN_tot)/molecular_weights)
            atomic_cons_current = np.multiply(self.W_atoms, atomic_cons_current)
            atomic_cons = np.vstack([atomic_cons,atomic_cons_current])

            time += dt

            if T_new > T0+200.0 and tau_ign_ANN==0.0:
                tau_ign_ANN = time

        # ANN equilibrium (last temperature)
        Teq_ann = T_new

        # Error on equilibrium temperature (percent)
        error_Teq = 100.0*abs(Teq_ref - Teq_ann)/Teq_ref

        # Use configurable prefix to label ignition-like delays
        return {
            "states_cvode": states,
            "state_ref": state_ref,
            "states_ann": state_save,
            f"tau_{label_prefix}_cvode": tau_ign_CVODE,
            f"tau_{label_prefix}_ann": tau_ign_ANN,
            "Teq_cvode": Teq_ref,
            "Teq_ann": Teq_ann,
            "error_Teq_percent": error_Teq,
            "atomic_cons": atomic_cons,
            "sumYs": sumYs,
            "progvar": np.array(progvar_vect),
            "ann_calls": ann_calls,
            "cvode_calls": cvode_calls,
        }

    def run_0D_ignition_case(self, phi, T0, pressure, dt, nb_ite=100):
        """Run one 0D ignition case (fuel+air, parametrized by phi).

        This is the core simulation routine for premixed ignition, without
        plotting or printing, so it can be reused programmatically for
        parameter sweeps and error aggregation.
        """

        # Mixture composition: premixed fuel + air parameterized by phi
        fuel_ox_ratio = (
            self.gas.n_atoms(self.fuel, 'C')
            + 0.25 * self.gas.n_atoms(self.fuel, 'H')
            - 0.5 * self.gas.n_atoms(self.fuel, 'O')
        )
        compo = f"{self.fuel}:{phi:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}"

        return self._run_0D_case(T0, pressure, dt, nb_ite, compo, label_prefix="ignition")

    def run_0D_pyrolysis_case(self, T0, pressure, dt, nb_ite=100, diluent=None, X_diluent=0.0):
        """Run one 0D pyrolysis case (pure fuel or fuel+diluent).

        This uses the same CVODE/ANN integration as the ignition case but
        initializes the gas with a composition that does not rely on an
        equivalence ratio.

        Parameters
        ----------
        T0 : float
            Initial temperature [K].
        pressure : float
            Initial pressure [Pa].
        dt : float
            Time step used by both CVODE and ANN loops.
        nb_ite : int, optional
            Number of time steps.
        diluent : str or None, optional
            Optional inert species name (e.g. "N2", "Ar"). If None, a pure
            fuel case is used.
        X_diluent : float, optional
            Mole fraction of diluent if present; fuel fraction is 1 - X_diluent.

        Returns
        -------
        dict
            Structured results in the same format as ignition cases, with
            ignition-like delays labeled under the "pyrolysis" prefix.
        """

        if diluent is None:
            # Pure fuel case
            compo = f"{self.fuel}:1.0"
        else:
            # Fuel + inert diluent, simple two-component mixture
            X_fuel = max(0.0, 1.0 - X_diluent)
            compo = f"{self.fuel}:{X_fuel:3.2f}, {diluent}:{X_diluent:3.2f}"

        return self._run_0D_case(T0, pressure, dt, nb_ite, compo, label_prefix="pyrolysis")

    def run_0D_ignition_series(self, cases):
        """Run a series of 0D ignition cases and aggregate errors.

        Parameters
        ----------
        cases : iterable of dict
            Each dict must contain keys compatible with run_0D_ignition_case,
            e.g.: {"phi": ..., "T0": ..., "pressure": ..., "dt": ..., "nb_ite": ...}.

        Returns
        -------
        results : list[dict]
            Per-case results (direct outputs from run_0D_ignition_case) with the
            corresponding input parameters attached under key "params".
        summary : dict
            Aggregate metrics across all cases: averages and standard deviations
            for equilibrium-temperature error, ignition delays, and solver calls.
        """

        results = []

        err_Teq_list = []
        tau_cvode_list = []
        tau_ann_list = []
        ann_calls_list = []
        cvode_calls_list = []

        for case in cases:
            res = self.run_0D_ignition_case(**case)
            # Attach input parameters to the result for traceability
            res_with_params = {"params": dict(case), **res}
            results.append(res_with_params)

            err_Teq_list.append(res["error_Teq_percent"])
            tau_cvode_list.append(res["tau_ign_cvode"])
            tau_ann_list.append(res["tau_ign_ann"])
            ann_calls_list.append(res["ann_calls"])
            cvode_calls_list.append(res["cvode_calls"])

        err_Teq_arr = np.array(err_Teq_list)
        tau_cvode_arr = np.array(tau_cvode_list)
        tau_ann_arr = np.array(tau_ann_list)
        ann_calls_arr = np.array(ann_calls_list)
        cvode_calls_arr = np.array(cvode_calls_list)

        # Ignition delay relative error (%), with simple guard for zero CVODE delay
        with np.errstate(divide="ignore", invalid="ignore"):
            tau_rel_err = 100.0 * np.where(
                tau_cvode_arr > 0.0,
                (tau_ann_arr - tau_cvode_arr) / tau_cvode_arr,
                0.0,
            )

        summary = {
            "n_cases": len(results),
            # Equilibrium temperature error stats (percent)
            "Teq_error_mean_percent": float(err_Teq_arr.mean()) if err_Teq_arr.size > 0 else 0.0,
            "Teq_error_std_percent": float(err_Teq_arr.std()) if err_Teq_arr.size > 0 else 0.0,
            # Ignition delay stats (absolute values)
            "tau_cvode_mean": float(tau_cvode_arr.mean()) if tau_cvode_arr.size > 0 else 0.0,
            "tau_ann_mean": float(tau_ann_arr.mean()) if tau_ann_arr.size > 0 else 0.0,
            # Ignition delay relative error stats (percent)
            "tau_rel_error_mean_percent": float(tau_rel_err.mean()) if tau_rel_err.size > 0 else 0.0,
            "tau_rel_error_std_percent": float(tau_rel_err.std()) if tau_rel_err.size > 0 else 0.0,
            # Solver calls
            "ann_calls_mean": float(ann_calls_arr.mean()) if ann_calls_arr.size > 0 else 0.0,
            "cvode_calls_mean": float(cvode_calls_arr.mean()) if cvode_calls_arr.size > 0 else 0.0,
        }

        return results, summary

    # OD IGNITION (wrapper with plotting)
    def test_0D_ignition(self, phi, T0, pressure, dt, nb_ite=100):
        result = self.run_0D_ignition_case(phi, T0, pressure, dt, nb_ite)

        states = result["states_cvode"]
        state_save = result["states_ann"]
        atomic_cons = result["atomic_cons"]
        sumYs = result["sumYs"]
        tau_ign_CVODE = result["tau_ignition_cvode"]
        tau_ign_ANN = result["tau_ignition_ann"]
        error_Teq = result["error_Teq_percent"]
        ann_calls = result["ann_calls"]
        cvode_calls = result["cvode_calls"]

        if self.verbose:
            print(f"\nError on equilibrium flame temperature is: {error_Teq} % \n")

        print("\n NUMBER OF CALLS TO SOLVERS:")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        #-------------------- PLOTTING --------------------------- 
        ftsize = 14

        # Create directory with plots
        folder = f'./plots_0D_T0#{T0}_phi#{phi}'
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        fig, ax = plt.subplots(1, 1)
        ax.plot(states.t, states.T, ls="--", color = "k", lw=2, label="CVODE")
        ax.plot(states.t, state_save[:,0], ls="-", color = "b", lw=2, marker='x', label="NN")
        ax.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        ax.set_ylabel('$T$ $[K]$', fontsize=ftsize)
        ax.legend()
        fig.tight_layout()

        fig.savefig(folder +  "/Temperature.png", dpi=500)

        for spec in self.spec_to_plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(states.t, states.Y[:,self.gas.species_index(spec)], ls="--", color = "k", lw=2, label="CVODE")
            ax.plot(states.t, state_save[:,self.spec_list_ANN.index(spec)+1], ls="-", color = "b", lw=2, marker='x', label="NN")
            ax.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
            ax.set_ylabel(f'{spec} mass fraction $[-]$', fontsize=ftsize)
            if spec=="N2": #because N2 is often constant
                ax.set_ylim([states.Y[0,self.gas.species_index(spec)]*0.95, states.Y[0,self.gas.species_index(spec)]*1.05])
            ax.legend()
            fig.tight_layout()

            fig.savefig(folder + f"/Y{spec}.png", dpi=500)

        # Mass conservation
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(states.t, sumYs, ls="-", color = "b", lw=2)
        ax2.plot(states.t, np.ones(len(states.t)), ls="--", color = "k", lw=2)
        ax2.plot(states.t, np.ones(len(states.t))+0.01, ls="--", color = "k", lw=2, alpha=0.5)
        ax2.plot(states.t, np.ones(len(states.t))-0.01, ls="--", color = "k", lw=2, alpha=0.5)
        ax2.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        ax2.set_ylabel(r'$\sum_{k} Y_k$ $[-]$', fontsize=ftsize)
        ax2.set_ylim([0.95,1.05])
        fig2.tight_layout()

        fig2.savefig(folder + "/SumYs.png", dpi=500)

        # Atomic conservation
        fig3, axs3 = plt.subplots(2, 2)

        ratio_scale_down = 0.9
        ratio_scale_up = 1.1

        axs3[0,0].plot(states.t, atomic_cons[:,0], lw=2, color="purple")
        axs3[0,0].set_ylabel('$Y_C$ $[-]$', fontsize=ftsize)
        axs3[0,0].set_ylim([atomic_cons[0,0]*ratio_scale_down,atomic_cons[0,0]*ratio_scale_up])
        axs3[0,0].xaxis.set_major_formatter(plt.NullFormatter())

        axs3[0,1].plot(states.t, atomic_cons[:,1], lw=2, color="purple")
        axs3[0,1].set_ylabel('$Y_H$ $[-]$', fontsize=ftsize)
        axs3[0,1].set_ylim([atomic_cons[0,1]*ratio_scale_down,atomic_cons[0,1]*ratio_scale_up])
        axs3[0,1].xaxis.set_major_formatter(plt.NullFormatter())

        axs3[1,0].plot(states.t, atomic_cons[:,2], lw=2, color="purple")
        axs3[1,0].set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        axs3[1,0].set_ylabel('$Y_O$ $[-]$', fontsize=ftsize)
        axs3[1,0].set_ylim([atomic_cons[0,2]*ratio_scale_down,atomic_cons[0,2]*ratio_scale_up])

        axs3[1,1].plot(states.t, atomic_cons[:,3], lw=2, color="purple")
        axs3[1,1].set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        axs3[1,1].set_ylabel('$Y_N$ $[-]$', fontsize=ftsize)
        axs3[1,1].set_ylim([atomic_cons[0,3]*ratio_scale_down,atomic_cons[0,3]*ratio_scale_up])

        fig3.tight_layout()
        fig3.savefig(folder + "/AtomCons.png", dpi=500)

        # Ignition delays comparison
        print(f" >> CVODE ignition delay: {tau_ign_CVODE}")
        print(f" >> ANN ignition delay: {tau_ign_ANN}")

    def test_0D_pyrolysis(self, T0, pressure, dt, nb_ite=100, diluent=None, X_diluent=0.0):
        """Wrapper around run_0D_pyrolysis_case with plotting.

        The plotting logic mirrors test_0D_ignition but folder naming and
        printed labels reflect the pyrolysis case and do not include phi.
        """

        result = self.run_0D_pyrolysis_case(T0, pressure, dt, nb_ite, diluent=diluent, X_diluent=X_diluent)

        states = result["states_cvode"]
        state_save = result["states_ann"]
        atomic_cons = result["atomic_cons"]
        sumYs = result["sumYs"]
        tau_pyro_CVODE = result["tau_pyrolysis_cvode"]
        tau_pyro_ANN = result["tau_pyrolysis_ann"]
        error_Teq = result["error_Teq_percent"]
        ann_calls = result["ann_calls"]
        cvode_calls = result["cvode_calls"]

        if self.verbose:
            print(f"\nError on equilibrium temperature (pyrolysis case) is: {error_Teq} % \n")

        print("\n NUMBER OF CALLS TO SOLVERS (pyrolysis):")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        #-------------------- PLOTTING ---------------------------
        ftsize = 14

        # Create directory with plots (no phi in naming)
        folder = f'./plots_0D_pyro_T0#{T0}'
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        fig, ax = plt.subplots(1, 1)
        ax.plot(states.t, states.T, ls="--", color="k", lw=2, label="CVODE")
        ax.plot(states.t, state_save[:, 0], ls="-", color="b", lw=2, marker='x', label="NN")
        ax.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        ax.set_ylabel('$T$ $[K]$', fontsize=ftsize)
        ax.legend()
        fig.tight_layout()

        fig.savefig(folder + "/Temperature.png", dpi=500)

        for spec in self.spec_to_plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(states.t, states.Y[:, self.gas.species_index(spec)], ls="--", color="k", lw=2, label="CVODE")
            ax.plot(states.t, state_save[:, self.spec_list_ANN.index(spec) + 1], ls="-", color="b", lw=2, marker='x', label="NN")
            ax.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
            ax.set_ylabel(f'{spec} mass fraction $[-]$', fontsize=ftsize)
            if spec == "N2":
                ax.set_ylim([
                    states.Y[0, self.gas.species_index(spec)] * 0.95,
                    states.Y[0, self.gas.species_index(spec)] * 1.05,
                ])
            ax.legend()
            fig.tight_layout()

            fig.savefig(folder + f"/Y{spec}.png", dpi=500)

        # Mass conservation
        fig2, ax2 = plt.subplots(1, 1)
        ax2.plot(states.t, sumYs, ls="-", color="b", lw=2)
        ax2.plot(states.t, np.ones(len(states.t)), ls="--", color="k", lw=2)
        ax2.plot(states.t, np.ones(len(states.t)) + 0.01, ls="--", color="k", lw=2, alpha=0.5)
        ax2.plot(states.t, np.ones(len(states.t)) - 0.01, ls="--", color="k", lw=2, alpha=0.5)
        ax2.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        ax2.set_ylabel(r'$\sum_{k} Y_k$ $[-]$', fontsize=ftsize)
        ax2.set_ylim([0.95, 1.05])
        fig2.tight_layout()

        fig2.savefig(folder + "/SumYs.png", dpi=500)

        # Atomic conservation
        fig3, axs3 = plt.subplots(2, 2)

        ratio_scale_down = 0.9
        ratio_scale_up = 1.1

        axs3[0, 0].plot(states.t, atomic_cons[:, 0], lw=2, color="purple")
        axs3[0, 0].set_ylabel('$Y_C$ $[-]$', fontsize=ftsize)
        axs3[0, 0].set_ylim([
            atomic_cons[0, 0] * ratio_scale_down,
            atomic_cons[0, 0] * ratio_scale_up,
        ])
        axs3[0, 0].xaxis.set_major_formatter(plt.NullFormatter())

        axs3[0, 1].plot(states.t, atomic_cons[:, 1], lw=2, color="purple")
        axs3[0, 1].set_ylabel('$Y_H$ $[-]$', fontsize=ftsize)
        axs3[0, 1].set_ylim([
            atomic_cons[0, 1] * ratio_scale_down,
            atomic_cons[0, 1] * ratio_scale_up,
        ])
        axs3[0, 1].xaxis.set_major_formatter(plt.NullFormatter())

        axs3[1, 0].plot(states.t, atomic_cons[:, 2], lw=2, color="purple")
        axs3[1, 0].set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        axs3[1, 0].set_ylabel('$Y_O$ $[-]$', fontsize=ftsize)
        axs3[1, 0].set_ylim([
            atomic_cons[0, 2] * ratio_scale_down,
            atomic_cons[0, 2] * ratio_scale_up,
        ])

        axs3[1, 1].plot(states.t, atomic_cons[:, 3], lw=2, color="purple")
        axs3[1, 1].set_xlabel('$t$ $[ms]$', fontsize=ftsize)
        axs3[1, 1].set_ylabel('$Y_N$ $[-]$', fontsize=ftsize)
        axs3[1, 1].set_ylim([
            atomic_cons[0, 3] * ratio_scale_down,
            atomic_cons[0, 3] * ratio_scale_up,
        ])

        fig3.tight_layout()
        fig3.savefig(folder + "/AtomCons.png", dpi=500)

        # Ignition-like delays comparison for pyrolysis case
        print(f" >> CVODE pyrolysis delay (T rise): {tau_pyro_CVODE}")
        print(f" >> ANN pyrolysis delay (T rise): {tau_pyro_ANN}")

    def run_1D_premixed_case(self, phi, T0, pressure, dt, T_threshold=0.0):
        """Run one 1D premixed flame case and return structured results.

        Core computation without plotting/printing, suitable for parameter sweeps.
        """

        #-------------------- 1D FLAME COMPUTATION ---------------------------
        # Cantera computation settings
        initial_grid = np.linspace(0.0, 0.03, 10)  # m
        tol_ss = [1.0e-4, 1.0e-9]  # [rtol atol] for steady-state problem
        tol_ts = [1.0e-5, 1.0e-5]  # [rtol atol] for time stepping
        loglevel = 0  # amount of diagnostic output (0 to 8)

        # Determining initial composition using phi
        fuel_ox_ratio = self.gas.n_atoms(self.fuel,'C') + 0.25*self.gas.n_atoms(self.fuel,'H') - 0.5*self.gas.n_atoms(self.fuel,'O')
        compo = f'{self.fuel}:{phi:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'

        self.gas.TPX = T0, pressure, compo

        f = ct.FreeFlame(self.gas, initial_grid)

        f.flame.set_steady_tolerances(default=tol_ss)
        f.flame.set_transient_tolerances(default=tol_ts)

        # Mixing model and Jacobian
        f.energy_enabled = True
        f.transport_model = 'UnityLewis'
        f.set_max_jac_age(10, 10)
        f.set_time_step(1e-5, [2, 5, 10, 20])

        f.set_refine_criteria(ratio=3, slope=0.1, curve=0.5)
        f.solve(loglevel=loglevel, refine_grid=True)

        f.set_refine_criteria(ratio=3, slope=0.05, curve=0.2)
        f.solve(loglevel=loglevel, refine_grid=True)

        f.set_refine_criteria(ratio=3, slope=0.05, curve=0.1, prune=0.03)
        f.solve(loglevel=loglevel, refine_grid=True)

        # Mass fractions and temperature at steady state
        Yt = f.Y
        Tt = f.T

        # Computing progress variable
        c = np.empty(len(f.grid))
        nb_pnts_total = len(f.grid)
        for i in range(nb_pnts_total):
            state = np.append(Tt[i], Yt[:,i])
            c[i] = self.compute_progvar(state, pressure)

        X_grid = f.grid

        # Initializing Y at t+dt
        Yt_dt_exact = np.zeros(Yt.shape)
        Yt_dt_ann = np.zeros((self.gas.n_species, Yt.shape[1]))

        #-------------------- EXACT REACTION RATES --------------------------- 
        nb_0_reactors = len(X_grid) # number of 1D reactors
        for i_reac in range(nb_0_reactors):
            self.gas.TPY = Tt[i_reac], pressure, Yt[:,i_reac]
            r = ct.IdealGasConstPressureReactor(self.gas)

            # Initializing reactor
            sim = ct.ReactorNet([r])
            time = 0.0
            states = ct.SolutionArray(self.gas, extra=['t'])

            # We advance solution by dt
            time = dt
            sim.advance(time)
            states.append(r.thermo.state, t=time * 1e3)

            Yt_dt_exact[:,i_reac] = states.Y

        # Reaction rates
        Omega_exact = (Yt_dt_exact-Yt)/dt

        #-------------------- ANN REACTION RATES ---------------------------
        ann_calls = 0
        cvode_calls = 0

        # Initial solution
        Yt_ann = Yt

        for i_reac in range(nb_0_reactors):
            # Computing current progress variable
            state = np.append(Tt[i_reac], Yt_ann[:,i_reac])
            progvar = self.compute_progvar(state, pressure)

            # Attribute cluster
            if self.nb_clusters>0:
                self.attribute_cluster(state, progvar)
            else:
                self.cluster = 0

            # advance to t + dt
            if Tt[i_reac] >= T_threshold:
                T_new, Y_new = self.advance_state_NN(Tt[i_reac], Yt_ann[:,i_reac], pressure, dt)

                # If hybrid, we check conservation and use CVODE if not satisfied
                if self.hybrid_ann_cvode:
                    cons_criterion = np.abs(np.sum(Y_new,axis=0) - 1.0)
                    if cons_criterion > self.hybrid_ann_cvode_tol:
                        # CVODE advance
                        T_new, Y_new = self.advance_state_CVODE(Tt[i_reac], Yt_ann[:,i_reac], pressure, dt)
                        cvode_calls += 1
                    else:
                        ann_calls +=1
                else:
                    ann_calls += 1
            else:
                T_new = Tt[i_reac]
                Y_new = Yt_ann[:,i_reac]

            Yt_dt_ann[:,i_reac] = np.reshape(Y_new,-1)

        # Reaction rates
        Omega_ann = (Yt_dt_ann-Yt_ann)/dt

        return {
            "flame": f,
            "X_grid": X_grid,
            "Tt": Tt,
            "Yt": Yt,
            "c": c,
            "Yt_dt_exact": Yt_dt_exact,
            "Yt_dt_ann": Yt_dt_ann,
            "Omega_exact": Omega_exact,
            "Omega_ann": Omega_ann,
            "ann_calls": ann_calls,
            "cvode_calls": cvode_calls,
        }

    def run_1D_premixed_series(self, cases, species_for_error=None, x_window=None):
        """Run a series of 1D premixed cases and aggregate reaction-rate errors.

        Parameters
        ----------
        cases : iterable of dict
            Each dict must contain keys compatible with run_1D_premixed_case,
            e.g.: {"phi": ..., "T0": ..., "pressure": ..., "dt": ..., "T_threshold": ...}.
        species_for_error : list[str] | None
            If provided, compute L2 errors on Omega for these species only.
            If None, use self.spec_to_plot.
        x_window : tuple[float, float] | None
            Optional spatial window [x_min, x_max] over which to compute the
            L2 error. If None, use the full flame domain.

        Returns
        -------
        results : list[dict]
            Per-case results (direct outputs from run_1D_premixed_case) with the
            corresponding input parameters attached under key "params".
        summary : dict
            Aggregate metrics across all cases (means/stds of L2 errors for each
            species and average solver call counts).
        """

        if species_for_error is None:
            species_for_error = list(self.spec_to_plot)

        results = []

        # Per-species error accumulator: {spec -> [err_case_0, err_case_1, ...]}
        species_err = {spec: [] for spec in species_for_error}
        ann_calls_list = []
        cvode_calls_list = []

        for case in cases:
            res = self.run_1D_premixed_case(**case)
            f = res["flame"]
            Omega_exact = res["Omega_exact"]
            Omega_ann = res["Omega_ann"]
            ann_calls_list.append(res["ann_calls"])
            cvode_calls_list.append(res["cvode_calls"])

            # Spatial window selection
            x = f.grid
            if x_window is not None:
                x_min, x_max = x_window
                mask = (x >= x_min) & (x <= x_max)
            else:
                mask = slice(None)

            dx = np.gradient(x[mask]) if not isinstance(mask, slice) else np.gradient(x)

            for spec in species_for_error:
                k = self.gas.species_index(spec)
                omega_ex = Omega_exact[k, mask]
                omega_nn = Omega_ann[k, mask]
                # L2 error in physical units over x-window
                diff = omega_nn - omega_ex
                l2_err = float(np.sqrt(np.sum(diff**2 * dx)))
                species_err[spec].append(l2_err)

            res_with_params = {"params": dict(case), **res}
            results.append(res_with_params)

        ann_calls_arr = np.array(ann_calls_list)
        cvode_calls_arr = np.array(cvode_calls_list)

        species_err_mean = {}
        species_err_std = {}
        for spec, errs in species_err.items():
            arr = np.array(errs)
            species_err_mean[spec] = float(arr.mean()) if arr.size > 0 else 0.0
            species_err_std[spec] = float(arr.std()) if arr.size > 0 else 0.0

        summary = {
            "n_cases": len(results),
            "species_l2_error_mean": species_err_mean,
            "species_l2_error_std": species_err_std,
            "ann_calls_mean": float(ann_calls_arr.mean()) if ann_calls_arr.size > 0 else 0.0,
            "cvode_calls_mean": float(cvode_calls_arr.mean()) if cvode_calls_arr.size > 0 else 0.0,
        }

        return results, summary

    # 1D PREMIXED (wrapper with plotting)
    def test_1D_premixed(self, phi, T0, pressure, dt, T_threshold=0.0):

        print(f"Computing Cantera 1D flame with T0 = {T0} K and phi = {phi}")

        result = self.run_1D_premixed_case(phi, T0, pressure, dt, T_threshold)

        f = result["flame"]
        Omega_exact = result["Omega_exact"]
        Omega_ann = result["Omega_ann"]
        ann_calls = result["ann_calls"]
        cvode_calls = result["cvode_calls"]

        print("\n NUMBER OF CALLS TO SOLVERS:")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        #-------------------- PLOTTING --------------------------- 
        folder = f'./plots_1D_prem_T0#{T0}_phi#{phi}'
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for spec in self.spec_to_plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(f.grid, Omega_exact[self.gas.species_index(spec),:], ls="-", color = "b", lw=2, label="Exact")
            ax.plot(f.grid, Omega_ann[self.gas.species_index(spec),:], ls="--", color = "purple", lw=2, label="Neural network")
            ax.set_xlabel('$x$ [m]')
            ax.set_ylabel(f'{spec} Reaction rate')
            ax.set_xlim([0.008, 0.014])
            ax.legend()
            fig.tight_layout()

            fig.savefig(folder + f"/Y{spec}.png", dpi=700)

            plt.show()

#-----------------------------------------------------------------------
#   IA MODEL HANDLING FUNCTIONS
#-----------------------------------------------------------------------
    # Functions to load models
    def load_models(self):
        self.models_list = []
        
        for i in range(self.nb_clusters):
            model = torch.load(os.path.join(self.models_folder, f"cluster{i}_model.pth"), weights_only=False)
            # Add to list
            self.models_list.append(model)

        # if self.verbose:
        #     for i in range(self.nb_clusters):
        #         print(f"\n ----------CLUSTER MODEL {i}---------- \n")
        #         summary(self.models_list[i], col_names=["input_size", "output_size", "num_params"])

    def load_scalers(self):

        self.Xscaler_list = []
        self.Yscaler_list = []

        for i_cluster in range(self.nb_clusters):

            with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:

                grp = h5file_r[f"CLUSTER_{i_cluster}"]

                Xscaler_array = grp['Xscaler'][:]
                Yscaler_array = grp['Yscaler'][:]

                Xscaler_mean = Xscaler_array[:, 0]
                Xscaler_var = Xscaler_array[:, 1]
                #
                Yscaler_mean = Yscaler_array[:, 0]
                Yscaler_var = Yscaler_array[:, 1]

                Xscaler_std = np.sqrt(Xscaler_var)
                Yscaler_std = np.sqrt(Yscaler_var)

                # shape (n_features, 2): column 0 = mean, column 1 = std
                Xscaler_mean_std = np.column_stack((Xscaler_mean, Xscaler_std))
                Yscaler_mean_std = np.column_stack((Yscaler_mean, Yscaler_std))

                self.Xscaler_list.append(Xscaler_mean_std)
                self.Yscaler_list.append(Yscaler_mean_std)


    def load_clustering_algorithm(self):

        if self.clustering_method=="kmeans":
            with open(self.models_folder + "/kmeans_model.pkl", "rb") as f:
                self.kmeans = pickle.load(f)
            self.kmeans_scaler = joblib.load(self.models_folder + "/Xscaler_kmeans.pkl")

        elif self.clustering_method=="progvar":
            with open(self.models_folder + "/c_bounds.pkl", "rb") as input_file:
                self.c_bounds = pickle.load(input_file)

    def attribute_cluster(self, state_vector=None, progvar=None):
        
        log_state = state_vector.copy()

        if self.clustering_method=="kmeans":
            
            # We remove N2 if necessary
            if self.remove_N2:
                n2_index = self.spec_list_ANN.index("N2")
                log_state = np.delete(log_state, n2_index+1)

            # Transformation
            if self.log_transform_X>0:
                log_state[log_state < self.threshold] = self.threshold
                if self.log_transform_X==1:
                    log_state[1:] = np.log(log_state[1:])
                elif self.log_transform_X==2:
                    log_state[1:] = (log_state[:, 1:]**self.lambda_bct - 1.0)/self.lambda_bct

            # Scaling vector
            vect_scaled = self.kmeans_scaler.transform(log_state.reshape(1, -1))
            # Applying k-means
            self.cluster = self.kmeans.predict(vect_scaled)[0]

        elif self.clustering_method=="progvar":
            self.cluster = 999
            for i in range(self.nb_clusters):
                if progvar>=self.c_bounds[i] and progvar<self.c_bounds[i+1]:
                    self.cluster = i
                elif progvar==1.0:
                    self.cluster = self.nb_clusters - 1
        # By default if no clustering, we assume that we only have a cluster 0
        else:
            self.cluster = 0


    def _inverse_scale(self, scaled_y, mean, std):
        """Inverse scaling helper for mass fractions (numpy version).

        scaled_y: array in scaled space
        mean, std: scaling statistics (numpy arrays)
        """
        y = mean + (std + 1e-7) * scaled_y

        return y

    def _scale(self, x, mean, std):
        """Scaling helper for mass fractions (numpy version).

        x: array in physical (original) space
        mean, std: scaling statistics (numpy arrays)
        """

        scaled_array = (x - mean) / (std + 1e-7)

        return scaled_array

#-----------------------------------------------------------------------
#   TIME ADVANCEMENT FUNTIONS 
#-----------------------------------------------------------------------
    # CVODE solver   
    def advance_state_CVODE(self, T_old, Y_old, pressure, dt):
        # Gas object modification
        self.gas.TPY= T_old, pressure, Y_old
            
        r = ct.IdealGasConstPressureReactor(self.gas)
        
        # Initializing reactor
        sim = ct.ReactorNet([r])
        states = ct.SolutionArray(self.gas, extra=['t'])
        
        # Advancing solution
        sim.advance(dt)
        states.append(r.thermo.state, t=dt)
        
        # New state
        T_new = states.T[0]
        Y_new = states.Y[0,:]

        return T_new, Y_new

    # NN model
    def advance_state_NN(self, T_old, Y_old, pressure, dt):
        # Boolean for IA computation success (used for hybrid model)
        self.ia_success = True

        # Gas object modification
        self.gas.TPY= T_old, pressure, Y_old

        # Grouping temperature and mass fractions in a "state" vector
        state_vector = np.append(T_old, Y_old)

        # If N2 is not considered, it needs to be removed from the state_vector
        if self.remove_N2:
            n2_index = self.spec_list_ANN.index("N2")
            n2_value = state_vector[n2_index+1]
            state_vector = np.delete(state_vector, n2_index+1)

        
        # NN update for Yk's
        if self.log_transform_X==1:
            state_vector[state_vector<self.threshold] = self.threshold
        elif self.log_transform_X==2:
            state_vector[state_vector<0.0] = 0.0

        # Build log_state input vector.
        # For dt_var=False: [T, species...]
        # For dt_var=True:  [T, species..., dt] with dt transformed consistently with database_processing.
        if self.dt_var:
            # state_vector currently contains [T, species...], dt comes from the argument.
            # We do NOT clip dt by species threshold; apply its own log transform with a small epsilon.
            dt_eps = 1e-300
            dt_logged = np.log(max(dt, dt_eps)) if self.log_transform_X > 0 else dt

            log_state = np.zeros(self.nb_species_ANN + 2)
            log_state[0] = state_vector[0]
            if self.log_transform_X == 1:
                log_state[1:self.nb_species_ANN+1] = np.log(state_vector[1:])
            elif self.log_transform_X == 2:
                log_state[1:self.nb_species_ANN+1] = (state_vector[1:]**self.lambda_bct - 1.0)/self.lambda_bct
            else:
                log_state[1:self.nb_species_ANN+1] = state_vector[1:]
            # Last entry is dt (optionally logged)
            log_state[-1] = dt_logged
        else:
            log_state = np.zeros(self.nb_species_ANN+1)
            log_state[0] = state_vector[0]
            if self.log_transform_X==1:
                log_state[1:] = np.log(state_vector[1:])
            elif self.log_transform_X==2:
                log_state[1:] = (state_vector[1:]**self.lambda_bct - 1.0)/self.lambda_bct
            else:
                log_state[1:] = state_vector[1:]
        
        # input of NN
        log_state = log_state.reshape(1, -1)
        mean_X = self.Xscaler_list[self.cluster][:, 0]
        std_X = self.Xscaler_list[self.cluster][:, 1]
        NN_input = self._scale(log_state, mean_X, std_X)

        # New state predicted by ANN
        NN_input = torch.tensor(NN_input, dtype=torch.float64).to(self.device)
        state_new = self.models_list[self.cluster](NN_input)
        state_new = state_new.detach().cpu().numpy()

        # Getting Y and inverse scaling (numpy version)
        mean_Y = self.Yscaler_list[self.cluster][:, 0]
        std_Y = self.Yscaler_list[self.cluster][:, 1]
        Y_new = self._inverse_scale(state_new, mean_Y, std_Y)

        # Log transform of species
        if self.log_transform_Y>0:
            if self.output_omegas:
                log_state_updated = log_state[0,1:] + Y_new
                if self.log_transform_Y==1:
                    Y_new = np.exp(log_state_updated)
                elif self.log_transform_Y==2:
                    Y_new = (self.lambda_bct*log_state_updated+1.0)**(1./self.lambda_bct)
            else:
                if self.log_transform_Y==1:
                    Y_new = np.exp(Y_new)
                elif self.log_transform_Y==2:
                    Y_new = (self.lambda_bct*Y_new+1.0)**(1./self.lambda_bct)
                
        # If reaction rate outputs from network
        if self.output_omegas and self.log_transform_Y==0:
            Y_new += state_vector[1:]   # Remark: state vector already contains information about N2 removal

        # Adding back N2 before computing temperature
        if self.remove_N2:
            Y_new = np.insert(Y_new, n2_index, n2_value)
        
        # Sum of Yk before renormalization (used for analysis)
        self.sum_Yk_before_renorm = Y_new.sum()

        # Hybrid model: checking if we satisfy the threshold
        if self.hybrid_ann_cvode:
            cons_criterion = np.abs(np.sum(Y_new,axis=0) - 1.0)
            if cons_criterion > self.hybrid_ann_cvode_tol:
                self.ia_success = False

        # Enforcing element conservation
        if self.yk_renormalization:
            Y_new = self.enforce_elements_balance(self.gas, Y_old, Y_new)
                
        # Deducing T from energy conservation
        T_new = state_vector[0] - (1/self.gas.cp)*np.sum(self.gas.partial_molar_enthalpies/self.gas.molecular_weights*(Y_new-Y_old))
        
        # Reshaping mass fraction vector
        Y_new = Y_new.reshape(Y_new.shape[0],1)

        return T_new, Y_new

#-----------------------------------------------------------------------
#   ANALYZING ERRORS
#-----------------------------------------------------------------------

    def compute_ann_errors(self, T, Y, pressure, dt, already_transformed=False):
        T_old = T.copy()

        # Dealing with the case where Y is already transformed (logged or bct) at input of function
        if already_transformed:
            if self.log_transform_X==1:
                Y_old = np.exp(Y)
            elif self.log_transform_X==2:
                Y_old = (self.lambda_bct*Y+1.0)**(1./self.lambda_bct)
        else:
            Y_old = Y.copy()

        # Assign to cluster
        # Computing current progress variable
        state = np.append(T_old, Y_old)
        progvar = self.compute_progvar(state, pressure, self.mechanism_type)
        self.progvar = progvar
        #
        if self.nb_clusters>0:
            self.attribute_cluster(state, progvar)
        else:
            self.cluster = 0
        #
        print(f"Current point in cluster: {self.cluster} \n")
            

        # CVODE run
        T_cvode, Y_cvode = self.advance_state_CVODE(T_old, Y_old, pressure, dt)

        # ANN run
        T_ann, Y_ann = self.advance_state_NN(T_old, Y_old, pressure, dt)

        # If N2 is not considered, it needs to be removed
        if self.remove_N2:
            n2_index = self.spec_list_ANN.index("N2")
            #
            Y_old = np.delete(Y_old, n2_index+1)
            Y_cvode = np.delete(Y_cvode, n2_index+1)
            Y_ann = np.delete(Y_ann, n2_index+1)

        # Taking log if necessary (if output is difference of logs)
        if self.log_transform_Y==1:
            Y_old[Y_old<self.threshold] = self.threshold
            Y_cvode[Y_cvode<self.threshold] = self.threshold
            Y_ann[Y_ann<self.threshold] = self.threshold
        elif self.log_transform_Y==2:
            Y_old[Y_old<0.0] = 0.0
            Y_cvode[Y_cvode<0.0] = 0.0
            Y_ann[Y_ann<0.0] = 0.0
        #
        if self.log_transform_Y==1:
            log_Y_old = np.log(Y_old)
            log_Y_cvode = np.log(Y_cvode)
            log_Y_ann = np.log(Y_ann)
        elif self.log_transform_Y==2:
            log_Y_old = (Y_old**self.lambda_bct - 1.0)/self.lambda_bct
            log_Y_cvode = (Y_cvode**self.lambda_bct - 1.0)/self.lambda_bct
            log_Y_ann = (Y_ann**self.lambda_bct - 1.0)/self.lambda_bct
        else:
            log_Y_old = Y_old
            log_Y_cvode = Y_cvode
            log_Y_ann = Y_ann


        # We compute difference of logs
        diff_log_Y_cvode = log_Y_cvode - log_Y_old
        diff_log_Y_ann = log_Y_ann - log_Y_old

        # Normalized differences in log space (using scaler statistics)
        mean_Y = self.Yscaler_list[self.cluster][:, 0]
        std_Y = self.Yscaler_list[self.cluster][:, 1]
        diff_log_Y_cvode_norm = self._scale(diff_log_Y_cvode, mean_Y, std_Y)
        diff_log_Y_ann_norm = self._scale(diff_log_Y_ann, mean_Y, std_Y)

        # Error in normalized log-space (MSE per species)
        err_Yk = (diff_log_Y_cvode_norm - diff_log_Y_ann_norm) ** 2

        err_T = 100.0*np.abs((T_ann-T_cvode)/T_cvode)

        return err_Yk, err_T


    def _load_validation_data(self):
        """Load X_val/Y_val for each cluster from training_data.h5.

        This mirrors NN_manager.read_training_data/get_scalers_stats and is
        used by evaluate_on_validation_set.
        """

        if self._val_data_loaded:
            return

        self._X_val = []
        self._Y_val = []

        with h5py.File(f"{self.dataset_path:s}/training_data.h5", 'r') as h5file_r:
            # Infer number of clusters from file (robust to future changes)
            top_level_groups = [k for k in h5file_r.keys() if isinstance(h5file_r[k], h5py.Group)]

            for grp_name in sorted(top_level_groups):
                grp = h5file_r[grp_name]
                X_val = grp['X_val'][:]
                Y_val = grp['Y_val'][:]
                self._X_val.append(X_val)
                self._Y_val.append(Y_val)

        self._val_data_loaded = True


    def evaluate_on_validation_set(self, scatter_folder="./plots_validation", max_points_scatter=50000):
        """Evaluate ANN predictions against Y_val stored in training_data.h5.

        For each cluster:
          * Read X_val / Y_val from training_data.h5
          * Run the corresponding ANN model
          * Compare predicted vs target values in scaled space and (approx.)
            physical space using the existing inverse-scaling helpers
          * Compute basic statistics (MSE, MAE, RMSE) per output dimension
          * Generate scatter plots of errors

        Parameters
        ----------
        scatter_folder : str
            Directory where scatter plots are written. Existing folder is
            removed.
        max_points_scatter : int
            Maximum number of validation points plotted per cluster for
            scatter plots (random subsampling if larger).
        """

        # Ensure models and scalers have been loaded
        self.load_models()
        self.load_scalers()

        # Load validation data from training_data.h5
        self._load_validation_data()

        # Prepare output folder
        if os.path.isdir(scatter_folder):
            shutil.rmtree(scatter_folder)
        os.makedirs(scatter_folder)

        all_stats = []

        for i_cluster in range(self.nb_clusters):
            X_val = self._X_val[i_cluster]
            Y_val = self._Y_val[i_cluster]

            # Convert to torch tensors on the correct device
            X_val_t = torch.tensor(X_val, dtype=torch.float64).to(self.device)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float64).to(self.device)

            # Model and scalers for current cluster
            model = self.models_list[i_cluster]
            model.to(self.device)
            model.eval()

            mean_X = torch.from_numpy(self.Xscaler_list[i_cluster][:, 0]).to(self.device).to(torch.float64)
            std_X = torch.from_numpy(self.Xscaler_list[i_cluster][:, 1]).to(self.device).to(torch.float64)
            mean_Y = torch.from_numpy(self.Yscaler_list[i_cluster][:, 0]).to(self.device).to(torch.float64)
            std_Y = torch.from_numpy(self.Yscaler_list[i_cluster][:, 1]).to(self.device).to(torch.float64)

            # Unscale inputs back to log/physical space to mimic runtime
            # convention, then rescale using local helper to be fully
            # consistent. This keeps this routine aligned with advance_state_NN.
            X_unscaled = self._inverse_scale(X_val_t, mean_X, std_X)

            # Forward pass
            with torch.no_grad():
                Y_pred_t = model(X_val_t)

            # Error metrics in scaled space
            diff_scaled = (Y_pred_t - Y_val_t).detach().cpu().numpy()
            mse_scaled = np.mean(diff_scaled**2, axis=0)
            mae_scaled = np.mean(np.abs(diff_scaled), axis=0)
            rmse_scaled = np.sqrt(mse_scaled)

            # Approximate physical-space outputs by inverse scaling
            Y_val_phys = self._inverse_scale(Y_val_t, mean_Y, std_Y)
            Y_pred_phys = self._inverse_scale(Y_pred_t, mean_Y, std_Y)
            diff_phys = (Y_pred_phys - Y_val_phys).detach().cpu().numpy()
            mse_phys = np.mean(diff_phys**2, axis=0)
            mae_phys = np.mean(np.abs(diff_phys), axis=0)
            rmse_phys = np.sqrt(mse_phys)

            # Aggregate statistics for this cluster
            stats_cluster = {
                "cluster": i_cluster,
                "mse_scaled": mse_scaled,
                "mae_scaled": mae_scaled,
                "rmse_scaled": rmse_scaled,
                "mse_phys": mse_phys,
                "mae_phys": mae_phys,
                "rmse_phys": rmse_phys,
            }
            all_stats.append(stats_cluster)

            # Scatter plot of physical-space errors (subsample if needed)
            n_pts = diff_phys.shape[0]
            if n_pts > max_points_scatter:
                idx = np.random.choice(n_pts, size=max_points_scatter, replace=False)
                diff_plot = diff_phys[idx, :]
            else:
                diff_plot = diff_phys

            fig, ax = plt.subplots(1, 1)
            # Flatten over species/output dimensions
            ax.scatter(np.arange(diff_plot.size), diff_plot.ravel(), s=1, alpha=0.5)
            ax.set_xlabel("Sample index (flattened)")
            ax.set_ylabel("Prediction error (physical units)")
            ax.set_title(f"Cluster {i_cluster} validation errors")
            fig.tight_layout()

            fig.savefig(os.path.join(scatter_folder, f"cluster_{i_cluster}_errors.png"), dpi=400)
            plt.close(fig)

        # Simple console summary
        for stats in all_stats:
            i_cluster = stats["cluster"]
            mse_phys_mean = float(np.mean(stats["mse_phys"]))
            rmse_phys_mean = float(np.mean(stats["rmse_phys"]))
            print(f"[Validation] Cluster {i_cluster}: mean MSE={mse_phys_mean:.3e}, mean RMSE={rmse_phys_mean:.3e}")

        return all_stats



#-----------------------------------------------------------------------
#   THERMO-CHEMICAL FUNCTIONS 
#-----------------------------------------------------------------------
    # Progress variable
    def compute_progvar(self, state, pressure):
        # Equilibrium state in present conditions
        pv_ind = self.pv_ind
        self.gas.TPY = state[0], pressure, state[1:]
        self.gas.equilibrate('HP')

        Yc_eq = 0.0
        Yc = 0.0
        for i in pv_ind:
            Yc_eq += self.gas.Y[i]
            Yc += state[i+1]
     
        progvar = Yc/Yc_eq

        # Clipping
        progvar = min(max(progvar, 0.0), 1.0)

        return progvar


    def enforce_elements_balance(self, gas, Y_old, Y_new):
        # Resulting vector
        Y_new_corr = Y_new.copy()

        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.parse_species_names(self.spec_list_ANN)

        # Species molecular weights
        molecular_weights = utils.get_molecular_weights(gas.species_names)

        # Initial elements mass fractions
        Y_a_in = np.dot(atomic_array,np.reshape(Y_old,-1)/molecular_weights)
        Y_a_in = np.multiply(self.W_atoms, Y_a_in)

        # Final elements mass fractions
        Y_a_out = np.dot(atomic_array,np.reshape(Y_new,-1)/molecular_weights)
        Y_a_out = np.multiply(self.W_atoms, Y_a_out)

        # Carbon: we correct CO2 (we test if present)
        if Y_a_in[0]>0:
            Y_new_corr[gas.species_index("CO2")] += -(Y_a_out[0] - Y_a_in[0])/ ((self.W_atoms[0]/molecular_weights[gas.species_index("CO2")]) * atomic_array[0,gas.species_index("CO2")])

        # Hydrogen: we correct H2
        Y_new_corr[gas.species_index("H2")] += -(Y_a_out[1] - Y_a_in[1])/ ((self.W_atoms[1]/molecular_weights[gas.species_index("H2")]) * atomic_array[1,gas.species_index("H2")])

        # Nitrogen: we correct N2
        Y_new_corr[gas.species_index("N2")] += -(Y_a_out[3] - Y_a_in[3])/ ((self.W_atoms[3]/molecular_weights[gas.species_index("N2")]) * atomic_array[3,gas.species_index("N2")])

        # We perform a new estimation of element mass fractions and then correct O using O2
        Y_a_out = np.dot(atomic_array,np.reshape(Y_new_corr,-1)/molecular_weights)
        Y_a_out = np.multiply(self.W_atoms, Y_a_out)
        Y_new_corr[gas.species_index("O2")] += -(Y_a_out[2] - Y_a_in[2])/ ((self.W_atoms[2]/molecular_weights[gas.species_index("O2")]) * atomic_array[2,gas.species_index("O2")])


        return Y_new_corr

