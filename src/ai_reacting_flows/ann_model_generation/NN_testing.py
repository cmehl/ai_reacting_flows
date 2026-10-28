import os
import sys
from pathlib import Path
import joblib
import pickle
import oyaml as yaml
import h5py

import numpy as np

import cantera as ct

import torch
from datetime import datetime
from torchinfo import summary
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, DeepONet, DeepONet_shift
import ai_reacting_flows.ann_model_generation.NN_testing_metrics as metrics
import ai_reacting_flows.ann_model_generation.NN_testing_plots as plots

import ai_reacting_flows.tools.utilities as utils
import ai_reacting_flows.tools.transforms as transforms
from ai_reacting_flows.databases_processing.processing_parameters import read_processing_parameters, MECHANISM_FILE


class NNTesting():

    def __init__(self, testing_parameters):

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.verbose = True

        self.run_folder = os.getcwd()
        # Simple text log file in the run folder to trace parameters actually
        # used by this testing helper. Use a per-run timestamped filename so
        # logs do not accumulate across different executions.
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.run_folder, f"nn_testing_{ts_run}.log")

        def _log(msg: str):
            """Append a timestamped message to the nn_testing.log file."""
            ts = datetime.now().isoformat(timespec="seconds")
            with open(self.log_path, "a") as _f:
                _f.write(f"[{ts}] {msg}\n")

        # Bind as instance method for reuse in other methods
        self._log = _log
        # Model folder name
        self.models_folder = f"{self.run_folder}/MODELS/{testing_parameters['models_folder']}"
        with open(os.path.join(self.models_folder, "networks_params.yaml"), "r") as file:
            networks_parameters = yaml.safe_load(file)
        self.dataset_path = os.path.join(self.run_folder, networks_parameters["database_path"])
        
        # Processing parameters of the learning database (stored in training_data.h5, see processing_parameters.py)
        params = read_processing_parameters(self.dataset_path, self.run_folder)

        self.dt_var = params["dt_var"]
        self.fuel = params["fuel"] # testing only takes 1 component fuel so far
        self.mech = os.path.join(self.dataset_path, MECHANISM_FILE)   # Mechanism copied in the database folder

        self.log_transform_X = params["log_transform_X"]
        self.log_transform_Y = params["log_transform_Y"]
        # Box-Cox parameter (kept consistent with NN_manager)
        self.lambda_bct = params["lambda_bct"]
        self.threshold = params["threshold"]
        self.remove_N2 = not params["with_N_chemistry"]
        self.output_omegas = params["output_omegas"]
        # General list of species for which input/output log/BCT should be
        # skipped, mirroring LearningDatabase.log_excluded_species.
        self.log_excluded_species = params["log_excluded_species"]

        self.clustering_method  = params["clustering_method"]
        # Keep track of the clustering feature layout used at database-processing
        # time ("phys", "dt", "all"). This is needed to build a consistent
        # k-means input vector here.
        self.clusterize_on = params["clusterize_on"]

        # Chemistry (maybe an information to get from model folder ?)
        self.spec_to_plot = testing_parameters["spec_to_plot"]

        # Hybrid computation
        self.hybrid_ann_cvode = testing_parameters["hybrid_ann_cvode"]
        self.hybrid_ann_cvode_tol = testing_parameters["hybrid_ann_cvode_tol"]

        # Renormalization of Yk to satisfy elements conservation
        self.yk_renormalization = testing_parameters["yk_renormalization"]

        # CANTERA solution object needed to access species indices
        self.gas = ct.Solution(self.mech)
        utils.check_elements(self.gas)   # Only species made of C, H, O and N are handled
        self.spec_list_ref = self.gas.species_names
        self.nb_species_ref = len(self.spec_list_ref)

        # Getting species and number of species
        self.spec_list_ANN = self.spec_list_ref
        self.nb_species_ANN = self.nb_species_ref
        if self.remove_N2:
            # We drop N2 from the ANN input/output space when with_N_chemistry
            # is disabled in dtb_processing.
            self.spec_list_ANN = [spec for spec in self.spec_list_ref if spec != "N2"]
            self.nb_species_ANN -= 1

        # Find number of clusters -> equal for instance to number of json files in model folder
        self.nb_clusters = len(list(Path(self.models_folder).glob("*.pth")))
        print(f">> {self.nb_clusters} models were found")

        # Check consistency with the processing parameters, could help identify an issue
        assert self.nb_clusters==params["nb_clusters"]

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
        self.pv_species = params["pv_species"]
        self.pv_ind = []
        for i in range(len(self.pv_species)):
            self.pv_ind.append(self.gas.species_index(self.pv_species[i]))

        # Some constants used in the code
        self.W_atoms = utils.get_atomic_weights() # Order: C, H, O, N

        # Cached validation data (optional, used by evaluate_on_validation_set)
        self._val_data_loaded = False
        self._X_val = []
        self._Y_val = []

        # Log a concise summary of key parameters used for testing so that
        # discrepancies with dtb_processing / NN_manager are easy to diagnose.
        self._log(
            "INIT: "
            f"models_folder={self.models_folder}, "
            f"dataset_path={self.dataset_path}, "
            f"dt_var={self.dt_var}, "
            f"fuel={self.fuel}, "
            f"output_omegas={self.output_omegas}, "
            f"remove_N2={self.remove_N2}, "
            f"clusterize_on={self.clusterize_on}, "
            f"clustering_method={self.clustering_method}, "
            f"nb_clusters={self.nb_clusters}, "
            f"nb_species_ref={self.nb_species_ref}, "
            f"nb_species_ANN={self.nb_species_ANN}"
        )

#-----------------------------------------------------------------------
#   0D TEST CASES
#-----------------------------------------------------------------------

    def _run_0D_case(self, T0, pressure, dt, nb_ite, gas_init_compo, reactor_type="ignition"):
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
        reactor_type : str
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

        # Progress variable species mass fraction in the initial (unburnt) mixture
        Yc_u = Y_ini[0, self.pv_ind].sum()

        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.get_atomic_array(self.gas)

        #-------------------- CVODE COMPUTATION---------------------------
        for _ in range(nb_ite):
            simtime += dt
            sim.advance(simtime)
            states.append(r.thermo.state, t=simtime*1.0e3)

            if reactor_type=="ignition" and states.T[-1] > T0+200.0 and tau_ign_CVODE==0.0:
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
        progvar = self.compute_progvar(np.append(T_ann, Y_k_ann), pressure, Yc_u)
        progvar_vect = [progvar]

        for model in self.models_list:
            model.to(self.device)
            model.eval()

        # atomic composition of species
        molecular_weights = self.gas.molecular_weights
        atomic_cons = np.dot(atomic_array,np.reshape(Y_k_ann,-1)/molecular_weights)
        atomic_cons = np.multiply(self.W_atoms, atomic_cons)

        # NEURAL NETWORK COMPUTATION
        time = 0.0
        state = np.append(T_ann, Y_k_ann)
        for i in range(nb_ite):
            # Old state
            T_old = state[0]
            Y_old = state[1:]

            # Attribute cluster for current state before ANN advancement.
            if self.nb_clusters > 0:
                self.attribute_cluster(state, dt, progvar)
            else:
                self.cluster = 0

            # Log a compact line for each ANN advancement (index, T_old,
            # dt, cluster, and whether dt is part of the clustering vector).
            self._log(
                "0D_STEP "
                f"i={i}, "
                f"T_old={float(T_old):.6e}, "
                f"dt={float(dt):.6e}, "
                f"cluster={self.cluster}, "
            )

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

            # Progress variable of the new state (used for clustering at next step)
            progvar = self.compute_progvar(state, pressure, Yc_u)

            # Saving values
            state_save = np.vstack([state_save,state])
            progvar_vect.append(progvar)

            # Mass conservation
            sumYs = np.append(sumYs, np.sum(Y_new,axis=0))

            # atomic composition of species
            atomic_cons_current = np.dot(atomic_array,Y_new.reshape(self.nb_species_ref)/molecular_weights)
            atomic_cons_current = np.multiply(self.W_atoms, atomic_cons_current)
            atomic_cons = np.vstack([atomic_cons,atomic_cons_current])

            time += dt

            if reactor_type=="ignition" and T_new > T0+200.0 and tau_ign_ANN==0.0:
                tau_ign_ANN = time

        # ANN equilibrium (last temperature)
        Teq_ann = T_new

        # Use configurable prefix to label ignition-like delays
        return {
            "time_vect": states.t,
            "states_ref": state_ref,
            "states_ann": state_save,
            "tau_ign_cvode": tau_ign_CVODE,
            "tau_ign_ann": tau_ign_ANN,
            "Teq_ref": Teq_ref,
            "Teq_ann": Teq_ann,
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

        Parameters
        ----------
        phi : float
            Equivalence ratio.
        T0 : float
            Initial temperature [K].
        pressure : float
            Initial pressure [Pa].
        dt : float
            Time step used by both CVODE and ANN loops.
        nb_ite : int, optional
            Number of time steps.

        Returns
        -------
        dict
            Structured results in a dictionary.
        """

        # Mixture composition: premixed fuel + air parameterized by phi
        fuel_ox_ratio = (
            self.gas.n_atoms(self.fuel, 'C')
            + 0.25 * self.gas.n_atoms(self.fuel, 'H')
            - 0.5 * self.gas.n_atoms(self.fuel, 'O')
        )
        compo = f"{self.fuel}:{phi}, O2:{fuel_ox_ratio}, N2:{fuel_ox_ratio * 0.79 / 0.21}"

        return self._run_0D_case(T0, pressure, dt, nb_ite, compo, reactor_type="ignition")


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
            Structured results in the same format as ignition cases.
        """

        if diluent is None:
            # Pure fuel case
            compo = f"{self.fuel}:1.0"
        else:
            # Fuel + inert diluent, simple two-component mixture
            X_fuel = max(0.0, 1.0 - X_diluent)
            compo = f"{self.fuel}:{X_fuel:3.2f}, {diluent}:{X_diluent:3.2f}"

        return self._run_0D_case(T0, pressure, dt, nb_ite, compo, reactor_type="pyrolysis")

    def run_0D_series(self, cases, mode="ignition"):
        """Run a series of 0D cases (ignition or pyrolysis) and aggregate errors.

        Parameters
        ----------
        cases : iterable of dict
            For ``mode="ignition"``: each dict must contain keys compatible
            with :meth:`run_0D_ignition_case`, e.g. ``{"phi": ..., "T0": ...,
            "pressure": ..., "dt": ..., "nb_ite": ...}``.

            For ``mode="pyrolysis"``: each dict must contain keys compatible
            with :meth:`run_0D_pyrolysis_case`, e.g. ``{"T0": ..., "pressure":
            ..., "dt": ..., "nb_ite": ..., "diluent": ..., "X_diluent": ...}``.

        mode : {"ignition", "pyrolysis"}
            Selects which 0D case runner to use and how to interpret the
            aggregated delay metrics.

        Returns
        -------
        results : list[dict]
            Per-case results (direct outputs from the selected single-case
            runner) with the corresponding input parameters attached under key
            "params".
        summary : dict
            Aggregate metrics across all cases: average and standard deviation
            of the equilibrium-temperature error.
        """

        if mode not in ("ignition", "pyrolysis"):
            raise ValueError(f"Unsupported 0D mode: {mode}")

        results = []

        err_Teq_list = []
        ann_calls_list = []
        cvode_calls_list = []

        for case in cases:
            if mode == "ignition":
                res = self.run_0D_ignition_case(**case)
            else:
                res = self.run_0D_pyrolysis_case(**case)

            # Attach input parameters to the result for traceability
            res_with_params = {"params": dict(case), **res}
            results.append(res_with_params)

            # equilibrium temperature error (percent)
            err_Teq_list.append(metrics.relative_error_percent(res["Teq_ref"], res["Teq_ann"]))

            ann_calls_list.append(res["ann_calls"])
            cvode_calls_list.append(res["cvode_calls"])

        err_Teq_mean, err_Teq_std = metrics.mean_std(err_Teq_list)
        ann_calls_arr = np.array(ann_calls_list)
        cvode_calls_arr = np.array(cvode_calls_list)

        # Summary
        summary = {
            "n_cases": len(results),
            # Equilibrium temperature error stats (percent)
            "Teq_error_mean_percent": err_Teq_mean,
            "Teq_error_std_percent": err_Teq_std,
            "mode": mode,
        }

        return results, summary
    

    # OD IGNITION (wrapper with plotting)
    def test_0D_ignition(self, phi, T0, pressure, dt, nb_ite=100):

        result = self.run_0D_ignition_case(phi, T0, pressure, dt, nb_ite)

        tau_ign_CVODE = result["tau_ign_cvode"]
        tau_ign_ANN = result["tau_ign_ann"]
        ann_calls = result["ann_calls"]
        cvode_calls = result["cvode_calls"]

        # Error on equilibrium temperature
        error_Teq = metrics.relative_error_percent(result["Teq_ref"], result["Teq_ann"])

        if self.verbose:
            print(f"\nError on equilibrium flame temperature is: {error_Teq} % \n")

        print("\n NUMBER OF CALLS TO SOLVERS:")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        # Plots
        plots.plot_0D_case(result, f'./plots_0D_T0#{T0}_phi#{phi}', self.spec_to_plot, self.spec_list_ref, self.remove_N2)

        # Ignition delays comparison
        print(f" >> CVODE ignition delay: {tau_ign_CVODE}")
        print(f" >> ANN ignition delay: {tau_ign_ANN}")


    def test_0D_pyrolysis(self, T0, pressure, dt, nb_ite=100, diluent=None, X_diluent=0.0):
        """Wrapper around run_0D_pyrolysis_case with plotting.

        Same plots as test_0D_ignition, but folder naming and printed labels
        reflect the pyrolysis case and do not include phi.
        """

        result = self.run_0D_pyrolysis_case(T0, pressure, dt, nb_ite, diluent=diluent, X_diluent=X_diluent)

        ann_calls = result["ann_calls"]
        cvode_calls = result["cvode_calls"]

        # Error on equilibrium temperature
        error_Teq = metrics.relative_error_percent(result["Teq_ref"], result["Teq_ann"])

        if self.verbose:
            print(f"\nError on equilibrium temperature (pyrolysis case) is: {error_Teq} % \n")

        print("\n NUMBER OF CALLS TO SOLVERS (pyrolysis):")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        # Plots (no phi in folder naming)
        plots.plot_0D_case(result, f'./plots_0D_pyro_P#{pressure}_T0#{T0}_dt#{dt}', self.spec_to_plot, self.spec_list_ref, self.remove_N2)

#-----------------------------------------------------------------------
#   1D PREMIXED TEST CASE
#-----------------------------------------------------------------------

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
        compo = f'{self.fuel}:{phi}, O2:{fuel_ox_ratio}, N2:{fuel_ox_ratio * 0.79 / 0.21}'

        self.gas.TPX = T0, pressure, compo

        # Progress variable species mass fraction in the fresh gases
        Yc_u = self.gas.Y[self.pv_ind].sum()

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
            c[i] = self.compute_progvar(state, pressure, Yc_u)

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

            # Computing current progress variable on physical state
            state_phys = np.append(Tt[i_reac], Yt_ann[:, i_reac])
            progvar = self.compute_progvar(state_phys, pressure, Yc_u)

            # Attribute cluster
            if self.nb_clusters > 0:
                self.attribute_cluster(state_phys, dt, progvar)
            else:
                self.cluster = 0

            # Log parameters used for clustering in 1D case at this spatial
            # location (x index only; grid position can be inferred from f.grid
            # if needed).
            self._log(
                "1D_CLUSTER "
                f"i_reac={i_reac}, "
                f"T={float(Tt[i_reac]):.6e}, "
                f"dt={float(dt):.6e}, "
                f"cluster={self.cluster}, "
            )

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

            # L2 error in physical units over x-window
            for spec in species_for_error:
                k = self.gas.species_index(spec)
                species_err[spec].append(metrics.l2_error(f.grid, Omega_ann[k], Omega_exact[k], x_window))

            res_with_params = {"params": dict(case), **res}
            results.append(res_with_params)

        ann_calls_arr = np.array(ann_calls_list)
        cvode_calls_arr = np.array(cvode_calls_list)

        species_err_mean = {}
        species_err_std = {}
        for spec, errs in species_err.items():
            species_err_mean[spec], species_err_std[spec] = metrics.mean_std(errs)

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

        # Plots
        plots.plot_1D_reaction_rates(f.grid, Omega_exact, Omega_ann, self.spec_to_plot, self.spec_list_ref, f'./plots_1D_prem_T0#{T0}_phi#{phi}')

#-----------------------------------------------------------------------
#   IA MODEL HANDLING FUNCTIONS
#-----------------------------------------------------------------------

    # Functions to load models
    def load_models(self):
        self.models_list = []
        
        for i_cluster in range(self.nb_clusters):
            model = torch.load(os.path.join(self.models_folder, f"cluster{i_cluster}_model.pth"), weights_only=False)
            # Add to list
            self.models_list.append(model)


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

                Xscaler_std = transforms.scaler_std(Xscaler_var)
                Yscaler_std = transforms.scaler_std(Yscaler_var)

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


    def attribute_cluster(self, state, dt, progvar):
        """Cluster of the state [T [K], Y_k [-]] (all species) advanced by dt [s], with progress variable progvar [-].

        k-means features built as in LearningDatabase.clusterize_dataset (tools/transforms.py).
        """

        if self.clustering_method=="kmeans":

            # Species of the network (N2 removed if necessary)
            Y = np.array([state[1 + self.spec_list_ref.index(spec)] for spec in self.spec_list_ANN])

            # Clustering on the time step only makes sense with multiple dt
            clusterize_on = self.clusterize_on if self.dt_var else "phys"

            feat = transforms.kmeans_features(np.array([state[0]]), Y.reshape(1, -1), np.array([dt]), self.spec_list_ANN,
                                              clusterize_on, self.log_transform_X, self.threshold, self.lambda_bct,
                                              self.log_excluded_species)

            # --------------------------------------------------------------
            # Scale and predict cluster
            # --------------------------------------------------------------
            # Scaling vector using the same StandardScaler that was fitted
            # during database_processing.clusterize_dataset.
            vect_scaled = self.kmeans_scaler.transform(feat.reshape(1, -1))
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
            n2_index = self.spec_list_ref.index("N2")
            n2_value = state_vector[n2_index+1]
            state_vector = np.delete(state_vector, n2_index+1)
        
        # Network input: [T, transformed species, (transformed dt)], as in LearningDatabase.process_database
        # (tools/transforms.py). Species of log_excluded_species stay on a linear scale.
        x_species_t = transforms.transform_species(state_vector[1:].reshape(1, -1), self.spec_list_ANN, self.log_transform_X,
                                                   self.threshold, self.lambda_bct, self.log_excluded_species)
        log_state = np.column_stack([[[state_vector[0]]], x_species_t])
        if self.dt_var:
            log_state = np.column_stack([log_state, [[transforms.transform_dt(dt, self.log_transform_X)]]])

        # input of NN
        mean_X = self.Xscaler_list[self.cluster][:, 0]
        std_X = self.Xscaler_list[self.cluster][:, 1]
        NN_input = transforms.scale(log_state, mean_X, std_X)
        
        # New state predicted by ANN
        NN_input = torch.tensor(NN_input, dtype=torch.float64).to(self.device)
        state_new = self.models_list[self.cluster](NN_input)
        state_new = state_new.detach().cpu().numpy()

        # Getting Y and inverse scaling (numpy version)
        mean_Y = self.Yscaler_list[self.cluster][:, 0]
        std_Y = self.Yscaler_list[self.cluster][:, 1]
        y_t = transforms.unscale(state_new, mean_Y, std_Y)

        # Mass fractions at t+dt (outputs as differences or not, in transformed space or not)
        Y_new = transforms.output_to_Yk(x_species_t, y_t, self.spec_list_ANN, self.output_omegas, self.log_transform_X,
                                        self.log_transform_Y, self.lambda_bct, self.log_excluded_species)

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
        Y_new = Y_new.T

        return T_new, Y_new

#-----------------------------------------------------------------------
#   ANALYZING ERRORS
#-----------------------------------------------------------------------

    #WARNING: function not up to date (actually not used), it does not consider log excluded species for instance
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
        # Computing current progress variable on physical state
        state_phys = np.append(T_old, Y_old)
        progvar = self.compute_progvar(state_phys, pressure, self.mechanism_type)
        self.progvar = progvar

        if self.nb_clusters > 0:
            self.attribute_cluster(state_phys, dt, progvar)
        else:
            self.cluster = 0

        # Log the parameters used in this error-computation call for
        # debugging consistency with dtb_processing / NN_manager.
        self._log(
            "ERR_CLUSTER "
            f"T={float(T_old):.6e}, "
            f"dt={float(dt):.6e}, "
            f"cluster={self.cluster}, "
            f"clusterize_on={self.clusterize_on}, "
        )
            

        # CVODE run
        T_cvode, Y_cvode = self.advance_state_CVODE(T_old, Y_old, pressure, dt)

        # ANN run
        T_ann, Y_ann = self.advance_state_NN(T_old, Y_old, pressure, dt)

        # If N2 is not considered, it needs to be removed
        if self.remove_N2:
            n2_index = self.spec_list_ref.index("N2")
            #
            Y_old = np.delete(Y_old, n2_index+1)
            Y_cvode = np.delete(Y_cvode, n2_index+1)
            Y_ann = np.delete(Y_ann, n2_index+1)

        # Taking log if necessary (if output is difference of logs)
        if self.log_transform_Y==1:
            # Apply threshold only to non-excluded species.
            for i, spec in enumerate(self.spec_list_ANN):
                if spec in self.log_excluded_species:
                    continue
                idx = i  # here Y_* already correspond to species only
                if idx >= Y_old.shape[0]:
                    break
                if Y_old[idx] < self.threshold:
                    Y_old[idx] = self.threshold
                if Y_cvode[idx] < self.threshold:
                    Y_cvode[idx] = self.threshold
                if Y_ann[idx] < self.threshold:
                    Y_ann[idx] = self.threshold
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
        diff_log_Y_cvode_norm = transforms.scale(diff_log_Y_cvode, mean_Y, std_Y)
        diff_log_Y_ann_norm = transforms.scale(diff_log_Y_ann, mean_Y, std_Y)

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
        plots.new_folder(scatter_folder)

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

            std_Y = self.Yscaler_list[i_cluster][:, 1]

            # Forward pass
            with torch.no_grad():
                Y_pred_t = model(X_val_t)

            # Prediction errors in scaled space
            diff_scaled = (Y_pred_t - Y_val_t).detach().cpu().numpy()

            # Statistics for this cluster, in scaled and unscaled spaces
            all_stats.append({"cluster": i_cluster, **metrics.validation_errors(diff_scaled, std_Y)})

            # Scatter plot of unscaled errors (the mean of the scaler cancels in the difference)
            plots.plot_validation_errors(diff_scaled * std_Y, i_cluster, scatter_folder, max_points_scatter)

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
    def compute_progvar(self, state, pressure, Yc_u):
        """Progress variable c = (Yc - Yc_u) / (Yc_eq - Yc_u) [-], clipped to [0, 1].

        Same definition as in the databases (stochastic particles and flamelets).
        state: [T [K], Y_k [-]], pressure [Pa], Yc_u: progress variable species mass fraction
        in the unburnt mixture [-]. Yc_eq is computed at equilibrium (HP) from the current state.
        """
        # Equilibrium state in present conditions
        self.gas.TPY = state[0], pressure, state[1:]
        self.gas.equilibrate('HP')

        Yc_eq = self.gas.Y[self.pv_ind].sum()
        Yc = state[1:][self.pv_ind].sum()

        denom = Yc_eq - Yc_u
        progvar = (Yc - Yc_u) / denom if denom > 1e-10 else 0.0

        # Clipping
        progvar = min(max(progvar, 0.0), 1.0)

        return progvar


    def enforce_elements_balance(self, gas, Y_old, Y_new):
        # Resulting vector
        Y_new_corr = Y_new.copy()

        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.get_atomic_array(gas)

        # Species molecular weights
        molecular_weights = gas.molecular_weights

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
