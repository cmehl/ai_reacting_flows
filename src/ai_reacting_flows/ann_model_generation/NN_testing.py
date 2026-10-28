import os
import itertools
import h5py
import oyaml as yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cantera as ct

import torch
from datetime import datetime
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, DeepONet, DeepONet_shift
from ai_reacting_flows.ann_model_generation.NN_inference import NNInference
import ai_reacting_flows.ann_model_generation.NN_testing_metrics as metrics
import ai_reacting_flows.ann_model_generation.NN_testing_plots as plots

import ai_reacting_flows.tools.utilities as utils
import ai_reacting_flows.tools.transforms as transforms


# Types of test cases, sections of nn_testing.yaml
CASE_TYPES = ["0D_ignition", "0D_pyrolysis", "1D_premixed", "1D_diffusion"]

# Keys of a series section of nn_testing.yaml that are not parameters of the cases
SERIES_OPTIONS = ["run", "plots", "x_window"]


def series_cases(series):
    """Cases (list of dicts of parameters) of a series section of nn_testing.yaml: parameters given
    as lists are swept and all combinations are returned, other parameters are the same for all cases."""

    params = {key: value for key, value in series.items() if key not in SERIES_OPTIONS}
    values = [value if isinstance(value, list) else [value] for value in params.values()]

    return [dict(zip(params.keys(), combination)) for combination in itertools.product(*values)]


def print_errors(errors):
    """Print the errors of one case."""

    for key, value in errors.items():
        print(f"   >>> {key}: {value}")


class NNTesting():

    def __init__(self, testing_parameters):
        """testing_parameters: content of nn_testing.yaml (see README). Only models_folder, spec_to_plot,
        yk_renormalization, hybrid_ann_cvode and hybrid_ann_cvode_tol are needed to run single cases."""

        self.testing_parameters = testing_parameters

        self.run_folder = os.getcwd()

        # Outputs (plots, summaries, log) in TESTING_<models_folder>/ of the run folder, with a copy of the parameters
        self.output_folder = os.path.join(self.run_folder, f"TESTING_{testing_parameters['models_folder']}")
        os.makedirs(self.output_folder, exist_ok=True)
        with open(os.path.join(self.output_folder, "nn_testing.yaml"), "w") as file:
            yaml.dump(testing_parameters, file, default_flow_style=False, sort_keys=False)

        # Simple text log file in the output folder to trace parameters actually
        # used by this testing helper. Use a per-run timestamped filename so
        # logs do not accumulate across different executions.
        ts_run = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.output_folder, f"nn_testing_{ts_run}.log")

        def _log(msg: str):
            """Append a timestamped message to the nn_testing.log file."""
            ts = datetime.now().isoformat(timespec="seconds")
            with open(self.log_path, "a") as _f:
                _f.write(f"[{ts}] {msg}\n")

        # Bind as instance method for reuse in other methods
        self._log = _log
        # Model folder name
        self.models_folder = f"{self.run_folder}/MODELS/{testing_parameters['models_folder']}"

        # ANN chemistry: networks, scalers, clustering and processing parameters (shared with the stochastic reactors)
        self.inference = NNInference(self.models_folder, self.run_folder)

        # Chemistry of the model, used by the reference (CVODE) computations
        self.fuel = self.inference.fuel
        self.gas = self.inference.gas
        self.spec_list_ref = self.inference.spec_list_ref
        self.nb_species_ref = self.inference.nb_species_ref
        self.nb_species_ANN = self.inference.nb_species_ANN
        self.remove_N2 = self.inference.remove_N2
        self.pv_ind = self.inference.pv_ind
        self.W_atoms = self.inference.W_atoms

        # Chemistry (maybe an information to get from model folder ?)
        self.spec_to_plot = testing_parameters["spec_to_plot"]

        # Hybrid computation
        self.hybrid_ann_cvode = testing_parameters["hybrid_ann_cvode"]
        self.hybrid_ann_cvode_tol = testing_parameters["hybrid_ann_cvode_tol"]

        # Renormalization of Yk to satisfy elements conservation
        self.yk_renormalization = testing_parameters["yk_renormalization"]

        # Standard deviations of T [K] and transformed Y_k of the network species on the training dataset
        # (all clusters), normalization of the simulation error E_sim (see case_errors)
        self.std_train = self._training_std()

        # Cached validation data (optional, used by evaluate_on_validation_set)
        self._val_data_loaded = False
        self._X_val = []
        self._Y_val = []

        # Log a concise summary of key parameters used for testing so that
        # discrepancies with dtb_processing / NN_manager are easy to diagnose.
        self._log(
            "INIT: "
            f"models_folder={self.models_folder}, "
            f"dataset_path={self.inference.dataset_path}, "
            f"dt_var={self.inference.dt_var}, "
            f"fuel={self.fuel}, "
            f"output_omegas={self.inference.output_omegas}, "
            f"remove_N2={self.remove_N2}, "
            f"clusterize_on={self.inference.clusterize_on}, "
            f"clustering_method={self.inference.clustering_method}, "
            f"nb_clusters={self.inference.nb_clusters}, "
            f"nb_species_ref={self.nb_species_ref}, "
            f"nb_species_ANN={self.nb_species_ANN}"
        )

#-----------------------------------------------------------------------
#   0D TEST CASES
#-----------------------------------------------------------------------

    def _run_0D_case(self, T0, pressure, dt, max_simulation_time, T_eq_tolerance, gas_init_compo, reactor_type="ignition"):
        """Internal helper to run one 0D constant-pressure reactor case.

        Parameters
        ----------
        T0 : float
            Initial temperature [K].
        pressure : float
            Initial pressure [Pa].
        dt : float
            Time step used by both CVODE and ANN loops [s].
        max_simulation_time : float
            Maximum end time [s], used only if CVODE does not reach equilibrium before.
        T_eq_tolerance : float
            CVODE stops when its temperature is within T_eq_tolerance [K] of the HP equilibrium
            temperature of the initial mixture. The ANN is run for the same number of steps.
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

        # Equilibrium temperature of the initial mixture (constant enthalpy and pressure) [K]
        self.gas.equilibrate('HP')
        T_eq = self.gas.T
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
        # Until equilibrium (or max_simulation_time if not reached)
        equilibrium_reached = False
        while not equilibrium_reached and simtime < max_simulation_time - 0.5 * dt:
            simtime += dt
            sim.advance(simtime)
            states.append(r.thermo.state, t=simtime*1.0e3)

            if reactor_type=="ignition" and states.T[-1] > T0+200.0 and tau_ign_CVODE==0.0:
                tau_ign_CVODE = simtime

            equilibrium_reached = abs(states.T[-1] - T_eq) <= T_eq_tolerance

        # Number of time steps, the same for the ANN
        nb_ite = len(states.T) - 1

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
            self.attribute_cluster(state, dt, progvar)

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
            "T_eq": T_eq,
            "equilibrium_reached": equilibrium_reached,
            "Teq_ann": Teq_ann,
            "atomic_cons": atomic_cons,
            "sumYs": sumYs,
            "progvar": np.array(progvar_vect),
            "ann_calls": ann_calls,
            "cvode_calls": cvode_calls,
        }

    def run_0D_ignition_case(self, phi, T0, pressure, dt, max_simulation_time, T_eq_tolerance=1.0):
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
        max_simulation_time : float
            Maximum end time [s] if equilibrium is not reached before.
        T_eq_tolerance : float
            Equilibrium criterion [K] ending the simulation (see _run_0D_case).

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

        return self._run_0D_case(T0, pressure, dt, max_simulation_time, T_eq_tolerance, compo, reactor_type="ignition")


    def run_0D_pyrolysis_case(self, T0, pressure, dt, max_simulation_time, T_eq_tolerance=1.0, diluent=None, X_diluent=0.0):
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
        max_simulation_time : float
            Maximum end time [s] if equilibrium is not reached before.
        T_eq_tolerance : float
            Equilibrium criterion [K] ending the simulation (see _run_0D_case).
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

        return self._run_0D_case(T0, pressure, dt, max_simulation_time, T_eq_tolerance, compo, reactor_type="pyrolysis")

    # OD IGNITION (wrapper with plotting)
    def test_0D_ignition(self, phi, T0, pressure, dt, max_simulation_time, T_eq_tolerance=1.0):
        """One 0D ignition case with printed errors and plots (see run_case)."""

        return self.test_case("0D_ignition", {"phi": phi, "T0": T0, "pressure": pressure, "dt": dt,
                                              "max_simulation_time": max_simulation_time, "T_eq_tolerance": T_eq_tolerance})


    def test_0D_pyrolysis(self, T0, pressure, dt, max_simulation_time, T_eq_tolerance=1.0, diluent=None, X_diluent=0.0):
        """One 0D pyrolysis case with printed errors and plots (see run_case)."""

        return self.test_case("0D_pyrolysis", {"T0": T0, "pressure": pressure, "dt": dt, "max_simulation_time": max_simulation_time,
                                               "T_eq_tolerance": T_eq_tolerance, "diluent": diluent, "X_diluent": X_diluent})

#-----------------------------------------------------------------------
#   1D TEST CASES
#-----------------------------------------------------------------------

    def run_1D_premixed_case(self, phi, T0, pressure, dt, T_threshold=0.0):
        """Run one 1D premixed flame case and return structured results.

        Freely propagating flame of fuel/air at equivalence ratio phi, fresh gases at T0 [K] and
        pressure [Pa], unity Lewis numbers. ANN and CVODE reaction rates compared a priori on the
        converged flame (see _a_priori_reaction_rates).
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

        return self._a_priori_reaction_rates(f, pressure, dt, T_threshold, np.full(len(f.grid), Yc_u))


    def run_1D_diffusion_case(self, strain, T_ox, T_fuel, pressure, dt, width=0.02, T_threshold=0.0):
        """Run one 1D counterflow diffusion flame case and return structured results.

        Pure fuel at T_fuel [K] against air at T_ox [K], at pressure [Pa], strain rate strain [1/s],
        distance between the inlets width [m], mixture-averaged transport: same setup as the
        flamelets database (database_flmts.compute_1d_diffusion). ANN and CVODE reaction rates
        compared a priori on the converged flame (see _a_priori_reaction_rates).
        """

        loglevel = 0  # amount of diagnostic output (0 to 8)

        # Stream compositions
        compo_ox = 'O2:0.21, N2:0.79'
        compo_fuel = f'{self.fuel}:1'

        self.gas.TPX = T_ox, pressure, compo_ox
        density_ox = self.gas.density
        Yc_ox = self.gas.Y[self.pv_ind].sum()

        self.gas.TPX = T_fuel, pressure, compo_fuel
        density_fuel = self.gas.density
        Yc_fuel = self.gas.Y[self.pv_ind].sum()

        # Stream mass flow rates [kg/m2/s]
        vel = strain * width / 2.0
        mdot_ox = density_ox * vel
        mdot_fuel = density_fuel * vel

        f = ct.CounterflowDiffusionFlame(self.gas, width=width)

        f.transport_model = 'mixture-averaged'
        f.soret_enabled = False
        f.radiation_enabled = False

        f.set_max_jac_age(10, 10)
        f.set_time_step(1e-5, [2, 5, 10, 20])

        # Set the state of the two inlets
        f.fuel_inlet.mdot = mdot_fuel
        f.fuel_inlet.X = compo_fuel
        f.fuel_inlet.T = T_fuel

        f.oxidizer_inlet.mdot = mdot_ox
        f.oxidizer_inlet.X = compo_ox
        f.oxidizer_inlet.T = T_ox

        f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)
        f.solve(loglevel=loglevel, auto=True)

        # Unburnt progress variable species mass fraction: mixing of the two streams at the local mixture fraction
        Z = f.mixture_fraction("Bilger")
        Yc_u = Z * Yc_fuel + (1.0 - Z) * Yc_ox

        return self._a_priori_reaction_rates(f, pressure, dt, T_threshold, Yc_u)


    def _a_priori_reaction_rates(self, f, pressure, dt, T_threshold, Yc_u):
        """Reaction rates [1/s] of the converged 1D flame f, (Y_k(t+dt) - Y_k) / dt, with each point
        advanced by dt [s] with CVODE and with the ANN (points below T_threshold [K] are frozen for the ANN).

        Yc_u: unburnt progress variable species mass fraction at each point [-] (progress variable for clustering).
        """

        # Mass fractions and temperature at steady state
        Yt = f.Y
        Tt = f.T

        # Computing progress variable
        c = np.empty(len(f.grid))
        nb_pnts_total = len(f.grid)
        for i in range(nb_pnts_total):
            state = np.append(Tt[i], Yt[:,i])
            c[i] = self.compute_progvar(state, pressure, Yc_u[i])

        X_grid = f.grid

        # Initializing T and Y at t+dt
        Yt_dt_exact = np.zeros(Yt.shape)
        Yt_dt_ann = np.zeros((self.gas.n_species, Yt.shape[1]))
        T_dt_exact = np.zeros(len(Tt))
        T_dt_ann = np.zeros(len(Tt))

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
            T_dt_exact[i_reac] = states.T[0]

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
            progvar = self.compute_progvar(state_phys, pressure, Yc_u[i_reac])

            # Attribute cluster
            self.attribute_cluster(state_phys, dt, progvar)

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
            T_dt_ann[i_reac] = T_new

        # Reaction rates
        Omega_ann = (Yt_dt_ann-Yt_ann)/dt

        return {
            "flame": f,
            "c": c,
            "Yt_dt_exact": Yt_dt_exact,
            "Yt_dt_ann": Yt_dt_ann,
            "T_dt_exact": T_dt_exact,
            "T_dt_ann": T_dt_ann,
            "Omega_exact": Omega_exact,
            "Omega_ann": Omega_ann,
            "ann_calls": ann_calls,
            "cvode_calls": cvode_calls,
        }


    # 1D PREMIXED (wrapper with plotting)
    def test_1D_premixed(self, phi, T0, pressure, dt, T_threshold=0.0, x_window=None):
        """One 1D premixed flame case with printed errors and plots (see run_case)."""

        return self.test_case("1D_premixed", {"phi": phi, "T0": T0, "pressure": pressure, "dt": dt,
                                              "T_threshold": T_threshold}, x_window)


    # 1D DIFFUSION (wrapper with plotting)
    def test_1D_diffusion(self, strain, T_ox, T_fuel, pressure, dt, width=0.02, T_threshold=0.0, x_window=None):
        """One 1D counterflow diffusion flame case with printed errors and plots (see run_case)."""

        return self.test_case("1D_diffusion", {"strain": strain, "T_ox": T_ox, "T_fuel": T_fuel, "pressure": pressure,
                                               "dt": dt, "width": width, "T_threshold": T_threshold}, x_window)

#-----------------------------------------------------------------------
#   SERIES OF TEST CASES (nn_testing.yaml)
#-----------------------------------------------------------------------

    def run_testing(self):
        """Run the series of test cases of the testing parameters (nn_testing.yaml) with run: true,
        then the evaluation on the validation set if selected. Returns the summaries of the series."""

        summaries = {}
        for case_type in CASE_TYPES:
            series = self.testing_parameters[case_type]
            if series["run"]:
                summaries[case_type] = self.run_series(case_type, series)

        validation = self.testing_parameters["validation_set"]
        if validation["run"]:
            self.evaluate_on_validation_set(max_points_scatter=validation["max_points_scatter"])

        return summaries


    def run_series(self, case_type, series):
        """Run all the cases of a series and write their errors in TESTING_<model>/summary_<case_type>.yaml.

        series: section of nn_testing.yaml: parameters of run_case (a list of values is swept, all
        combinations are run), plots (plots of each case), x_window ([x_min, x_max] [m] of the
        1D errors and plots, full domain if null).
        Returns the summary: errors of each case, mean and standard deviation over the cases, and the
        simulation error aggregated per dt (E(dt)). The errors of each case are also written in
        simulation_errors_<case_type>.csv, and E(dt) in simulation_errors_<case_type>_by_dt.csv (plotted in
        simulation_errors_<case_type>_by_dt.png).
        """

        x_window = series.get("x_window")
        cases = series_cases(series)

        print(f"\n>> Series {case_type}: {len(cases)} cases")

        entries = []
        for case in cases:

            print(f"\n>> {case_type}: {case}")
            result = self.run_case(case_type, case)
            errors = self.case_errors(case_type, result, x_window)
            print_errors(errors)

            if series["plots"]:
                self.plot_case(case_type, case, result, x_window)
                plt.close("all")

            entries.append({"params": case, "errors": errors})

        # Mean and standard deviation over the cases (cases without value, e.g. no ignition, are skipped)
        mean, std = {}, {}
        for key in entries[0]["errors"]:
            values = [entry["errors"][key] for entry in entries if entry["errors"][key] is not None]
            mean[key], std[key] = metrics.mean_std(values)

        summary = {"n_cases": len(entries), "mean": mean, "std": std, "cases": entries}

        # Table of the errors of each simulation (one row per case: parameters and errors)
        table = pd.DataFrame([{**entry["params"], **entry["errors"]} for entry in entries])
        table.to_csv(os.path.join(self.output_folder, f"simulation_errors_{case_type}.csv"), index=False)

        # 0D: simulation error aggregated over the simulations of each dt, E(dt)
        if "E_sim" in table:
            by_dt = table.groupby("dt")["E_sim"].agg(E_mean="mean", E_std=lambda e: e.std(ddof=0), nb_simulations="count")
            by_dt.to_csv(os.path.join(self.output_folder, f"simulation_errors_{case_type}_by_dt.csv"))
            plots.plot_simulation_errors_by_dt(table, by_dt, case_type, self.output_folder)
            summary["E_sim_by_dt"] = {float(dt): {key: float(value) for key, value in row.items()} for dt, row in by_dt.iterrows()}

        with open(os.path.join(self.output_folder, f"summary_{case_type}.yaml"), "w") as file:
            yaml.dump(summary, file, default_flow_style=False, sort_keys=False)

        return summary


    def test_case(self, case_type, case, x_window=None):
        """Run one case, print its errors and write its plots. Returns the errors."""

        result = self.run_case(case_type, case)
        errors = self.case_errors(case_type, result, x_window)
        print_errors(errors)
        self.plot_case(case_type, case, result, x_window)

        return errors


    def run_case(self, case_type, case):
        """Result of one case of type case_type ("0D_ignition", "0D_pyrolysis", "1D_premixed" or
        "1D_diffusion"), case: dict of the parameters of the corresponding run_*_case method."""

        if case_type == "0D_ignition":
            return self.run_0D_ignition_case(**case)

        elif case_type == "0D_pyrolysis":
            return self.run_0D_pyrolysis_case(**case)

        elif case_type == "1D_premixed":
            return self.run_1D_premixed_case(**case)

        elif case_type == "1D_diffusion":
            return self.run_1D_diffusion_case(**case)

        raise ValueError(f"Unknown case type {case_type}, expected one of {CASE_TYPES}")


    def case_errors(self, case_type, result, x_window=None):
        """Errors of the ANN compared to CVODE for the result of one case (dict of floats, None if not defined).

        0D: simulation error E_sim [-] (see NN_testing_metrics.normalized_trajectory_error) over all the
        time steps, i.e. until CVODE equilibrium (equilibrium_steps, equilibrium_reached), error on the final temperature [%], ignition delays [s]
        (time at which T exceeds T0 + 200 K) and their error [%] (ignition only), numbers of ANN and CVODE calls.
        1D: a priori error E_sim [-], normalized as for 0D, on the states after one step dt from each grid
        point in x_window (mean over the points instead of the time steps), L2 error of the reaction
        rates of spec_to_plot over x_window (see NN_testing_metrics.l2_error), numbers of ANN and CVODE calls.
        """

        errors = {}

        # 0D reactors
        if case_type.startswith("0D"):

            # Simulation error, normalized with the training standard deviations, over the steps (initial state excluded)
            x_ref = self._error_variables(result["states_ref"][1:])
            x_ann = self._error_variables(result["states_ann"][1:])
            errors["E_sim"] = metrics.normalized_trajectory_error(x_ann, x_ref, self.std_train)
            errors["equilibrium_steps"] = len(result["states_ref"]) - 1
            errors["equilibrium_reached"] = bool(result["equilibrium_reached"])

            errors["T_final_error_percent"] = float(metrics.relative_error_percent(result["Teq_ref"], result["Teq_ann"]))

            # Ignition delays (no delay if CVODE does not ignite)
            if case_type == "0D_ignition":
                errors["tau_ign_cvode"] = float(result["tau_ign_cvode"])
                errors["tau_ign_ann"] = float(result["tau_ign_ann"])
                errors["tau_ign_error_percent"] = None
                if result["tau_ign_cvode"] > 0.0:
                    errors["tau_ign_error_percent"] = float(metrics.relative_error_percent(result["tau_ign_cvode"], result["tau_ign_ann"]))

        # 1D flames
        else:
            grid = result["flame"].grid

            # Grid points of x_window [m] (all points if None)
            in_window = np.ones(len(grid), dtype=bool)
            if x_window is not None:
                in_window = (grid >= x_window[0]) & (grid <= x_window[1])

            # A priori error, normalized with the training standard deviations, after one step dt
            x_ref = self._error_variables(np.column_stack([result["T_dt_exact"], result["Yt_dt_exact"].T])[in_window])
            x_ann = self._error_variables(np.column_stack([result["T_dt_ann"], result["Yt_dt_ann"].T])[in_window])
            errors["E_sim"] = metrics.normalized_trajectory_error(x_ann, x_ref, self.std_train)

            for spec in self.spec_to_plot:
                k = self.spec_list_ref.index(spec)
                errors[f"L2_error_omega_{spec}"] = metrics.l2_error(grid, result["Omega_ann"][k], result["Omega_exact"][k], x_window)

        errors["ann_calls"] = result["ann_calls"]
        errors["cvode_calls"] = result["cvode_calls"]

        return errors


    def plot_case(self, case_type, case, result, x_window=None):
        """Plots of one case in TESTING_<model>/<case_type>_<parameters>/."""

        folder = os.path.join(self.output_folder, case_type + "_" + "_".join(f"{key}#{value}" for key, value in case.items()))

        # 0D reactors: time evolutions
        if case_type.startswith("0D"):
            plots.plot_0D_case(result, folder, self.spec_to_plot, self.spec_list_ref, self.remove_N2)

        # 1D flames: reaction rates along the flame
        else:
            plots.plot_1D_reaction_rates(result["flame"].grid, result["Omega_exact"], result["Omega_ann"],
                                         self.spec_to_plot, self.spec_list_ref, folder, x_window)

#-----------------------------------------------------------------------
#   ANN AND CVODE ADVANCEMENT OF ONE STATE
#-----------------------------------------------------------------------

    def attribute_cluster(self, state, dt, progvar):
        """Cluster (self.cluster) of the state [T [K], Y_k [-]] (all species) advanced by dt [s],
        with progress variable progvar [-] (see NNInference.attribute_clusters)."""

        self.cluster = int(self.inference.attribute_clusters(np.array([state[0]]), np.reshape(state[1:], (1, -1)),
                                                             dt, np.array([progvar]))[0])


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
        """ANN advancement by dt [s] of the state T_old [K], Y_old [-] (all species) at pressure [Pa],
        with the network of self.cluster (see NNInference.advance)."""

        T_new, Y_new, sum_Yk = self.inference.advance(np.array([T_old]), np.reshape(Y_old, (1, -1)), pressure, dt,
                                                      np.array([self.cluster]), self.yk_renormalization)

        # Sum of Yk before renormalization (used for analysis)
        self.sum_Yk_before_renorm = sum_Yk[0]

        # Hybrid model: ANN computation fails if mass conservation is not satisfied
        self.ia_success = True
        if self.hybrid_ann_cvode:
            if np.abs(sum_Yk[0] - 1.0) > self.hybrid_ann_cvode_tol:
                self.ia_success = False

        return T_new[0], Y_new[0]

#-----------------------------------------------------------------------
#   ANALYZING ERRORS
#-----------------------------------------------------------------------

    #WARNING: function not up to date (actually not used), it does not consider log excluded species for instance
    def compute_ann_errors(self, T, Y, pressure, dt, already_transformed=False):

        T_old = T.copy()

        # Dealing with the case where Y is already transformed (logged or bct) at input of function
        if already_transformed:
            if self.inference.log_transform_X==1:
                Y_old = np.exp(Y)
            elif self.inference.log_transform_X==2:
                Y_old = (self.inference.lambda_bct*Y+1.0)**(1./self.inference.lambda_bct)
        else:
            Y_old = Y.copy()

        # Assign to cluster
        # Computing current progress variable on physical state
        state_phys = np.append(T_old, Y_old)
        progvar = self.compute_progvar(state_phys, pressure, self.mechanism_type)
        self.progvar = progvar

        self.attribute_cluster(state_phys, dt, progvar)

        # Log the parameters used in this error-computation call for
        # debugging consistency with dtb_processing / NN_manager.
        self._log(
            "ERR_CLUSTER "
            f"T={float(T_old):.6e}, "
            f"dt={float(dt):.6e}, "
            f"cluster={self.cluster}, "
            f"clusterize_on={self.inference.clusterize_on}, "
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
        if self.inference.log_transform_Y==1:
            # Apply threshold only to non-excluded species.
            for i, spec in enumerate(self.inference.spec_list_ANN):
                if spec in self.inference.log_excluded_species:
                    continue
                idx = i  # here Y_* already correspond to species only
                if idx >= Y_old.shape[0]:
                    break
                if Y_old[idx] < self.inference.threshold:
                    Y_old[idx] = self.inference.threshold
                if Y_cvode[idx] < self.inference.threshold:
                    Y_cvode[idx] = self.inference.threshold
                if Y_ann[idx] < self.inference.threshold:
                    Y_ann[idx] = self.inference.threshold
        elif self.inference.log_transform_Y==2:
            Y_old[Y_old<0.0] = 0.0
            Y_cvode[Y_cvode<0.0] = 0.0
            Y_ann[Y_ann<0.0] = 0.0
        #
        if self.inference.log_transform_Y==1:
            log_Y_old = np.log(Y_old)
            log_Y_cvode = np.log(Y_cvode)
            log_Y_ann = np.log(Y_ann)
        elif self.inference.log_transform_Y==2:
            log_Y_old = (Y_old**self.inference.lambda_bct - 1.0)/self.inference.lambda_bct
            log_Y_cvode = (Y_cvode**self.inference.lambda_bct - 1.0)/self.inference.lambda_bct
            log_Y_ann = (Y_ann**self.inference.lambda_bct - 1.0)/self.inference.lambda_bct
        else:
            log_Y_old = Y_old
            log_Y_cvode = Y_cvode
            log_Y_ann = Y_ann


        # We compute difference of logs
        diff_log_Y_cvode = log_Y_cvode - log_Y_old
        diff_log_Y_ann = log_Y_ann - log_Y_old

        # Normalized differences in log space (using scaler statistics)
        mean_Y = self.inference.Yscaler_list[self.cluster][:, 0]
        std_Y = self.inference.Yscaler_list[self.cluster][:, 1]
        diff_log_Y_cvode_norm = transforms.scale(diff_log_Y_cvode, mean_Y, std_Y)
        diff_log_Y_ann_norm = transforms.scale(diff_log_Y_ann, mean_Y, std_Y)

        # Error in normalized log-space (MSE per species)
        err_Yk = (diff_log_Y_cvode_norm - diff_log_Y_ann_norm) ** 2

        err_T = 100.0*np.abs((T_ann-T_cvode)/T_cvode)

        return err_Yk, err_T


    def _training_std(self):
        """Standard deviations of [T, transformed Y_k of the network species] on the training inputs of all
        clusters, pooled from the scaler statistics of each cluster in training_data.h5."""

        cols = ["Temperature_X"] + [f"{spec}_X" for spec in self.inference.spec_list_ANN]

        means, variances, counts = [], [], []
        with h5py.File(f"{self.inference.dataset_path:s}/training_data.h5", 'r') as h5file_r:
            for i_cluster in range(self.inference.nb_clusters):
                grp = h5file_r[f"CLUSTER_{i_cluster}"]

                # Columns of the training inputs (T, species, possibly dt)
                cols_X = [c.decode() if isinstance(c, bytes) else str(c) for c in grp["X_train"].attrs["cols"]]
                idx = [cols_X.index(col) for col in cols]

                means.append(grp["Xscaler"][idx, 0])
                variances.append(grp["Xscaler"][idx, 1])
                counts.append(grp["X_train"].shape[0])

        _, var = metrics.pooled_mean_var(np.array(means), np.array(variances), counts)

        return transforms.scaler_std(var)


    def _error_variables(self, states):
        """[T [K], transformed Y_k of the network species] (n, 1 + N_S) of states [T, Y_k of all species]
        (n, 1 + n_species), transformed as the network inputs."""

        cols_ANN = [1 + self.spec_list_ref.index(spec) for spec in self.inference.spec_list_ANN]
        Y_t = transforms.transform_species(states[:, cols_ANN], self.inference.spec_list_ANN, self.inference.log_transform_X,
                                           self.inference.threshold, self.inference.lambda_bct, self.inference.log_excluded_species)

        return np.column_stack([states[:, 0], Y_t])


    def _load_validation_data(self):
        """Load X_val/Y_val for each cluster from training_data.h5.

        This mirrors NN_manager.read_training_data/get_scalers_stats and is
        used by evaluate_on_validation_set.
        """

        if self._val_data_loaded:
            return

        self._X_val = []
        self._Y_val = []

        with h5py.File(f"{self.inference.dataset_path:s}/training_data.h5", 'r') as h5file_r:
            # Infer number of clusters from file (robust to future changes)
            top_level_groups = [k for k in h5file_r.keys() if isinstance(h5file_r[k], h5py.Group)]

            for grp_name in sorted(top_level_groups):
                grp = h5file_r[grp_name]
                X_val = grp['X_val'][:]
                Y_val = grp['Y_val'][:]
                self._X_val.append(X_val)
                self._Y_val.append(Y_val)

        self._val_data_loaded = True


    def evaluate_on_validation_set(self, scatter_folder=None, max_points_scatter=50000):
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
        scatter_folder : str or None
            Directory where scatter plots are written (default
            TESTING_<model>/plots_validation). Existing folder is removed.
        max_points_scatter : int
            Maximum number of validation points plotted per cluster for
            scatter plots (random subsampling if larger).
        """

        # Load validation data from training_data.h5
        self._load_validation_data()

        # Prepare output folder (default: TESTING_<model>/plots_validation)
        if scatter_folder is None:
            scatter_folder = os.path.join(self.output_folder, "plots_validation")
        plots.new_folder(scatter_folder)

        all_stats = []

        for i_cluster in range(self.inference.nb_clusters):
            X_val = self._X_val[i_cluster]
            Y_val = self._Y_val[i_cluster]

            # Convert to torch tensors on the correct device
            X_val_t = torch.tensor(X_val, dtype=torch.float64).to(self.inference.device)
            Y_val_t = torch.tensor(Y_val, dtype=torch.float64).to(self.inference.device)

            # Model and scalers for current cluster
            model = self.inference.models_list[i_cluster]

            std_Y = self.inference.Yscaler_list[i_cluster][:, 1]

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
