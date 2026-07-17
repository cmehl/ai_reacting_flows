import os, sys
from typing import Dict
from scipy.stats import qmc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cantera as ct
import h5py

import seaborn as sns
sns.set_style("darkgrid")

from sklearn.model_selection import train_test_split

from ai_reacting_flows.tools.utilities import compute_X_element

class DatabaseFlamelets(object):

    """Generate flamelet databases using Cantera and MPI.

    This class is instantiated from ``flmts_reactors_data_gen.main.generate_flmts_database``
    and encapsulates 0D, 1D premixed, and 1D diffusion flame simulations,
    optional data augmentation, and HDF5 export.
    """

    def __init__(self, data_gen_parameters: Dict, comm) -> None:
        """Initialize the flamelet database generator.

        Parameters
        ----------
        data_gen_parameters:
            Parsed YAML configuration dictionary used by
            ``generate_flmts_database``.
        comm:
            MPI communicator used to distribute the workload.
        """

        # MPI communicator
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()

        # Core input parameters
        self.mech_file = data_gen_parameters["mech_file"]
        self.fuel = data_gen_parameters["fuel"][0]
        self.time_step = data_gen_parameters["time_step"]

        self.include_zerod = data_gen_parameters["include_zerod"]
        self.include_oned_prem = data_gen_parameters["include_oned_prem"]
        self.include_oned_diff = data_gen_parameters["include_oned_diff"]

        self.augment_dataset = data_gen_parameters["augment_dataset"]

        self.folder = "FLAMELETS_DTB_" + data_gen_parameters["results_folder_suffix"]
        self.dtb_file = data_gen_parameters["dtb_file"]

        # We work with constant pressure here
        self.p = data_gen_parameters["pressure"]

        # Initialize solution array
        gas = ct.Solution(self.mech_file)
        self.species_names = gas.species_names
        self.nb_spec = len(self.species_names)
        cols = (
            ['Temperature']
            + ['Pressure']
            + self.species_names
            + ['enthalpy']
            + ['Progress_variable']
            + ['Heat_release_rate']
            + ['reactor_type']
            + ['Simulation number']
        )
        self.df = pd.DataFrame(data=[], columns=cols)

        # Progress variable setup
        self.pv_species = data_gen_parameters["pv_species"]
        self.npvspec = len(self.pv_species)
        # int: list of progress variable species indices
        self.pv_ind = [gas.species_index(spec) for spec in self.pv_species]


    def compute_0d_reactors(self, zerod_params):

        # 0D reactors parameters
        phi_bounds = zerod_params["phi_bounds"]
        T0_bounds = zerod_params["T0_bounds"]
        n_samples = zerod_params["n_samples"]
        max_sim_time = zerod_params["max_sim_time"]
        solve_mode = zerod_params["solve_mode"]
         
        self.n_samples_0D = n_samples
        self.sim_numbers_0D = np.arange(n_samples)

         # DOE 0D REACTORS
         # Initially lhs gives numbers between 0 and 1
        seed = 42
        sampler = qmc.LatinHypercube(d=2, seed=seed) 
        samples = sampler.random(n=self.n_samples_0D)
        self.df_ODE_0D = pd.DataFrame(data=samples, columns=['Phi', 'T0'])
        
        self.df_ODE_0D["sim_number"] = self.sim_numbers_0D 
        
        # Rescale 
        self.df_ODE_0D['Phi'] = phi_bounds[0] + (phi_bounds[1] - phi_bounds[0]) * self.df_ODE_0D['Phi']
        self.df_ODE_0D['T0'] = T0_bounds[0] + (T0_bounds[1] - T0_bounds[0]) * self.df_ODE_0D['T0']

        # Setting aside test reactors conditions (20%) - validation dataset is selected on a point basis in database_processing
        self.id_sim_train_0D, self.id_sim_test_0D = train_test_split(self.sim_numbers_0D, test_size=0.2, random_state=24)
        #
        self.df_ODE_train_0D = self.df_ODE_0D[self.df_ODE_0D["sim_number"].isin(self.id_sim_train_0D)]
        self.df_ODE_test_0D = self.df_ODE_0D[self.df_ODE_0D["sim_number"].isin(self.id_sim_test_0D)]
        #
        if self.rank==0:
            self.df_ODE_train_0D.to_csv(os.path.join(self.folder,"sim_train_0D.csv"), index=False)
            self.df_ODE_test_0D.to_csv(os.path.join(self.folder,"sim_test_0D.csv"), index=False)

            fig, ax = plt.subplots()
            ax.scatter(self.df_ODE_train_0D['T0'], self.df_ODE_train_0D['Phi'], color="k", label="Train")
            ax.scatter(self.df_ODE_test_0D['T0'], self.df_ODE_test_0D['Phi'], color="r", label="Test")
            fig.legend(ncol=3)
            ax.set_xlabel("T0 [K]")
            ax.set_ylabel("Phi [-]")
            fig.savefig(os.path.join(self.folder,"doe_0D_reactors.png"))


        # Make sure every rank waits until rank 0 has written the DOE files/plot
        self.comm.Barrier()
         
        # PERFORMING SIMULATIONS
        # Hard coded for the moment
        equil_tol = 0.5

        # Chemical mechanisms
        gas = ct.Solution(self.mech_file)
        gas_equil = ct.Solution(self.mech_file)

        data = []

        # Split the training rows across MPI ranks (simple round-robin split)
        rows = list(self.df_ODE_train_0D.iterrows())
        rows_local = rows[self.rank::self.size]

        data_local = []
        curves_local = []  # (t_array, T_array) pairs, gathered later for the combined plot

        for i, row in rows_local:

            phi_ini = row['Phi']
            temperature_ini = row['T0']

            print(f"[rank {self.rank}] T0={temperature_ini}; phi={phi_ini}")

            # Initial gas state
            fuel_ox_ratio = gas.n_atoms(species=self.fuel, element='C') \
                            + 0.25 * gas.n_atoms(species=self.fuel, element='H') \
                            - 0.5 * gas.n_atoms(species=self.fuel, element='O')
            compo_ini = f'{self.fuel}:{phi_ini:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
            gas.TPX = temperature_ini, self.p, compo_ini

            nb_spec = len(gas.X)   # Number of species
            Y0 = gas.Y   # Initial mass fractions
            h0 = gas.HP[0]

            # 0D reactor
            r = ct.IdealGasConstPressureReactor(gas)

            # Initializing reactor
            sim = ct.ReactorNet([r])
            time = 0.0
            states = ct.SolutionArray(gas, extra=['t'])

            # Computing equilibrium (to get end of simulation criterion)
            gas_equil.TPX = temperature_ini, self.p, compo_ini
            Yc_u  = gas_equil.Y[self.pv_ind].sum()
            gas_equil.equilibrate('HP')
            Yc_eq = gas_equil.Y[self.pv_ind].sum()
            state_equil = np.append(gas_equil.X, gas_equil.T)

            equil_bool = False
            n_iter = 0

            progvar_vect = []
            hrr_vect = []

            while (equil_bool == False) and (time < max_sim_time):
                
                if solve_mode=="dt_cfd":
                    time += self.time_step
                    sim.advance(time)
                    states.append(r.thermo.state, t=time)
                elif solve_mode=="dt_cvode":
                    t_cvode = sim.step()
                    time = t_cvode
                    if n_iter%5==0:   # dt_cvode gives too many points
                        states.append(r.thermo.state, t=t_cvode)
                else:
                    sys.exit("solve_mode should be dt_cvode or dt_cfd")

                # checking if equilibrium is reached
                state_current = np.append(r.thermo.X, r.T)
                residual = 100.0*np.linalg.norm(state_equil - state_current,ord=np.inf)/np.linalg.norm(state_equil,
                                                                                                           ord=np.inf)
                
                # Compute progress variable
                Yc = gas.Y[self.pv_ind].sum(axis=0)
                if Yc_eq - Yc_u > 1e-10:
                    progvar = (Yc - Yc_u) / (Yc_eq - Yc_u)
                else:
                    progvar = 0.0
                progvar_vect.append(progvar)
                
                # Compute heat release rate
                hrr = 0.0       
                for spec in gas.species_names:
                    standard_enthalpy_spec = gas.standard_enthalpies_RT[gas.species_index(spec)] * ct.gas_constant * gas.T
                    hrr += -gas.net_production_rates[gas.species_index(spec)] * standard_enthalpy_spec
                hrr_vect.append(hrr)

                
                n_iter +=1
                # max iteration                    
                if residual < equil_tol:
                    equil_bool = True
            
            # ============================== Construction of the database =========
            # Get the total number of rows for the current simulation
            n_rows = len(states.t) + 1
            # nb_spec + [Temperature, Pressure, enthalpy, progress variable, HRR, reactor type, sim number] (=> +7)
            n_cols = nb_spec + 7
            # empty array saving
            arr = np.empty(shape=(n_rows,n_cols))
            #  initial conditions
            arr[0,0] = temperature_ini
            arr[0,1] = self.p
            arr[0,2:2+nb_spec] = Y0
            arr[0,-5] = h0
            arr[0,-4] = 0.0
            arr[0,-3] = 0.0
            arr[0,-2] = 0
            arr[0,-1] = i
            #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
            arr[1:, 0] = states.T
            arr[1:, 1] = states.P
            arr[1:, 2:2 + nb_spec] = states.Y
            arr[1:, -5] = states.enthalpy_mass
            arr[1:, -4] = np.asarray(progvar_vect)
            arr[1:, -3] = np.asarray(hrr_vect)
            arr[1:, -2] = 0
            arr[1:, -1] = i

            # Save in pandas dataframe
            cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['Progress_variable'] + ['Heat_release_rate'] + ['reactor_type'] + ['Simulation number']
            df = pd.DataFrame(data=arr, columns=cols)

            # List of dataframes
            data_local.append(df)

            # Store the raw curve (small numpy arrays, easy to pickle/gather)
            curves_local.append((np.array(states.t), np.array(states.T), np.asarray(progvar_vect), np.asarray(hrr_vect)))


        # Gather results from all ranks onto rank 0
        all_data = self.comm.gather(data_local, root=0)
        all_curves = self.comm.gather(curves_local, root=0)

        if self.rank == 0:
            # Flatten list of lists -> single list of dataframes
            data = [df for sublist in all_data for df in sublist]
            curves = [c for sublist in all_curves for c in sublist]
    
            df_ODE_0D = pd.concat(data, axis=0).reset_index(drop=True)
            # Sort back by simulation number to keep a deterministic order
            df_ODE_0D = df_ODE_0D.sort_values(['Simulation number'], kind='stable').reset_index(drop=True)
    
            if self.df.empty:
                self.df = df_ODE_0D.copy()
            else:
                self.df = pd.concat([self.df, df_ODE_0D], axis=0).reset_index(drop=True)
    
            # Build the combined trajectories plot from every rank's curves
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel("t [s]")
            ax1.set_ylabel("T [K]")
            ax2.set_ylabel("c [-]")
            ax3.set_ylabel("HRR [W/m3]")
            for t_arr, T_arr, progvar_arr, hrr_arr in curves:
                ax1.plot(t_arr, T_arr)
                ax2.plot(t_arr, progvar_arr.squeeze())
                ax3.plot(t_arr, hrr_arr.squeeze())
            fig.tight_layout()
            fig.savefig(os.path.join(self.folder, "0D_trajectories.png"))
    
        # Broadcast the merged dataframe back to every rank so self.df stays
        # consistent across ranks for subsequent steps in the pipeline
        self.df = self.comm.bcast(self.df if self.rank == 0 else None, root=0)


    def compute_1d_premixed(self, oned_prem_params):

        # 0D reactors parameters
        phi_bounds = oned_prem_params["phi_bounds"]
        T0_bounds = oned_prem_params["T0_bounds"]
        n_samples = oned_prem_params["n_samples"]
         
        self.n_samples_1D_prem = n_samples
        self.sim_numbers_1D_prem = np.arange(n_samples)

         # DOE 1D_prem REACTORS
         # Initially lhs gives numbers between 0 and 1
        seed = 42
        sampler = qmc.LatinHypercube(d=2, seed=seed) 
        samples = sampler.random(n=self.n_samples_1D_prem)
        self.df_ODE_1D_prem = pd.DataFrame(data=samples, columns=['Phi', 'T0'])
        
        self.df_ODE_1D_prem["sim_number"] = self.sim_numbers_1D_prem 
        
        # Rescale 
        self.df_ODE_1D_prem['Phi'] = phi_bounds[0] + (phi_bounds[1] - phi_bounds[0]) * self.df_ODE_1D_prem['Phi']
        self.df_ODE_1D_prem['T0'] = T0_bounds[0] + (T0_bounds[1] - T0_bounds[0]) * self.df_ODE_1D_prem['T0']

        # Setting aside test reactors conditions (20%) - validation dataset is selected on a point basis in database_processing
        self.id_sim_train_1D_prem, self.id_sim_test_1D_prem = train_test_split(self.sim_numbers_1D_prem, test_size=0.2, random_state=24)
        #
        self.df_ODE_train_1D_prem = self.df_ODE_1D_prem[self.df_ODE_1D_prem["sim_number"].isin(self.id_sim_train_1D_prem)]
        self.df_ODE_test_1D_prem = self.df_ODE_1D_prem[self.df_ODE_1D_prem["sim_number"].isin(self.id_sim_test_1D_prem)]
        #
        if self.rank==0:
            self.df_ODE_train_1D_prem.to_csv(os.path.join(self.folder,"sim_train_1D_prem.csv"), index=False)
            self.df_ODE_test_1D_prem.to_csv(os.path.join(self.folder,"sim_test_1D_prem.csv"), index=False)

            fig, ax = plt.subplots()
            ax.scatter(self.df_ODE_train_1D_prem['T0'], self.df_ODE_train_1D_prem['Phi'], color="k", label="Train")
            ax.scatter(self.df_ODE_test_1D_prem['T0'], self.df_ODE_test_1D_prem['Phi'], color="r", label="Test")
            fig.legend(ncol=3)
            ax.set_xlabel("T0 [K]")
            ax.set_ylabel("Phi [-]")
            fig.savefig(os.path.join(self.folder,"doe_1D_prem_flames.png"))

        # Make sure every rank waits until rank 0 has written the DOE files/plot
        self.comm.Barrier()
         
        # PERFORMING SIMULATIONS

        # Chemical mechanism
        gas = ct.Solution(self.mech_file)

        data = []

        initial_grid = np.linspace(0.0, 0.03, 10)  # m
        tol_ss = [1.0e-4, 1.0e-9]  # [rtol atol] for steady-state problem
        tol_ts = [1.0e-5, 1.0e-5]  # [rtol atol] for time stepping
        loglevel = 0  # amount of diagnostic output (0 to 8)

        # Split the training rows across MPI ranks (simple round-robin split)
        rows = list(self.df_ODE_train_1D_prem.iterrows())
        rows_local = rows[self.rank::self.size]

        data_local = []
        curves_local = []  # (t_array, T_array) pairs, gathered later for the combined plot

        for i, row in rows_local:

            crashed = False

            phi_ini = row['Phi']
            temperature_ini = row['T0']

            print(f"[rank {self.rank}] T0={temperature_ini}; phi={phi_ini}")

            # Initial gas state
            fuel_ox_ratio = gas.n_atoms(species=self.fuel, element='C') \
                            + 0.25 * gas.n_atoms(species=self.fuel, element='H') \
                            - 0.5 * gas.n_atoms(species=self.fuel, element='O')
            compo_ini = f'{self.fuel}:{phi_ini:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
            gas.TPX = temperature_ini, self.p, compo_ini

            nb_spec = len(gas.X)   # Number of species

            f = ct.FreeFlame(gas, initial_grid)

            f.flame.set_steady_tolerances(default=tol_ss)
            f.flame.set_transient_tolerances(default=tol_ts)
            f.inlet.set_steady_tolerances(default=tol_ss)
            f.inlet.set_transient_tolerances(default=tol_ts)
            f.outlet.set_steady_tolerances(default=tol_ss)
            f.outlet.set_transient_tolerances(default=tol_ts)

            f.transport_model = 'mixture-averaged'
            # f.transport_model = 'multicomponent'
            f.soret_enabled = False

            f.set_max_jac_age(10, 10)
            f.set_time_step(1e-5, [2, 5, 10, 20])

            f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)

            try:
                f.solve(loglevel=loglevel, auto=True)
            except ct.CanteraError:
                print(f" WARNING: Computation crashed for T0={temperature_ini}, phi={phi_ini} => skipped")
                crashed = True

            if crashed==False:

                # Computation of progress variable (same Yceq as we are adiabatic)
                gas_equil = ct.Solution(self.mech_file)
                gas_equil.TPX = temperature_ini, self.p, compo_ini
                Yc_u  = gas_equil.Y[self.pv_ind].sum()
                gas_equil.equilibrate('HP')
                Yc_eq = gas_equil.Y[self.pv_ind].sum()
                #
                Yc = f.Y[self.pv_ind, :].sum(axis=0)
                if Yc_eq - Yc_u > 1e-10:
                    progvar = (Yc - Yc_u) / (Yc_eq - Yc_u)
                else:
                    progvar = np.zeros_like(Yc)

                # Computation of HRR
                hrr = np.zeros(f.grid.size)
                for n in range(f.grid.size):
                    gas.TPY = f.T[n], f.P, f.Y[:, n]
                    h = gas.standard_enthalpies_RT * ct.gas_constant * f.T[n]
                    hrr[n] = -np.dot(gas.net_production_rates, h)

                # ============================== Construction of the database =========
                # Get the total number of rows for the current simulation
                n_rows = len(f.grid)
                # nb_spec + [Temperature, Pressure, enthalpy, progress variable, HRR, reactor type, sim number] (=> +7)
                n_cols = nb_spec + 7
                # empty array saving
                arr = np.empty(shape=(n_rows,n_cols))
                #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
                arr[:, 0] = f.T
                arr[:, 1] = f.P
                arr[:, 2:2 + nb_spec] = np.transpose(f.Y)
                arr[:, -5] = f.enthalpy_mass
                arr[:, -4] = progvar
                arr[:, -3] = hrr
                arr[:, -2] = 1
                arr[:, -1] = i

                # Save in pandas dataframe
                cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['Progress_variable'] + ['Heat_release_rate'] + ['reactor_type'] + ['Simulation number']
                df = pd.DataFrame(data=arr, columns=cols)

                # List of dataframes
                data_local.append(df)

                # Store the raw curve (small numpy arrays, easy to pickle/gather)
                curves_local.append((np.array(f.grid), np.array(f.T), progvar, hrr))

        # Gather results from all ranks onto rank 0
        all_data = self.comm.gather(data_local, root=0)
        all_curves = self.comm.gather(curves_local, root=0)

        if self.rank == 0:
            # Flatten list of lists -> single list of dataframes
            data = [df for sublist in all_data for df in sublist]
            curves = [c for sublist in all_curves for c in sublist]
    
            df_ODE_1D_prem = pd.concat(data, axis=0).reset_index(drop=True)
            # Sort back by simulation number to keep a deterministic order
            df_ODE_1D_prem = df_ODE_1D_prem.sort_values(['Simulation number'], kind='stable').reset_index(drop=True)
    
            if self.df.empty:
                self.df = df_ODE_1D_prem.copy()
            else:
                self.df = pd.concat([self.df, df_ODE_1D_prem], axis=0).reset_index(drop=True)
    
            # Build the combined trajectories plot from every rank's curves
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel("x [m]")
            ax1.set_ylabel("T [K]")
            ax2.set_ylabel("c [-]")
            ax3.set_ylabel("HRR [W/m3]")
            for x_arr, T_arr, progvar_arr, hrr_arr in curves:
                ax1.plot(x_arr, T_arr)
                ax2.plot(x_arr, progvar_arr)
                ax3.plot(x_arr, hrr_arr)
            fig.tight_layout()
            fig.savefig(os.path.join(self.folder, "1D_prem_trajectories.png"))
    
        # Broadcast the merged dataframe back to every rank so self.df stays
        # consistent across ranks for subsequent steps in the pipeline
        self.df = self.comm.bcast(self.df if self.rank == 0 else None, root=0)


    def compute_1d_diffusion(self, oned_diff_params):

        # 0D reactors parameters
        strain_bounds = oned_diff_params["strain_bounds"]
        T0_bounds = oned_diff_params["T0_bounds"]
        n_samples = oned_diff_params["n_samples"]
        width = oned_diff_params["width"]
         
        self.n_samples_1D_diff = n_samples
        self.sim_numbers_1D_diff = np.arange(n_samples)

         # DOE 1D_diff REACTORS
         # Initially lhs gives numbers between 0 and 1
        seed = 42
        sampler = qmc.LatinHypercube(d=2, seed=seed) 
        samples = sampler.random(n=self.n_samples_1D_diff)
        self.df_ODE_1D_diff = pd.DataFrame(data=samples, columns=['Strain', 'T0'])
        
        self.df_ODE_1D_diff["sim_number"] = self.sim_numbers_1D_diff 
        
        # Rescale 
        self.df_ODE_1D_diff['Strain'] = strain_bounds[0] + (strain_bounds[1] - strain_bounds[0]) * self.df_ODE_1D_diff['Strain']
        self.df_ODE_1D_diff['T0'] = T0_bounds[0] + (T0_bounds[1] - T0_bounds[0]) * self.df_ODE_1D_diff['T0']

        # Setting aside test reactors conditions (20%) - validation dataset is selected on a point basis in database_processing
        self.id_sim_train_1D_diff, self.id_sim_test_1D_diff = train_test_split(self.sim_numbers_1D_diff, test_size=0.2, random_state=24)
        #
        self.df_ODE_train_1D_diff = self.df_ODE_1D_diff[self.df_ODE_1D_diff["sim_number"].isin(self.id_sim_train_1D_diff)]
        self.df_ODE_test_1D_diff = self.df_ODE_1D_diff[self.df_ODE_1D_diff["sim_number"].isin(self.id_sim_test_1D_diff)]
        #
        if self.rank==0:

            self.df_ODE_train_1D_diff.to_csv(os.path.join(self.folder,"sim_train_1D_diff.csv"), index=False)
            self.df_ODE_test_1D_diff.to_csv(os.path.join(self.folder,"sim_test_1D_diff.csv"), index=False)

            fig, ax = plt.subplots()
            ax.scatter(self.df_ODE_train_1D_diff['T0'], self.df_ODE_train_1D_diff['Strain'], color="k", label="Train")
            ax.scatter(self.df_ODE_test_1D_diff['T0'], self.df_ODE_test_1D_diff['Strain'], color="r", label="Test")
            fig.legend(ncol=3)
            ax.set_xlabel("T0 [K]")
            ax.set_ylabel("Strain [s-1]")
            fig.savefig(os.path.join(self.folder,"doe_1D_diff_flames.png"))

        # Make sure every rank waits until rank 0 has written the DOE files/plot
        self.comm.Barrier()
         
        # PERFORMING SIMULATIONS

        # Chemical mechanism
        gas = ct.Solution(self.mech_file)

        data = []

        loglevel = 0  # amount of diagnostic output (0 to 8)

        # Split the training rows across MPI ranks (simple round-robin split)
        rows = list(self.df_ODE_train_1D_diff.iterrows())
        rows_local = rows[self.rank::self.size]

        data_local = []
        curves_local = []  # (t_array, T_array) pairs, gathered later for the combined plot

        for i, row in rows_local:

            crashed = False

            strain = row['Strain']
            temperature_ini = row['T0']

            print(f"[rank {self.rank}] T0_ox={temperature_ini}; strain={strain}")

            gas.TP = gas.T, self.p

            nb_spec = len(gas.X)   # Number of species

            # Stream compositions
            compo_ini_o = 'O2:0.21, N2:0.79'
            compo_ini_f = f'{self.fuel}:1'

            gas.TPX = temperature_ini, self.p, compo_ini_o
            density_o = gas.density
            Y_o = gas.Y.copy()
            h_o = gas.enthalpy_mass                   # J/kg
            gas.TPX = 300.0, self.p, compo_ini_f
            density_f = gas.density
            Y_f = gas.Y.copy()
            h_f = gas.enthalpy_mass                   # J/kg

            # Stream mass flow rates
            vel = strain * width / 2.0
            mdot_o = density_o * vel
            mdot_f = density_f * vel

            f = ct.CounterflowDiffusionFlame(gas, width=width)

            f.transport_model = 'mixture-averaged'
            # f.transport_model = 'multicomponent'
            f.soret_enabled = False

            f.radiation_enabled = False

            f.set_max_jac_age(10, 10)
            f.set_time_step(1e-5, [2, 5, 10, 20])

            # Set the state of the two inlets
            f.fuel_inlet.mdot = mdot_f
            f.fuel_inlet.X = compo_ini_f
            f.fuel_inlet.T = 300.0

            f.oxidizer_inlet.mdot = mdot_o
            f.oxidizer_inlet.X = compo_ini_o
            f.oxidizer_inlet.T = temperature_ini

            f.set_refine_criteria(ratio=2, slope=0.06, curve=0.12, prune=0.04)

            try:
                f.solve(loglevel=loglevel, auto=True)
            except ct.CanteraError:
                print(f" WARNING: Computation crashed for T0_ox={temperature_ini}, a={strain} => skipped")
                crashed = True

            if crashed==False:
                
                #NOT WORKING: c reaches 2 ...
                # # Computation of progress variable: we compute unburnt and burnt values from a local state (based on Z)
                # #Unburnt is on the mixing line and burnt is computed from an equilibrium computation
                # Z_arr = f.mixture_fraction("Bilger")
                # progvar = np.zeros_like(Z_arr)
                # gas_equil = ct.Solution(self.mech_file)
                # for n in range(f.grid.size):

                #     Z = Z_arr[n]

                #     # Local cold unburnt state at same Z
                #     Y_mix = Z * Y_f + (1.0 - Z) * Y_o
                #     h_mix = Z * h_f + (1.0 - Z) * h_o
                #     gas_equil.HPY = h_mix, self.p, Y_mix
                #     Yc_u = gas_equil.Y[self.pv_ind].sum()

                #     # Equilibrium from that local state
                #     gas_equil.equilibrate('HP')
                #     Yc_eq = gas_equil.Y[self.pv_ind].sum()

                #     # Current flame point
                #     Yc = f.Y[self.pv_ind, n].sum()

                #     if Yc_eq - Yc_u > 1e-10:
                #         progvar[n] = (Yc - Yc_u) / (Yc_eq - Yc_u)
                #     else:
                #         progvar[n] = 0.0

                # Easier solution: we normalize by min and max of Yc in flame
                # For what we need to do with progvar (undersample database mostly) it should be ok
                Yc_arr = f.Y[self.pv_ind, :].sum(axis=0)
                Yc_min = Yc_arr.min()
                Yc_max = Yc_arr.max()
                if Yc_max - Yc_min > 1e-10:
                    progvar = (Yc_arr - Yc_min) / (Yc_max - Yc_min)
                else:
                    progvar = np.zeros_like(Yc_arr)

                # Computation of HRR
                hrr = np.zeros(f.grid.size)
                for n in range(f.grid.size):
                    gas.TPY = f.T[n], f.P, f.Y[:, n]
                    h = gas.standard_enthalpies_RT * ct.gas_constant * f.T[n]
                    hrr[n] = -np.dot(gas.net_production_rates, h)

                # ============================== Construction of the database =========
                # Get the total number of rows for the current simulation
                n_rows = len(f.grid)
                # nb_spec + [Temperature, Pressure, enthalpy, progress variable, HRR, reactor type, sim number] (=> +7)
                n_cols = nb_spec + 7
                # empty array saving
                arr = np.empty(shape=(n_rows,n_cols))
                #  solution for each time step (each couple of S(t) and S(t+dt) for the dtCVODE case)
                arr[:, 0] = f.T
                arr[:, 1] = f.P
                arr[:, 2:2 + nb_spec] = np.transpose(f.Y)
                arr[:, -5] = f.enthalpy_mass
                arr[:, -4] = progvar
                arr[:, -3] = hrr
                arr[:, -2] = 2
                arr[:, -1] = i
                
                # Save in pandas dataframe
                cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy'] + ['Progress_variable'] + ['Heat_release_rate'] + ['reactor_type'] + ['Simulation number']
                df = pd.DataFrame(data=arr, columns=cols)

                # List of dataframes
                data_local.append(df)

                # Store the raw curve (small numpy arrays, easy to pickle/gather)
                curves_local.append((np.array(f.grid), np.array(f.T), progvar, hrr))

        # Gather results from all ranks onto rank 0
        all_data = self.comm.gather(data_local, root=0)
        all_curves = self.comm.gather(curves_local, root=0)

        if self.rank == 0:
            # Flatten list of lists -> single list of dataframes
            data = [df for sublist in all_data for df in sublist]
            curves = [c for sublist in all_curves for c in sublist]
    
            df_ODE_1D_diff = pd.concat(data, axis=0).reset_index(drop=True)
            # Sort back by simulation number to keep a deterministic order
            df_ODE_1D_diff = df_ODE_1D_diff.sort_values(['Simulation number'], kind='stable').reset_index(drop=True)
    
            if self.df.empty:
                self.df = df_ODE_1D_diff.copy()
            else:
                self.df = pd.concat([self.df, df_ODE_1D_diff], axis=0).reset_index(drop=True)
    
            # Build the combined trajectories plot from every rank's curves
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
            for ax in [ax1, ax2, ax3]:
                ax.set_xlabel("x [m]")
            ax1.set_ylabel("T [K]")
            ax2.set_ylabel("c [-]")
            ax3.set_ylabel("HRR [W/m3]")
            for x_arr, T_arr, progvar_arr, hrr_arr in curves:
                ax1.plot(x_arr, T_arr)
                ax2.plot(x_arr, progvar_arr)
                ax3.plot(x_arr, hrr_arr)
            fig.tight_layout()
            fig.savefig(os.path.join(self.folder, "1D_diff_trajectories.png"))
    
        # Broadcast the merged dataframe back to every rank so self.df stays
        # consistent across ranks for subsequent steps in the pipeline
        self.df = self.comm.bcast(self.df if self.rank == 0 else None, root=0)



    def augment_data(self):

        self.df_flamelet = self.df.copy()

        gas = ct.Solution(self.mech_file)

        cols = ['Temperature'] + ['Pressure'] + gas.species_names + ['enthalpy']+ ['Progress_variable'] + ['Heat_release_rate'] + ['reactor_type'] + ['Simulation number']
        self.df_augmented = pd.DataFrame(data=[], columns=cols)


        if self.include_zerod:
            max_values_0D = self.df[self.df["reactor_type"]==0].max()    # We should probably do it by simulation type !!
            min_values_0D = self.df[self.df["reactor_type"]==0].min()

        if self.include_oned_prem:
            max_values_1D_prem = self.df[self.df["reactor_type"]==1].max()    # We should probably do it by simulation type !!
            min_values_1D_prem = self.df[self.df["reactor_type"]==1].min()

        if self.include_oned_diff:
            max_values_1D_diff = self.df[self.df["reactor_type"]==2].max()    # We should probably do it by simulation type !!
            min_values_1D_diff = self.df[self.df["reactor_type"]==2].min()

        h_idx = 2 + self.nb_spec
        bounds = {}
        if self.include_zerod:
            bounds[0] = (max_values_0D.iloc[h_idx], min_values_0D.iloc[h_idx])
        if self.include_oned_prem:
            bounds[1] = (max_values_1D_prem.iloc[h_idx], min_values_1D_prem.iloc[h_idx])
        if self.include_oned_diff:
            bounds[2] = (max_values_1D_diff.iloc[h_idx], min_values_1D_diff.iloc[h_idx])


        # Split row indices across ranks
        rows = list(self.df.iterrows())
        rows_local = rows[self.rank::self.size]    

        local_rows_new = []
        local_idx_new = []

        # Data augmentation based on the work of Ding et al. (HFRD method)
        # Subtelty here: when you do that i correspond to the original row index; and therefore by saving i we are able to rebuild 
        # df_augmented with the rows having same positions than original dataframe
        for i, row in rows_local: 
            
            tries = 0
            accepted = False
            rng = np.random.default_rng(42 + i) # per row rng needed to avoid processor dependent results
            while tries<=10 and accepted==False:
                
                tries += 1
                accepted = True

                T = row.iloc[0]
                p = row.iloc[1]
                Yk = row.iloc[2:2 + self.nb_spec].values
                h = row.iloc[2 + self.nb_spec]
                i_reac = row.iloc[5 + self.nb_spec]

                hmax, hmin = bounds[i_reac]
                
                # Random numbers
                c = rng.uniform(-1, 1)
                d = rng.uniform(-1, 1)

                # Parameters
                a = 8
                b = 5
                
                # New state
                hnew = h + (c/a) * (hmax - hmin)
                Yk = np.clip(Yk, 0, 1)
                Yknew = Yk**(1.+d/b)
                Yknew = Yknew / Yknew.sum()

                # New temperature
                try:
                    gas.HPY = hnew, p, Yknew
                    Tnew = gas.T
                except ct.CanteraError:
                    print(f">> T computation crashed for row {i}")
                    accepted = False
                    continue


                # Skip under some conditions
                if Tnew<300.0:
                    accepted = False
                    continue

                # Elements (C, H, O, N)
                X_el = compute_X_element(gas.species_names, Yknew)
                O_N_ratio = X_el[2]/X_el[3]
                # if O_N_ratio<0.25 or O_N_ratio > 0.28:
                #     accepted = False


                if accepted:
                    row_new = np.empty(7+self.nb_spec)
                    row_new[0] = Tnew
                    row_new[1] = p
                    row_new[2:2 + self.nb_spec] = Yknew
                    row_new[2 + self.nb_spec] = hnew
                    row_new[3 + self.nb_spec] = row.iloc[3 + self.nb_spec]   #CM: recompute proper progvar 
                    row_new[4 + self.nb_spec] = row.iloc[4 + self.nb_spec]   #CM: recompute proper HRR
                    row_new[5 + self.nb_spec] = row.iloc[5 + self.nb_spec]
                    row_new[6 + self.nb_spec] = row.iloc[6 + self.nb_spec]

                    local_rows_new.append(row_new)
                    local_idx_new.append(i)

        # Gather results from all ranks onto rank 0
        all_rows_new = self.comm.gather(local_rows_new, root=0)
        all_idx_new = self.comm.gather(local_idx_new, root=0)

        if self.rank == 0:
            flat_rows = [r for sub in all_rows_new for r in sub]
            flat_idx = [idx for sub in all_idx_new for idx in sub]

            if flat_rows:
                self.df_augmented = pd.DataFrame(data=flat_rows, columns=cols, index=flat_idx)
                self.df_augmented = self.df_augmented.sort_index()
            else:
                self.df_augmented = pd.DataFrame(data=[], columns=cols)
        else:
            self.df_augmented = None

        # Broadcast self.df_augmented from rank 0 to every rank
        self.df_augmented = self.comm.bcast(self.df_augmented, root=0)

        # Now every rank has an identical self.df_augmented, so this is consistent everywhere
        self.df = pd.concat([self.df, self.df_augmented], axis=0).reset_index(drop=True)




    def save_database(self):

        self.df.to_csv(os.path.join(self.folder,"database_simus.csv"))



    def generate_XY_h5(self):

        # Compute X and Y arrays
        self.X, self.Y = self._get_X_Y()

        # Shuffle data — compute the permutation on rank 0 only, then broadcast it
        # so every rank applies the EXACT same shuffle (otherwise each rank's
        # np.random call would draw a different permutation independently).
        if self.rank == 0:
            # Use a fixed RNG seed so the shuffle is reproducible
            rng = np.random.default_rng(42)
            permutation_train = rng.permutation(self.X.shape[0])
        else:
            permutation_train = None
        
        permutation_train = self.comm.bcast(permutation_train, root=0)

        self.X = self.X.iloc[permutation_train].reset_index(drop=True)
        self.Y = self.Y.iloc[permutation_train].reset_index(drop=True)

        # Store initial solution in h5 file
        if self.rank==0:
            cols = ['Temperature'] + ['Pressure'] + self.species_names + ['Prog_var'] + ['HRR']
            f = h5py.File(f"{self.folder}/{self.dtb_file}","w")
            grp = f.create_group("ITERATION_00000")    # naming to be compatible with stochastic database writing
            dset = grp.create_dataset("X", data=self.X.values)
            dset.attrs["cols"] = cols
            dset = grp.create_dataset("Y", data=self.Y.values)
            dset.attrs["cols"] = cols
            f.close()



    def _get_X_Y(self):

        # Cantera gas object
        self.gas = ct.Solution(self.mech_file)

        df = self.df.iloc[:, :-2]

        cols = df.columns

        X = df.copy()

        X.columns = [str(col) + '_X' for col in cols]
        self.X_cols = X.columns.tolist()

        Y_np = self._advance_dt_cfd(X)
        Y_columns = [str(col) + '_Y' for col in cols]
        Y = pd.DataFrame(data=Y_np, columns = Y_columns)
        self.Y_cols = Y.columns.tolist()

        # Remove non needed items
        list_to_remove = ['enthalpy_X']
        [self.X_cols.remove(elt) for elt in list_to_remove]   
        X = X[self.X_cols]

        # Remove non needed lables for Y
        list_to_remove = ['enthalpy_Y']   # Temperature computed from enthalpy conservation
        # Removing unwanted items
        [self.Y_cols.remove(elt) for elt in list_to_remove]
        Y= Y[self.Y_cols]

        return X, Y
    


    def _advance_dt_cfd(self, X):

        n_rows = X.shape[0]
        n_cols = X.shape[1]

        # Split rows across MPI ranks (X is already identical on every rank)
        global_indices = np.arange(n_rows)
        local_indices = global_indices[self.rank::self.size]   # round-robin split

        Y_local = np.empty((len(local_indices), n_cols))

        for k_local, k_global in enumerate(local_indices):

            row = X.iloc[k_global]

            T = row.iloc[0]
            p = row.iloc[1]
            Yk = row.iloc[2:2 + self.nb_spec].values

            self.gas.TPY = T, p, Yk

            # Constant pressure reactor
            r = ct.IdealGasConstPressureReactor(self.gas)

            # Initializing reactor
            sim = ct.ReactorNet([r])

            # Advancing to dt
            sim.advance(self.time_step)

            # Updated state
            Y_local[k_local, 0] = self.gas.T
            Y_local[k_local, 1] = self.gas.P
            Y_local[k_local, 2:2 + self.nb_spec] = self.gas.Y
            Y_local[k_local, 3 + self.nb_spec] = -1.0 # We add progvar to the dataset but we set meaningless value as it is not needed later
            Y_local[k_local, 4 + self.nb_spec] = -1.0 # We add hrr to the dataset but we set meaningless value as it is not needed later

        # Gather local results + their global row indices on every rank
        # (Allgather so every rank ends up with the full Y, ready to use)
        all_Y_local = self.comm.allgather(Y_local)
        all_indices = self.comm.allgather(local_indices)

        Y = np.empty((n_rows, n_cols))
        for Y_chunk, idx_chunk in zip(all_Y_local, all_indices):
            Y[idx_chunk, :] = Y_chunk

        return Y
    


    def scatter_plot_data(self, x_var, y_var):

        fig, ax = plt.subplots()

        if self.augment_dataset:
            if self.include_zerod:
                ax.scatter(self.df_augmented[x_var][self.df_augmented["reactor_type"]==0], self.df_augmented[y_var][self.df_augmented["reactor_type"]==0], color="blue", alpha=0.2,  s=3)
            if self.include_oned_prem:
                ax.scatter(self.df_augmented[x_var][self.df_augmented["reactor_type"]==1], self.df_augmented[y_var][self.df_augmented["reactor_type"]==1], color="green", alpha=0.2,  s=3)
            if self.include_oned_diff:
                ax.scatter(self.df_augmented[x_var][self.df_augmented["reactor_type"]==2], self.df_augmented[y_var][self.df_augmented["reactor_type"]==2], color="purple", alpha=0.2,  s=3)
            flamelet_data = self.df_flamelet
        else:
            flamelet_data = self.df

        if self.include_zerod:
            ax.scatter(flamelet_data[x_var][flamelet_data["reactor_type"]==0], flamelet_data[y_var][flamelet_data["reactor_type"]==0], color="blue", s=3, label="0D")
        if self.include_oned_prem:
            ax.scatter(flamelet_data[x_var][flamelet_data["reactor_type"]==1], flamelet_data[y_var][flamelet_data["reactor_type"]==1], color="green", s=3, label="1D premixed")
        if self.include_oned_diff:
            ax.scatter(flamelet_data[x_var][flamelet_data["reactor_type"]==2], flamelet_data[y_var][flamelet_data["reactor_type"]==2], color="purple", s=3, label="1D diffusion")

        ax.set_xlabel(x_var, fontsize=14)
        ax.set_ylabel(y_var, fontsize=14)

        ax.legend()

        fig.savefig(os.path.join(self.folder,f"scatter_plot_{x_var}_{y_var}.png"),dpi=500)
