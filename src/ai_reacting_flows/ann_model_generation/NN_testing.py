import glob
import os
import sys
import joblib
import pickle
import shutil
import oyaml as yaml

import numpy as np
import matplotlib.pyplot as plt

import cantera as ct

import torch
# from torchinfo import summary
from ai_reacting_flows.ann_model_generation.NN_models import MLPModel, DeepONet, DeepONet_shift
# from ai_reacting_flows.ann_model_generation.cantera_runs import compute_nn_cantera_0D_homo

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
        self.log_transform_X = dtb_processing_params["log_transform_X"]
        self.log_transform_Y = dtb_processing_params["log_transform_Y"]
        self.threshold = dtb_processing_params["threshold"]
        self.remove_N2 = not dtb_processing_params["with_N_chemistry"]
        self.clustering_method  = dtb_processing_params["clustering_method"]
        self.output_omegas = dtb_processing_params["output_omegas"]

        # Getting parameters from ANN model
        with open(os.path.join(self.run_folder, f"STOCH_DTB_{dtb_processing_params['dtb_folder_suffix']}","dtb_params.yaml"), "r") as file:
            dtb_parameters = yaml.safe_load(file)
        
        self.fuel = dtb_parameters["fuel"][0] # fuel is a list, testing only takes 1 component fuel so far
        self.mech = dtb_parameters["mech_file"]

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

#-----------------------------------------------------------------------
#   MAIN TESTING FUNCTIONS
#-----------------------------------------------------------------------
    # OD IGNITION
    def test_0D_ignition(self, phi, T0, pressure, dt, nb_ite=100):
        # Initialization of ignition delays
        tau_ign_CVODE = 0.0
        tau_ign_ANN = 0.0

        #-------------------- INITIALISATION---------------------------
        # Setting composition
        fuel_ox_ratio = self.gas.n_atoms(self.fuel,'C') + 0.25*self.gas.n_atoms(self.fuel,'H') - 0.5*self.gas.n_atoms(self.fuel,'O')
        compo = f'{self.fuel}:{phi:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
        self.gas.TPX = T0, pressure, compo

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
        # Remark: we don't use the method we wrote below

        for i in range(nb_ite):
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
        # Counting calls
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
        i=0
        time = 0.0
        state = np.append(T_ann, Y_k_ann)
        for i in range(nb_ite):
            print(f"ITERATION {i} \n")

            # Old state
            T_old = state[0]
            Y_old = state[1:]

            # Computing current progress variable
            progvar = self.compute_progvar(state, pressure)

            # Attribute cluster
            if self.nb_clusters>0:
                self.attribute_cluster(state, progvar)
            else:
                self.cluster = 0

            print(f"Current point in cluster: {self.cluster} \n")

            T_new, Y_new = self.advance_state_NN(T_old, Y_old, pressure, dt)

            # If hybrid, we check conservation and use CVODE if not satisfied
            if self.hybrid_ann_cvode:
                if self.ia_success is False:
                    # CVODE advance
                    T_new, Y_new = self.advance_state_CVODE(T_old, Y_old, pressure, dt)
                    cvode_calls += 1
                    print(">> Hybrid model: CVODE is used")
                else:
                    ann_calls +=1
                    print(">> Hybrid model: ANN is used")
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
            
            # Incrementation
            i+=1
            time += dt

            if T_new > T0+200.0 and tau_ign_ANN==0.0:
                tau_ign_ANN = time
        
        # ANN equilibrium (last temperature)
        Teq_ann = T_new

        if self.verbose:
            # Error on equilibrium temperature
            error_Teq = 100.0*abs(Teq_ref - Teq_ann)/Teq_ref
            print(f"\nError on equilibrium flame temperature is: {error_Teq} % \n")

        # Print number of calls
        print("\n NUMBER OF CALLS TO SOLVERS:")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        #-------------------- PLOTTING --------------------------- 
        # Common args
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

    # 1D PREMIXED
    def test_1D_premixed(self, phi, T0, pressure, dt, T_threshold=0.0):

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

        print(f"Computing Cantera 1D flame with T0 = {T0} K and phi = {phi}")
                
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

        # Mass fractions at time t
        Yt = f.Y

        # Temperature at time t
        Tt = f.T
            
        # Computing progress variable
        c = np.empty(len(f.grid))
        nb_pnts_total = len(f.grid)
        for i in range(nb_pnts_total):
            state = np.append(Tt[i], Yt[:,i])
            c[i] = self.compute_progvar(state, pressure, "detailed")

        # x axis
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
        # Counting calls
        ann_calls = 0
        cvode_calls = 0

        # Initial solution
        Yt_ann = Yt

        for i_reac in range(nb_0_reactors):
            # Computing current progress variable
            state = np.append(Tt[i_reac], Yt_ann[:,i_reac])
            progvar = self.compute_progvar(state, pressure, self.mechanism_type)

            # Attribute cluster
            if self.nb_clusters>0:
                self.attribute_cluster(state, progvar)
            else:
                self.cluster = 0

            print(f"Current point in cluster: {self.cluster} \n")
            
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
                        print(">> Hybrid model: CVODE is used")
                    else:
                        ann_calls +=1
                        print(">> Hybrid model: ANN is used")
                else:
                    ann_calls += 1
            else:
                T_new = Tt[i_reac]
                Y_new = Yt_ann[:,i_reac]
            
            Yt_dt_ann[:,i_reac] = np.reshape(Y_new,-1)

        # Reaction rates
        Omega_ann = (Yt_dt_ann-Yt_ann)/dt

        # Print number of calls
        print("\n NUMBER OF CALLS TO SOLVERS:")
        print(f"   >>> Number of ANN calls: {ann_calls}")
        print(f"   >>> Number of CVODE calls: {cvode_calls}")

        #-------------------- PLOTTING --------------------------- 
        # Create directory with plots
        folder = f'./plots_1D_prem_T0#{T0}_phi#{phi}'
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for spec in self.spec_to_plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(f.grid, Omega_exact[self.gas.species_index(spec),:], ls="-", color = "b", lw=2, label="Exact")
            if self.mechanism_type=="reduced":
                ax.plot(f.grid, Omega_ann[self.gas_reduced.species_index(spec),:], ls="--", color = "purple", lw=2, label="Neural network")
            else:
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

        for i in range(self.nb_clusters):
            Xscaler = joblib.load(f"{self.dataset_path}/cluster{i}/Xscaler.save")
            Yscaler = joblib.load(f"{self.dataset_path}/cluster{i}/Yscaler.save") 

            self.Xscaler_list.append(Xscaler)
            self.Yscaler_list.append(Yscaler)

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
        NN_input = self.Xscaler_list[self.cluster].transform(log_state)
                
        # New state predicted by ANN
        NN_input = torch.tensor(NN_input, dtype=torch.float64).to(self.device)
        state_new = self.models_list[self.cluster](NN_input)
        state_new = state_new.detach().cpu().numpy()

        # Getting Y and scaling
        Y_new = self.Yscaler_list[self.cluster].inverse_transform(state_new)

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

        # We apply the normalizer
        diff_log_Y_cvode = diff_log_Y_cvode.reshape(1,-1)
        diff_log_Y_ann = diff_log_Y_ann.reshape(1,-1)
        #
        diff_log_Y_cvode_norm = self.Yscaler_list[self.cluster].transform(diff_log_Y_cvode)
        diff_log_Y_ann_norm = self.Yscaler_list[self.cluster].transform(diff_log_Y_ann)
        #
        diff_log_Y_cvode_norm = diff_log_Y_cvode_norm.reshape(-1)
        diff_log_Y_ann_norm = diff_log_Y_ann_norm.reshape(-1)

        # Error (MSE)
        # mse = MeanSquaredError()
        # err_Yk = mse(diff_log_Y_cvode_norm, diff_log_Y_ann)  

        # Test: we compute error on actual mass fractions
        err_Yk = (Y_cvode-Y_ann)**2

        err_T = 100.0*np.abs((T_ann-T_cvode)/T_cvode)

        return err_Yk, err_T

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

    # Closure model for fictive species
    def close_fictive_species(self, Y_ref, Y_reduced):
        # Transposing to have a shape nb_points,nb_species, which is more convenient
        Y_ref = Y_ref.transpose()
        Y_reduced = Y_reduced.transpose()

        gas = ct.Solution(self.mech)
        gas_reduced = ct.Solution(self.reduced_mechanism)

        # Adding fictive species by conservation of properties
        A_atomic = utils.get_molar_mass_atomic_matrix(gas.species_names, [self.fuel], not self.remove_N2)
        Ya = np.dot(A_atomic, Y_ref.transpose()).transpose()
        h = np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*Y_ref, axis=1)
        #
        A_atomic_reduced = utils.get_molar_mass_atomic_matrix(gas_reduced.species_names, [self.fuel], not self.remove_N2)
        Ya_reduced = np.dot(A_atomic_reduced, Y_reduced.transpose()).transpose()
        h_reduced = np.sum(gas_reduced.partial_molar_enthalpies/gas_reduced.molecular_weights*Y_reduced, axis=1)
        #
        Delta_Ya = Ya - Ya_reduced
        Delta_h = h - h_reduced
        #
        Delta = np.concatenate([Delta_Ya, Delta_h.reshape(-1,1)], axis=1)
        #
        A_atomic_fictive = utils.get_molar_mass_atomic_matrix(self.fictive_species, [self.fuel], not self.remove_N2)
        partial_molar_enthalpies_fictive = np.empty(self.nb_spec_fictive)
        molecular_weights_fictive = np.empty(self.nb_spec_fictive)
        for i, spec in enumerate(self.fictive_species):
            partial_molar_enthalpies_fictive[i] = gas_reduced.partial_molar_enthalpies[gas_reduced.species_index(spec)]
            molecular_weights_fictive[i] = gas_reduced.molecular_weights[gas_reduced.species_index(spec)]
        #
        delta_h_f = partial_molar_enthalpies_fictive/molecular_weights_fictive
        #
        matrix_linear_system = np.concatenate([A_atomic_fictive, delta_h_f.reshape(1,-1)])
        #
        matrix_inv = np.linalg.inv(matrix_linear_system)

        # Getting mass fractions of fictive species
        Yk_fictive = np.dot(matrix_inv, Delta.transpose()).transpose()
        #
        j = 0
        for i, spec in enumerate(gas_reduced.species_names):
            if spec in self.fictive_species:
                Y_reduced[:,i] = Yk_fictive[:,j]
                j+=1

        # Transposing back
        Y_reduced = Y_reduced.transpose()

        return Y_reduced

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











