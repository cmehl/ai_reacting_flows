import glob
import os
import sys
import joblib
import pickle
import shutil
import shelve

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import model_from_json

import cantera as ct

from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation
from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation_RR
from ai_reacting_flows.ann_model_generation.tensorflow_custom import AtomicConservation_RR_lsq
from ai_reacting_flows.ann_model_generation.tensorflow_custom import GetN2Layer, ZerosLayer, GetLeftPartLayer, GetRightPartLayer

import ai_reacting_flows.tools.utilities as utils


class ModelTesting(object):


    def __init__(self, testing_parameters):

        self.verbose = True
        
        # Model folder name
        self.models_folder = testing_parameters["models_folder"]

        # Chemistry (maybe an information to get from model folder ?)
        self.fuel = testing_parameters["fuel"]
        self.with_N_chemistry = testing_parameters["with_N_chemistry"]

        self.spec_to_plot = testing_parameters["spec_to_plot"]

        # Getting parameters from ANN model
        shelfFile = shelve.open(self.models_folder + "/model_params")
        self.output_omegas = shelfFile["output_omegas"]
        self.log_transform = shelfFile["log_transform"]
        self.threshold = shelfFile["threshold"]
        self.remove_N2 = shelfFile["remove_N2"]
        self.hard_constraints_model = shelfFile["hard_constraints_model"]
        self.mechanism_type = shelfFile["mechanism_type"]
        shelfFile.close()

        # Chemical mechanisms
        self.mech = self.models_folder + "/mech_detailed.yaml"
        if self.mechanism_type=="reduced":
            self.reduced_mechanism = self.models_folder + "/mech_reduced.yaml"

        # CANTERA solution object needed to access species indices
        gas = ct.Solution(self.mech)
        self.spec_list_ref = gas.species_names
        self.nb_species_ref = len(self.spec_list_ref)

        # Getting species and number of species
        if self.mechanism_type=="reduced":
            gas_reduced = ct.Solution(self.reduced_mechanism)
            self.spec_list_ANN = gas_reduced.species_names
            self.nb_species_ANN_tot = len(self.spec_list_ANN)
            self.nb_species_ANN = self.nb_species_ANN_tot
            if self.remove_N2:
                self.nb_species_ANN = self.nb_species_ANN_tot - 1
        else:
            self.spec_list_ANN = self.spec_list_ref
            self.nb_species_ANN_tot = self.nb_species_ref
            if self.remove_N2:
                self.nb_species_ANN = self.nb_species_ref - 1



        # Find number of clusters -> equal for instance to number of json files in model folder
        self.nb_clusters = len(glob.glob1(self.models_folder,"*.json"))
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


        # Load models
        self.load_models()


        # Load scalers
        self.load_scalers()


        # Load clustering algorithm if relevant
        if self.nb_clusters>1:
           self.load_clustering_algorithm()


        # Progress variable definition (we do it in any case, even if no clustering based on progress variable)
        self.pv_species = testing_parameters["pv_species"]
        self.pv_ind_detailed = []
        if self.mechanism_type=="reduced":
            self.pv_ind_reduced = []
        for i in range(len(self.pv_species)):
            self.pv_ind_detailed.append(gas.species_index(self.pv_species[i])) 
            if self.mechanism_type=="reduced":
                self.pv_ind_reduced.append(gas_reduced.species_index(self.pv_species[i])) 


        # Getting fictive species names
        if self.mechanism_type=="reduced":
            self.fictive_species = []
            for spec in gas_reduced.species_names:
                if spec.endswith("_F"):
                    self.fictive_species.append(spec)

            self.nb_spec_fictive = len(self.fictive_species)


#-----------------------------------------------------------------------
#   MAIN TESTING FUNCTIONS
#-----------------------------------------------------------------------

    # OD IGNITION
    def test_0D_ignition(self, phi, T0, pressure, dt, nb_ite=100):

        #-------------------- INITIALISATION---------------------------
        
        # CANTERA gas object
        gas = ct.Solution(self.mech)

        # Setting composition
        fuel_ox_ratio = gas.n_atoms(self.fuel,'C') + 0.25*gas.n_atoms(self.fuel,'H') - 0.5*gas.n_atoms(self.fuel,'O')
        compo = f'{self.fuel}:{phi:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
        gas.TPX = T0, pressure, compo

        # Defining reactor
        r = ct.IdealGasConstPressureReactor(gas)
            
        sim = ct.ReactorNet([r])
        simtime = 0.0
        states = ct.SolutionArray(gas, extra=['t'])

        # Initial state (saved for later use in NN computation)
        states.append(r.thermo.state, t=0)
        T_ini = states.T
        Y_ini = states.Y

        # For atomic conservation we need atom molecular weights
        # Order: C, H, O, N
        W_atoms = np.array([12.011, 1.008, 15.999, 14.007])
        
        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.parse_species_names(self.spec_list_ANN)

        #-------------------- CVODE COMPUTATION---------------------------
        # Remark: we don't use the method we wrote below

        for i in range(nb_ite):
            simtime += dt
            sim.advance(simtime)
            states.append(r.thermo.state, t=simtime*1e3)
                
        # Equilibrium temperature
        Teq_ref = states.T[len(states.T)-1]
        
        # Convert in a big array (m,NS+1) for later normalization
        state_ref = states.T.reshape(nb_ite+1, 1)
        
        for spec in self.spec_list_ref:
            Y = states.Y[:,gas.species_index(spec)].reshape(nb_ite+1, 1)
            state_ref = np.concatenate((state_ref, Y), axis=-1)


        #-------------------- ANN SOLVER ---------------------------

        # Initial solution
        if self.mechanism_type=="reduced":
            gas_reduced = ct.Solution(self.reduced_mechanism)
            gas_reduced.TPX = T0, pressure, compo
            #
            T_ann = gas_reduced.T
            Y_k_ann = gas_reduced.Y
        else:
            Y_k_ann = Y_ini
            T_ann = T_ini

        # Vectors to store time data
        state_save = np.append(T_ann, Y_k_ann)
        sumYs = [1.0]
        progvar_vect = [0.0]

        # atomic composition of species
        if self.mechanism_type=="reduced":
            molecular_weights = utils.get_molecular_weights(gas_reduced.species_names)
        else:
            molecular_weights = utils.get_molecular_weights(gas.species_names)
        atomic_cons = np.dot(atomic_array,np.reshape(Y_k_ann,-1)/molecular_weights)
        atomic_cons = np.multiply(W_atoms, atomic_cons)

        # NEURAL NETWORK COMPUTATION
        i=0
        time = 0.0
        state = np.append(T_ann, Y_k_ann)
        for i in range(nb_ite):

            # Old state
            T_old = state[0]
            Y_old = state[1:]

            # Computing current progress variable
            progvar = self.compute_progvar(state, pressure, self.mechanism_type)

            # Attribute cluster
            self.attribute_cluster(state, progvar)

            print(f"Current point in cluster: {self.cluster} \n")

            T_new, Y_new = self.advance_state_NN(T_old, Y_old, pressure, dt)

            # Vector with all variable (T and Yks)
            state = np.append(T_new, Y_new)
                
            # Saving values
            state_save = np.vstack([state_save,state])
            progvar_vect.append(progvar)
                    
            # Mass conservation
            sumYs.append(np.sum(Y_new,axis=0))

            
            # atomic composition of species  WHAT IS SELF.NO_SPECIES_NN ?
            atomic_cons_current = np.dot(atomic_array,Y_new.reshape(self.nb_species_ANN_tot)/molecular_weights)
            atomic_cons_current = np.multiply(W_atoms, atomic_cons_current)
            atomic_cons = np.vstack([atomic_cons,atomic_cons_current])
            
            # Incrementation
            i+=1
            time += dt
        
        
        # ANN equilibrium (last temperature)
        Teq_ann = T_new

        if self.verbose:
            # Error on equilibrium temperature
            error_Teq = 100.0*abs(Teq_ref - Teq_ann)/Teq_ref
            print(f"\nError on equilibrium flame temperature is: {error_Teq} % \n")


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
            ax.plot(states.t, states.Y[:,gas.species_index(spec)], ls="--", color = "k", lw=2, label="CVODE")
            ax.plot(states.t, state_save[:,self.spec_list_ANN.index(spec)+1], ls="-", color = "b", lw=2, marker='x', label="NN")
            ax.set_xlabel('$t$ $[ms]$', fontsize=ftsize)
            ax.set_ylabel(f'{spec} mass fraction $[-]$', fontsize=ftsize)
            if spec=="N2": #because N2 is often constant
                ax.set_ylim([states.Y[0,gas.species_index(spec)]*0.95, states.Y[0,gas.species_index(spec)]*1.05])
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



    # 1D PREMIXED
    def test_1D_premixed(self, phi, T0, pressure, dt, T_threshold=0.0):

        # CANTERA gas object
        gas = ct.Solution(self.mech)

        #-------------------- 1D FLAME COMPUTATION ---------------------------

        # Cantera computation settings
        initial_grid = np.linspace(0.0, 0.03, 10)  # m
        tol_ss = [1.0e-4, 1.0e-9]  # [rtol atol] for steady-state problem
        tol_ts = [1.0e-5, 1.0e-5]  # [rtol atol] for time stepping
        loglevel = 0  # amount of diagnostic output (0 to 8)
            
        # Determining initial composition using phi
        fuel_ox_ratio = gas.n_atoms(self.fuel,'C') + 0.25*gas.n_atoms(self.fuel,'H') - 0.5*gas.n_atoms(self.fuel,'O')
        compo = f'{self.fuel}:{phi:3.2f}, O2:{fuel_ox_ratio:3.2f}, N2:{fuel_ox_ratio * 0.79 / 0.21:3.2f}'
            
        gas.TPX = T0, pressure, compo

        print(f"Computing Cantera 1D flame with T0 = {T0} K and phi = {phi}")
                
        f = ct.FreeFlame(gas, initial_grid)
                
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
        if self.mechanism_type=="reduced":
            gas_reduced = ct.Solution(self.reduced_mechanism)
            Yt_dt_ann = np.zeros((gas_reduced.n_species, Yt.shape[1]))
        else:
            Yt_dt_ann = np.zeros((gas.n_species, Yt.shape[1]))


        #-------------------- EXACT REACTION RATES --------------------------- 

        nb_0_reactors = len(X_grid) # number of 1D reactors
        for i_reac in range(nb_0_reactors):
                
            gas.TPY = Tt[i_reac], pressure, Yt[:,i_reac]

            r = ct.IdealGasConstPressureReactor(gas)
                
            # Initializing reactor
            sim = ct.ReactorNet([r])
            time = 0.0
            states = ct.SolutionArray(gas, extra=['t'])
            
            # We advance solution by dt
            time = dt
            sim.advance(time)
            states.append(r.thermo.state, t=time * 1e3)
            
            Yt_dt_exact[:,i_reac] = states.Y
            
        # Reaction rates
        Omega_exact = (Yt_dt_exact-Yt)/dt


        #-------------------- ANN REACTION RATES ---------------------------

        # Initial solution (depends on the use of reduced mechanism)
        if self.mechanism_type=="reduced":
            Yt_ann = np.zeros((gas_reduced.n_species, Yt.shape[1]))
            # Real species
            for i, spec in enumerate(gas_reduced.species_names):
                if spec in gas.species_names:
                    Yt_ann[i,:] = Yt[gas.species_index(spec),:]

            # Close conservation using fictive species present in mechanism
            Yt_ann = self.close_fictive_species(Yt, Yt_ann)
        else:
            Yt_ann = Yt

        for i_reac in range(nb_0_reactors):

            # Computing current progress variable
            state = np.append(Tt[i_reac], Yt_ann[:,i_reac])
            progvar = self.compute_progvar(state, pressure, self.mechanism_type)

            # Attribute cluster
            self.attribute_cluster(state, progvar)

            print(f"Current point in cluster: {self.cluster} \n")
            
            
            # advance to t + dt
            if Tt[i_reac] >= T_threshold:
                T_new, Y_new = self.advance_state_NN(Tt[i_reac], Yt_ann[:,i_reac], pressure, dt)
            else:
                T_new = Tt[i_reac]
                Y_new = Yt_ann[:,i_reac]
            
            Yt_dt_ann[:,i_reac] = np.reshape(Y_new,-1)

        # Reaction rates
        Omega_ann = (Yt_dt_ann-Yt_ann)/dt


        #-------------------- PLOTTING --------------------------- 
        
        # Create directory with plots
        folder = f'./plots_1D_prem_T0#{T0}_phi#{phi}'
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        for spec in self.spec_to_plot:
            fig, ax = plt.subplots(1, 1)
            ax.plot(f.grid, Omega_exact[gas.species_index(spec),:], ls="-", color = "b", lw=2, label="Exact")
            if self.mechanism_type=="reduced":
                ax.plot(f.grid, Omega_ann[gas_reduced.species_index(spec),:], ls="--", color = "purple", lw=2, label="Neural network")
            else:
                ax.plot(f.grid, Omega_ann[gas.species_index(spec),:], ls="--", color = "purple", lw=2, label="Neural network")
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

            # Model reconstruction from JSON file
            with open(self.models_folder + f'/model_architecture_cluster{i}.json', 'r') as f:

                if self.hard_constraints_model==1:
                    if self.output_omegas==True:
                        model = model_from_json(f.read(), custom_objects={'AtomicConservation_RR': AtomicConservation_RR,
                                                                           'GetN2Layer' : GetN2Layer,
                                                                           'ZerosLayer': ZerosLayer,
                                                                           'GetLeftPartLayer': GetLeftPartLayer,
                                                                           'GetRightPartLayer': GetRightPartLayer})
                    else:
                        model = model_from_json(f.read(), custom_objects={'AtomicConservation': AtomicConservation,
                                                                           'GetN2Layer' : GetN2Layer,
                                                                           'GetLeftPartLayer': GetLeftPartLayer,
                                                                           'GetRightPartLayer': GetRightPartLayer})
                elif self.hard_constraints_model==2:
                    if self.output_omegas==True:
                        model = model_from_json(f.read(), custom_objects={'AtomicConservation_RR_lsq': AtomicConservation_RR_lsq,
                                                                           'GetN2Layer' : GetN2Layer,
                                                                           'ZerosLayer': ZerosLayer,
                                                                           'GetLeftPartLayer': GetLeftPartLayer,
                                                                           'GetRightPartLayer': GetRightPartLayer})
                    else:
                        sys.exit("hard_constraints_model=2 not written for output_omegas=False")
                else:
                    model = model_from_json(f.read(), custom_objects={'GetN2Layer' : GetN2Layer,
                                                                           'ZerosLayer': ZerosLayer,
                                                                           'GetLeftPartLayer': GetLeftPartLayer,
                                                                            'GetRightPartLayer': GetRightPartLayer})
            
            # Load weights into the new model
            model.load_weights(self.models_folder + f'/model_weights_cluster{i}.h5')

            # Add to list
            self.models_list.append(model)

        if self.verbose:
            for i in range(self.nb_clusters):
                print(f"\n ----------CLUSTER MODEL {i}---------- \n")
                self.models_list[i].summary()



    def load_scalers(self):

        self.Xscaler_list = []
        self.Yscaler_list = []

        for i in range(self.nb_clusters):
            Xscaler = joblib.load(self.models_folder + f"/Xscaler_cluster{i}.pkl")
            Yscaler = joblib.load(self.models_folder + f"/Yscaler_cluster{i}.pkl") 

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

        if self.clustering_method=="kmeans":
            # Scaling vector
            vect_scaled = self.kmeans_scaler.transform(state_vector.reshape(1, -1))
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

        # CANTERA gas object
        gas = ct.Solution(self.mech)
        
        # Gas object modification
        gas.TPY= T_old, pressure, Y_old
            
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

        # CANTERA gas object
        if self.mechanism_type=="reduced":
            gas = ct.Solution(self.reduced_mechanism)
        else:
            gas = ct.Solution(self.mech)
                
        # Gas object modification
        gas.TPY= T_old, pressure, Y_old

        # Grouping temperature and mass fractions in a "state" vector
        state_vector = np.append(T_old, Y_old)

        # If N2 is not considered, it needs to be removed from the state_vector
        if self.remove_N2:
            n2_index = self.spec_list_ANN.index("N2")
            n2_value = state_vector[n2_index+1]
            state_vector = np.delete(state_vector, n2_index+1)

        
        # NN update for Yk's
        if self.log_transform==1:
            state_vector[state_vector<self.threshold] = self.threshold
        elif self.log_transform==2:
            state_vector[state_vector<0.0] = 0.0
            
        log_state = np.zeros(self.nb_species_ANN+1)
        log_state[0] = state_vector[0]
        if self.log_transform==1:
            log_state[1:] = np.log(state_vector[1:])
        elif self.log_transform==2:
            log_state[1:] = (state_vector[1:]**self.lambda_bct - 1.0)/self.lambda_bct
        else:
            log_state[1:] = state_vector[1:]
        

        # input of NN
        log_state = log_state.reshape(1, -1)
        
        NN_input = self.Xscaler_list[self.cluster].transform(log_state)
                
        # New state predicted by ANN
        state_new = self.models_list[self.cluster].predict(NN_input, batch_size=1)
                
        # Getting Y and scaling
        Y_new = self.Yscaler_list[self.cluster].inverse_transform(state_new)

        # Log transform of species
        if self.log_transform>0:
            if self.output_omegas:
                log_state_updated = log_state[0,1:] + Y_new
                if self.log_transform==1:
                    Y_new = np.exp(log_state_updated)
                elif self.log_transform==2:
                    Y_new = (self.lambda_bct*log_state_updated+1.0)**(1./self.lambda_bct)
            else:
                if self.log_transform==1:
                    Y_new = np.exp(Y_new)
                elif self.log_transform==2:
                    Y_new = (self.lambda_bct*Y_new+1.0)**(1./self.lambda_bct)
                
                
        # If reaction rate outputs from network
        if self.output_omegas and self.log_transform==0:
            if self.remove_N2:
                Y_old_wo_N2 = np.delete(Y_old, n2_index)
                Y_new += Y_old_wo_N2
            else:
                Y_new += Y_old

        # Adding back N2 before computing temperature
        if self.remove_N2:
            Y_new = np.insert(Y_new, n2_index, n2_value)
                
        # Deducing T from energy conservation
        T_new = state_vector[0] - (1/gas.cp)*np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*(Y_new-Y_old))
        
        # Reshaping mass fraction vector
        Y_new = Y_new.reshape(Y_new.shape[0],1)


        return T_new, Y_new


#-----------------------------------------------------------------------
#   THERMO-CHEMICAL FUNTIONS 
#-----------------------------------------------------------------------

    # TO CLEAN
    def compute_progvar(self, state, pressure, mechanism_type):

        # Equilibrium state in present conditions
        if mechanism_type=="reduced":
            gas = ct.Solution(self.reduced_mechanism)
            pv_ind = self.pv_ind_reduced
        else:
            gas = ct.Solution(self.mech)
            pv_ind = self.pv_ind_detailed
        gas.TPY = state[0], pressure, state[1:]
        gas.equilibrate('HP')

        Yc_eq = 0.0
        Yc = 0.0
        for i in pv_ind:
            Yc_eq += gas.Y[i]
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
        A_atomic = utils.get_molar_mass_atomic_matrix(gas.species_names, self.fuel, self.with_N_chemistry)
        Ya = np.dot(A_atomic, Y_ref.transpose()).transpose()
        h = np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*Y_ref, axis=1)
        #
        A_atomic_reduced = utils.get_molar_mass_atomic_matrix(gas_reduced.species_names, self.fuel, self.with_N_chemistry)
        Ya_reduced = np.dot(A_atomic_reduced, Y_reduced.transpose()).transpose()
        h_reduced = np.sum(gas_reduced.partial_molar_enthalpies/gas_reduced.molecular_weights*Y_reduced, axis=1)
        #
        Delta_Ya = Ya - Ya_reduced
        Delta_h = h - h_reduced
        #
        Delta = np.concatenate([Delta_Ya, Delta_h.reshape(-1,1)], axis=1)
        #
        A_atomic_fictive = utils.get_molar_mass_atomic_matrix(self.fictive_species, self.fuel, self.with_N_chemistry)
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



































