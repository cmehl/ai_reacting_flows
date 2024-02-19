import os
import numpy as np 
import pandas as pd
import shelve
import glob 
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import time  
import matplotlib.pyplot as plt
import joblib
import shutil 
import pickle
import sys 
import cantera as ct
import ai_reacting_flows.tools.utilities as utils
from collections import OrderedDict

torch.set_printoptions(precision=10) # MK NECESSARY FOR THE FORWARD DEF IF NOT LOSS PRECISION 

class ClusterModels :
    def __init__(self, params : dict) -> None:
        
        
        #Communs parameters for learning and testing 
        self.folder = params["folder"]
        self.hidden_layers = params["nb_units_in_layers_list"] 
        self.activation = params["layers_activation_list"] 
        self.layers_type = params["layers_type"]            
        self.path = params["dataset_path"] 
        self.dt_simu = params['dt_simu']
        self.fuel = params['fuel']
        self.mechanism_type = params['mechanism_type']
        self.device = params['device']
        self.removeN2 = params["remove_N2"]
        self.hard_constraints_model=params['hard_constraints_model']
        self.log_X = params['log_X']
        self.log_Y = params['log_Y']
        self.framwork = params['Framework']
    

    # MK Only testing parameters 
    def init_training(self,params : dict) -> None : 

        self.new_model = params["new_model"]
        self.batch_size = params["batch_size"]        
        self.shuffle = params["shuffle"]  
        self.loss_function = params["loss_function"]      
        self.epoch = params["epoch"]       
        self.optimizer = params["optimizer"]
        self.init_lr = params["initial_learning_rate"]
        self.decay_steps = params["decay_steps"]
        self.decay_rate = params["decay_rate"]
        
        self.nb_clusters = len(next(os.walk(self.path))[1])
        print("CLUSTERING:")
        print(f">> Number of clusters is: {self.nb_clusters}")

        self.directory = "./" + self.folder

        if self.new_model==True:
            print(">> A new folder is created.")
            # Remove folder if already exists
            shutil.rmtree(self.directory, ignore_errors=True)
            # Create folder
            os.makedirs(self.directory)
            os.mkdir(self.directory + "/training")
            os.mkdir(self.directory + "/training/training_curves")
            os.mkdir(self.directory + "/evaluation" )
            with open(self.directory + "/__init__.py",'w') : pass 


            #copy
            if self.mechanism_type == "detailed" : 
                self.mech = self.path + "/mech_detailed.yaml"
                shutil.copy(self.path +"/mech_detailed.yaml",self.directory)

            elif self.mechanism_type == "reduced" : 
                self.mech = self.path +"/mech_reduced.yaml"
                shutil.copy(self.path + "/mech_reduced.yaml",self.directory)


            # MK : CHANGE THIS PART : NO EXPORT POSSIBLE BETWEEN ENER AND LOCAL 
            shelfFile = shelve.open(self.path +"/dtb_params")
            self.log_X = shelfFile["log_transform_X"]
            self.log_Y = shelfFile["log_transform_Y"]
            self.output_omegas = shelfFile["output_omegas"]
            self.clustering_type = shelfFile["clusterization_method"]
            shelfFile.close()

            if self.clustering_type=="progvar":
                shutil.copy(self.path + "/c_bounds.pkl", self.directory)
            elif self.clustering_type=="kmeans":
                shutil.copy(self.path + "/kmeans_model.pkl", self.directory)
                shutil.copy(self.path + "/Xscaler_kmeans.pkl", self.directory)
                shutil.copy(self.path + "/kmeans_norm.dat", self.directory)
                shutil.copy(self.path + "/km_centroids.dat", self.directory)

            for i in range(0,self.nb_clusters) : 
                os.mkdir(self.directory+f"/my_model_cluster_{i}")
            
            with open(self.directory +"/__init__.py",'w') : pass 
        else:
            if not os.path.exists(self.directory):
                sys.exit(f'ERROR: new_model_folder is set to False but model {self.directory} does not exist')
            print(f">> Existing model folder {self.directory} is used. \n")

    def init_testing(self,params : dict)-> None:
        print("TESTING")

        self.verbose = True 

        self.spec_to_plot = params['spec_to_plot']       
        self.pv_specices = params['pv_species']
        self.yk = params['yk_renormalization']
        self.hybrid = params['hybrid_ann_cvode']
        self.hybrid_tol = params['hybrid_ann_cvode_tol']
        self.phi = params['phi']
        self.T0 = params["T0"]
        self.pressure = params['pressure']
        self.dt = params['dt']
        self.nb_ite = params['nb_ite']
        self.output_omegas = params['output_omegas']
        self.threshold = params['threshold']

        self.directory = "./" + self.folder

        # MK : CHANGE THIS PART NO EXPORT BETWEEN ENER AND LOCAL 

        #shelfFile = shelve.open(self.path +"/dtb_params")
        #self.output_omegas = shelfFile["output_omegas"]
        #self.log_X = shelfFile["log_transform_X"]
        #self.log_Y = shelfFile["log_transform_Y"]
        #self.threshold = shelfFile["threshold"]
        #shelfFile.close()

        if self.mechanism_type == "detailed" : 
            self.mech = self.folder +"/mech_detailed.yaml"
        elif self.mechanism_type == "reduced" : 
            self.mech = self.folder +"/mech_reduced.yaml"

        gas = ct.Solution(self.mech)
        self.spec_list_ref = gas.species_names 
        self.nb_species_ref = len(self.spec_list_ref)
        # Getting species and number of species

        if self.mechanism_type=="reduced":
            gas_reduced = ct.Solution(self.mech)
            self.spec_list_ANN = gas_reduced.species_names
            self.nb_species_ANN_tot = len(self.spec_list_ANN)
            self.nb_species_ANN = self.nb_species_ANN_tot
            if self.removeN2:
                self.nb_species_ANN = self.nb_species_ANN_tot - 1
        else:
            self.spec_list_ANN = self.spec_list_ref
            self.nb_species_ANN_tot = self.nb_species_ref
            if self.removeN2:
                self.nb_species_ANN = self.nb_species_ref - 1
        
        # Find number of clusters -> equal for instance to number of json files in model folder
        self.nb_clusters = len(glob.glob1(self.directory,"*.pt"))
        print(f">> {self.nb_clusters} models were found")

        #If more than one cluster, we need to check the clustering method
        if self.nb_clusters>1:
            if os.path.exists(self.directory + "/kmeans_model.pkl"):
                self.clustering_method = "kmeans"
            elif os.path.exists(self.directory + "/c_bounds.pkl"):
                self.clustering_method = "progvar"
            else:
                sys.exit("ERROR: There are more than one clusters and no clustering method file in the model folder")
            print(f"   >> Clustering method {self.clustering_method} is used \n")
        else:
            self.clustering_method = None

        #loads Models 
        self.loads_models() 
        
        #load scaler for X and Y 
        self.loads_scalers() 

        # Load clustering algorithm if relevant
        if self.nb_clusters>1:
           self.load_clustering_algorithm()

        self.pv_ind_detailed = [] 
        if self.mechanism_type=="reduced":
            self.pv_ind_reduced = []
        for i in range(len(self.pv_specices)):
            self.pv_ind_detailed.append(gas.species_index(self.pv_specices[i])) 
            if self.mechanism_type=="reduced":
                self.pv_ind_reduced.append(gas_reduced.species_index(self.pv_specices[i])) 


        # Getting fictive species names
        if self.mechanism_type=="reduced":
            self.fictive_species = []
            for spec in gas_reduced.species_names:
                if spec.endswith("_F"):
                    self.fictive_species.append(spec)

            self.nb_spec_fictive = len(self.fictive_species)


        # Some constants used in the code
        self.W_atoms = np.array([12.011, 1.008, 15.999, 14.007]) # Order: C, H, O, N
   
    def loads_models(self) : 
        self.lists_models = [] 
        for i in range(0,self.nb_clusters) :
            model_params_communs= {
                "folder" : self.folder,
                "nb_units_in_layers_list": self.hidden_layers[i],
                "layers_activation_list" :  self.activation[i],
                "layers_type" : self.layers_type,
                "dataset_path" : self.path,
                "remove_N2" : self.removeN2,
                "log_X" : self.log_X, 
                "log_Y" : self.log_Y,
                "device" : self.device,
                "dt_simu" : self.dt_simu,
                "fuel" : self.fuel,
                "mechanism_type" : self.mechanism_type,
                "hard_constraints_model" : self.hard_constraints_model,
            }
            
            # MK CHose between PYTORCH MODEL or  TENSORFLOW MODEL (exportation has to be done before)

            if self.framwork == "Pytorch" : 
                name = f"/Model_{i}_Pytorch"
                print(name)
                state_dict = torch.load(self.directory+name)

                for key in state_dict :
                    print(key)
                
                # MK THERE IS MANY KEYS PROBLEME BETWEEN ENER AND LOCAL ==> TO BE INVASTIGATED 
                state_dict['model.0.weight'] = state_dict.pop('0.weight')
                state_dict['model.2.weight'] = state_dict.pop('2.weight')
                state_dict['model.4.weight'] = state_dict.pop('4.weight')
                state_dict['model.0.bias'] = state_dict.pop('0.bias')
                state_dict['model.2.bias'] = state_dict.pop('2.bias')
                state_dict['model.4.bias'] = state_dict.pop('4.bias')
                

            if self.framwork == "Tensorflow" :
                name = f"/Model_{i}_Pytorch_Tensorflow"
                print(name)

                with open(f"./ENER/POIDS_TENSORFLOW_CONVERTI/model_weights_cluster_{i}_tensorflow_MARTIN.pkl", 'rb') as f : 
                    dico = pickle.load(f)

                state_dict = dico

                for key in state_dict : 
                    state_dict[key] = torch.tensor(state_dict[key].T)
            
    
            model = MLPModel(model_params_communs,i)

            model.load_state_dict(state_dict,strict = True )

            # Check if 1: load has been succefuly loaded 2 : precision of the weight between Pt and Pt or Tf and PT 3 : different from save() ==> Model_testing 
            model.save_testing(self.framwork)
            
            print(f"model_{i} = ")
                     
            

            # FOR WEIGHT INTO .CSV 
            #if self.framwork == "Tensorflow" : 
            #    for a,b in model.named_parameters() : 
            #        pd.DataFrame(b.cpu().detach().numpy()).to_csv(f"./Poids_Model_Tensorflow_converti/Model_{i}_{a}.csv")
                

            self.lists_models.append(model)

        print(self.lists_models)

    def loads_scalers(self) : 
        self.Xscaler_list = []
        self.Yscaler_list = []

        for i in range(self.nb_clusters):
            Xscaler = joblib.load(self.directory+f"/my_model_cluster_{i}" + f"/Xscaler_cluster{i}.pkl")
            Yscaler = joblib.load(self.directory+f"/my_model_cluster_{i}" + f"/Yscaler_cluster{i}.pkl") 
            #Xscaler = joblib.load(self.directory+f"/tf_pt" + f"/Xscaler_cluster{i}.pkl")
            #Yscaler = joblib.load(self.directory+f"/tf_pt" + f"/Yscaler_cluster{i}.pkl")
            
            self.Xscaler_list.append(Xscaler)
            self.Yscaler_list.append(Yscaler)

    def load_clustering_algorithm(self) : 

        if self.clustering_method=="kmeans":
            with open(self.directory + "/kmeans_model.pkl", "rb") as f:
                self.kmeans = pickle.load(f)
            self.kmeans_scaler = joblib.load(self.directory + "/Xscaler_kmeans.pkl")

        elif self.clustering_method=="progvar":
            with open(self.directory + "/c_bounds.pkl", "rb") as input_file:
                self.c_bounds = pickle.load(input_file)

        pass 

    def train_all_models(self) :
        
        
        self.models=[]
        
        for icluster in range(0,self.nb_clusters) :
            model_params_communs= {
                "folder" : self.folder,
                "nb_units_in_layers_list": self.hidden_layers[icluster],
                "layers_activation_list" :  self.activation[icluster],
                "layers_type" : self.layers_type,
                "dataset_path" : self.path,
                "remove_N2" : self.removeN2,
                "log_X" : self.log_X, 
                "log_Y" : self.log_Y,
                "device" : self.device,
                "dt_simu" : self.dt_simu,
                "fuel" : self.fuel,
                "mechanism_type" : self.mechanism_type,
                "hard_constraints_model" : self.hard_constraints_model,
                "log_X" : self.log_X, 
                "log_Y" : self.log_Y,
            }
            model_params_train = {
                'new_model'  : self.new_model, 
                'batch_size' : self.batch_size , 
                'shuffle' : self.shuffle , 
                'loss_function' : self.loss_function, 
                'epoch' : self.epoch[icluster],
                'optimizer' : self.optimizer, 
                'initial_learning_rate' : self.init_lr, 
                'decay_steps' : self.decay_steps,
                'decay_rate' : self.decay_rate,



            }

            model = MLPModel(model_params_communs, icluster)
            model.init_training(model_params_train)
            model.trainning()
            name = model.save()
            self.models.append(name)

        print(self.models) 
    
    def Test_0D_ignition_Pytorch(self,phi,T0,pressure,dt,nb_ite) :

        #Initialization of ignition delays
        tau_ign_CVODE = 0.0
        tau_ign_ANN = 0.0

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

        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.parse_species_names(self.spec_list_ANN)

        #-------------------- CVODE COMPUTATION---------------------------
        # Remark: we don't use the method we wrote below

        for i in range(nb_ite):
            simtime += dt
            sim.advance(simtime)
            states.append(r.thermo.state, t=simtime*1e3)

            if states.T[-1] > T0+200.0 and tau_ign_CVODE==0.0:
                tau_ign_CVODE = simtime
                
        # Equilibrium temperature
        Teq_ref = states.T[len(states.T)-1]
        
        # Convert in a big array (m,NS+1) for later normalization
        state_ref = states.T.reshape(nb_ite+1, 1)
        
        for spec in self.spec_list_ref:
            Y = states.Y[:,gas.species_index(spec)].reshape(nb_ite+1, 1)
            state_ref = np.concatenate((state_ref, Y), axis=-1)


        #-------------------- ANN SOLVER ---------------------------
            
        # Counting calls
        ann_calls = 0
        cvode_calls = 0

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
        sumYs = np.array([1.0])
        progvar_vect = [0.0]

        # atomic composition of species
        if self.mechanism_type=="reduced":
            molecular_weights = utils.get_molecular_weights(gas_reduced.species_names)
        else:
            molecular_weights = utils.get_molecular_weights(gas.species_names)
        atomic_cons = np.dot(atomic_array,np.reshape(Y_k_ann,-1)/molecular_weights)
        atomic_cons = np.multiply(self.W_atoms, atomic_cons)

        # NEURAL NETWORK COMPUTATION
        i=0
        time = 0.0
        state = np.append(T_ann, Y_k_ann)

        for i in range(nb_ite) :

            print(f"ITERATION {i}\n")
            # Old state
            T_old = state[0]
            Y_old = state[1:]

            # Computing current progress variable
            progvar = self.compute_progvar(state, pressure, self.mechanism_type)

            # Attribute cluster
            if self.nb_clusters>0:
                self.attribute_cluster(state, progvar)
            else:
                self.cluster = 0
            
            print(f'Current point in cluster : {self.cluster}\n')

            T_new, Y_new = self.advance_state_NN(T_old, Y_old, pressure, dt)
           

            # If hybrid, we check conservation and use CVODE if not satisfied
            if self.hybrid:
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
        folder = f'./plots_0D_T0#{T0}_phi#{phi}_pytorch_sensi3'
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        #MK VOIR SI CA MARCHE POUR POUVOIR FAIRE DES COURBE DE COMPARAISON
        np.save(folder +"/all_state_save.dat",state_save,allow_pickle=True)
        
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


        # Ignition delays comparison
        print(f" >> CVODE ignition delay: {tau_ign_CVODE}")
        print(f" >> ANN ignition delay: {tau_ign_ANN}")

    def Test_1D_ignintion_Pytorch(self,phi,T0,pressure,dt,T_threshold=0.0 ) : 
        # CANTERA gas object
        gas = ct.Solution(self.mech)

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

        # Counting calls
        ann_calls = 0
        cvode_calls = 0

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
            if self.nb_clusters>0:
                self.attribute_cluster(state, progvar)
            else:
                self.cluster = 0

            print(f"Current point in cluster: {self.cluster} \n")
            
            
            # advance to t + dt
            if Tt[i_reac] >= T_threshold:
                T_new, Y_new = self.advance_state_NN(Tt[i_reac], Yt_ann[:,i_reac], pressure, dt)

                # If hybrid, we check conservation and use CVODE if not satisfied
                if self.hybrid:
                    cons_criterion = np.abs(np.sum(Y_new,axis=0) - 1.0)
                    if cons_criterion > self.hybrid_tol:
                        # CVODE advance
                        #T_new, Y_new = self.advance_state_CVODE(Tt[i_reac], Yt_ann[:,i_reac], pressure, dt)
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






            
        pass 

    def compute_progvar(self, state,pressure,mechanism_type) : 
        # Equilibrium state in present conditions

        if mechanism_type=="reduced":
            gas = ct.Solution(self.mech)
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
  
    def attribute_cluster(self, state_vector=None, progvar=None):
    
        log_state = state_vector.copy()

        if self.clustering_method=="kmeans":
            
            # We remove N2 if necessary
            if self.removeN2:
                n2_index = self.spec_list_ANN.index("N2")
                log_state = np.delete(log_state, n2_index+1)

            # Transformation
            if self.log_X>0:
                log_state[log_state < self.threshold] = self.threshold
                if self.log_X==1:
                    log_state[1:] = np.log(log_state[1:])
                elif self.log_X==2:
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

    def advance_state_NN(self,T_old,Y_old,pressure,dt): 
         # Boolean for IA computation success (used for hybrid model)
        self.ia_success = True

        # CANTERA gas object
        if self.mechanism_type=="reduced":
            gas = ct.Solution(self.mech)
        else:
            gas = ct.Solution(self.mech)
        
        # Gas object modification
        gas.TPY= T_old, pressure, Y_old
        
        print(f"Y_old = {Y_old}\n")
        print(f"T_old = {T_old}\n")
        # Grouping temperature and mass fractions in a "state" vector
        state_vector = np.append(T_old, Y_old)

        # If N2 is not considered, it needs to be removed from the state_vector
        if self.removeN2:
            n2_index = self.spec_list_ANN.index("N2")
            n2_value = state_vector[n2_index+1]
            state_vector = np.delete(state_vector, n2_index+1)
        
        # NN update for Yk's
        if self.log_X==1:
            state_vector[state_vector<self.threshold] = self.threshold
        elif self.log_X==2:
            state_vector[state_vector<0.0] = 0.0

        log_state = np.zeros(self.nb_species_ANN+1)
        log_state[0] = state_vector[0]
        if self.log_X==1:
            log_state[1:] = np.log(state_vector[1:])

        elif self.log_X==2:
            log_state[1:] = (state_vector[1:]**self.lambda_bct - 1.0)/self.lambda_bct
        else:
            log_state[1:] = state_vector[1:]

        # input of NN
        log_state = log_state.reshape(1, -1)

        

        NN_input = self.Xscaler_list[self.cluster].transform(log_state)
        
        #NN_input  = torch.tensor([[3.02697274, -0.99733966 ,2.18745969,-1.11672509 ,2.43641303,-1.13591278,-7.67748582 ,-9.12762476, -5.59838819]],dtype=torch.double).to(self.device)
        
        print(f"NN_input 1 : {NN_input}\n")
        #Change to tensor 
        NN_input = torch.tensor(NN_input,dtype=torch.float64).to(self.device)

        # New state predicted by ANN
        
        #print(f"NN_input 2 : {NN_input}\n")
        state_new = self.lists_models[self.cluster].forward_toto(NN_input)
	
        print(f"state_new = {state_new}\n")

        # Getting Y and scaling
        Y_new = self.Yscaler_list[self.cluster].inverse_transform((state_new.cpu()).detach().numpy())
        print(f"Y_new = {Y_new}\n")        

        # Log transform of species
        if self.log_Y>0:
            if self.output_omegas:
                log_state_updated = log_state[0,1:] + Y_new
                if self.log_Y==1:
                    Y_new = np.exp(log_state_updated)
                elif self.log_Y==2:
                    Y_new = (self.lambda_bct*log_state_updated+1.0)**(1./self.lambda_bct)
            else:
                if self.log_Y==1:
                    Y_new = np.exp(Y_new)
                elif self.log_Y==2:
                    Y_new = (self.lambda_bct*Y_new+1.0)**(1./self.lambda_bct)
         

        # If reaction rate outputs from network
        if self.output_omegas and self.log_Y==0:
            Y_new += state_vector[1:]   # Remark: state vector already contains information about N2 removal

        # Adding back N2 before computing temperature
        if self.removeN2:
            Y_new = np.insert(Y_new, n2_index, n2_value)
    
        # Sum of Yk before renormalization (used for analysis)
        self.sum_Yk_before_renorm = Y_new.sum()

        # Hybrid model: checking if we satisfy the threshold
        if self.hybrid:
            cons_criterion = np.abs(np.sum(Y_new,axis=0) - 1.0)
            if cons_criterion > self.hybrid_tol:
                self.ia_success = False

        # Enforcing element conservation
        if self.yk:
            Y_new = self.enforce_elements_balance(gas, Y_old, Y_new)

                
        # Deducing T from energy conservation
        T_new = state_vector[0] - (1/gas.cp)*np.sum(gas.partial_molar_enthalpies/gas.molecular_weights*(Y_new-Y_old))
        print(f"T_new = {T_new}\n") 
        # Reshaping mass fraction vector
        Y_new = Y_new.reshape(Y_new.shape[0],1)


        return T_new, Y_new



# Define the MLP model
class MLPModel(nn.Module):
    
    # Def __init__
    def __init__(self, params : dict, icluster : int )-> None:
        super(MLPModel, self).__init__()


        self.folder = params["folder"]
        self.hidden_layers = params["nb_units_in_layers_list"] 
        self.activation = params["layers_activation_list"] 
        self.layers_type = params["layers_type"]            
        self.path = params["dataset_path"] 
        self.dt_simu = params['dt_simu']
        self.fuel = params['fuel']
        self.mechanism_type = params['mechanism_type']
        self.device = params['device']
        self.removeN2 = params["remove_N2"]
        self.hard_constraints_model=params['hard_constraints_model']
        self.log_X = params['log_X']
        self.log_Y = params['log_Y']

        self.icluster = icluster        
        self.directory = "./" + self.folder 

        input_size = self.get_input_size()
        output_size =self.get_output_size()
        

        
            
        layers = []

        # Add input layer
        layers.append(nn.Linear(input_size, self.hidden_layers[0],dtype=torch.float64))
        layers.append(self.get_activation(self.activation[0]))

        # Add hidden layers

        for i in range(1, len(self.hidden_layers)):
            if self.layers_type == "dense":
                layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i],dtype=torch.float64))

            elif self.layers_type == "resnet":

                layers.append(ResidualBlock(self.hidden_layers[i - 1], self.hidden_layers[i]))
        
            layers.append(self.get_activation(self.activation [i]))

        # Add output layer
            
        layers.append(nn.Linear(self.hidden_layers[-1], output_size,dtype=torch.float64))

        # Combine all layers

        self.model = nn.Sequential(*layers).to(self.device)        
        
    # Def for training 
    def init_training(self,params :dict) -> None : 


        self.new_model = params["new_model"]
        self.batch_size = params["batch_size"]        
        self.shuffle = params["shuffle"]  
        self.loss_function = params["loss_function"]      
        self.epoch = params["epoch"]       
        self.optimizer = params["optimizer"]
        self.init_lr = params["initial_learning_rate"]
        self.decay_steps = params["decay_steps"]
        self.decay_rate = params["decay_rate"]
        

        
        self.time_stamp = int(time.time())


        # Get Data 
        X_train, X_val , Y_train, Y_val = self.get_data(self.path, self.icluster)
            
        # Remove N2 
        if self.removeN2 == True :
            X_train = X_train.drop("N2_X", axis=1)
            X_val = X_val.drop("N2_X", axis=1)

            Y_train = Y_train.drop("N2_Y", axis=1)
            Y_val = Y_val.drop("N2_Y", axis=1)
        
        self.Y_cols = Y_train.columns
        self.X_cols = X_train.columns
        
        # Nomalized and Tensor 
            
        X_train, X_val = self.format_X_data_training(X_train, X_val)
        Y_train, Y_val = self.format_Y_data_training(Y_train, Y_val)

        # Data Loader parameters 

        dataloader_params= {"batch_size" : self.batch_size,
                            "shuffle" : self.shuffle,
                            "num_workers" : 0, 
                            }

        # list lock 

        data_train = list(zip(X_train,Y_train))
        data_val = list(zip(X_val,Y_val))

        # init of the data loader 

        self.data_train = DataLoader(data_train, **dataloader_params)  
        self.data_val= DataLoader(data_val,**dataloader_params)


    #get input size from X.csv
    def get_input_size(self) -> int : 
        with open(self.path + f"/dtb/cluster{self.icluster}/X_train.csv", 'r') as f : 
            colums = f.readline().split(',')
        
        input_size = len(colums)
        if self.removeN2 and 'N2_X' in colums :
            input_size-=1

        return input_size

    #get ouput size from Y.csv 
    def get_output_size(self) -> int: 

        with open(self.path + f"/dtb/cluster{self.icluster}/Y_train.csv", 'r') as f : 
            colums = f.readline().split(',')
        
        output_size = len(colums)
        if self.removeN2 and 'N2_Y' in colums :
            output_size-=1

        return output_size

    # Forward 
    def forward_toto(self, x):
        return self.model(x)
    
    # Type of cell activation 
    def get_activation(self, activation):
        if activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
    
    # Optimizer  
    def get_optimizer(self) : 
        
        if self.optimizer== 'adam':

            return torch.optim.Adam(self.parameters(),lr = self.init_lr)
        else:
            raise ValueError(f"Unsupported optmizer function: {self.optimizer}")

    # Scheduler 
    def get_scheduler(self,optimizer) : 
        
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_steps, gamma=self.decay_rate)
    
    # Loss function
    def get_loss_function(self) : 
        if self.loss_function == 'mean_squared_error':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")       

    # Get the ML data 
    def get_data(self, path : str, i_cluster : int)-> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        X_train = pd.read_csv(filepath_or_buffer= self.path + f"/cluster{i_cluster}/X_train.csv")
        Y_train = pd.read_csv(filepath_or_buffer= self.path + f"/cluster{i_cluster}/Y_train.csv")
        
        X_val = pd.read_csv(filepath_or_buffer= self.path + f"/cluster{i_cluster}/X_val.csv")
        Y_val = pd.read_csv(filepath_or_buffer= self.path + f"/cluster{i_cluster}/Y_val.csv")


        return X_train, X_val, Y_train, Y_val

    #Scaling the X Data
    def format_X_data_training(self, X_train :pd.DataFrame, X_val : pd.DataFrame) -> tuple [torch.tensor, torch.tensor]: 

        # NORMALIZING X
        Xscaler = StandardScaler()
        # Fit scaler
        Xscaler.fit(X_train)
        
        # Transform data (remark: automatically transform to numpy array)
        X_train = Xscaler.transform(X_train)
        
        X_val = Xscaler.transform(X_val)

        X_train = torch.tensor(X_train,dtype=torch.float64).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float64).to(self.device)
        joblib.dump(Xscaler, self.directory+ f'/my_model_cluster_{self.icluster}/Xscaler_cluster{self.icluster}.pkl')
        np.savetxt(self.directory + f'/my_model_cluster_{self.icluster}/norm_param_X_cluster{self.icluster}.dat', np.vstack([Xscaler.mean_, Xscaler.var_]).T)
        

        return X_train, X_val 
        
    # Scaling the Y data 
    def format_Y_data_training(self,Y_train : pd.DataFrame, Y_val : pd.DataFrame) -> tuple [torch.tensor, torch.tensor] :
        # NORMALIZING Y
        # Choose scaler
        Yscaler = StandardScaler()

        # Fit scaler
        Yscaler.fit(Y_train)
    
        # Transform data (remark: automatically transform to numpy array)
        Y_train = Yscaler.transform(Y_train)
        Y_val = Yscaler.transform(Y_val)

        Y_train = torch.tensor(Y_train,dtype=torch.float64).to(self.device)
        Y_val = torch.tensor(Y_val, dtype=torch.float64).to(self.device)

        joblib.dump(Yscaler, self.directory+ f'/my_model_cluster_{self.icluster}/Yscaler_cluster{self.icluster}.pkl')
        np.savetxt(self.directory + f'/my_model_cluster_{self.icluster}/norm_param_Y_cluster{self.icluster}.dat', np.vstack([Yscaler.mean_, Yscaler.var_]).T)
        
        

        return Y_train, Y_val
    
    # plotting
    def plot(self,loss_train,loss_val,min_train,max_train,min_val,max_val,icluster) :
        
        plt.title('Cluster_'+str(icluster))
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale('log')
        plt.plot(loss_train,"b")
        plt.plot(loss_val,"r")
        #plt.fill_between(range(len(loss_train)),min_train,max_train, alpha =.3,color="b")
        plt.legend(['train','validation'])
        plt.show()
        plt.savefig( self.directory + f"/training/training_curves/loss_cluster{self.icluster}.png") 

    # save model 
    def save(self) -> str:
        name = f"Model_{self.icluster}_Pytorch"
        torch.save(self.model.state_dict(), f"{self.directory}/{name}")
        return name
    
    def save_testing(self,framework:str) -> str:
        name = f"Model_{self.icluster}_Pytorch_framework_{framework}_testing"
        torch.save(self.model.state_dict(), f"{self.directory}/{name}")
        return name
 
    # Load model 
    def load(self):
        name = f"Model_{self.icluster}_Pytorch"
        self.load_state_dict(torch.load(f"{self.directory}/{name}"))
        self.eval()
        return self



    # Trainning NN 
    def trainning(self) :
        
        
        
        # Calcul temps 
        start_time = time.perf_counter()

        train_losses_global = []
        min_train_losses_global = []
        max_train_losses_global = []

        val_losses_global = []
        min_val_losses_global = []
        max_val_losses_global = []

        history = { "epoch" : [], "loss_train" : [], "loss_val" :[]} 

        

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        loss_function = self.get_loss_function()
        
        


        for epoch in range(self.epoch) :

            train_losses_epoch = []
            val_losses_epoch = []

            #Stop if no variation in loss 
            #if len(train_losses_global)>3 and round(max(train_losses_global[-3:]), 3) == round(min(train_losses_global[-3:]), 3):
            #    print("Stopped by stable loss")
            #    break

            #Training loop 

            for _X,_Y in self.data_train :

                #data loading
                X = _X 
                Y = _Y
                #if self.use_cuda :
                #    X = X.to(self.device)
                #    Y = Y.to(self.device)
                    

                #prediction 
                
                Y_pred = self(X)

                # back propagation 
                optimizer.zero_grad()
                loss = loss_function(Y, Y_pred)
                loss.backward()
                optimizer.step()

                #Display and metrics 

                train_losses_epoch.append(float(loss))
                print('Epoch {:>4} of {}, Train Loss: {:5.2f}, Val Loss: {:5.2f}, Elapsed time: {:.2f}'.format(epoch+1,self.epoch , train_losses_epoch[-1],0, time.perf_counter()-start_time),end="\r",flush=True)
            
            

            train_losses_global.append(np.mean(train_losses_epoch))
            min_train_losses_global.append(np.min(train_losses_epoch))
            max_train_losses_global.append(np.max(train_losses_epoch))

            print('Epoch {:>4} of {}, Train Loss: {:5.2f}, Val loss: {:5.2f}, Elapsed time: {:.2f}'.format(epoch+1, self.epoch, train_losses_global[-1],0, time.perf_counter()-start_time),end="\r",flush=True)


            # Validation Loop 
            for _X_Val,_Y_Val in self.data_val :
                #data loading
                X_Val = _X_Val 
                Y_Val = _Y_Val

                #if self.use_cuda :
                #    X_Val = X_Val.to(self.device)
                #    Y_Val = Y_Val.to(self.device)

                #prediction 
                    
                Y_pred = self(X_Val)               

                #Display and metrics 
                loss = loss_function(Y_Val, Y_pred)
                val_losses_epoch.append(float(loss))
                print('Epoch {:>4} of {}, Train Loss: {:5.2f}, Val Loss: {:5.2f}, Elapsed time: {:.2f}'.format(epoch+1,self.epoch , train_losses_epoch[-1],val_losses_epoch[-1], time.perf_counter()-start_time),end="\r",flush=True)
            
            val_losses_global.append(np.mean(val_losses_epoch))
            min_val_losses_global.append(np.min(val_losses_epoch))
            max_val_losses_global.append(np.max(val_losses_epoch))

            history["epoch"].append(epoch+1)
            history["loss_train"].append(train_losses_global[-1])
            history["loss_val"].append(val_losses_global[-1])
            
            print('Epoch {:>4} of {}, Train Loss: {:5.2f}, Val loss: {:5.2f}, Elapsed time: {:.2f}'.format(epoch+1, self.epoch, train_losses_global[-1],val_losses_global[-1], time.perf_counter()-start_time),end="\n",flush=True)

            scheduler.step()
            
        self.plot(train_losses_global,val_losses_global,min_train_losses_global,max_train_losses_global, min_val_losses_global, max_val_losses_global,self.icluster)
        print(f'Temps Cluster {self.icluster} = {time.perf_counter()-start_time}')
        #Save architecture 

        #Save losses 
        pd.DataFrame(history).to_csv(self.directory+f'/training/training_curves/histroy_model_{self.icluster}.csv')
        
        #Save Model  : weights and bias 
        torch.save(MLPModel, self.directory +'/train_model_cluster_'+str(self.icluster)+'.pt')
        

        
        ### PREDICTION ERROR : UNSCALE INTO DTB CSV 
        Xscaler, Yscaler = self.load_scaler_error()

        if self.log_X > 0 : 
            X_val_unscaled = Xscaler.inverse_transform((X_Val.cpu()).detach().numpy())
            if self.log_X == 1 : #LOG 
                X_val_unscaled = np.float128(X_val_unscaled)
                X_val_unscaled = np.exp(X_val_unscaled)
            elif self.log_X == 2 : #BCT 
                pass 


        
        if self.log_Y > 0 :
            Y_val_unscaled = Yscaler.inverse_transform((Y_Val.cpu()).detach().numpy())
            Y_pred_unscaled = Yscaler.inverse_transform((Y_pred.cpu()).detach().numpy())
            if self.log_Y == 1 : # Log
                Y_val_unscaled = np.float128(Y_val_unscaled)
                Y_pred_unscaled = np.float128(Y_pred_unscaled)
                Y_val_unscaled = np.exp(Y_val_unscaled)
                Y_pred_unscaled = np.exp(Y_pred_unscaled)
            elif self.log_Y == 2 : #BTC 
                pass 

        errors_pred_val = np.absolute(Y_pred_unscaled - Y_val_unscaled)
        data_array = np.concatenate((X_val_unscaled, Y_val_unscaled, Y_pred_unscaled, errors_pred_val), axis = 1)
        error_cols = [str(col) + '_err' for col in self.Y_cols]
        Y_pred_cols = [str(col) + '_pred' for col in self.Y_cols]
        columns = list(self.X_cols) + list(self.Y_cols) + Y_pred_cols + error_cols
        dtb_val = pd.DataFrame(data_array, columns = columns)
        dtb_val.to_csv( self.directory  + f"/evaluation/validation_predictions_cluster{self.icluster}.csv" ,sep=';',index=False)


        return train_losses_global, val_losses_global
    
    def load_scaler_error(self) : 

        Xscaler = joblib.load(self.directory+f"/my_model_cluster_{self.icluster}" + f"/Xscaler_cluster{self.icluster}.pkl")
        Yscaler = joblib.load(self.directory+f"/my_model_cluster_{self.icluster}" + f"/Yscaler_cluster{self.icluster}.pkl")

        return Xscaler, Yscaler 

    

    
       
# Define a simple residual block for the resnet architecture
        
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        
    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x += identity
        x = torch.relu(x)
        return x

