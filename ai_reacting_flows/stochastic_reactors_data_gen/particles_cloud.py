import sys
import os
from time import perf_counter
import itertools

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import h5py
import seaborn as sns
import cantera as ct

from ai_reacting_flows.stochastic_reactors_data_gen.particle import Particle
import ai_reacting_flows.tools.utilities as utils
from ai_reacting_flows.tools.utilities import PRINT

from ai_reacting_flows.stochastic_reactors_data_gen.ann_model import ModelANN

matplotlib.use('Agg')
sns.set()

class ParticlesCloud(object):
    
# =============================================================================
#   INITIALIZATION
# =============================================================================

    # Initialize particle cloud
    def __init__(self, data_gen_parameters, comm):

        # MPI communicator
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
            
        # =====================================================================
        #     READING GENERAL INPUT DATA    
        # =====================================================================
        
        # Folder where result are stored
        self.run_folder = os.getcwd()
        self.results_folder = f"{self.run_folder:s}/STOCH_DTB_" + data_gen_parameters["results_folder_suffix"]
        self.solution_file = data_gen_parameters["dtb_file"]
        
        # Methodology to solve chemistry: Cantera or NN
        self.ML_inference_flag = data_gen_parameters["ML_inference_flag"]
        if self.ML_inference_flag:
            self.ML_models = data_gen_parameters["ML_models"]
            self.pg_thresholds = data_gen_parameters["pg_thresholds"]
        
        # Initialize time and general parameters
        self.time_max = data_gen_parameters["time_max"]
        self.time = 0.0
        self.iteration = 0
        self.dt = data_gen_parameters["time_step"]
        self.calc_mean_traj = data_gen_parameters["calc_mean_traj"]
        
        # Statistics convergence status
        self.stats_converged = False

        # Mechanism file
        self.mech_file  = data_gen_parameters["mech_file"]
        self.gas = ct.Solution(self.mech_file)

        # Dabatase building
        self.build_ml_dtb = data_gen_parameters["build_ml_dtb"]

        # =====================================================================
        #     READING INLET DATA    
        # =====================================================================
        
        # Reading excel file with inlet data
        inlets_file_xl = pd.ExcelFile(data_gen_parameters["inlets_file"])
        df = inlets_file_xl.parse("Sheet1")
        
        # Number of inlets
        self.nb_inlets = df.shape[0]
        
        # Species names
        self.species_names = df.columns[4:].to_list()

        # Get the data as a numpy array (we discard index column)
        data = df.to_numpy()[:,1:]
        
        # Total number of particles in simulation
        self.nb_parts_tot = data[:,0].sum()
        
        # Dict with number of particles per inlet
        self.nb_parts_per_inlet = dict.fromkeys(range(1,self.nb_inlets+1))
        for inlet in self.nb_parts_per_inlet:
            i = inlet-1 # because of pythonic convention
            self.nb_parts_per_inlet[inlet] = data[i,0]
        
        # Composition for each inlet
        self.state_per_inlet = dict.fromkeys(range(1,self.nb_inlets+1))
        for inlet in self.state_per_inlet:
            i = inlet-1
            self.state_per_inlet[inlet] = np.array(data[i,1:], dtype="float")
        
        # Number of states variables (using first inlet as it is necessarily present)
        self.nb_state_vars = len(self.state_per_inlet[1])


        # =====================================================================
        #     GENERATION OF PARTICLES LIST 
        # =====================================================================
        
        # Populating particles list
        self.particles_list = []
        num_part = 0
        for i in range(1,self.nb_inlets+1):  # Loop on inlet
            for j in range(self.nb_parts_per_inlet[i]):  # creating Ni particles for inlet i
                state_ini = self.state_per_inlet[i]
                particle_current = Particle(state_ini, self.species_names, i, num_part, data_gen_parameters, self)
                self.particles_list.append(particle_current)
                num_part += 1
        
        # =====================================================================
        #     MIXING MODEL INITIALIZATION 
        # =====================================================================
                
        # Mixing model
        self.mixing_model = data_gen_parameters["mixing_model"]
        self.mixing_time = data_gen_parameters["mixing_time"]
        if self.mixing_model == "EMST":
            import ai_reacting_flows.stochastic_reactors_data_gen.EMST.emst_mixing as emst_mixing
            self.emst = emst_mixing
        
        # Chemistry temperature threshold
        self.T_threshold = data_gen_parameters["T_threshold"]

        # CURL with differential diffusion
        if self.mixing_model=="CURL_MODIFIED_DD":
            
            # Ren DD model needs mixing times to estimate number of particles pairs mixing
            # We arbitrarily take particle 0, assuming Lewis numbers do not vary a lot
            part = self.particles_list[0]
            part.compute_lewis_numbers(self)

            self.tau_k = self.mixing_time * part.Le_k
            self.tau_min = np.min(self.tau_k)  #* np.min(self.Le_k) #np.min(self.tau_k)
            # self.tau_min =  self.mixing_time

            # List of particle pairs
            self.particle_pairs = list(itertools.product(np.arange(self.nb_parts_tot), np.arange(self.nb_parts_tot)))

            # Removing pairs of type (i,i)
            for pair in self.particle_pairs:
                if pair[0]==pair[1]:
                    self.particle_pairs.remove(pair)

        else:
            self.tau_min = self.mixing_time
        
        # Number of particles pair to use for CURL model
        if self.mixing_model in ["CURL","CURL_MODIFIED","CURL_MODIFIED_DD"]:
            # Estimating number of pairs (float)
            # if self.mixing_model=="CURL_MODIFIED_DD":  # 1.5 factor used in Ren et al. paper
            #     N_pairs = 1.5 * self.nb_parts_tot * self.dt/self.tau_min
            # else:
            #     N_pairs = self.nb_parts_tot * self.dt/self.tau_min
            N_pairs = 1.5 * self.nb_parts_tot * self.dt/self.tau_min # 1.5 factor used in Ren et al. paper
            
            # Number of mixed particle
            self.Npairs_curl = int(round(N_pairs))

        if self.mixing_model!="CURL_MODIFIED_DD" and self.mixing_model!="EMST":  
            # Do or read mixing
            read_mixing = data_gen_parameters["read_mixing"]
            do_CURL_rate = True if "CURL_MODIFIED" in self.mixing_model else False
            if read_mixing:
                with open(f"{self.run_folder:s}/pairs_list.pkl", "rb") as f:
                    self.pairs_list = pickle.load(f)
                if do_CURL_rate:
                    with open(f"{self.run_folder:s}/CURL_rate_list.pkl", "rb") as f:
                        self.CURL_rate_list = pickle.load(f)
            else:
                n_ite = int(self.time_max / self.dt) + 1
                self.pairs_list = []
                self.CURL_rate_list = []
                for i in range(n_ite): # DAK: vectorize pair generation ?
                    self.pairs_list.append(utils.sample_comb2((self.nb_parts_tot,self.nb_parts_tot), self.Npairs_curl))
                    if do_CURL_rate:
                        self.CURL_rate_list.append(np.random.random())
                
                with open(self.results_folder + "/pairs_list.pkl", 'wb') as f:
                    pickle.dump(self.pairs_list, f)
                if do_CURL_rate:
                    with open(self.results_folder + "/CURL_rate_list.pkl", 'wb') as f:
                        pickle.dump(self.CURL_rate_list, f)
            
            
            
            # If number is very small, N_pairs_curl might be 0, in this case we set it
            # to 1 and perform diffusion every round(1/N_pairs) time steps
            if self.Npairs_curl==0:
                self.Npairs_curl=1
                self.diffusion_freq = int(round(1/N_pairs))
            else:
                self.diffusion_freq = 1
            
            
        # Initializing particles age in EMST model
        if data_gen_parameters["mixing_model"]=="EMST":
            self._init_EMST()
            
        
        # =====================================================================
        #     ATOMIC CONSERVATION ANALYSIS
        # =====================================================================
        
        # Analysis of atomic content
        # Atom molecular weights (Order: C, H, O, N)
        # Matrix with number of each atom in species (order of rows: C, H, O, N)
        atomic_array = utils.parse_species_names(self.species_names)
        
        # Atomic mass per elements (dictionary)
        mass_per_atom_array = np.array([12.011, 1.008, 15.999, 14.007])  # Having the array is also convenient
        
        # Atomic mass per species (numpy array)
        mol_weights = utils.get_molecular_weights(self.species_names)
        
        # Matrix for computing atomic conservation
        self.A_atomic = np.copy(atomic_array)
        for j in range(4):
            self.A_atomic[j,:] *=  mass_per_atom_array[j]
        for k in range(len(self.species_names)):
            self.A_atomic[:,k] /=  mol_weights[k]
            
        for part in self.particles_list:
            part.atomic_mass_fractions = np.dot(self.A_atomic, part.Y)
        
        
        # =====================================================================
        #     INPUT/OUTPUT INITIALIZATION
        # =====================================================================

        # Columns of data array with solution and post-processing variables
        self.cols_all_states = ['Temperature'] + ['Pressure'] + self.species_names + ['Mix_frac'] + ['Equiv_ratio'] + ['Prog_var'] + ['HRR'] +  ['Time'] + ['Particle_number'] + ['Inlet_number'] + ['Y_C', 'Y_H', 'Y_O', 'Y_N'] + ['Mass']
        
        # Initialize mean trajectories
        if self.calc_mean_traj:
            self._compute_means()
            # Only rank 0 deals with mean trajectories as it is a post-processing
            if self.rank==0: 
                self.mean_trajectories = np.zeros((1, self.nb_inlets, self.nb_state_vars+4))
                self.mean_trajectories[0,:,:] = self.mean_states
        
        
        # Initialize vector for tracking mean/std of temperature 
        self.mean_T_vect = []
        self.stdev_T_vect= []
        
        # Physical time per iterations
        self.timings = pd.DataFrame(data=[], columns=["Total", "Reaction", "Diffusion", "MeanTraj", "Writing", "Plotting"])
        
        # Start of simulation printing
        if self.rank == 0:
            self.print_initial()


        # =====================================================================
        #     CHECKING INPUTS   
        # =====================================================================

        # Compatibility between inlet file and mech file species names
        if self.rank == 0:
            self.check_species()
    
# =============================================================================
#   TIME MARCHING FUNCTIONS
# =============================================================================
                
    # Advance solution to next time step     
    #TODO: dt inside and outside class, maybe there is a cleaner way
    def advance_time(self, dt):
        
        # Initial time
        t0 = perf_counter()

        # Write current iteration solution
        if self.rank==0:
            self._write_solution()
        
        # Store state before chemistry to build X.csv
        if self.build_ml_dtb and self.rank==0:
            self._update_dtb_states("X")

        t1 = perf_counter()

        # Advance reaction rate
        self._apply_reactions(dt)
        t2 = perf_counter()

        # Updating particle associated post-processing variables
        self._update_vars()

        # Store state after chemistry to build Y.csv
        if self.build_ml_dtb and self.rank==0:
            self._update_dtb_states("Y")
        
        # Advance molecular diffusion
        # Diffusion is applied by rank 0 and then redistributed to make sure every processor has the same states
        if self.rank==0:
            self._apply_diffusion()

        self.comm.bcast(self.particles_list, root=0)
        
        t3 = perf_counter()
        
        # Updating particle associated post-processing variables
        self._update_vars()
        
        # Compute mean trajectories
        if self.calc_mean_traj:
            self._compute_means()
            # Only rank 0 needs to know the means as it is post-processing
            if self.rank==0:
                self._append_means_to_trajectory()
        t4 = perf_counter()

        # TODO REMOVE
        t5 = perf_counter()
        
        # Additional statistics on particles
        self.calc_statistics()
        
        t6 = perf_counter()
        
        # Store computation timings
        last_loc = self.timings.shape[0]
        self.timings.loc[last_loc] = np.array([t6-t0, t2-t1, t3-t2, t4-t3, t1-t0, t6-t5])
        
        # Print
        if self.rank==0:
            self.print_iteration()
        
        # Check termination of computation
        self.check_termination()

        # Update time and iteration
        self.time += dt
        self.iteration += 1
        
    # Apply reaction rate to particles
    def _apply_reactions(self, dt):

        # We divide the list of particles into chunks to be treated by each processor
        lists_particles = np.array_split(np.array(self.particles_list), self.size)

        # We verify size of list to avoid errors
        assert len(lists_particles) == self.size

        # We scatter the chunks on each processors. Remark: rank=root also takes a chunk.
        lists_particles = self.comm.scatter(lists_particles, root=0)

        # We perform chemical reactions -> each processor computes on its chunks
        for part in lists_particles:
            if self.ML_inference_flag:
                part.react_NN_wrapper(self)
            else:
                part.react(dt, self)

        # Allgather regroups the chunks and give the whole list to all the processors (<=> gather + broadcast)
        result = self.comm.allgather(lists_particles)

        # As we have a list of lists of particles we need to concatenate
        self.particles_list = list(itertools.chain.from_iterable(result))

    # Apply diffusion to particles
    def _apply_diffusion(self):
        
        if self.mixing_model=="CURL" or self.mixing_model=="CURL_MODIFIED":

            if self.iteration%self.diffusion_freq==0:
                self._mix_curl()

        elif self.mixing_model=="CURL_MODIFIED_DD":

            if self.iteration%self.diffusion_freq==0:
                self._mix_curl_dd()

        elif self.mixing_model=="EMST":
            self._mix_EMST()

# =============================================================================
#   MIXING MODELING
# =============================================================================
    
    # CURL model: unity lewis numbers model => we work directly with mass fractions
    def _mix_curl(self):

        # Randomly select mixing pairs
        pairs = self.pairs_list[self.iteration]
        
        # Carrying out diffusion
        for pair in pairs:
            
            # Finding particles
            part_1 = self.particles_list[pair[0]]
            part_2 = self.particles_list[pair[1]]
            # for part in self.particles_list:
            #     if part.num_part==pair[0]:
            #         part_1 = part
            #     if part.num_part==pair[1]:
            #         part_2 = part

            # Initial values
            Y1_ini = part_1.Y.copy()
            h1_ini = part_1.hs
            Y2_ini = part_2.Y.copy()
            h2_ini = part_2.hs
                    
            if self.mixing_model=="CURL":
                
                # Update particles states
                part_1.Y = 0.5 * (Y2_ini + Y1_ini)
                part_1.hs = 0.5 * (h2_ini + h1_ini)
                #
                part_2.Y = 0.5 * (Y2_ini + Y1_ini)
                part_2.hs = 0.5 * (h2_ini + h1_ini)
                
            elif self.mixing_model=="CURL_MODIFIED":
                    
                # Random value between 0 and 1
                a = self.CURL_rate_list[self.iteration]
                        
                # Update particles states
                part_1.Y += 0.5 * a * (Y2_ini - Y1_ini)
                part_1.hs += 0.5 * a * (h2_ini - h1_ini)
                #
                part_2.Y += 0.5 * a * (Y1_ini - Y2_ini)
                part_2.hs += 0.5 * a * (h1_ini - h2_ini)
            
            
            part_1.state[0] = part_1.hs
            part_1.state[2:] = part_1.Y
            part_1.update_ThermoStates(self)
            
            part_2.state[0] = part_2.hs
            part_2.state[2:] = part_2.Y
            part_2.update_ThermoStates(self)

    # CURL model: with differential diffusion => we work with species masses
    def _mix_curl_dd(self):

        masses_of_pairs = np.empty(len(self.particle_pairs))
        for i, pair in enumerate(self.particle_pairs):
            masses_of_pairs[i] = self.particles_list[pair[0]].mass + self.particles_list[pair[1]].mass
        prob_of_pairs = masses_of_pairs / masses_of_pairs.sum()

        # Randomly select mixing pairs
        pairs_index = np.random.choice(len(self.particle_pairs), size=self.Npairs_curl, p=prob_of_pairs,replace=False)
        
        # Carrying out diffusion
        for i in pairs_index:
            
            # Particles in the pair
            part_1 = self.particles_list[self.particle_pairs[i][0]]
            part_2 = self.particles_list[self.particle_pairs[i][1]]

            # Compute Lewis numbers of mixing particles only (saves costs)
            part_1.compute_lewis_numbers(self)
            part_2.compute_lewis_numbers(self)
            Le_k_m = 0.5*(part_1.Le_k + part_2.Le_k)   # To conserve mass we need one Lewis, we take the average as a test

            # Updating mixing times using current particles' Lewis numbers
            self.tau_k = self.mixing_time * Le_k_m
            self.tau_min = np.min(self.tau_k)
            
            # Random value between 0 and min(Le_k) -> to avoid overshooting
            # min_Le_1 = np.min(part_1.Le_k)
            # min_Le_2 = np.min(part_2.Le_k)
            # alpha = np.min(Le_k_m) * np.random.random()   # SIMPLE MODEL
            alpha = np.random.random()   # REN MODEL

            # Recording initial enthalpy and mass of particle 1 (which will be updated first)
            mass_k_1_ini = part_1.mass_k.copy()
            Hs_1_ini = part_1.Hs

            # REN ET AL MODEL
            # Variables needed to update particles
            Y_12 = (part_1.mass_k + part_2.mass_k)/(part_1.mass + part_2.mass)
            hs_12 = (part_1.Hs + part_2.Hs)/(part_1.mass + part_2.mass)
            theta_k = (3.0-(9.0-8.0*(self.tau_min/self.tau_k))**0.5)/2
            theta_hs = (3.0-(9.0-8.0*(self.tau_min/self.mixing_time))**0.5)/2

            # Updating mass and total enthalpy of particle 1
            part_1.mass_k = (1.0-alpha*theta_k)*part_1.mass_k + alpha*theta_k*part_1.mass*Y_12
            part_1.Hs = (1.0-alpha*theta_hs)*part_1.Hs + alpha*theta_hs*part_1.mass*hs_12

            flux_sum = - np.sum(part_1.mass_k - mass_k_1_ini)
            flux = (part_1.mass_k - mass_k_1_ini) + flux_sum * part_1.Y
            part_1.Y += flux
            part_2.Y -= flux
            
            part_2.mass_k = part_2.mass_k - (part_1.mass_k - mass_k_1_ini)
            part_2.Hs = part_2.Hs - (part_1.Hs - Hs_1_ini)

            # # SIMPLE MODEL
            # # Total enthalpy updated using alpha as a coefficient
            # part_1.Hs += 0.5 * alpha * (part_2.Hs - part_1.Hs)

            # # Species masses updated using alpha weighted by lewis numbers
            # part_1.mass_k += 0.5 * (alpha/Le_k_m) * (part_2.mass_k - part_1.mass_k)

            # # Particle 2 update
            # part_2.Hs += 0.5 * alpha * (Hs_1_ini - part_2.Hs)
            # part_2.mass_k += 0.5 * (alpha/Le_k_m) * (mass_k_1_ini - part_2.mass_k)

            # Dealing with negative masses (we put them in the other particle)
            for j in range(len(part_1.mass_k)):

                if part_1.mass_k[j] < 0.0:
                    part_2.mass_k[j] += part_1.mass_k[j]
                    part_1.mass_k[j] = 0.0

                elif part_2.mass_k[j] < 0.0:
                    part_1.mass_k[j] += part_2.mass_k[j]
                    part_2.mass_k[j] = 0.0


            # Updating variables associated to particles
            for part in [part_1, part_2]:

                # Total masses of species
                part.mass = np.sum(part.mass_k)

                # Mass fractions
                part.Y = part.mass_k / part.mass

                # Specific sensible enthalpies
                part.hs = part.Hs / part.mass
                
                # Main state
                part.state[0] = part.hs
                part.state[2:] = part.Y
                
                # Temperature
                part.update_ThermoStates(self)

    # EMST model: initialization
    def _init_EMST(self):
        
        nparts = len(self.particles_list)
        ncompo = len(self.particles_list[0].state) - 1  # particle 0 arbitrary (pressure not considered)
        
        state = np.zeros(nparts, order='F', dtype='float32')
        wt = np.empty(nparts, order='F', dtype='float32')
        fscale = np.empty(ncompo, order='F', dtype='float32')
        f = np.empty((nparts, ncompo), order='F', dtype='float32')
        for part in self.particles_list:
            i = part.num_part
            wt[i] = 1.0/nparts
            state_part = np.append([part.hs], part.Y)
            f[i, :] = state_part
            
        
        # Scaling factor (to be defined better)
        fscale[0] = 1.0e16
        fscale[1:] = 0.1
        
        # control vars for expert users
        cvars = np.zeros(6, order='F', dtype='float32')
        
        # Normalized time scale
        C_phi = 2.0  # model constant
        omdt = self.dt / (C_phi*self.mixing_time)
        
        # Using EMST routine to initialize age of particles
        status = self.emst.emst(mode=1,f=f,state=state,wt=wt,omdt=omdt,fscale=fscale,cvars=cvars,np=nparts,nc=ncompo)
        
        # Checking EMST error status
        assert status == 0, "EMST mixing failure"
        
        # Dispatching age of particles
        for part in self.particles_list:
            i = part.num_part
            part.age = state[i]
        
    # EMST model: mixing
    def _mix_EMST(self):
        
        nparts = len(self.particles_list)
        ncompo = len(self.particles_list[0].state) - 1  # particle 0 arbitrary (pressure not considered)
        
        state = np.empty(nparts, order='F', dtype='float32')
        wt = np.empty(nparts, order='F', dtype='float32')
        fscale = np.empty(ncompo, order='F', dtype='float32')
        f = np.empty((nparts, ncompo), order='F', dtype='float32')
        for part in self.particles_list:
            i = part.num_part
            state[i] = part.age
            wt[i] = 1.0/nparts
            state_part = np.append([part.hs], part.Y)
            f[i, :] = state_part
            
        
        # Scaling factor (to be defined better)
        fscale[0] = 1.0e16
        fscale[1:] = 0.1
        
        # control vars for expert users
        cvars = np.zeros(6, order='F', dtype='float32')
        
        # Normalized time scale
        C_phi = 2.0  # model constant
        omdt = self.dt / (C_phi*self.mixing_time)
        
        # Using EMST routine to initialize age of particles
        status = self.emst.emst(mode=2,f=f,state=state,wt=wt,omdt=omdt,fscale=fscale,cvars=cvars,np=nparts,nc=ncompo)
        
        # Checking EMST error status
        assert status == 0, "EMST mixing failure" 
        
        # Updating particle associated variables
        for part in self.particles_list:
            
            # Main characteristics
            i = part.num_part
            part.age = state[i]
            part.hs = f[i, 0]
            part.Y = f[i, 1:]
            part.state[0] = part.hs
            part.state[2:] = part.Y
            
            # Temperature
            part.update_ThermoStates(self)

# =============================================================================
#   MISC
# =============================================================================

    def _update_vars(self):

        # We parallelize as the progress variable computation is expensive (indeed, it relies on an "equil" computation)

        # We divide the list of particles into chunks to be treated by each processor
        lists_particles = np.array_split(np.array(self.particles_list), self.size)

        # We scatter the chunks on each processors. Remark: rank=root also takes a chunk.
        lists_particles = self.comm.scatter(lists_particles, root=0)

        # We perform variable updates of particles
        for part in lists_particles:
            part.compute_equiv_ratio()
            part.compute_progress_variable(self)
            part.compute_mixture_fraction()
            part.compute_heat_release_rate(self)

        # Allgather regroups the chunks and give the whole list to all the processors (<=> gather + broadcast)
        result = self.comm.allgather(lists_particles)

        # As we have a list of lists of particles we need to concatenate
        self.particles_list = list(itertools.chain.from_iterable(result))

# =============================================================================
#   MACHINE LEARNING RELATED FUNCTIONS
# =============================================================================        
    
    def _init_ML_modelling(self):
        
        # Initializing ANN model
        self.ann_model = ModelANN(self.ML_model)
        
        # Loading model
        self.ann_model.load_ann_model()

        # Load scalers
        self.ann_model.load_scalers()

# =============================================================================
#   MEAN TRAJECTORIES
# =============================================================================
    
    def _compute_means(self):
        
        # Shape of mean_states
        self.mean_states = np.zeros((1, self.nb_inlets, self.nb_state_vars+4))

        # Loops to accumulate averages
        for part in self.particles_list:
            for inlet in range(1, self.nb_inlets+1): # Warning: inlets are numbered from 1
                
                # Correspondance for inlet in array
                idx_inlet = inlet-1
                
                if part.num_inlet==inlet: 
                    self.mean_states[0, idx_inlet,1] += part.T/self.nb_parts_per_inlet[inlet] 
                    self.mean_states[0, idx_inlet,2] += part.P/self.nb_parts_per_inlet[inlet]
                    self.mean_states[0, idx_inlet,3:self.nb_state_vars+1] += part.Y/self.nb_parts_per_inlet[inlet]
                    self.mean_states[0, idx_inlet,self.nb_state_vars+1] += part.mix_frac/self.nb_parts_per_inlet[inlet]
                    self.mean_states[0, idx_inlet,self.nb_state_vars+2] += part.equiv_ratio/self.nb_parts_per_inlet[inlet]
                    self.mean_states[0, idx_inlet,self.nb_state_vars+3] += part.prog_var/self.nb_parts_per_inlet[inlet]
        
        # Add time
        for inlet in range(1, self.nb_inlets+1):
            self.mean_states[0, inlet-1, 0] = self.time

    def _append_means_to_trajectory(self):
        
        self.mean_trajectories = np.concatenate([self.mean_trajectories, self.mean_states], axis=0)
        
    # Export trajectories in HDF5 format
    def write_trajectories(self):
                
        # Create h5 file
        f = h5py.File("mean_trajectories.h5","w")
        
        # Create group
        grp = f.create_group("TRAJECTORIES")
        
        for inlet in range(1, self.nb_inlets+1):
            grp.create_dataset(f"INLET {inlet:d}",data=self.mean_trajectories[:, inlet-1, :])
        
        # Closing file
        f.close()

# =============================================================================
#   DATABASE HANDLING
# =============================================================================    
    
    def _write_solution(self):
        
        # Set current iteration results in a dataframe
        arr = np.empty((self.nb_parts_tot, len(self.cols_all_states)))
        for i in range(self.nb_parts_tot):
            # i-th particle
            part = self.particles_list[i]
            arr[i,0] = part.T
            arr[i,1] = part.P
            arr[i,2:part.nb_state_vars] = part.Y
            arr[i,part.nb_state_vars] = part.mix_frac
            arr[i,part.nb_state_vars+1] = part.equiv_ratio
            arr[i,part.nb_state_vars+2] = part.prog_var
            arr[i,part.nb_state_vars+3] = part.hrr
            arr[i,part.nb_state_vars+4] = self.time
            arr[i,part.nb_state_vars+5] = part.num_part
            arr[i,part.nb_state_vars+6] = part.num_inlet
            arr[i,part.nb_state_vars+7:part.nb_state_vars+11] = part.atomic_mass_fractions
            arr[i,part.nb_state_vars+11] = part.mass

        # Store initial solution in h5 file
        f = h5py.File(f"{self.results_folder}/{self.solution_file}","a")
        grp = f.create_group(f"ITERATION_{self.iteration:05d}")    # Group created because this function is called first
        dset = grp.create_dataset("all_states",data=arr)
        dset.attrs["cols"] = self.cols_all_states
        f.close()

    def _update_dtb_states(self, which_state):
        
        # Set current iteration results in a dataframe
        cols = ['Temperature'] + ['Pressure'] + self.species_names + ['Prog_var'] + ['HRR']
        arr = np.empty((self.nb_parts_tot, len(cols)))
        for i in range(self.nb_parts_tot):
            # i-th particle
            part = self.particles_list[i]
            arr[i,0] = part.T
            arr[i,1] = part.P
            arr[i,2:part.nb_state_vars] = part.Y
            arr[i,part.nb_state_vars] = part.prog_var
            arr[i,part.nb_state_vars+1] = part.hrr

        # Store initial solution in h5 file
        f = h5py.File(f"{self.results_folder}/{self.solution_file}","a")
        grp = f.get(f"ITERATION_{self.iteration:05d}")
        dset = grp.create_dataset(which_state,data=arr)
        dset.attrs["cols"] = cols
        f.close()

# =============================================================================
#   PARTICLES STATISTICS COMPUTATION 
# =============================================================================

    def calc_statistics(self):

        # Temperature vector
        Temp_vect = np.empty(self.nb_parts_tot)
        for i, part in enumerate(self.particles_list):
            Temp_vect[i] = part.T
        
        # Mean and standard deviation of temperature
        self.mean_T = Temp_vect.mean()
        self.stdev_T = Temp_vect.std()
        self.mean_T_vect.append(self.mean_T)
        self.stdev_T_vect.append(self.stdev_T)
        self.ratio_T_stdev = self.stdev_T / self.mean_T

    def plot_stats(self):
            
        if not os.path.isdir(self.results_folder + "/statistics"):
            os.mkdir(self.results_folder + "/statistics")
            
        # Creating the time vector
        N = len(self.mean_T_vect)
        time_vect = np.linspace(0, self.time_max, N)
        
        # List -> array conversions
        self.mean_T_vect = np.array(self.mean_T_vect)
        self.stdev_T_vect = np.array(self.stdev_T_vect)
        
        # Temperature stats
        fig1, ax1 = plt.subplots()
        ax1.plot(time_vect, self.mean_T_vect, label="mean", color='k')
        ax1.plot(time_vect, self.stdev_T_vect, label="std", color='b')
        ax1.set_xlabel("$t$ $[s]$")
        ax1.set_ylabel("$T$ $[K]$")
        fig1.legend()
        fig1.tight_layout()
        fig1.savefig(self.results_folder + "/statistics/stats_T.png")
        plt.close()
        
        
        # Ratio T_mean / T_stdev
        fig2, ax2 = plt.subplots()
        ax2.plot(time_vect, self.stdev_T_vect/self.mean_T_vect, color='purple')
        ax2.set_xlabel("$t$ $[s]$")
        ax2.set_ylabel("$T_{std} / T_{mean}$ $[-]$")
        ax2.set_ylim([0,1])
        fig2.legend()
        fig2.tight_layout()
        fig2.savefig(self.results_folder + "/statistics/ratio_T.png")
        plt.close()
            
# =============================================================================
#   DISPLAYING
# =============================================================================

    def print_initial(self):
        
        PRINT("")
        PRINT("START OF SIMULATION")
        PRINT(f">> Total number of iterations: {int(self.time_max/self.dt)}")
        PRINT(f">> Total number of particles: {self.nb_parts_tot}")
        if self.mixing_model=="CURL":
            PRINT(f">> Number of particles pair used for CURL diffusion is: {self.Npairs_curl}")
            PRINT(f">> Curl diffusion performed every {self.diffusion_freq} iterations \n")
        PRINT("")

    def print_iteration(self):
        
        # Get computation timings
        last_loc = self.timings.shape[0]
        last_line = self.timings.loc[last_loc-1]
        
        PRINT("")
        PRINT(f"ITERATION {self.iteration:d}")
        PRINT("Physics:")
        PRINT(f"  >> Current physical time: {self.time:4.3e} s")
        PRINT("PARTICLES STATISTICS:")
        PRINT(f"  >> Mean temperature of particles: {self.mean_T:4.1f} K")
        PRINT(f"  >> Temperature standard deviation: {self.stdev_T:4.1f} K")
        PRINT(f"             => Ratio: {self.ratio_T_stdev:4.3f}")
        PRINT("CPU costs:")
        PRINT(f"  >> Time spent for diffusion: {last_line['Diffusion']:5.4f} s")
        PRINT(f"  >> Time spent for reactions: {last_line['Reaction']:5.4f} s")
        PRINT(f"  >> Time spent for computing mean trajectory: {last_line['MeanTraj']:5.4f} s")
        PRINT(f"  >> Time spent for writing results: {last_line['Writing']:5.4f} s")
        PRINT(f"  >> Time spent for post-processing results: {last_line['Plotting']:5.4f} s")
        PRINT(f"  >> Total CPU time: {last_line['Total']:5.4f} s")
        PRINT("")      
        
    def check_termination(self):
        
        # Termination based on temperature standard deviation
        threshold = 0.02
        if self.ratio_T_stdev<threshold:
            self.stats_converged = True
            print("-----------LOW TEMPERATURE VARIANCE: END OF COMPUTATION-----------")
        
# =============================================================================
#   INPUT CHECKING
# =============================================================================

    def check_species(self):

        # Inlet file species names
        species_inlet = self.species_names
        
        # CANTERA mechanism species
        mech_file = self.mech_file
        gas = ct.Solution(mech_file)
        species_mech = gas.species_names

        try:
            sorted(species_inlet)==sorted(species_mech)
        except:
            print("Species list in mech file and inlet file are not the same !")
            sys.exit(1)