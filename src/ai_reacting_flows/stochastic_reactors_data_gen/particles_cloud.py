import os
from time import perf_counter
import itertools
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import h5py
import seaborn as sns
import cantera as ct

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from mpi4py import MPI

from ai_reacting_flows.stochastic_reactors_data_gen.particle import Particle
from ai_reacting_flows.stochastic_reactors_data_gen.pre_processing import Inlet
import ai_reacting_flows.stochastic_reactors_data_gen.emst_pope as emst_pope
import ai_reacting_flows.tools.utilities as utils
from ai_reacting_flows.tools.utilities import PRINT
from ai_reacting_flows.ann_model_generation.NN_inference import NNInference

matplotlib.use('Agg')
sns.set()

# Attributes of the particles changed during the simulation, exchanged between MPI ranks as one array of floats
# per particle (each rank keeps its own copy of the particles; the other attributes are constant)
EXCHANGED_SCALARS = ["T", "P", "hs", "Hs", "mass", "Yc_u", "Yc_eq", "recompute_Yc_eq", "prog_var", "equiv_ratio",
                     "mix_frac", "hrr", "emst_age"]
EXCHANGED_ARRAYS = ["Y", "X", "mass_k", "state"]

# EMST_POPE: scale of the mass fractions for the tree [-] (example of emst.f: the tree is built on Y / 0.1,
# minor species have a negligible weight in it)
EMST_SPECIES_SCALE = 0.1

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
        
        # Initialize time and general parameters
        self.time_max = float(data_gen_parameters["time_max"])
        self.time = 0.0
        self.iteration = 0
        self.dt_input = data_gen_parameters["time_step"]
        self.calc_mean_traj = data_gen_parameters["calc_mean_traj"]
        
        # Statistics convergence status
        self.stats_converged = False
        self.automatic_termination = data_gen_parameters.get("automatic_termination", True)

        # Mechanism file
        self.mech_file  = data_gen_parameters["mech_file"]
        self.gas = ct.Solution(self.mech_file)

        # Only species made of C, H, O and N are handled (error if e.g. Ar or He)
        utils.check_elements(self.gas)

        # Dabatase building
        self.build_ml_dtb = data_gen_parameters["build_ml_dtb"]

        # Species names
        self.species_names = self.gas.species_names
        self.nb_species = len(self.species_names)

        # Progress variables species
        self.pv_species = data_gen_parameters["pv_species"]
        self.pv_ind = [self.gas.species_index(spec) for spec in self.pv_species]

        # ANN chemistry: networks of MODELS/<ML_model_folder> in the run folder (see NN_inference.py)
        if self.ML_inference_flag:
            self.ML_yk_renormalization = data_gen_parameters["ML_yk_renormalization"]
            self.ML_hybrid_ann_cvode = data_gen_parameters["ML_hybrid_ann_cvode"]
            self.ML_hybrid_ann_cvode_tol = data_gen_parameters["ML_hybrid_ann_cvode_tol"]

            models_folder = f"{self.run_folder}/MODELS/{data_gen_parameters['ML_model_folder']}"
            self.nn_inference = NNInference(models_folder, self.run_folder)

            # Same species and progress variable (used for clustering) as the model
            if self.nn_inference.spec_list_ref != self.species_names:
                raise ValueError(f"Species of {models_folder} differ from those of mech_file")
            if self.nn_inference.pv_species != self.pv_species:
                raise ValueError(f"pv_species of {models_folder} ({self.nn_inference.pv_species}) differ from pv_species")

        # Setting the time-step array (it is an array to account for possible different values of dt in user-presribed time windows)
        # This is useful in cases where slow chemistry takes places first and more reactive particles are added later
        self._set_time_step_array()

        # =====================================================================
        #     GENERATING INLETS    
        # =====================================================================

        # Reading inlets configuration from 
        self.inlets_config = data_gen_parameters["inlets"]

        self.nb_inlets = len(self.inlets_config)

        # Required inputs for each inlet type
        FIELDS_BY_TYPE = {
            "blank":          ["T", "P"],
            "air":            ["T", "P"],
            "fuel":           ["T", "P", "fuel"],
            "cold_premixed":  ["T", "P", "fuel", "phi"],
            "burnt_premixed": ["T", "P", "fuel", "phi"],
        }

        inlet_list = []
        for inlet_name in self.inlets_config:

            cfg = self.inlets_config[inlet_name]

            inlet_type = cfg["type"]
            nb_part = int(cfg["nb_particles"])
            activation_time = float(cfg.get("activation_time", 0.0))  #default to 0 if parameter not present

            inlet = Inlet(inlet_type, nb_particles=nb_part, activation_time = activation_time, pv_ind= self.pv_ind)

            state_kwargs = {"mech": self.mech_file}
            for field in FIELDS_BY_TYPE[inlet_type]:
                state_kwargs[field] = cfg[field]
   
            inlet.set_state(**state_kwargs)

            inlet_list.append(inlet)
        
        # Total number of particles in simulation
        self.nb_parts_per_inlet = dict.fromkeys(range(1,self.nb_inlets+1))
        self.state_per_inlet = dict.fromkeys(range(1,self.nb_inlets+1))
        for i, inlet in enumerate(inlet_list):
            self.nb_parts_per_inlet[i+1] = inlet.nb_particles
            self.state_per_inlet[i+1] = inlet.state[1:]


        # Total number of particles
        self.nb_parts_tot = sum(self.nb_parts_per_inlet.values())

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
                activation_time = inlet_list[i-1].activation_time
                Yc_u_ini = inlet_list[i-1].Yc_u
                particle_current = Particle(state_ini, self.species_names, i, num_part, data_gen_parameters, activation_time, Yc_u_ini, self)
                self.particles_list.append(particle_current)
                num_part += 1

        # Setting is_active flag on each particle (and counting active particles)
        self._count_active_particles()

        # Activation times for each particle
        activation_times = np.array([p.activation_time for p in self.particles_list])

        # At least one inlet must be active from the start of the simulation
        if activation_times.min() > 0.0:
            raise ValueError("At least one inlet must have activation_time = 0")
        
        # =====================================================================
        #     MIXING MODEL INITIALIZATION 
        # =====================================================================
                
        # Mixing model
        self.mixing_model = data_gen_parameters["mixing_model"]
        self.mixing_time = float(data_gen_parameters["mixing_time"])

        # EMST_POPE: ages of the particles in the stationary state. Same seed on all ranks (mixing is done
        # by rank 0, the particles and their ages are then sent to all ranks)
        if self.mixing_model == "EMST_POPE":
            self.emst_rng = np.random.default_rng(0)
            for part, age in zip(self.particles_list, emst_pope.init_ages(self.nb_parts_tot, self.emst_rng)):
                part.emst_age = age
        
        # Chemistry temperature threshold
        self.T_threshold = float(data_gen_parameters["T_threshold"])

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

            N_pairs = np.zeros_like(self.dt)
            max_attained = []
            time = 0.0
            for i in range(self.n_ite):

                    # Number of active particles (time at the start of iteration i, as in advance_time)
                    active_idx = np.where(activation_times <= time)[0]   # true particle indices, active at this time
                    n_parts_active = active_idx.size
                    time += self.dt[i]
            
                    N_pairs[i] = 1.5 * n_parts_active * self.dt[i]/self.tau_min # 1.5 factor used in Ren et al. paper

                    # Case where we select all pairs (pairs where i==j and (i,j)=(j,i) are not considered)
                    N_pairs_max = n_parts_active * (n_parts_active - 1) // 2
                    if N_pairs[i] > N_pairs_max:
                        max_attained.append(i)
                        N_pairs[i] = N_pairs_max

            if self.rank==0 and max_attained:
                PRINT(f"WARNING: number of selected pairs for mixing exceeds maximum for time steps {max_attained} \n -> setting N_pairs = N_pairs_max (the mixing time will be altered)")
            
            # Number of mixed particle
            self.Npairs_curl = np.round(N_pairs).astype(int)

            # If number is very small, N_pairs_curl might be 0, in this case we set it
            # to 1 and perform diffusion every round(1/N_pairs) time steps
            self.diffusion_freq = np.zeros_like(self.Npairs_curl)
            for i in range(self.n_ite):
                if self.Npairs_curl[i]==0:
                    self.Npairs_curl[i]=1
                    self.diffusion_freq[i] = int(round(1/N_pairs[i]))
                else:
                    self.diffusion_freq[i] = 1

        

        if self.mixing_model not in ["CURL_MODIFIED_DD", "EMST_SIMPLIFIED", "EMST_POPE"]:
            # Do or read mixing
            read_mixing = data_gen_parameters["read_mixing"]
            do_CURL_rate = True if "CURL_MODIFIED" in self.mixing_model else False
            if read_mixing:
                with open(f"{self.run_folder:s}/pairs_list.pkl", "rb") as f:
                    self.pairs_list = pickle.load(f)
                if do_CURL_rate:
                    with open(f"{self.run_folder:s}/CURL_rate_list.pkl", "rb") as f:
                        self.CURL_rate_list = pickle.load(f)
            elif self.rank==0:
                # Pairs drawn and saved by rank 0 only, as mixing is only performed by rank 0

                self.pairs_list = []
                self.CURL_rate_list = []

                time = 0.0
                for i in range(self.n_ite): # DAK: vectorize pair generation ?

                    # Particles active at the start of iteration i (as in advance_time)
                    active_idx = np.where(activation_times <= time)[0]   # true particle indices, active at this time
                    time += self.dt[i]

                    # local_pairs = utils.sample_comb2((active_idx.size, active_idx.size), self.Npairs_curl[i]) # Pairs using active particles indices
                    local_pairs = utils.sample_pairs(active_idx.size, self.Npairs_curl[i])
                    real_pairs = active_idx[local_pairs]   # Going back to particles indices
                    self.pairs_list.append(real_pairs)

                    # One random mixing extent per pair (Janicka et al. 1979, Pope 1982)
                    if do_CURL_rate:
                        self.CURL_rate_list.append(np.random.random(len(real_pairs)))
                

                with open(self.results_folder + "/pairs_list.pkl", 'wb') as f:
                    pickle.dump(self.pairs_list, f)
                if do_CURL_rate:
                    with open(self.results_folder + "/CURL_rate_list.pkl", 'wb') as f:
                        pickle.dump(self.CURL_rate_list, f)

                    
        # =====================================================================
        #     ATOMIC CONSERVATION ANALYSIS
        # =====================================================================
        
        # Matrix for computing elements mass fractions (order of rows: C, H, O, N), from Cantera
        self.A_atomic = utils.get_elements_matrix(self.gas)
        
        
        # =====================================================================
        #     INPUT/OUTPUT INITIALIZATION
        # =====================================================================

        # Columns of data array with solution and post-processing variables
        self.cols_all_states = ['Temperature'] + ['Pressure'] + self.species_names + ['Mix_frac'] + ['Equiv_ratio'] + ['Prog_var'] + ['HRR'] +  ['Time'] + ['Particle_number'] + ['Inlet_number'] + ['Y_C', 'Y_H', 'Y_O', 'Y_N'] + ['Mass'] + ['EMST_age']
        
        # Initialize mean trajectories
        if self.calc_mean_traj:
            self._compute_means()
            # Only rank 0 deals with mean trajectories as it is a post-processing
            if self.rank==0: 
                self.mean_trajectories = np.zeros((1, self.nb_inlets, self.nb_state_vars+4))
                self.mean_trajectories[0,:,:] = self.mean_states
        
        
        # Initialize vector for tracking mean/std of temperature 
        self.time_stats_vect = []
        self.mean_T_vect = []
        self.stdev_T_vect= []
        
        # Physical time per iterations
        self.timings = pd.DataFrame(data=[], columns=["Total", "Reaction", "Diffusion", "MeanTraj", "Writing", "Plotting"])
        
        # Start of simulation printing
        if self.rank == 0:
            self.print_initial()



    def _set_time_step_array(self) -> None:
        """
        Build self.dt, an array giving the time step used at each iteration,
        and self.n_ite, the total number of iterations.

        self.dt_input can be:
            - a number (int/float): constant time step for the whole run.
            - a dict {t0: dt0, t1: dt1, ...}: dt0 is used from t0 to t1,
            dt1 from t1 to t2, etc., and the last dt is used until
            self.time_max. Keys need not be pre-sorted, but the first
            key must be 0.

        Raises
        ------
        TypeError
            If self.dt_input is neither a number nor a dict.
        ValueError
            If the dict is empty, doesn't start at t=0, or contains a
            non-positive dt or time interval.
        """

        # Constant dt case
        if isinstance(self.dt_input, (int, float)):
            dt_cte = float(self.dt_input)
            if dt_cte <= 0:
                raise ValueError(f"[Error in dt array] dt must be positive, got {dt_cte}")
            
            # One extra iteration: the summed time may stay slightly below time_max after ceil(time_max/dt) steps
            self.n_ite = math.ceil(self.time_max / dt_cte) + 1
            self.dt = np.ones(self.n_ite) * dt_cte

        # dt changing over time in user-prescribed time intervals
        elif isinstance(self.dt_input, dict):
            if not self.dt_input:
                    raise ValueError("[Error in dt array] dt_input dict is empty")
            
            dt_dict = dict(sorted(self.dt_input.items()))

            dt_change_times = list(dt_dict.keys())
            dt_values = list(dt_dict.values())
            nb_dt_changes = len(dt_change_times)

            if dt_change_times[0] != 0:
                raise ValueError("[Error in dt array] The first time step change must start at t=0")

            dt_chunks = []
            for i in range(nb_dt_changes):

                dt_current = dt_values[i]
                if dt_current <= 0:
                    raise ValueError(f"[Error in dt array] dt must be positive, got {dt_current}")

                t_start = dt_change_times[i]
                t_end = (
                    self.time_max if i == nb_dt_changes - 1
                    else dt_change_times[i + 1]
                )
                period = t_end - t_start

                if period <= 0:
                    raise ValueError(
                        f"[Error in dt array] Time intervals must be increasing (issue at t={dt_change_times[i]})"
                    )
                
                #Number of iteration in this time window
                n_ite_period = math.ceil(period / dt_current) 
                dt_chunks.append(np.full(n_ite_period, dt_current))

            self.dt = np.concatenate(dt_chunks) if dt_chunks else np.array([])
            self.dt = np.append(self.dt, self.dt[-1])
            self.n_ite = len(self.dt)

        # Wrong type (e.g. 1e-5 is read as a string by YAML, 1.0e-5 is a float)
        else:
            raise TypeError(f"[Error in dt array] time_step must be a number or a dict, got {self.dt_input!r} (write 1.0e-5, not 1e-5)")


# =============================================================================
#   TIME MARCHING FUNCTIONS
# =============================================================================
                
    # Advance solution to next time step     
    #TODO: dt inside and outside class, maybe there is a cleaner way
    def advance_time(self, dt):
        
        # Initial time
        t0 = perf_counter()

        # Counting number of active particles
        self._count_active_particles()

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
        # Diffusion is applied by rank 0, the mixed states are then sent to every processor
        if self.rank==0:
            self._apply_diffusion()
        self._share_states_of_root()

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
        if self.automatic_termination:
            self.check_termination()    # we suspend that for now, we have cases where we want to run with low temperature variance

        # Update time and iteration
        self.time += dt
        self.iteration += 1
        
    # Apply reaction rate to particles
    def _apply_reactions(self, dt):

        # Each processor computes the chemistry of its chunk of particles
        chunk = self._particle_chunk()
        particles = [self.particles_list[i] for i in chunk]

        # Reaction only if time is larger than activation time
        active_particles = [part for part in particles if self.time >= part.activation_time]

        if self.ML_inference_flag:
            # Chemistry with the ANN
            self._react_ANN(active_particles, dt)
        else:
            # Chemistry with Cantera (CVODE)
            for part in active_particles:
                part.react(dt, self)

        # States of all chunks given to all the processors
        self._gather_states(chunk)

    # Apply reaction rate to particles with the ANN
    def _react_ANN(self, particles, dt):
        """Advance the chemistry of particles by dt [s] with the ANN (one batch, see NNInference).

        Particles at T <= T_threshold are frozen, as in Particle.react. With ML_hybrid_ann_cvode,
        particles whose predicted mass fractions sum differs from 1 by more than ML_hybrid_ann_cvode_tol
        are advanced with Cantera instead.
        """

        particles = [part for part in particles if part.T > self.T_threshold]
        if len(particles) == 0:
            return

        T = np.array([part.T for part in particles])
        Y = np.array([part.Y for part in particles])
        P = np.array([part.P for part in particles])
        progvar = np.array([part.prog_var for part in particles])

        clusters = self.nn_inference.attribute_clusters(T, Y, dt, progvar)
        T_new, Y_new, sum_Yk = self.nn_inference.advance(T, Y, P, dt, clusters, self.ML_yk_renormalization)

        for i, part in enumerate(particles):

            # Hybrid: Cantera if the ANN does not conserve mass
            if self.ML_hybrid_ann_cvode and np.abs(sum_Yk[i] - 1.0) > self.ML_hybrid_ann_cvode_tol:
                part.react(dt, self)
            else:
                part.set_state_from_T_Y(T_new[i], Y_new[i], self)

    # Apply diffusion to particles
    def _apply_diffusion(self):
        
        if self.mixing_model=="CURL" or self.mixing_model=="CURL_MODIFIED":

            if self.iteration%self.diffusion_freq[self.iteration]==0:
                self._mix_curl()

        elif self.mixing_model=="CURL_MODIFIED_DD":

            if self.iteration%self.diffusion_freq[self.iteration]==0:
                self._mix_curl_dd()

        elif self.mixing_model=="EMST_SIMPLIFIED":
            self._mix_emst()

        elif self.mixing_model=="EMST_POPE":
            self._mix_emst_pope()



    def _count_active_particles(self) -> None:

        activation_times = np.array([p.activation_time for p in self.particles_list])
        is_active_mask = activation_times <= self.time
        self.nb_active_particles = int(is_active_mask.sum())

        for particle, active in zip(self.particles_list, is_active_mask):
            particle.is_active = bool(active)


# =============================================================================
#   MIXING MODELING
# =============================================================================
    
    # CURL model: unity lewis numbers model => we work directly with mass fractions
    def _mix_curl(self):

        # Randomly select mixing pairs
        pairs = self.pairs_list[self.iteration]
        
        # Carrying out diffusion
        for k, pair in enumerate(pairs):
            
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
            #
            Yc_u_1_ini = part_1.Yc_u.copy()
            Yc_u_2_ini = part_2.Yc_u.copy()
                    
            if self.mixing_model=="CURL":
                
                # Update particles states
                part_1.Y = 0.5 * (Y2_ini + Y1_ini)
                part_1.hs = 0.5 * (h2_ini + h1_ini)
                part_1.Yc_u = 0.5 * (Yc_u_1_ini + Yc_u_2_ini)
                #
                part_2.Y = 0.5 * (Y2_ini + Y1_ini)
                part_2.hs = 0.5 * (h2_ini + h1_ini)
                part_2.Yc_u = 0.5 * (Yc_u_1_ini + Yc_u_2_ini)
                
            elif self.mixing_model=="CURL_MODIFIED":
                    
                # Random value between 0 and 1, drawn for each pair
                a = self.CURL_rate_list[self.iteration][k]
                        
                # Update particles states
                part_1.Y += 0.5 * a * (Y2_ini - Y1_ini)
                part_1.hs += 0.5 * a * (h2_ini - h1_ini)
                part_1.Yc_u += 0.5 * a * (Yc_u_2_ini - Yc_u_1_ini)
                #
                part_2.Y += 0.5 * a * (Y1_ini - Y2_ini)
                part_2.hs += 0.5 * a * (h1_ini - h2_ini)
                part_2.Yc_u += 0.5 * a * (Yc_u_1_ini - Yc_u_2_ini)
            
            
            part_1.state[0] = part_1.hs
            part_1.state[2:] = part_1.Y
            part_1.update_ThermoStates(self)
            
            part_2.state[0] = part_2.hs
            part_2.state[2:] = part_2.Y
            part_2.update_ThermoStates(self)

            # Yc_eq will need to be recomputed as particles have mixed
            part_1.recompute_Yc_eq = True
            part_2.recompute_Yc_eq = True
            

    # CURL model: with differential diffusion => we work with species masses
    def _mix_curl_dd(self):

        masses_of_pairs = np.empty(len(self.particle_pairs))
        for i, pair in enumerate(self.particle_pairs):
            masses_of_pairs[i] = self.particles_list[pair[0]].mass + self.particles_list[pair[1]].mass
        prob_of_pairs = masses_of_pairs / masses_of_pairs.sum()

        # Randomly select mixing pairs
        pairs_index = np.random.choice(len(self.particle_pairs), size=self.Npairs_curl[self.iteration], p=prob_of_pairs,replace=False)
        
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




    # EMST mixing model (EMST_SIMPLIFIED)
    def _mix_emst(self):
        """Perform one EMST micro-mixing (simplified version, EMST_SIMPLIFIED; see _mix_emst_pope for the model
        of Subramaniam & Pope).

        The implementation follows the Euclidean Minimum Spanning Tree (EMST)
        approach:
          - build an MST in normalized composition space,
          - compute an exponential pairwise relaxation coefficient,
          - distribute mixing conservatively along tree edges.
        """

        # --------------------------------------------------------------
        # Gather particle properties
        # --------------------------------------------------------------
        active_particles = [p for p in self.particles_list if p.is_active]
        Y_all = np.array([p.Y for p in active_particles])
        #
        hs_all = np.array([p.hs for p in active_particles])
        Yc_u_all = np.array([p.Yc_u for p in active_particles])

        # Question: should we inlude enthalpy in tree construction ?

        # --------------------------------------------------------------
        # Phase-space used to build the EMST (normalized)
        # --------------------------------------------------------------
        # Question: do we need to normalize ?
        # phi_tree_raw = np.column_stack((Y_all))

        # sigma = phi_tree_raw.std(axis=0)
        # sigma[sigma < 1e-30] = 1.0

        # phi_tree = phi_tree_raw  / sigma
        # phi_tree = phi_tree_raw.copy()

        phi_tree = Y_all.copy()

        # Variables actually mixed (can include extra scalars such as Yc_u)
        phi_old = np.column_stack((Y_all, hs_all, Yc_u_all))

        # --------------------------------------------------------------
        # Compute Euclidean MST
        # --------------------------------------------------------------
        distances = squareform(pdist(phi_tree))
        mst = minimum_spanning_tree(distances)
        mst = mst.toarray()

        # ------------------------------------------------------------------
        # Extract tree edges
        # ------------------------------------------------------------------
        edges = np.transpose(np.nonzero(mst))

        # ------------------------------------------------------------------
        # Conservative pairwise mixing along each edge
        # ------------------------------------------------------------------

        mixing_coeff = self.dt[self.iteration] / self.mixing_time
        
        # Stability safeguard (prevents overshooting on a single edge)
        mixing_coeff = min(mixing_coeff, 0.5)

        phi_new = phi_old.copy()

        # Randomize edge order
        rng = np.random.default_rng(seed=42)
        edge_order = rng.permutation(len(edges))

        for idx in edge_order:
            
            i, j = edges[idx]
            delta = phi_new[i] - phi_new[j]

            phi_new[i] -= mixing_coeff * delta
            phi_new[j] += mixing_coeff * delta

        # --------------------------------------------------------------
        # Write back particle states
        # --------------------------------------------------------------
        for k, p in enumerate(active_particles):

            p.Y = phi_new[k, :self.nb_species]
            p.hs = phi_new[k, self.nb_species]
            p.Yc_u = phi_new[k, self.nb_species + 1]

            # For safety
            # p.Y = np.clip(p.Y, 0.0, None)
            # p.Y /= p.Y.sum() + 1e-30

            p.state[0] = p.hs
            p.state[2:] = p.Y

            p.update_ThermoStates(self)

            # EMST requires recomputation of equilibrium progress variable
            p.recompute_Yc_eq = True


    # EMST mixing model of Subramaniam & Pope (EMST_POPE)
    def _mix_emst_pope(self):
        """EMST mixing of the active particles over the time step, model of Subramaniam & Pope with ages
        (see emst_pope.py): the variance function of the tree variables decays as exp(-dt / mixing_time).

        Mixed variables: Y, hs and Yc_u. As in the example of emst.f, the tree is built on the mass
        fractions only, scaled by EMST_SPECIES_SCALE; hs and Yc_u are mixed but not used for the tree.
        """

        active_particles = [p for p in self.particles_list if p.is_active]

        f = np.column_stack((np.array([p.Y for p in active_particles]),
                             np.array([p.hs for p in active_particles]),
                             np.array([p.Yc_u for p in active_particles])))
        w = np.array([p.mass for p in active_particles])
        ages = np.array([p.emst_age for p in active_particles])

        # Scales: mass fractions for the tree, hs and Yc_u not used for the tree
        fscale = np.append(np.full(self.nb_species, EMST_SPECIES_SCALE), [emst_pope.NO_TREE_SCALE] * 2)

        # Mixing over the normalized time dt / mixing_time
        ds = self.dt[self.iteration] / self.mixing_time
        f_new, ages_new = emst_pope.emst_mix(f, fscale, w, ages, ds, self.emst_rng)

        # Write back particle states
        for k, p in enumerate(active_particles):

            p.Y = f_new[k, :self.nb_species]
            p.hs = f_new[k, self.nb_species]
            p.Yc_u = f_new[k, self.nb_species + 1]
            p.emst_age = ages_new[k]

            p.state[0] = p.hs
            p.state[2:] = p.Y

            p.update_ThermoStates(self)

            # Mixing changes the equilibrium progress variable
            p.recompute_Yc_eq = True


# =============================================================================
#   MISC
# =============================================================================

    def _update_vars(self):

        # We parallelize as the progress variable computation is expensive (indeed, it relies on an "equil" computation)
        chunk = self._particle_chunk()

        # We perform variable updates of particles of the chunk of this processor
        for i in chunk:
            part = self.particles_list[i]
            part.compute_equiv_ratio()
            part.compute_progress_variable(self)
            part.compute_mixture_fraction()
            part.compute_heat_release_rate(self)

        # States of all chunks given to all the processors
        self._gather_states(chunk)

# =============================================================================
#   MPI EXCHANGES OF PARTICLE STATES
# =============================================================================

    def _particle_chunk(self, rank=None):
        """Indices of the particles treated by processor rank (this processor by default), dealt in turn
        (round-robin): rank, rank + size, rank + 2 size, ... Particles of an inlet (e.g. hot particles, the only
        ones that react at first) are then spread over all processors."""

        rank = self.rank if rank is None else rank

        return np.arange(rank, self.nb_parts_tot, self.size)

    def _pack_states(self, particles):
        """Array (n, n_values) of the attributes of the particles changed during the simulation
        (EXCHANGED_SCALARS, then EXCHANGED_ARRAYS)."""

        return np.array([np.concatenate([[float(getattr(part, name)) for name in EXCHANGED_SCALARS]]
                                        + [getattr(part, name) for name in EXCHANGED_ARRAYS])
                         for part in particles])

    def _unpack_states(self, particles, states):
        """Set the attributes of the particles from the array states (see _pack_states)."""

        for part, values in zip(particles, states):
            for k, name in enumerate(EXCHANGED_SCALARS):
                setattr(part, name, values[k])
            part.recompute_Yc_eq = bool(part.recompute_Yc_eq)

            start = len(EXCHANGED_SCALARS)
            for name in EXCHANGED_ARRAYS:
                n = len(getattr(part, name))
                setattr(part, name, values[start:start + n].copy())
                start += n

    def _share_states_of_root(self):
        """States of the particles of processor 0 (e.g. after mixing) given to all the processors."""

        if self.size == 1:
            return

        # Number of exchanged values per particle, the same for all particles and processors
        part = self.particles_list[0]
        n_values = len(EXCHANGED_SCALARS) + sum(len(getattr(part, name)) for name in EXCHANGED_ARRAYS)

        if self.rank == 0:
            states = self._pack_states(self.particles_list)
        else:
            states = np.empty((self.nb_parts_tot, n_values))
        self.comm.Bcast(states, root=0)

        if self.rank != 0:
            self._unpack_states(self.particles_list, states)

    def _gather_states(self, chunk):
        """States of the particles of the chunk of each processor given to all the processors."""

        if self.size == 1:
            return

        # Array of all the states, filled by the chunks of all processors one after the other
        # (no pickling: array of floats)
        local = self._pack_states([self.particles_list[i] for i in chunk])
        n_values = local.shape[1]
        chunks = [self._particle_chunk(rank) for rank in range(self.size)]
        counts = np.array([len(c) for c in chunks]) * n_values
        displacements = np.concatenate([[0], np.cumsum(counts)[:-1]])
        states = np.empty((self.nb_parts_tot, n_values))
        self.comm.Allgatherv(local, [states, counts, displacements, MPI.DOUBLE])

        # Rows of states in the order of the particles, then particles of the other processors
        particle_of_row = np.concatenate(chunks)
        others = np.setdiff1d(np.arange(self.nb_parts_tot), chunk)
        row_of_particle = np.empty(self.nb_parts_tot, dtype=int)
        row_of_particle[particle_of_row] = np.arange(self.nb_parts_tot)
        self._unpack_states([self.particles_list[i] for i in others], states[row_of_particle[others]])

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

        # Keep only active particles
        active_particles = [p for p in self.particles_list if p.is_active]
        nb_active = len(active_particles)
    
        
        # Set current iteration results in a dataframe
        arr = np.empty((nb_active, len(self.cols_all_states)))
        for i, part in enumerate(active_particles):
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
            arr[i,part.nb_state_vars+7:part.nb_state_vars+11] = np.dot(self.A_atomic, part.Y)   # Elements mass fractions (C, H, O, N)
            arr[i,part.nb_state_vars+11] = part.mass
            arr[i,part.nb_state_vars+12] = part.emst_age   # Age property of EMST_POPE (NaN for other mixing models)

        # Store initial solution in h5 file
        f = h5py.File(f"{self.results_folder}/{self.solution_file}","a")
        grp = f.create_group(f"ITERATION_{self.iteration:05d}")    # Group created because this function is called first
        dset = grp.create_dataset("all_states",data=arr)
        dset.attrs["cols"] = self.cols_all_states
        f.close()

    def _update_dtb_states(self, which_state):

        # Keep only active particles
        active_particles = [p for p in self.particles_list if p.is_active]
        nb_active = len(active_particles)
        
        # Set current iteration results in a dataframe
        cols = ['Temperature'] + ['Pressure'] + self.species_names + ['Prog_var'] + ['HRR']
        arr = np.empty((nb_active, len(cols)))
        for i, part in enumerate(active_particles):
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

        # Temperature vector (active particles only)
        Temp_vect = np.array([part.T for part in self.particles_list if part.is_active])
        
        # Mean and standard deviation of temperature
        self.mean_T = Temp_vect.mean()
        self.stdev_T = Temp_vect.std()
        self.time_stats_vect.append(self.time)
        self.mean_T_vect.append(self.mean_T)
        self.stdev_T_vect.append(self.stdev_T)
        self.ratio_T_stdev = self.stdev_T / self.mean_T

    def plot_stats(self):
            
        if not os.path.isdir(self.results_folder + "/statistics"):
            os.mkdir(self.results_folder + "/statistics")
            
        # Time of each statistics [s]
        time_vect = np.array(self.time_stats_vect)
        
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
        PRINT(f">> Total number of iterations: {self.n_ite}")
        PRINT(f">> Number of inlets: {self.nb_inlets}")
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
        PRINT(f"Number of active particles: {self.nb_active_particles} (/{self.nb_parts_tot})")
        PRINT("Physics:")
        PRINT(f"  >> Current physical time: {self.time:6.5e} s")
        PRINT(f"  >> Current time step: {self.dt[self.iteration]:4.3e} s")
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
        """
        Check whether the simulation should be flagged as converged. 
        
        Only active particles (particle.is_active) are considered.

        Termination criteria:
          - Temperature standard deviation ratio below threshold.
          - Progress variable close to 1 for all active particles.
        """        

        # Termination based on temperature standard deviation
        threshold = 0.02
        temp_converged = self.ratio_T_stdev < threshold

        if temp_converged: # and prog_var_converged:
            self.stats_converged = True
            print("-----------LOW TEMPERATURE VARIANCE AND HIGH PROGRESS VARIABLE: END OF COMPUTATION-----------")
