import os
import oyaml as yaml

import shutil
from time import perf_counter

from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud
from ai_reacting_flows.tools.utilities import PRINT, react_multi_dt

import numpy as np
import h5py
import pandas as pd
import cantera as ct
import scipy as sc

from mpi4py import MPI

h5py.get_config().mpi

# Disabling TF warning (in particular warnings related to unused CPU features)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_stochastic_database(comm : 'MPI.Comm'):

    # Initial computation time
    t_ini = perf_counter()

    rank = comm.Get_rank()

    run_folder = os.getcwd()
    with open(os.path.join(run_folder, "dtb_params.yaml"), "r") as file:
        data_gen_parameters = yaml.safe_load(file)

    #==========================================================================
    # CATCHING SOME INPUT ERRORS
    #==========================================================================

    if rank==0:

        if (data_gen_parameters["mixing_model"] not in ["CURL","CURL_MODIFIED","CURL_MODIFIED_DD","EMST"]):
            # Mixing model choice
            try:
                raise ValueError("Input error.")
            except ValueError:
                print("")
                print(">> case_type entry should be one of the following:")
                print("     - CURL")
                print("     - CURL_MODIFIED")
                print("     - CURL_MODIFIED_DD")
                print("     - EMST")
                print("")
                raise


        if (data_gen_parameters["mixing_time"] < data_gen_parameters["time_step"]):
            # Mixing model choice
            try:
                raise ValueError("Input error.")
            except ValueError:
                print("")
                print(">> Curl model diffusion time cannot be lower than the numerical time step.")
                print("")
                raise
            

    #==========================================================================
    # INITIALIZING RUN
    #==========================================================================

    if rank==0:

        PRINT("")
        PRINT(60 * "=")
        PRINT(" SIMULATION OF STOCHASTIC PARTICLES ")
        PRINT(60 * "=")

        # Checking input parameters

        # Remove existing results dir if already exists
        results_folder = "STOCH_DTB_" + data_gen_parameters["results_folder_suffix"]
        if os.path.isdir(results_folder):
            PRINT("WARNING: existing results directory has been removed. \n")
            shutil.rmtree(results_folder)
        os.mkdir(results_folder)

        # Copying input files to results folder to document run
        shutil.copy(data_gen_parameters["inlets_file"], results_folder)
        shutil.copy(data_gen_parameters["mech_file"], results_folder)


    # Particle cloud initialization
    particle_cloud = ParticlesCloud(data_gen_parameters, comm)

    #==============================================================================
    # PARTICLE CLOUD TIME MARCHING
    #==============================================================================

    # Initializations
    dt = data_gen_parameters["time_step"]
    time = 0.0

    while (time<data_gen_parameters["time_max"] and (not particle_cloud.stats_converged)):
        
        # Advancing particles state by one dt
        time += dt
        particle_cloud.advance_time(dt)

    # End of simulation operations
    if rank==0:

        # Save mean trajectories
        if data_gen_parameters["calc_mean_traj"]:
            particle_cloud.write_trajectories()
            shutil.move("./mean_trajectories.h5", results_folder + "/mean_trajectories.h5")

        # Plot some statistics on particles
        particle_cloud.plot_stats()


    #==============================================================================
    # ENDING THE SIMULATION
    #==============================================================================

    # End computation time
    t_end = perf_counter()

    shutil.copy(os.path.join(run_folder, "dtb_params.yaml"), f"{run_folder:s}/STOCH_DTB_{data_gen_parameters['results_folder_suffix']}/dtb_params.yaml")

    if rank==0:
        #  End of simulation printing
        PRINT("")
        PRINT("END OF SIMULATION")
        PRINT("CPU costs:")
        PRINT(f"Total CPU time of simulation: {t_end-t_ini:4.3f} s")
        PRINT(f"  >> Cumulated time spent in simulation for diffusion: {particle_cloud.timings['Diffusion'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for reactions: {particle_cloud.timings['Reaction'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for trajectories averaging: {particle_cloud.timings['MeanTraj'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for writing results: {particle_cloud.timings['Writing'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for plotting results: {particle_cloud.timings['Plotting'].sum():5.4f} s")
        PRINT("")


def GenerateVariable_dt(params, comm : 'MPI.Comm'):
    rank = comm.Get_rank()
    size = comm.Get_size()

    np.random.seed(1234)

    # Opening h5 file
    run_folder = os.getcwd()
    stoch_results_folder = f"{run_folder:s}/STOCH_DTB_" + params["results_folder_suffix"]
    h5file_r = h5py.File(f"{stoch_results_folder:s}/{params['dtb_file'].split('.')[0]:s}_resamp{int(params['T_threshold'])}K.h5", 'r')

    # Solution 0 read to get columns names
    col_names_X = h5file_r["ITERATION_00000/X"].attrs["cols"][:-2]
    col_names_Y = h5file_r["ITERATION_00000/Y"].attrs["cols"][:-2]

    # Loop on solutions
    list_df_X = []

    for i in range(len(h5file_r.keys())):
        data_X = h5file_r.get(f"ITERATION_{i:05d}/X")[:,:-2]
        list_df_X.append(pd.DataFrame(data=data_X, columns=col_names_X))

    h5file_r.close()

    X = pd.concat(list_df_X, ignore_index=True).to_numpy()
    Y = np.empty(X.shape)
    np.random.shuffle(X) # DAK: check why we shuffle X
    state_list = np.dstack((X,Y)) # DAK : check why we dstack with Y (np.empty)

    #adding dt to the features
    T_thresh = params['T_threshold']

    dt_simu = params['time_step']
    if params['time_step_type'] == 'set':
        dt_list = params['time_step_range']
        if dt_simu not in dt_list:
            dt_list.append(dt_simu)
    else:
        dt_range = params['time_step_range']
        nb_dt = params['nb_dt']

    state_list = np.array_split(state_list, size)
    # assert len(state_list) == size

    state_list = comm.scatter(state_list, root=0)
    
    gas = ct.Solution(params["mech_file"])
    # print('appel_gas')
    tot_0 = len(state_list)

    count = 0
    all_new_states = []
    #reacting all states X for dt in dt_range
    for state in state_list: 
            count +=1
            if rank == 0 and count%5000 == 0:
                print('Opération (proc 0)', count, '/', int(tot_0))
            if params['time_step_type'] == 'random':
                dt_list = sc.stats.loguniform(dt_range[0], dt_range[1]).rvs(size = nb_dt) 
                dt_list.sort()
            new_states = react_multi_dt(state, gas, T_thresh, dt_list, dt_simu)
            if new_states is not None:
                all_new_states.append(new_states)
    
    
    all_new_states = np.concatenate(all_new_states)
    X_new, Y_new = np.dsplit(all_new_states, 2)
    X_new, Y_new = np.squeeze(X_new), np.squeeze(Y_new)
    Y_new = Y_new[:,:-1]
    print('READY', rank)
    
    comm.Barrier()

    if rank ==0:

        f = h5py.File(f"{stoch_results_folder:s}/{params['new_file_name']:s}","w")

    comm.Barrier()
    
    for i in range(size):
        if rank == i:
            print('WRITING', rank)
            f = h5py.File(f"{stoch_results_folder:s}/{params['new_file_name']:s}","a")
            grp = f.create_group(f"ITERATION_{(rank):05d}") 
            dset_X = grp.create_dataset('X', data = X_new)
            dset_Y = grp.create_dataset('Y', data = Y_new)
            dset_X.attrs["cols"] = np.append(col_names_X, 'dt')
            dset_Y.attrs["cols"] = col_names_Y
            f.close()
            print('WRITTEN', rank)
            
        comm.Barrier()
    
    return 'Done'