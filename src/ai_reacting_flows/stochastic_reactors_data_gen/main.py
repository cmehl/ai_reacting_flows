import os
import oyaml as yaml

import shutil
from time import perf_counter

from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

from ai_reacting_flows.tools.utilities import PRINT

import numpy as np
import h5py
import pandas as pd
import cantera as ct
import scipy as sc

from mpi4py import MPI

h5py.get_config().mpi


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

        # Remove existing results dir if already exists
        results_folder = "STOCH_DTB_" + data_gen_parameters["results_folder_suffix"]
        if os.path.isdir(results_folder):
            PRINT("WARNING: existing results directory has been removed. \n")
            shutil.rmtree(results_folder)
        os.mkdir(results_folder)

        # Copying input files to results folder to document run
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