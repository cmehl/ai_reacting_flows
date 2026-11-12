import os
import oyaml as yaml

import shutil
from time import perf_counter

from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud

from ai_reacting_flows.tools.utilities import PRINT

import h5py

from mpi4py import MPI

h5py.get_config().mpi

# Mixing models of ParticlesCloud
MIXING_MODELS = ["CURL", "CURL_MODIFIED", "CURL_MODIFIED_DD", "EMST_SIMPLIFIED", "EMST_POPE"]


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

    # Mixing model (checked on all ranks: they all stop on the same error, none waits for the others)
    if data_gen_parameters["mixing_model"] not in MIXING_MODELS:
        raise ValueError(f"mixing_model must be one of {MIXING_MODELS}, got {data_gen_parameters['mixing_model']}")


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
    
    while (particle_cloud.time<data_gen_parameters["time_max"] and (not particle_cloud.stats_converged)):
        
        # Advancing particles state by one dt
        dt = particle_cloud.dt[particle_cloud.iteration]    # to account for possible multi-dt
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

    if rank==0:

        # Copy of the input parameters in the results folder (rank 0 only: one writer of the file)
        shutil.copy(os.path.join(run_folder, "dtb_params.yaml"), f"{run_folder:s}/STOCH_DTB_{data_gen_parameters['results_folder_suffix']}/dtb_params.yaml")

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