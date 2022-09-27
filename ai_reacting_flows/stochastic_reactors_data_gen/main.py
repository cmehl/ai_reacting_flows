import os

import shutil
from time import perf_counter

from ai_reacting_flows.stochastic_reactors_data_gen.particles_cloud import ParticlesCloud
from ai_reacting_flows.tools.utilities import PRINT

# Disabling TF warning (in particular warnings related to unused CPU features)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def generate_stochastic_database(data_gen_parameters, comm):

    # Initial computation time
    t_ini = perf_counter()

    rank = comm.Get_rank()

    #==========================================================================
    # CATCHING SOME INPUT ERRORS
    #==========================================================================

    if rank==0:

        if (data_gen_parameters["mixing_model"] not in ["CURL","CURL_MODIFIED","EMST"]):
            # Mixing model choice
            try:
                raise ValueError("Input error.")
            except ValueError:
                print("")
                print(">> case_type entry should be one of the following:")
                print("     - CURL")
                print("     - CURL_MODIFIED")
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
        os.mkdir(results_folder + "/figures")

        # Copying input files to results folder to document run
        shutil.copy("./" + data_gen_parameters["inlets_file"], results_folder)
        shutil.copy(data_gen_parameters["mech_file"], results_folder)


    # Particle cloud initialization
    particle_cloud = ParticlesCloud(data_gen_parameters, comm)

    #==============================================================================
    # PARTICLE CLOUD TIME MARCHING
    #==============================================================================

    # Initializations
    dt = data_gen_parameters["time_step"]
    time = 0.0

    while (time<data_gen_parameters["time_max"] and particle_cloud.stats_converged==False):
        
        # Advancing particles state by one dt
        time += dt
        particle_cloud.advance_time(dt)

    # End of simulation operations
    if rank==0:
    
        # Export learning datasets (X and Y)
        if data_gen_parameters["build_ml_dtb"]:
            particle_cloud.export_learning_datasets()
            shutil.move("./X_dtb.csv", results_folder + "/X_dtb.csv")
            shutil.move("./Y_dtb.csv", results_folder + "/Y_dtb.csv")
    
        # Save pandas dataframe with all states
        particle_cloud.export_database()
        shutil.move("./database_states.csv", results_folder + "/database_states.csv")

        # Save mean trajectories
        if data_gen_parameters["calc_mean_traj"]:
            particle_cloud.export_trajectories()
            shutil.move("./mean_trajectories.h5", results_folder + "/mean_trajectories.h5")

        # Plot some statistics on particles
        particle_cloud.plot_stats()

        # Plot final database pdf's
        if data_gen_parameters["ML_inference_flag"]==False:
            particle_cloud.plot_pdf_dtb_final()


        # Plot some graphs to represent the data
        particle_cloud.plot_TZ_scatter_inst()
        particle_cloud.plot_TZ_scatter_all()
        if data_gen_parameters["calc_mean_traj"]:
            particle_cloud.plot_TZ_trajectories()
            particle_cloud.plot_T_time_trajectories()
        if data_gen_parameters["ML_inference_flag"]==False:
            particle_cloud.plot_pdf_T_inst()
            particle_cloud.plot_pdf_T_all()



    # Number of chemical states in database
    nb_states_dtb = particle_cloud.df_all_states.shape[0]

    #==============================================================================
    # ENDING THE SIMULATION
    #==============================================================================

    # End computation time
    t_end = perf_counter()

    if rank==0:
        #  End of simulation printing
        PRINT("")
        PRINT("END OF SIMULATION")
        PRINT("Database:")
        PRINT(f"  >> Number of chemical states stored in database: {nb_states_dtb:d}")
        PRINT("CPU costs:")
        PRINT(f"Total CPU time of simulation: {t_end-t_ini:4.3f} s")
        PRINT(f"  >> Cumulated time spent in simulation for diffusion: {particle_cloud.timings['Diffusion'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for reactions: {particle_cloud.timings['Reaction'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for trajectories averaging: {particle_cloud.timings['MeanTraj'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for writing results: {particle_cloud.timings['Writing'].sum():5.4f} s")
        PRINT(f"  >> Cumulated time spent in simulation for plotting results: {particle_cloud.timings['Plotting'].sum():5.4f} s")
        PRINT("")

