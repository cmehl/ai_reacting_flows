# This script is used to call the stochastic reactors database generation. It is written as a python script rather than a notebook 
# in order to be parallelized with MPI processes.


from mpi4py import MPI

from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database


# MPI world communicator
comm = MPI.COMM_WORLD

# Dictionary to store inputs
data_gen_parameters = {}

# General parameters
data_gen_parameters["mech_file"] = "/work/mehlc/2_IA_KINETICS/ai_reacting_flows/data/chemical_mechanisms/mech_CH4_gri.yaml"      # Mechanism file
data_gen_parameters["inlets_file"] = "inlets_file.xlsx"      # file defining the stochastic reactors inlets
data_gen_parameters["results_folder_suffix"] = "PREMIXED_CH4_TEST"    # Name of the resulting database (used as a suffix of "STOCH_DTB")
data_gen_parameters["build_ml_dtb"] = True     # Flag to generate ML database or do a stand-alone simulation
data_gen_parameters["time_step"] = 0.5e-6       # Constant time step of the simulation
data_gen_parameters["time_max"] = 4e-6          # Simulation end time

# Modeling parameters
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"         # Mixing model (CURL or CURL_MODIFIED)
data_gen_parameters["mixing_time"] = 0.4e-3                 # Mixing time-scale
data_gen_parameters["T_threshold"] = 200.0                  # Temperature threshold for database
data_gen_parameters["calc_progvar"] = True                  # Flag to decide if progress variable is computed
data_gen_parameters["pv_species"] = ["CO", "CO2"]                 # Species used to define progress variable

# post-processing parameters
data_gen_parameters["calc_mean_traj"] = True              # Flag to decide if mean trajectories are computed in the post-processing step

# ML database parameters
data_gen_parameters["dt_ML"] = 0.5e-6                      # Time step of the ML database (may be different from the simulation time step)

# ML inference             # WORK IN PROGRESS
data_gen_parameters["ML_inference_flag"] = False
data_gen_parameters["ML_models"] = ("MODEL_H2_phi04_CURL_1l_200u_tanh_n2_cte_newdtb")
data_gen_parameters["prog_var_thresholds"] = (0.25, 0.95)



# Call to database generation function
generate_stochastic_database(data_gen_parameters, comm)



