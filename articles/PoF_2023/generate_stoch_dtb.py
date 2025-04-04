from mpi4py import MPI
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database

# MPI world communicator
comm = MPI.COMM_WORLD

# Dictionary to store inputs
data_gen_parameters = {}

# General parameters
data_gen_parameters["mech_file"] = "../../data/chemical_mechanisms/mech_H2.yaml"      # Mechanism file
data_gen_parameters["inlets_file"] = "inlets_file.xlsx" # file defining the stochastic reactors inlets
data_gen_parameters["results_folder_suffix"] = "H2_POF_2023"   # Name of the resulting database (used as a suffix of "STOCH_DTB")
data_gen_parameters["dtb_file"] = "solutions.h5"
data_gen_parameters["build_ml_dtb"] = True     # Flag to generate ML database or do a stand-alone simulation
data_gen_parameters["time_step"] = 0.5e-6       # Constant time step of the simulation
data_gen_parameters["time_max"] = 4e-3          # Simulation end time

# Modeling parameters
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"         # Mixing model (CURL or CURL_MODIFIED)
data_gen_parameters["mixing_time"] = 0.4e-3                 # Mixing time-scale
data_gen_parameters["read_mixing"] = False                 # Use pre-computed mixing pairs
data_gen_parameters["T_threshold"] = 600.0                  # Temperature threshold for database
data_gen_parameters["calc_progvar"] = True                  # Flag to decide if progress variable is computed
data_gen_parameters["pv_species"] = ["H2O"]                 # Species used to define progress variable

# post-processing parameters
data_gen_parameters["calc_mean_traj"] = False              # Flag to decide if mean trajectories are computed in the post-processing step

# ML database parameters
data_gen_parameters["dt_ML"] = 0.5e-6                      # Time step of the ML database (may be different from the simulation time step)

# ML inference
data_gen_parameters["ML_inference_flag"] = False

# Call to database generation function
generate_stochastic_database(data_gen_parameters, comm)



