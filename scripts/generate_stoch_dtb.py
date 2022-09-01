from mpi4py import MPI

from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database


# MPI world communicator
comm = MPI.COMM_WORLD

# Dictionary to store inputs
data_gen_parameters = {}

# General parameters
data_gen_parameters["mech_file"] = "/work/mehlc/2_IA_KINETICS/ai_reacting_flows/data/chemical_mechanisms/mech_H2.cti"
data_gen_parameters["inlets_file"] = "inlets_file.xlsx"
data_gen_parameters["results_folder_suffix"] = "hotspot_H2_DEV"
data_gen_parameters["build_ml_dtb"] = True
data_gen_parameters["time_step"] = 0.5e-6
data_gen_parameters["time_max"] = 1e-6
data_gen_parameters["nb_procs"] = 2

# Modeling parameters
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"
data_gen_parameters["mixing_time"] = 0.4e-3
data_gen_parameters["T_threshold"] = 200.0
data_gen_parameters["calc_progvar"] = True
data_gen_parameters["pv_species"] = ["H2O"]

# post-processing parameters
data_gen_parameters["plot_freq"] = 999999
data_gen_parameters["calc_mean_traj"] = True

# ML database parameters
data_gen_parameters["dt_ML"] = 0.5e-6

# ML inference 
data_gen_parameters["ML_inference_flag"] = False
data_gen_parameters["ML_models"] = ("MODEL_H2_phi04_CURL_1l_200u_tanh_n2_cte_newdtb")
data_gen_parameters["prog_var_thresholds"] = (0.25, 0.95)



# Call to database generation function
generate_stochastic_database(data_gen_parameters, comm)



