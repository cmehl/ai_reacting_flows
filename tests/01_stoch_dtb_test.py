from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database, GenerateVariable_dt
from misc import extract_h2_dtb_histograms, extract_h2_dtvar_dtb_histograms

import DummyComm as dc
import numpy as np
import os

# Dictionary to store inputs
data_gen_parameters = {}

# General parameters
data_gen_parameters["mech_file"] = "/../data/chemical_mechanisms/mech_H2.yaml"      # Mechanism file
data_gen_parameters["inlets_file"] = "/inlets_file.xlsx"      # file defining the stochastic reactors inlets
data_gen_parameters["results_folder_suffix"] = "TEST"    # Name of the resulting database (used as a suffix of "STOCH_DTB")
data_gen_parameters["dtb_file"] = "solutions.h5" #
data_gen_parameters["build_ml_dtb"] = True     # Flag to generate ML database or do a stand-alone simulation
data_gen_parameters["time_step"] = 5.0e-6       # Constant time step of the simulation
data_gen_parameters["time_max"] = 1e-3          # Simulation end time

# Multiple dt parameters
data_gen_parameters['time_step_type'] = 'random'  # set or random, if 'set': input the values of dt in a table. If 'random' : input the range (min and max)
data_gen_parameters['time_step_range'] = [0.1e-6, 1e-6]
data_gen_parameters['nb_dt'] = 5 # number of dt to choose in the range for each data point (only if time_step_type = 'random')
data_gen_parameters['new_file_name'] = f"mech_h2_{data_gen_parameters['nb_dt']}dt.h5"

# Modeling parameters
data_gen_parameters["mixing_model"] = "CURL_MODIFIED"         # Mixing model (CURL or CURL_MODIFIED)
data_gen_parameters["mixing_time"] = 1.0e-4                 # Mixing time-scale
data_gen_parameters["read_mixing"] = True                 # Use pre-computed mixing pairs
data_gen_parameters["T_threshold"] = 200.0                  # Temperature threshold for database
data_gen_parameters["calc_progvar"] = True                  # Flag to decide if progress variable is computed
data_gen_parameters["pv_species"] = ["H2O"]                 # Species used to define progress variable

# post-processing parameters
data_gen_parameters["calc_mean_traj"] = False              # Flag to decide if mean trajectories are computed in the post-processing step

# ML inference
data_gen_parameters["ML_inference_flag"] = False
data_gen_parameters["ML_models"] = ("")
data_gen_parameters["prog_var_thresholds"] = (0.25, 0.95)

def test_h2_dtb_computation():
    # required for CI on GitHub
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    tmp_mech_file = data_gen_parameters["mech_file"]
    tmp_inlets_file = data_gen_parameters["inlets_file"]
    data_gen_parameters["mech_file"] = f"{current_dir:s}{data_gen_parameters['mech_file']:s}"
    data_gen_parameters["inlets_file"] = f"{current_dir:s}{data_gen_parameters['inlets_file']:s}"

    # Call to database generation function
    generate_stochastic_database(data_gen_parameters, dc.DummyComm())

    data_gen_parameters["inlets_file"] = tmp_inlets_file  # required for multiple function in same file
    data_gen_parameters["mech_file"] = tmp_mech_file  # required for multiple function in same file

    tol = np.array([139.75, 3.74, 50.4, 4.94, 166.15, 9.9, 124.38, 5.9, 4.62, 2.04])
    assert np.max(extract_h2_dtb_histograms(f"./STOCH_DTB_{data_gen_parameters['results_folder_suffix']:s}/{data_gen_parameters['dtb_file']:s}","H2_TEST_histograms", tol)) < np.float64(0.0)

def test_h2_multi_dt_dtb_computation():
    # required for CI on GitHub
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    data_gen_parameters["mech_file"] = f"{current_dir:s}{data_gen_parameters['mech_file']:s}"
    data_gen_parameters["inlets_file"] = f"{current_dir:s}{data_gen_parameters['inlets_file']:s}"

    GenerateVariable_dt(data_gen_parameters, dc.DummyComm())

    tol = np.array([168.86, 8.89, 50.17, 20.32, 139.32, 52.21, 61.66, 13.04, 11.16])
    assert np.max(extract_h2_dtvar_dtb_histograms(f"./STOCH_DTB_{data_gen_parameters['results_folder_suffix']:s}/mech_h2_5dt.h5","H2_dtvar_TEST_histograms", tol)) < np.float64(0.0)