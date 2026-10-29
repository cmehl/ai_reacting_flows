"""
Helper for tests: small H2 stochastic reactors database (tests 01 to 04), built by the session fixtures
of conftest.py in a temporary folder.

generate_h2_stoch_database: 500 particles (lean H2/air, 450 fresh and 50 burnt), CURL_MODIFIED mixing,
mixing pairs read from pairs_list.pkl / CURL_rate_list.pkl for reproducibility, written in
STOCH_DTB_TEST/solutions.h5.
resample_h2_stoch_database: temperature threshold and HRR-based undersampling in (PC1, PC2) space,
written in STOCH_DTB_TEST/solutions_resamp200K.h5.
"""
import os
import shutil
import oyaml as yaml
from ai_reacting_flows.stochastic_reactors_data_gen.main import generate_stochastic_database
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from misc import fHRR

import DummyComm as dc

tests_dir = os.path.dirname(os.path.abspath(__file__))
mech_file = os.path.join(tests_dir, "../data/chemical_mechanisms/mech_H2.yaml")

# Stochastic reactors inlets
inlet_1 = {'type':'cold_premixed',
           'nb_particles': 450,
           'fuel': 'H2',
           'phi': 0.6,
           'T': 300.0,
           'P': 101325.0}
inlet_2 = {'type':'burnt_premixed',
           'nb_particles': 50,
           'fuel': 'H2',
           'phi': 0.6,
           'T': 300.0,
           'P': 101325.0}

# Database generation parameters
data_gen_parameters = {}

# General parameters
data_gen_parameters["mech_file"] = mech_file      # Mechanism file
data_gen_parameters["inlets"] = {'inlet_1': inlet_1, 'inlet_2': inlet_2}      # dict defining the stochastic reactors inlets
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
data_gen_parameters["pv_species"] = ["H2O"]                 # Species used to define progress variable
data_gen_parameters["fuel"] = ["H2"]

# post-processing parameters
data_gen_parameters["calc_mean_traj"] = False              # Flag to decide if mean trajectories are computed in the post-processing step

# ML inference
data_gen_parameters["ML_inference_flag"] = False

# Resampling parameters (LearningDatabase)
resampling_parameters = {}

# Input / output database files
resampling_parameters["database_params"] = {}
resampling_parameters["database_params"]["database_type"] = "stoch"              # Type of database
resampling_parameters["database_params"]["dtb_folder_suffix"] = "TEST"       # Stochastic reactors simulation folder
resampling_parameters["database_params"]["database_name"] = "dtb_resampled"   # Resulting database name
resampling_parameters["database_params"]["dtb_file"] = "solutions.h5"
resampling_parameters["database_params"]["dt_var"] = False
resampling_parameters["database_params"]["fuel"] = "H2"
resampling_parameters["database_params"]["mech_file"] = mech_file

# Data processing
resampling_parameters["data_processing"] = {}
resampling_parameters["data_processing"]["log_transform_X"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
resampling_parameters["data_processing"]["log_transform_Y"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
resampling_parameters["data_processing"]["threshold"] = 1.0e-10            # Threshold to be applied in case of logarithm transform
resampling_parameters["data_processing"]["T_threshold"] = 200.0

# Physics described: (omega vs Y) and N chemistry
resampling_parameters["data_processing"]["output_omegas"] = False          # True: output differences, False: output mass fractions
resampling_parameters["data_processing"]["with_N_chemistry"] = False        # Considering Nitrogen chemistry or not (if not, N not considered in atom balance for reduction). In MLP, it will change treatment of N2.

# Clustering
resampling_parameters["data_clustering"] = {}
resampling_parameters["data_clustering"]["clustering_method"] = "kmeans"
resampling_parameters["data_clustering"]["nb_clusters"] = 2
resampling_parameters["data_clustering"]["clusterize_on"] = 'phys'

resampling_parameters["train_set_size"] = 0.75

# Resampled database file, in STOCH_DTB_TEST/
resampled_file = "solutions_resamp200K.h5"


def generate_h2_stoch_database(folder):
    """Generate STOCH_DTB_TEST/solutions.h5 in folder (run folder, with the mixing pairs files)."""

    os.chdir(folder)
    for name in ["pairs_list.pkl", "CURL_rate_list.pkl"]:
        shutil.copy(os.path.join(tests_dir, name), folder)

    with open("dtb_params.yaml", "w") as file:
        yaml.dump(data_gen_parameters, file, default_flow_style=False)

    generate_stochastic_database(dc.DummyComm())


def resample_h2_stoch_database(folder):
    """Write STOCH_DTB_TEST/solutions_resamp200K.h5 from STOCH_DTB_TEST/solutions.h5 in folder."""

    os.chdir(folder)
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(resampling_parameters, file, default_flow_style=False)

    database = LearningDatabase()
    database.apply_temperature_threshold()
    database.undersample_HRR("PC1", "PC2", hrr_func = fHRR, keep_low_c = True, n_samples = 25000, n_bins = 100, plot_distrib = True)
    database.database_to_h5(os.path.join(folder, "STOCH_DTB_TEST") + "/", resampled_file)
