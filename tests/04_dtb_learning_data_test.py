import os
import numpy as np
from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase
from misc import check_h2_training_histograms

# Dictionary to store data processing parameters
dtb_processing_parameters = {}

# Input / output database files
dtb_processing_parameters["results_folder_suffix"] = "TEST"       # Stochastic reactors simulation folder
dtb_processing_parameters["database_name"] = "dtb_resampled" # "database_log_log_resampled_kmeans_3clusters"                   # Resulting database name
dtb_processing_parameters["dtb_file"] = "solutions.h5"

# Data processing
dtb_processing_parameters["log_transform_X"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
dtb_processing_parameters["log_transform_Y"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
dtb_processing_parameters["threshold"] = 1.0e-10            # Threshold to be applied in case of logarithm transform
dtb_processing_parameters["T_threshold"] = 200.0

# Physics described: (omega vs Y) and N chemistry
dtb_processing_parameters["output_omegas"] = False          # True: output differences, False: output mass fractions
dtb_processing_parameters["with_N_chemistry"] = False        # Considering Nitrogen chemistry or not (if not, N not considered in atom balance for reduction). In MLP, it will change treatment of N2.

# NN training related processings
dtb_processing_parameters["clusterize_on"] = 'phys'
dtb_processing_parameters["train_set_size"] = 0.75

# to be read from database file (to be added in attributes)
dtb_processing_parameters["mech_file"] = "/../data/chemical_mechanisms/mech_H2.yaml"        # Mechanism used for the database generation (/!\ YAML format)
dtb_processing_parameters["fuel"] = "H2"           # Fuel name
dtb_processing_parameters['dt_var'] = False

def test_dtvar_dtb_processing():
    # required for CI on GitHub
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    dtb_processing_parameters["mech_file"] = f"{current_dir:s}{dtb_processing_parameters['mech_file']:s}"

    database = LearningDatabase(dtb_processing_parameters)
    database.process_database()

    tol = []
    tol.append(np.array([102.17, 105.63, 29.49, 64.26, 327.54, 66.86, 60.28, 428.57, 74.37, 77.45]))
    tol.append(np.array([110.47, 30.55, 65.14, 339.27, 68.61, 61.46, 427.93, 76.23, 80.23]))
    tol.append(np.array([33.25, 35.6, 11.43, 22.57, 110.85, 22.97, 20.3, 142.02, 25.08, 25.32]))
    tol.append(np.array([37.1, 10.95, 22.37, 112.63, 22.68, 20.57, 141.72, 26.18, 26.39]))
    assert check_h2_training_histograms(f"./STOCH_DTB_{dtb_processing_parameters['results_folder_suffix']:s}/dtb_resampled/cluster0","H2_TEST_training_data", tol)