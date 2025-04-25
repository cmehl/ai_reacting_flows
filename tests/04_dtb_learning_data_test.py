import os
import oyaml as yaml
import numpy as np
from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase
from misc import check_h2_training_histograms

# Dictionary to store data processing parameters
dtb_processing_parameters = {}

# Input / output database files
dtb_processing_parameters["dtb_folder_suffix"] = "TEST"       # Stochastic reactors simulation folder
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
dtb_processing_parameters["clustering_method"] = "kmeans"
dtb_processing_parameters["nb_clusters"] = 1
dtb_processing_parameters["clusterize_on"] = 'phys'
dtb_processing_parameters["train_set_size"] = 0.75

def test_dtvar_dtb_processing():
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    database = LearningDatabase()
    database.process_database()

    tol = []
    tol.append(np.array([43.32, 29.68, 20.52, 22.88, 56.19, 26.34, 40.55, 22.9, 20.68]))
    tol.append(np.array([31.14, 19.51, 22.92, 58.46, 26.6, 40.58, 22.56, 19.82]))
    tol.append(np.array([15.82, 10.69, 7.92, 8.32, 19.92, 8.42, 12.74, 8.0, 7.06]))
    tol.append(np.array([10.76, 7.95, 9.16, 20.1, 10.25, 14.15, 8.52, 8.15]))
    assert check_h2_training_histograms(f"./STOCH_DTB_{dtb_processing_parameters['dtb_folder_suffix']:s}/dtb_resampled/cluster0","H2_TEST_training_data", tol)