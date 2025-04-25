import os
import oyaml as yaml
import numpy as np
from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase
from misc import fHRR, extract_h2_dtvar_dtb_histograms

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
dtb_processing_parameters["nb_clusters"] = 2
dtb_processing_parameters["clusterize_on"] = 'phys'
dtb_processing_parameters["train_set_size"] = 0.75

def test_dtb_resampling():
    # required for CI on GitHub
    current_dir = os.path.dirname(__file__)

    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    database = LearningDatabase()

    database.apply_temperature_threshold()

    # Check n_samples, needed ? (current > npoints in database)
    database.undersample_HRR("PC1", "PC2", hrr_func = fHRR, keep_low_c = True, n_samples = 25000, n_bins = 100, plot_distrib = True) # set distrib to false for CI

    database.database_to_h5(f"{current_dir:s}/STOCH_DTB_{dtb_processing_parameters['dtb_folder_suffix']:s}/", f"{dtb_processing_parameters['dtb_file'].split('.')[0]:s}_resamp{int(dtb_processing_parameters['T_threshold'])}K.h5")

    tol = np.array([35.71, 2.62, 11.83, 5.0, 28.87, 12.84, 10.97, 3.56, 2.57])
    assert np.max(extract_h2_dtvar_dtb_histograms(f"./STOCH_DTB_{dtb_processing_parameters['dtb_folder_suffix']:s}/{dtb_processing_parameters['dtb_file'].split('.')[0]:s}_resamp{int(dtb_processing_parameters['T_threshold'])}K.h5","H2_resampled_TEST_histograms", tol)) < np.float64(0.0)