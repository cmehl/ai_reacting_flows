import os
import numpy as np
from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase
from misc import fHRR, extract_h2_dtvar_dtb_histograms, check_h2_training_histograms

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

def test_dtb_resampling():
    # required for CI on GitHub
    current_dir = os.path.dirname(__file__)
    os.chdir(current_dir)
    dtb_processing_parameters["mech_file"] = f"{current_dir:s}{dtb_processing_parameters['mech_file']:s}"

    database = LearningDatabase(dtb_processing_parameters)

    # T_threshold should be in attribute for check (threshold_dtb < threshold_ppro) and warning (threshold_dtb != threshold_ppro)
    database.apply_temperature_threshold(dtb_processing_parameters["T_threshold"]) # input should be optional (default = threshold_dtb)

    # To be removed for actual CI
    # database.plot_pdf_var('Temperature')
    # database.plot_pdf_var('HRR')

    # database.print_data_size()
    # End : to be removed

    # Check n_samples, needed ? (current > npoints in database)
    database.undersample_HRR("PC1", "PC2", hrr_func = fHRR, keep_low_c = True, n_samples = 25000, n_bins = 100, plot_distrib = True) # set distrib to false for CI

    # To be removed for actual CI
    # database.plot_pdf_var('Temperature')
    # database.plot_pdf_var('HRR')

    # database.print_data_size()
    # End : to be removed

    database.database_to_h5(f"{current_dir:s}/STOCH_DTB_{dtb_processing_parameters['results_folder_suffix']:s}/", f"{dtb_processing_parameters['dtb_file'].split('.')[0]:s}_resamp{int(dtb_processing_parameters['T_threshold'])}K.h5")

    tol = np.array([35.71, 2.62, 11.83, 5.0, 28.87, 12.84, 10.97, 3.56, 2.57])
    assert np.max(extract_h2_dtvar_dtb_histograms(f"./STOCH_DTB_{dtb_processing_parameters['results_folder_suffix']:s}/{dtb_processing_parameters['dtb_file'].split('.')[0]:s}_resamp{int(dtb_processing_parameters['T_threshold'])}K.h5","H2_resampled_TEST_histograms", tol)) < np.float64(0.0)