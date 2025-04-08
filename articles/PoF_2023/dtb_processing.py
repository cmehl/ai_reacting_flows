from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase

# Function applied on the HRR
def fHRR(x):
    return 1.0+5.0*x

# Dictionary to store data processing parameters
dtb_processing_parameters = {}

# Input / output database files
dtb_processing_parameters["results_folder_suffix"] = "H2_POF_2023"       # Stochastic reactors simulation folder
dtb_processing_parameters["database_name"] = "dtb_resampled" # "database_log_log_resampled_kmeans_3clusters"                   # Resulting database name
dtb_processing_parameters["dtb_file"] = "solutions.h5"

# Data processing
dtb_processing_parameters["log_transform_X"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
dtb_processing_parameters["log_transform_Y"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
dtb_processing_parameters["threshold"] = 1.0e-10            # Threshold to be applied in case of logarithm transform
dtb_processing_parameters["T_threshold"] = 600.0

# Physics described: (omega vs Y) and N chemistry
dtb_processing_parameters["output_omegas"] = False   # True: output differences, False: output mass fractions
dtb_processing_parameters["with_N_chemistry"] = False  # Considering Nitrogen chemistry or not (if not, N not considered in atom balance for reduction). In MLP, it will change treatment of N2.

# NN training related processings
dtb_processing_parameters["clusterize_on"] = 'phys'
dtb_processing_parameters["train_set_size"] = 0.75

dtb_processing_parameters["mech_file"] = "../../data/chemical_mechanisms/mech_H2.yaml"        # Mechanism used for the database generation (/!\ YAML format)
dtb_processing_parameters["fuel"] = "H2"           # Fuel name
dtb_processing_parameters['dt_var'] = False

database = LearningDatabase(dtb_processing_parameters)
# database.density_scatter("PC1","PC2")
database.apply_temperature_threshold(dtb_processing_parameters["T_threshold"])
database.undersample_HRR("Temperature", "H2O", hrr_func = fHRR, keep_low_c = True, n_samples = 1000000, n_bins = 100, plot_distrib = True)
# database.density_scatter("PC1","PC2")
database.compare_resampled_pdfs("Temperature")
database.clusterize_dataset("kmeans", 2) # Log_transform (if required) > Scale (StandardScaler, same for all dtb) > Clusterize
database.process_database(plot_distributions=True, distribution_species=["Temperature","H2","H2O","OH"])