
from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase

# %%
# Dictionary to store data processing parameters
dtb_processing_parameters = {}

dtb_processing_parameters["dtb_folder"] = "../scripts/STOCH_DTB_HOTSPOT_H2_HRR"       # Stochastic reactors simulation folder
dtb_processing_parameters["database_name"] = "test" # "database_log_log_resampled_kmeans_3clusters"                   # Resulting database name
dtb_processing_parameters["log_transform_X"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
dtb_processing_parameters["log_transform_Y"] = 1              # 0: no transform, 1: Logarithm transform, 2: Box-Cox transform
dtb_processing_parameters["threshold"] = 1.0e-10            # Threshold to be applied in case of logarithm transform
dtb_processing_parameters["output_omegas"] = True          # True: output differences, False: output mass fractions
dtb_processing_parameters["detailed_mechanism"] = "/work/mehlc/2_IA_KINETICS/ai_reacting_flows/data/chemical_mechanisms/mech_H2.yaml"        # Mechanism used for the database generation (/!\ YAML format)
dtb_processing_parameters["fuel"] = "H2"           # Fuel name
dtb_processing_parameters["with_N_chemistry"] = False        # Considering Nitrogen chemistry or not (if not, N not considered in atom balance for reduction). In MLP, it will change treatment of N2.


# Read database
database = LearningDatabase(dtb_processing_parameters)

# Temperature threshold
database.apply_temperature_threshold(600.0)


# # chemistry reduction
# fictive_species = ["O2", "H2O", "CO2", "CH4"]
# subset_species = ["N2", "O2", "H2O", "CO2", "CH4", "CO"]
# database.reduce_species_set(subset_species, fictive_species)

# # 
# print(database.X[subset_species + [spec+"_F" for spec in fictive_species]].sum(axis=1))

# Plot point density of original dataset
database.density_scatter("Temperature" , "H2O", sort = True, bins = 100)

database.plot_pdf_var("Temperature")


# Function applied on the HRR
def f(x):
    return 1.0 + 5.0*x
    # return x

database.undersample_HRR("PC1", "PC2", hrr_func = f, keep_low_c = True, n_samples = 1000000, n_bins = 100, plot_distrib = False)

# Plot point density of resampled dataset
database.density_scatter("PC1", "PC2", sort = True, bins = 100)

database.plot_pdf_var("Temperature")


# 
database.print_data_size()

# K-means clustering
database.clusterize_dataset("kmeans", 2)

# 
database.visualize_clusters("PC1", "PC2")

# 
database.print_data_size()

# Generate ML database
database.process_database(plot_distributions = True, distribution_species=["Temperature", "O2", "H2"])

