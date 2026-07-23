from ai_reacting_flows.databases_processing.database_processing import LearningDatabase

# Function applied on the HRR
def fHRR(x):
    return 1.0+5.0*x

database = LearningDatabase()
database.density_scatter("PC1","PC2")
database.apply_temperature_threshold()

# database.undersample_HRR("Temperature", "H2O", hrr_func = fHRR, keep_low_c = True, n_samples = 500000, n_bins = 100, plot_distrib = True)
# database.undersample_1D("Temperature", seed=1991, plot_distrib=True)
database.undersample_2D("PC1","PC2", seed=1991, min_count=300, plot_distrib=False)

database.density_scatter("PC1","PC2")
database.compare_resampled_pdfs("Temperature")
database.clusterize_dataset() # Log_transform (if required) > Scale (StandardScaler, same for all dtb) > Clusterize
database.visualize_clusters("PC1","PC2")
database.visualize_clusters("NH3","Temperature")
database.print_data_size()
database.process_database(plot_distributions=True, distribution_species=["Temperature","NH3"])