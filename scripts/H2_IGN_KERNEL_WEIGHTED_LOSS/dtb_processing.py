from ai_reacting_flows.databases_processing.database_processing import LearningDatabase

database = LearningDatabase()
# database.density_scatter("PC1","PC2")
database.apply_temperature_threshold()
# Same number of points in each bin of log10(Omega), the norm of the scaled increments used by the loss weights.
# Omega_min: floor of Omega [-] (equilibrium, solver noise); bins with fewer than min_count points are kept whole
database.undersample_Omega(Omega_min = 1.0e-4, min_count = 2000, plot_distrib = False)
# database.density_scatter("PC1","PC2")
database.compare_resampled_pdfs("Temperature")
database.scatter_Omega_progvar()
database.clusterize_dataset() # Log_transform (if required) > Scale (StandardScaler, same for all dtb) > Clusterize
database.process_database(plot_distributions=True, distribution_species=["Temperature","H2","H2O","OH"])
