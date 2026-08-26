from ai_reacting_flows.databases_processing.database_processing import LearningDatabase

database = LearningDatabase()
database.apply_temperature_threshold()
database.undersample_1D("Temperature")
database.compare_resampled_pdfs("Temperature")
database.clusterize_dataset()  # Log_transform (if required) > Scale (StandardScaler, same for all dtb) > Clusterize
database.process_database(plot_distributions=True, distribution_species=["Temperature", "H2", "H2O", "OH"])
