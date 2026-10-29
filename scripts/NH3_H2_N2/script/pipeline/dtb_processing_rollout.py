from ai_reacting_flows.databases_processing.database_processing import LearningDatabase

# States already = the mc50000 T-undersampled set (preselect_T1D.py): no further undersampling.
database = LearningDatabase()
database.apply_temperature_threshold()
database.clusterize_dataset()
database.process_database(plot_distributions=True, distribution_species=["Temperature", "H2", "H2O", "OH"])
