from ai_reacting_flows.databases_processing.database_processing import LearningDatabase

# Function applied on the HRR
def fHRR(x):
    return 1.0+5.0*x

database = LearningDatabase()
database.density_scatter("PC1","PC2")
# database.apply_temperature_threshold()
database.pdf_Omega()                 # PDFs of Omega and log10(Omega), the norm of the scaled increments of the loss weights
database.scatter_Omega_progvar()     # log10(Omega) versus progress variable, colored by temperature
# database.undersample_HRR("Temperature", "H2O", hrr_func = fHRR, keep_low_c = True, n_samples = 1000000, n_bins = 100, plot_distrib = True)
# # database.density_scatter("PC1","PC2")
# database.compare_resampled_pdfs("Temperature")
# database.clusterize_dataset() # Log_transform (if required) > Scale (StandardScaler, same for all dtb) > Clusterize
database.process_database(plot_distributions=True, distribution_species=["Temperature","H2","H2O","OH"])