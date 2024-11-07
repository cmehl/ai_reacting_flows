# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: env_ia
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Generation of learning database from a stochastic reactors simulation
#
# This notebook builds a database to be used as a training database for the ML algorithm. In order for this script to be used, a stochastic reactors simulation with *build_ML_dtb=True* must have neen run beforehand. This simulation produces files *X.csv* and *Y.csv* with raw states $T$, $Y_k$ and necessary information to clusterize or re-sample the data (e.g. progress variable, heat release rate,...). 
#
# The current script generate the final database and enables to choose several options:
#
# + Prediction of $Y_k(t+dt)$ or $Y_k(t+dt)-Y_k(t)$
#
# + Application of a transform such as logarithm or Box-Cox
#
# + Possibility to apply a temperature threshold to the data to avoid non-reacting zones
#
# + Possibility to clusterize the data based on (i) k-means algorithm; (ii) progress variable values.
#
# + Re-sampling of the database: if the database is too imbalanced, a strategy based on heat release rate (HRR) may be used to focus the data on the regions of interest.
#
# Files *X_train*, *Y_train*, *X_val* and *Y_val* are created for each cluster. Note that if no clustering is applied, the default single cluster is cluster 0.

# %% [markdown]
# ## Data reading

# %%
from ai_reacting_flows.stochastic_reactors_data_gen.database_processing import LearningDatabase

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# The parameters of the database processing are first set:

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

# %% [markdown]
# The database is then created as a *LearningDatabase* object:

# %%
database = LearningDatabase(dtb_processing_parameters)

# %% [markdown]
# ## Data simple filtering

# %% [markdown]
# We can apply in a first step some simple filters to reduce the size of the dataset. Reducing the number of data points at this early stage leads to faster database treatment and may be recommended. 
#
# We can for instance apply a temperature threshold, here $600$ K for instance:

# %%
database.apply_temperature_threshold(600.0)

# %% [markdown]
# ## Reduction of the species space

# %% [markdown]
# We have the possibility to apply ANN only to a reduced subset of species. In order to preserve atomic masses and enthalpy, a set of fictive species is added to the database. Their mass fractions are computed so that conservation of quantities is ensured. In order for the problem to have a solution, the following rules must be obeyed when selecting the fictive species:
#
# + The number of fictive species must be *number of atoms + 1* (for the enthalpy). At the moment, the number of atoms is 4 ($C$, $H$, $O$, $N$), except for $H_2$, where carbon is not considered. Another possibility is to discard $N$, this is done by setting the *with_N_chemistry* parameter above to *False*.
#
# + Each atom must be represented at least once.
#
# The reduction operation can be done with the following lines:

# %%
fictive_species = ["O2", "H2O", "CO2", "CH4"]
subset_species = ["N2", "O2", "H2O", "CO2", "CH4", "CO"]
database.reduce_species_set(subset_species, fictive_species)

# %% [markdown]
# We can for instance check that the sum of species mass fractions is 1:

# %%
print(database.X[subset_species + [spec+"_F" for spec in fictive_species]].sum(axis=1))

# %% [markdown]
# Note that the check for the sum being 1 is here just illustrative, as an advanced verification on the individual atomic mass fractions and the enthalpy is made in the *reduce_species_set* routine.

# %% [markdown]
# ## Data re-sampling

# %% [markdown]
# In order to reach an adequate data distribution we might need to perform a re-sampling of the data. This could happen for instance when in the stochastic particles database a lot of particles are at the equilibrium state. The points of interest are the points where the HRR is high and are usually fewer in the initial database. 
#
# In order to diagnosis the issue, we can analyze the joint probability density function (PDF) of two variables and the PDF of one variable:

# %%
database.density_scatter("Temperature" , "H2O", sort = True, bins = 100)

database.plot_pdf_var("Temperature")


# %% [markdown]
# The methodology proposed here to perform data re-sampling is based on heat release rate. The idea is to promote points with high HRR in the database, and penalize the presence of points with high density in the initial database along with low HRR. The methodology is adapted from the method proposed in the the paper of [Chi et al.](https://www.sciencedirect.com/science/article/pii/S0010218022003406). We note $|\dot{q}|^n$ the min-max normalized norm of the absolute HRR. We consider two variables $\psi_1$ and $\psi_2$ for which marginal PDFs will be computed.
#
# For a given sample $\mathbf{x}$, the sampling PDF, conditioned on $\psi_1$ and $\psi_2$, is written $f_s(\mathbf{x})$.
#
# A PDF for the HRR is then built by considering the distribution of HRR conditioned on $\psi_1$ and $\psi_2$. The variable for which the PDF is built is a function of the HRR, $a=f(|\dot{q}|^n)$. In the paper of Chi et al., the function $f(x)=1+\alpha x$ is proposed as it helps to promote points in the flame region. In practice, we consider bins in $\psi_1$ and $\psi_2$ space and then compute the averaged $a$ of particles in each bins. This gives us a PDF $f_q(\mathbf{x})$. We propose to define the weighting function for the re-sampling as:
#
# $$
# f_m(\mathbf{x}) = \frac{f_q(\mathbf{x})}{f_s(\mathbf{x})}
# $$
#
# $N_s$ samples are then drawn from the original points. Each sample $x$ is assigned the weight $f_m(\mathbf{x})$ and a random selection is then performed. $N_s$ is defined by the user and should be carefully selected to (i) have enough point resolution in the dataset (reacting regions should be well populated); (ii) the distributions of variables must be better than the initial distribution.
#
# One shortcoming of the method happens when we want to predict ignition. Indeed, in the initial ignition phase, we have a relatively low number of points with low HRR. Therefore the method will tend to discard these points, which are crucial for predicting ignition correctly. The simple correction proposed here is to affect a large weight to points with progress variable $c<c_{0}$ (typically $c_{0}=0.1$ or $0.2$). This way, points with low progress variable will be selected into the database. In order to distinguish igniting from non-igniting points, a temperature condition is also added. 
#
# The procedure may be applied to our database as follows:

# %%
# Function applied on the HRR
def f(x):
    return 1.0 + 5.0*x
    # return x

database.undersample_HRR("PC1", "PC2", hrr_func = f, keep_low_c = True, n_samples = 1000000, n_bins = 100, plot_distrib = False)

# %% [markdown]
# The new joint PDF and PDF can then be checked:

# %%
database.density_scatter("PC1", "PC2", sort = True, bins = 100)

database.plot_pdf_var("Temperature")

# %% [markdown]
# We can check also the size of the database:

# %%
database.print_data_size()

# %% [markdown]
# ## Dataset clustering

# %% [markdown]
# We can clusterize the dataset based on a progress variable if needed:

# %%
database.clusterize_dataset("progvar", 2, c_bounds=[0,0.95,1.0])

# %% [markdown]
# Alternatively, we could have used k-means: (commented because double clustering is banned)

# %%
database.clusterize_dataset("kmeans", 6)

# %% [markdown]
# We can visualize the cluster:

# %%
database.visualize_clusters("PC1", "PC2")

# %% [markdown]
# We can print the size of the database (count made for each cluster):

# %%
database.print_data_size()

# %% [markdown]
# ## Database processing

# %% [markdown]
# Finally, the database is processed in order ot be used in ML pipeline: (useless dataframe columns are suppressed and the transformation of the data is performed)

# %%
database.process_database(plot_distributions = True, distribution_species=["Temperature", "O2", "H2"])
