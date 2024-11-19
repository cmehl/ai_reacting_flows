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
# # Stochastic reactors database post-analysis
#
# Some functions have been implemented to visualize generated stochastic reactors databases.
#
# Additional functions may be added by coding them directly in the file *stochastic_reactors_data_gen/post_processing*.

# %%
from ai_reacting_flows.stochastic_reactors_data_gen.post_processing import StochDatabase

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# We first define the database to be processed and the folder where the plots will be stored:

# %%
stoch_dtb_folder = "../scripts/STOCH_DTB_H2_DD_TEST_2"
save_folder = "./postpro_dtb"

# %% [markdown]
# The database is then read in a *StochDatabase* object which contains the methods to perform the post-processing:

# %%
stoch_dtb = StochDatabase(stoch_dtb_folder, save_folder)

# %% [markdown]
# Function to scatter plot $Y_k=f(T)$ for all time steps on a list on specified species: (here $H_2$ and $H$)

# %%
stoch_dtb.compute_1D_premixed(0.4, 101325.0, 300.0, "H2", "../data/chemical_mechanisms/mech_H2.yaml", "Mix")

# %%
stoch_dtb.plot_T_Yk(["H2", "H", "H2O", "O2"])

# %% [markdown]
# A generic function is available to scatter plot *var_x* as a function of *var_y*, colored by *var_c*:

# %%
var_x = "Temperature"
var_y = "H2"
var_c = "log_abs_HRR"
stoch_dtb.plot_generic(var_x, var_y, var_c)

# %% [markdown]
# Some functions are also available to plot individual iterations, here is an example for $(T,Y_k)$ plots:

# %%
stoch_dtb.plot_T_Yk_indiv(["H2", "H"], iteration=3)

# %% [markdown]
# Function to plot $T_{mean}=f(t)$ for each inlet:

# %%
stoch_dtb.plot_traj_T_time()

# %% [markdown]
# Function to plot the evolution of some particle variables (here $H_2$) over time for a given inlet:

# %%
nb_inlet = 2
stoch_dtb.plot_indiv_traj(nb_inlet, "H2")

# %% [markdown]
# Functions are also available to plot probability density functions (PDF) of the data. At the moment, only temperature PDF is implemented but other variables could easily be used. It may be done for a given iteration:

# %%
stoch_dtb.plot_pdf_T_inst(iteration=3)

# %% [markdown]
# Or for the entire dataset:

# %%
stoch_dtb.plot_pdf_T_all()

# %% [markdown]
# Plotting the PDF of heat release rate:

# %%
stoch_dtb.plot_pdf_HRR_all()

# %% [markdown]
# A function to plot the density of points is also available:

# %%
stoch_dtb.density_scatter("Temperature" , "H2", sort = True, bins = 20)
