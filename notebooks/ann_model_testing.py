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
# # Testing of ANN models for combustion
#
# Simple tests for the generated ANN models are made possible on python. These include:
#
# + Test on 0D ignition simulations
# + Test on the prediction of reaction rate of stationary 1D flames (*a priori* testing)
# + Comparison of the ANN predictions to the validation dataset

# %%
import os
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# The model is read from *MODELS/<models_folder>* in the case folder, trained in the notebook *ann_model_learning.ipynb*. The processing parameters (fuel, transforms, progress variable species, ...) are read from the learning database. We therefore move to the case folder:

# %%
os.chdir("../scripts/H2_IGN_KERNEL")

# %% [markdown]
# Parameters for the testing:

# %%
# Dictionary with parameters
testing_parameters = {}

testing_parameters["models_folder"] = "MODEL_H2_POF_2023"     # Folder for the ML model

testing_parameters["spec_to_plot"] = ["N2", "H2", "O2", "H2O", "H2O2"]          # List of species to plot

testing_parameters["yk_renormalization"] = True

testing_parameters["hybrid_ann_cvode"] = False             # CVODE helping ANN based on a conservation criterion
testing_parameters["hybrid_ann_cvode_tol"] = 5.0e-05      # Tolerance for the CVODE switching

# %% [markdown]
# The model testing is first initialized:

# %%
test = NNTesting(testing_parameters)

# %% [markdown]
# We can test the ability of the ANN model to reproduce the ignition of a homogeneous 0-D mixture. The mixture initial state is defined by its temperature $T_0$, its pressure $p$ and its equivalence ratio $\phi$. The simulation is here advanced in time using (i) the CVODE algorithm from CANTERA; (ii) the ANN model.

# %%
phi = 0.4
T0 = 1200.0
pressure = 101325.0
dt = 0.5e-6
nb_ite = 1000

test.test_0D_ignition(phi, T0, pressure, dt, nb_ite)

# %% [markdown]
# Another test, if the ANN has been built for that, is to assess its ability to reproduce laminar premixed flames. The test here is *a priori*, in the sense that we take a CANTERA converged solution, and compare the exact reaction rate to the ANN reaction rate. For a species $k$, if the exact solution is written $Y_k$, the local reaction rate is estimated as:
#
# $$ \dot{\omega}_k = \frac{Y_k(dt) - Y_k}{dt} $$
#
# where $Y_k(dt)$ is the solution $Y_k$ advanced by a time increment $dt$, estimated using either CVODE or ANN.

# %%
phi = 0.4
T0 = 300.0
pressure = 101325.0
dt = 0.5e-6
T_threshold = 600.0

test.test_1D_premixed(phi, T0, pressure, dt, T_threshold)

# %% [markdown]
# Finally, the ANN predictions can be compared to the validation dataset stored in *training_data.h5*, cluster by cluster. Error statistics are returned and scatter plots are saved in *scatter_folder*:

# %%
stats = test.evaluate_on_validation_set(scatter_folder="./plots_validation")
