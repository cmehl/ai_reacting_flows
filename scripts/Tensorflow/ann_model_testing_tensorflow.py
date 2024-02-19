from ai_reacting_flows.ann_model_generation.model_testing_mk import ModelTesting


# Dictionary with parameters
testing_parameters = {}

testing_parameters["models_folder"] = "/work/kotlarcm/WORK/AI/ENER/ai_reacting_flows_tensorflow/scripts/ENER/MODEL_LEARNING_TENSORFLOW_H2_HOTSPOT_MARTIN"     # Folder for the ML model


testing_parameters["input_weight"] = False # Import Weight to pytorch or other framework 
testing_parameters["fuel"] = "H2"            # Considered fuel
testing_parameters["with_N_chemistry"] = False         # Nitrogen chemistry considered or not

testing_parameters["spec_to_plot"] =  ["H2", "O2", "N2"] #["N2", "H2", "O2", "H2O", "H2O2"]          # List of species to plot
testing_parameters["pv_species"] = ["H2O"]            # Species used for progress variable computation

testing_parameters["yk_renormalization"] = False

testing_parameters["hybrid_ann_cvode"] = False             # CVODE helping ANN based on a conservation criterion
testing_parameters["hybrid_ann_cvode_tol"] = 2.0e-04      # Tolerance for the CVODE switching


# %% [markdown]
# The model testing is first initialized:

# %%
test = ModelTesting(testing_parameters)

# %% [markdown]
# We can test the ability of the ANN model to reproduce the ignition of a homogeneous 0-D mixture. The mixture initial state is defined by its temperature $T_0$, its pressure $p$ and its equivalence ratio $\phi$. The simulation is here advanced in time using (i) the CVODE algorithm from CANTERA; (ii) the ANN model.

# %%
phi = 0.4
T0 = 1200.0
pressure = 101325.0
dt = 0.5e-6
nb_ite = 10

test.test_0D_ignition(phi, T0, pressure, dt, nb_ite)

# %% [markdown]
# Another test, if the ANN has been built for that, is to assess its ability to reproduce laminar premixed flames. The test here is *a priori*, in the sense that we take a CANTERA converged solution, and compare the exact reaction rate to the ANN reaction rate. For a species $k$, if the exact solution is written $Y_k$, the local reaction rate is estimated as:
# 
# $$ \dot{\omega}_k = \frac{Y_k(dt) - Y_k}{dt} $$
# 
# where $Y_k(dt)$ is the solution $Y_k$ advanced by a time increment $dt$, estimated using either CVODE or ANN.

# %%
#phi = 0.4
#T0 = 300.0
#pressure = 101325.0
#dt = 0.5e-6
#T_threshold = 600.0

#test.test_1D_premixed(phi, T0, pressure, dt, T_threshold)


