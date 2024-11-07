# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: 'Python 3.9.5 (''venv'': venv)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ANN model generation
#
# This notebook shows how to generate a ML model for chemical kinetics prediction. It assumes a processed database is available. See the notebook *generate_ML_dtb.ipynb* to get more information.
#
# The training is here done consecutively for several clusters, defined in the database generation process. 

# %%
from ai_reacting_flows.ann_model_generation.MLP_model import MLPModel

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# The training parameters are first defined:

# %%
# Dictionary with parameters
training_parameters = {}

training_parameters["model_name_suffix"] = "TO_REMOVE"    # Name of the resulting model folder (as a suffix of MODEL)
    
training_parameters["dataset_path"] = "/work/mehlc/2_IA_KINETICS/ai_reacting_flows/scripts/STOCH_DTB_HOTSPOT_H2_HRR/database_test"    # path of the database

training_parameters["new_model_folder"] = False      # New model folders are training models inside existing folder

training_parameters["dt_simu"] = 5.0e-07     # Time step of the prediction
    
training_parameters["fuel"] = "H2"           # Fuel
training_parameters["mechanism_type"] = "detailed"    # Mechanism type for ANN chemistry: detailed or reduced

training_parameters["remove_N2"] = True    # if True, N2 is removed from the neural network prediction

training_parameters["nb_units_in_layers_list"] = [[80,80],[20,20],[40,40],[50,50]]   # Network shape: number of units in each layer
training_parameters["layers_activation_list"] = [['tanh','tanh'],['tanh', 'tanh'],['tanh', 'tanh'],['tanh', 'tanh']]    # Activation functions
training_parameters["layers_type"] = "dense"               # "dense" or "resnet"


training_parameters["batch_size"] = 512         # Batch size for the gradient descent
training_parameters["initial_learning_rate"] = 1.0e-3           # Initial learnign rate
training_parameters["decay_steps"] = 100        # Exponential decay: done each decay_steps
training_parameters["decay_rate"] = 0.92        # Exponential decay: rate
training_parameters["staircase"] = True         # Stair case or continuous

training_parameters["alpha_reg"] = 0.0    # L2 regularization coefficient
        
training_parameters["epochs_list"] = [1,1,1,400]    # number of epochs for each cluster


training_parameters["hard_constraints_model"] = 0   # Hard constraint: 0=no constraints, 1=constraints on atomic balance

# %% [markdown]
# The training is initialized:

# %%
mlp_model = MLPModel(training_parameters)

# %% [markdown]
# We can train all models in one go:

# %%
mlp_model.train_model_all_clusters()

# %% [markdown]
# Or we can train only the model for a given cluster:

# %%
# i_cluster = 0
# mlp_model.train_model_cluster_i(i_cluster)
