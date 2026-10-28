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
# # ANN model generation
#
# This notebook shows how to train a ML model for chemical kinetics prediction. It assumes a processed learning database is available. See the notebook *generate_ML_dtb.ipynb* to get more information.
#
# One network is trained per cluster defined in the database processing, consecutively for all clusters.

# %%
import os
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager

# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# The training parameters are read from the file *networks_params.yaml* of the case folder (see the README for all keys): path of the learning database, name of the model folder, network of each cluster and learning parameters. The processing parameters of the database (transforms, clustering, ...) are read from *training_data.h5*. We therefore move to the case folder:

# %%
os.chdir("../scripts/H2_IGN_KERNEL")
print(open("networks_params.yaml").read())

# %% [markdown]
# The training is initialized. The model folder *MODELS/MODEL_<model_name_suffix>* is created:

# %%
nn_manager = NN_manager()

# %% [markdown]
# We train the networks of all clusters. Losses and conservation diagnostics are plotted in *MODELS/MODEL_<model_name_suffix>/training/cluster_<i>/*:

# %%
nn_manager.train_all_clusters()
