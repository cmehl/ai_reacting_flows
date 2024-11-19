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
# # Analysis of ML database
#
# In this script we write some code to analyze processed databases. The first aim it to help debugging but it may be adapted also to generate some plots for publications.

# %%
import os
import pandas as pd
import seaborn as sns
sns.set_style("darkgrid")

# %% [markdown]
# Path to the database:

# %%
dtb_path = "../scripts/STOCH_DTB_PREMIXED_CH4/database_3"

# %% [markdown]
# We can count the number of clusters:

# %%
nb_clusters = len(next(os.walk(dtb_path))[1])
print(f">> There are {nb_clusters} cluster(s)")

# %% [markdown]
# Get data from a given cluster:

# %%
i_cluster = 0
X_train = pd.read_csv(dtb_path + f"/cluster{i_cluster}/X_train.csv")
Y_train = pd.read_csv(dtb_path + f"/cluster{i_cluster}/Y_train.csv")
X_val = pd.read_csv(dtb_path + f"/cluster{i_cluster}/X_val.csv")
Y_val = pd.read_csv(dtb_path + f"/cluster{i_cluster}/Y_val.csv")


# %% [markdown]
# Here we propose some plots to display data distribution:

# %%
sns.histplot(data=X_val, x="CH4_F_X", bins=200)

# %%
sns.histplot(data=Y_val, x="CH4_F_Y", bins=200)
