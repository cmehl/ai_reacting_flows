# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ANN vs CVODE — single-step comparison on a held-out CFD snapshot
#
# Loads the output of `test_ann_vs_cvode.py`: for a sample of cells drawn from
# a CFD snapshot not used during training/validation, each cell was advanced
# by `dt` two ways — CVODE (`IdealGasConstPressureReactor`, ground truth) and
# the trained ANN model. This compares the two: parity plots per species,
# temperature error, and an element-conservation check on the ANN output.

# %%
import os

import numpy as np
import h5py
import oyaml as yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interpn

import ai_reacting_flows.tools.utilities as utils

sns.set_style("darkgrid")

# %% [markdown]
# ## Configuration / loading

# %%
with open("test_ann_vs_cvode.yaml", "r") as f:
    test_params = yaml.safe_load(f)

results_file = test_params["output_file"]

with h5py.File(results_file, "r") as f:
    species_names = [s for s in f.attrs["species"]]
    time_step = f.attrs["time_step"]
    T_threshold = f.attrs["T_threshold"]
    test_data_file = f.attrs["test_data_file"]
    models_folder = f.attrs["models_folder"]

    T_ini = f["T_ini"][()]
    P_ini = f["P_ini"][()]
    Y_ini = f["Y_ini"][()]
    T_cvode = f["T_cvode"][()]
    Y_cvode = f["Y_cvode"][()]
    T_ann = f["T_ann"][()]
    Y_ann = f["Y_ann"][()]
    cluster = f["cluster"][()]

n_sample = T_ini.shape[0]
n_species = len(species_names)

print(f"Model            : {models_folder}")
print(f"Test snapshot    : {test_data_file}")
print(f"dt               : {time_step:g} s")
print(f"T_threshold      : {T_threshold:g} K")
print(f"Samples          : {n_sample}")
print(f"Species          : {species_names}")
print(f"Cluster counts   : {np.bincount(cluster)}")

# %% [markdown]
# ## Temperature

# %%
def density_scatter(ax, x, y, bins=150, **kwargs):
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=False)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )
    z[np.isnan(z)] = 0.0
    idx = z.argsort()
    im = ax.scatter(np.asarray(x)[idx], np.asarray(y)[idx], c=z[idx], s=4, **kwargs)
    return im

# %%
fig, axes = plt.subplots(ncols=2, figsize=(11, 4.5))

im = density_scatter(axes[0], T_cvode, T_ann)
lims = [T_cvode.min(), T_cvode.max()]
axes[0].plot(lims, lims, color="k", lw=1, ls="--")
axes[0].set_xlabel("$T$ CVODE [K]")
axes[0].set_ylabel("$T$ ANN [K]")
axes[0].set_title("Parity")
fig.colorbar(im, ax=axes[0], label="density")

sns.histplot(T_ann - T_cvode, bins=100, ax=axes[1], color="C0")
axes[1].set_xlabel(r"$T_{ANN} - T_{CVODE}$ [K]")
axes[1].set_title("Error distribution")

fig.tight_layout()

print(f"Temperature error: mean|err|={np.mean(np.abs(T_ann-T_cvode)):.3g} K, "
      f"max|err|={np.max(np.abs(T_ann-T_cvode)):.3g} K, "
      f"RMSE={np.sqrt(np.mean((T_ann-T_cvode)**2)):.3g} K")

# %% [markdown]
# ## Species parity plots
#
# Log-log, since mass fractions span many orders of magnitude (major species
# down to ppm-level radicals/NOx).

# %%
ncols = 3
nrows = -(-n_species // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
axes = axes.flat

floor = 1e-12
for ax, spec in zip(axes, species_names):
    j = species_names.index(spec)
    x = np.clip(Y_cvode[:, j], floor, None)
    y = np.clip(Y_ann[:, j], floor, None)

    density_scatter(ax, np.log10(x), np.log10(y), bins=80)
    lims = [np.log10(x).min(), np.log10(x).max()]
    ax.plot(lims, lims, color="k", lw=1, ls="--")
    ax.set_xlabel(f"log10 $Y_{{{spec}}}$ CVODE")
    ax.set_ylabel(f"log10 $Y_{{{spec}}}$ ANN")

for ax in list(axes)[n_species:]:
    ax.set_visible(False)

fig.tight_layout()

# %% [markdown]
# ## Error summary per species

# %%
print(f"{'species':8s} {'mean|err|':>12s} {'max|err|':>12s} {'mean Y (cvode)':>16s}")
for j, spec in enumerate(species_names):
    err = np.abs(Y_ann[:, j] - Y_cvode[:, j])
    print(f"{spec:8s} {err.mean():12.3e} {err.max():12.3e} {Y_cvode[:, j].mean():16.3e}")

# %% [markdown]
# ## Mass-fraction sum
#
# CVODE conserves $\sum_k Y_k = 1$ by construction; the ANN output does not,
# since nothing enforces it at inference time here.

# %%
fig, ax = plt.subplots(figsize=(5, 4))
sns.histplot(Y_ann.sum(axis=1), bins=100, ax=ax, color="C0")
ax.axvline(1.0, color="k", ls="--")
ax.set_xlabel(r"$\sum_k Y_k$ (ANN)")
fig.tight_layout()

print(f"sum(Y_k) ANN: min={Y_ann.sum(axis=1).min():.6f} max={Y_ann.sum(axis=1).max():.6f}")

# %% [markdown]
# ## Element conservation (ANN output vs input state)
#
# Same diagnostic as `analyze_solution.ipynb` / `NN_manager.plot_losses_conservation`,
# applied here to actual held-out CFD states instead of training-time validation data.

# %%
atomic_array = utils.parse_species_names(species_names)          # (4, n_species): rows C, H, O, N
mol_weights = utils.get_molecular_weights(species_names)          # (n_species,)
mass_per_atom = np.array([12.011, 1.008, 15.999, 14.007]).reshape((4, 1))

A_atomic = atomic_array * mass_per_atom
A_atomic = A_atomic / mol_weights[np.newaxis, :]

elements = ["C", "H", "O", "N"]

Y_atomic_in = Y_ini @ A_atomic.T
Y_atomic_ann = Y_ann @ A_atomic.T

residual = Y_atomic_ann - Y_atomic_in

fig, axes = plt.subplots(ncols=4, figsize=(16, 3.5))
for i, (ax, el) in enumerate(zip(axes, elements)):
    sns.histplot(residual[:, i], bins=100, ax=ax, color="C1")
    ax.set_xlabel(f"$Y_{{{el}}}(ANN) - Y_{{{el}}}(t)$")
fig.tight_layout()

print("Max |elemental mass fraction drift| per element (ANN vs input state):")
for el, r in zip(elements, residual.T):
    print(f"  {el:>2s}: max={np.abs(r).max():.3e}  mean={r.mean():.3e}  std={r.std():.3e}")
