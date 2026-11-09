"""
Test 32: undersampling on the norm of the scaled increments Omega (LearningDatabase.undersample_Omega).

The H2 stochastic database of the session (fixture h2_stoch_dtb, linked in a temporary run folder) is undersampled
on log10(Omega), Omega computed independently from the mass fractions at t and t+dt. Check: with the bins of undersample_1D, bins with at least
min_count points keep the size of the smallest of them, sparser bins are kept whole; values below Omega_min are in
the first bin; the temporary column is removed; the scatter plot of log10(Omega) versus Prog_var is written.

Independent of the other tests.
"""
import os

import numpy as np
import oyaml as yaml

import ai_reacting_flows.tools.utilities as utils
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from h2_stoch_database import mech_file

threshold = 1.0e-10
Omega_min = 1.0e-3
min_count = 50


def log10_Omega(X, Y, species, sigma):
    """log10 of the norm of the increments of ln Y_k scaled by sigma, floored at Omega_min."""

    omega = np.log(np.maximum(Y[species].values, threshold)) - np.log(np.maximum(X[species].values, threshold))
    Omega = np.sqrt(np.sum((omega / sigma)**2, axis=1))

    return np.log10(np.maximum(Omega, Omega_min))


def test_undersample_Omega(h2_stoch_dtb, tmp_path):
    # Own run folder (dtb_processing.yaml of the session folder is used by other tests), linked to the database
    os.chdir(tmp_path)
    os.symlink(h2_stoch_dtb / "STOCH_DTB_TEST", tmp_path / "STOCH_DTB_TEST")
    dtb_processing_parameters = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb_omega",
                            "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": 1, "log_transform_Y": 1, "threshold": threshold, "T_threshold": 200.0,
                            "output_omegas": True, "with_N_chemistry": False},
        "data_clustering": {"clustering_method": "kmeans", "nb_clusters": 1, "clusterize_on": "phys"},
        "train_set_size": 0.75}
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    database = LearningDatabase()
    species = [spec for spec in database.species_names if spec != "N2"]

    # Bins of log10(Omega) of the whole database, as in undersample_1D
    omega = np.log(np.maximum(database.Y[species].values, threshold)) - np.log(np.maximum(database.X[species].values, threshold))
    sigma = omega.std(axis=0)
    v = log10_Omega(database.X, database.Y, species, sigma)
    binwidth = utils.smart_binwidth(v)
    edges = np.arange(v.min(), v.max() + binwidth, binwidth)
    counts, _ = np.histogram(v, bins=edges)
    eligible = counts >= min_count
    n_keep = counts[eligible].min()
    assert v.min() == np.log10(Omega_min)

    database.scatter_Omega_progvar()
    assert os.path.isfile(tmp_path / "STOCH_DTB_TEST" / "Omega_vs_progvar.png")

    database.undersample_Omega(Omega_min, min_count=min_count)

    database.scatter_Omega_progvar()
    assert os.path.isfile(tmp_path / "STOCH_DTB_TEST" / "Omega_vs_progvar_resampled.png")

    assert "log10_Omega" not in database.X.columns and "log10_Omega" not in database.X_old.columns
    assert len(database.X) == len(database.Y) < len(v)

    # Same number of points in the bins with at least min_count points, sparse bins kept whole
    counts_new, _ = np.histogram(log10_Omega(database.X, database.Y, species, sigma), bins=edges)
    assert np.all(counts_new[eligible] == n_keep)
    assert np.all(counts_new[~eligible] == counts[~eligible])
