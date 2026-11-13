"""
Test 40: HRR and Omega undersampling without interpolation (LearningDatabase.undersample_HRR_bins,
undersample_Omega_bins).

The H2 stochastic database of the session (fixture h2_stoch_dtb, linked in a temporary run folder) is undersampled in
(Temperature, H2O) with a constant hrr_func: the probability of a point is 1 / N_b, N_b the number of points of its bin.
Check: the selection is the one of these probabilities with the same seed; the points of the outer half-bins in
temperature (fresh and burnt gases), which undersample_HRR (interpolation between bin centers) almost never draws,
are drawn. Omega version with omega_func = 1 + Omega, Omega computed independently from the mass fractions at t
and t+dt (as in test 32): the selection is the one of the probabilities mean_b(1 + Omega) / N_b with the same seed.

Independent of the other tests.
"""
import os

import numpy as np
import oyaml as yaml

from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from h2_stoch_database import mech_file

n_bins = 100
threshold = 1.0e-10


def hrr_constant(x):
    return np.ones_like(x)


def learning_database(h2_stoch_dtb, tmp_path):
    """LearningDatabase of the H2 stochastic database, in its own run folder, after the temperature threshold."""

    # Own run folder (dtb_processing.yaml of the session folder is used by other tests), linked to the database
    os.chdir(tmp_path)
    os.symlink(h2_stoch_dtb / "STOCH_DTB_TEST", tmp_path / "STOCH_DTB_TEST")
    dtb_processing_parameters = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb_hrr_bins",
                            "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": 1, "log_transform_Y": 1, "threshold": threshold, "T_threshold": 200.0,
                            "output_omegas": True, "with_N_chemistry": False},
        "data_clustering": {"clustering_method": "kmeans", "nb_clusters": 1, "clusterize_on": "phys"},
        "train_set_size": 0.75}
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    database = LearningDatabase()
    database.apply_temperature_threshold()

    return database


def bins_of_points(x, y):
    """Bin index of each point along each axis, as in LearningDatabase._undersample_bins, and number of points of each bin."""

    counts, x_e, y_e = np.histogram2d(x, y, bins=n_bins)
    ix = np.digitize(x, x_e[1:-1])
    iy = np.digitize(y, y_e[1:-1])

    return ix, iy, counts, x_e


def test_undersample_HRR_bins(h2_stoch_dtb, tmp_path):
    database = learning_database(h2_stoch_dtb, tmp_path)
    n = database.X.shape[0]
    n_samples = n // 20

    # Expected selection: probabilities 1 / N_b (mean of hrr_func of the bin = 1, divided by N_b), same seed
    T = database.X["Temperature"].values
    H2O = database.X["H2O"].values
    ix, iy, counts, T_e = bins_of_points(T, H2O)
    p = counts[ix, iy] / counts[ix, iy]**2
    p = p / p.sum()
    expected_choice = np.random.default_rng(1991).choice(n, replace=False, size=n_samples, p=p)

    # Outer half-bins in temperature: fresh and burnt gases
    half_bin = 0.5 * (T_e[1] - T_e[0])

    def edge_share(X):
        T_new = X["Temperature"].values
        return np.mean((T_new < T_e[0] + half_bin) | (T_new > T_e[-1] - half_bin))

    # With interpolation: edge points almost never drawn
    database.undersample_HRR("Temperature", "H2O", hrr_func=hrr_constant, keep_low_c=False, n_samples=n_samples, n_bins=n_bins)
    share_interp = edge_share(database.X)

    # Without interpolation: expected selection, edge points drawn
    database.X, database.Y = database.X_old, database.Y_old
    database.undersample_HRR_bins("Temperature", "H2O", hrr_func=hrr_constant, keep_low_c=False, n_samples=n_samples, n_bins=n_bins)
    share_bins = edge_share(database.X)

    print(f"Share of edge points: interpolation {share_interp:.4f}, bins {share_bins:.4f}")
    assert len(database.X) == len(database.Y) == n_samples
    assert np.array_equal(database.X["Temperature"].values, T[expected_choice])
    assert share_bins > 0.0
    assert share_interp < 0.1 * share_bins


def test_undersample_Omega_bins(h2_stoch_dtb, tmp_path):
    database = learning_database(h2_stoch_dtb, tmp_path)
    n = database.X.shape[0]
    n_samples = n // 20

    # Omega of each point: increments of ln Y_k (N2 excluded) scaled by their standard deviations
    species = [spec for spec in database.species_names if spec != "N2"]
    omega = np.log(np.maximum(database.Y[species].values, threshold)) - np.log(np.maximum(database.X[species].values, threshold))
    Omega = np.sqrt(np.sum((omega / omega.std(axis=0))**2, axis=1))

    # Expected selection: probabilities mean_b(1 + Omega) / N_b, same seed
    T = database.X["Temperature"].values
    ix, iy, counts, _ = bins_of_points(T, database.X["H2O"].values)
    g_sum = np.zeros((n_bins, n_bins))
    np.add.at(g_sum, (ix, iy), 1.0 + Omega)
    p = g_sum[ix, iy] / counts[ix, iy]**2
    p = p / p.sum()
    expected_choice = np.random.default_rng(1991).choice(n, replace=False, size=n_samples, p=p)

    database.undersample_Omega_bins("Temperature", "H2O", omega_func=lambda x: 1.0 + x, n_samples=n_samples, n_bins=n_bins)

    assert len(database.X) == len(database.Y) == n_samples
    assert np.array_equal(database.X["Temperature"].values, T[expected_choice])
