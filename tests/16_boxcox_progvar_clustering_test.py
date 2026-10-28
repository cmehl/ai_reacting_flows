"""
Test 16: Box-Cox transform and progress variable clustering in LearningDatabase.

Builds a small synthetic database (some negative mass fractions, progress variable slightly
outside [0, 1]) and processes it with a Box-Cox transform (lambda_bct = 0.5) and 2 clusters
on the progress variable.
Check:
- lambda_bct is read from dtb_processing.yaml and negative mass fractions are clipped to 0
  before the Box-Cox transform (no NaN): the minimum of the transformed H2 is -1/lambda_bct.
- the k-means features also clip negative values to 0 for Box-Cox.
- no point is lost by the progress variable clustering.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import oyaml as yaml
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
import ai_reacting_flows.tools.transforms as transforms

# Mechanism file (only copied by LearningDatabase)
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

lambda_bct = 0.5

dtb_processing_parameters = {
    "database_params": {"database_type": "stoch", "dtb_folder_suffix": "BCT_TEST", "database_name": "dtb_bct",
                        "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
    "data_processing": {"log_transform_X": 2, "log_transform_Y": 2, "lambda_bct": lambda_bct, "threshold": 1.0e-10,
                        "T_threshold": 200.0, "output_omegas": False, "with_N_chemistry": False},
    "data_clustering": {"clustering_method": "progvar", "nb_clusters": 2, "clusterize_on": "phys"},
    "train_set_size": 0.75,
}


def write_synthetic_database(dtb_folder):
    """Small constant-dt database with the same columns as a stochastic reactors database."""

    rng = np.random.default_rng(0)
    n = 200
    species = ["H2", "O2", "H2O", "N2"]
    cols = ["Temperature", "Pressure"] + species + ["Prog_var", "HRR"]

    data = np.empty((n, len(cols)))
    data[:, 0] = rng.uniform(300.0, 2000.0, n)            # T [K]
    data[:, 1] = 101325.0                                  # P [Pa]
    data[:, 2:6] = 10.0**rng.uniform(-8.0, -0.5, (n, 4))   # Yk [-]
    data[:, 6] = rng.uniform(-0.01, 1.01, n)               # Prog_var [-], slightly outside [0, 1]
    data[:, 7] = rng.uniform(0.0, 1.0e9, n)                # HRR [W/m3]
    data[0, 2] = -1.0e-20                                  # Negative H2 mass fraction (as from Cantera)

    os.makedirs(dtb_folder)
    with h5py.File(os.path.join(dtb_folder, "solutions.h5"), "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=data).attrs["cols"] = cols
        grp.create_dataset("Y", data=data).attrs["cols"] = cols

    return n


def test_boxcox_progvar_clustering(tmp_path):
    os.chdir(tmp_path)
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    n = write_synthetic_database(tmp_path / "STOCH_DTB_BCT_TEST")

    database = LearningDatabase()

    # k-means features [T, Box-Cox of H2]: negative values clipped to 0
    features = transforms.kmeans_features(np.array([300.0, 300.0]), np.array([[-1.0e-20], [0.25]]), None, ["H2"], "phys",
                                          2, database.threshold, database.lambda_bct, [])
    assert np.allclose(features[:, 1], [-1.0/lambda_bct, (0.25**lambda_bct - 1.0)/lambda_bct])

    database.clusterize_dataset(c_bounds=[0.0, 0.5, 1.0])
    database.process_database()

    n_points = 0
    H2_min = np.inf
    with h5py.File(tmp_path / "STOCH_DTB_BCT_TEST" / "dtb_bct" / "training_data.h5", "r") as f:
        for i_cluster in range(2):
            grp = f[f"CLUSTER_{i_cluster}"]
            X = np.vstack([grp["X_train"][()], grp["X_val"][()]])
            assert not np.isnan(X).any()
            n_points += X.shape[0]

            # Unscaled Box-Cox transformed H2
            i_H2 = list(grp["X_train"].attrs["cols"]).index("H2_X")
            mean, var = grp["Xscaler"][i_H2]
            H2_min = min(H2_min, (mean + np.sqrt(var) * X[:, i_H2]).min())

    assert n_points == n
    assert np.isclose(H2_min, -1.0/lambda_bct)
