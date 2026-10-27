"""
Test 05: k-means clustering features (constant dt).

Builds a small synthetic database in a temporary folder and runs LearningDatabase.clusterize_dataset.
Check: k-means is fitted on [T, log(Yk)] without N2, i.e. the features built at inference
in NNTesting.attribute_cluster.

Independent of the other tests.
"""
import os
import h5py
import joblib
import numpy as np
import oyaml as yaml
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase

# Mechanism file (only copied by LearningDatabase)
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

# Dictionary to store data processing parameters
dtb_processing_parameters = {}

dtb_processing_parameters["database_params"] = {}
dtb_processing_parameters["database_params"]["database_type"] = "stoch"
dtb_processing_parameters["database_params"]["dtb_folder_suffix"] = "KMEANS_TEST"
dtb_processing_parameters["database_params"]["database_name"] = "dtb_kmeans"
dtb_processing_parameters["database_params"]["dtb_file"] = "solutions.h5"
dtb_processing_parameters["database_params"]["dt_var"] = False
dtb_processing_parameters["database_params"]["fuel"] = "H2"
dtb_processing_parameters["database_params"]["mech_file"] = mech_file

dtb_processing_parameters["data_processing"] = {}
dtb_processing_parameters["data_processing"]["log_transform_X"] = 1
dtb_processing_parameters["data_processing"]["log_transform_Y"] = 1
dtb_processing_parameters["data_processing"]["threshold"] = 1.0e-10
dtb_processing_parameters["data_processing"]["T_threshold"] = 200.0
dtb_processing_parameters["data_processing"]["output_omegas"] = False
dtb_processing_parameters["data_processing"]["with_N_chemistry"] = False

dtb_processing_parameters["data_clustering"] = {}
dtb_processing_parameters["data_clustering"]["clustering_method"] = "kmeans"
dtb_processing_parameters["data_clustering"]["nb_clusters"] = 2
dtb_processing_parameters["data_clustering"]["clusterize_on"] = "phys"

dtb_processing_parameters["train_set_size"] = 0.75


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
    data[:, 6] = rng.uniform(0.0, 1.0, n)                  # Prog_var [-]
    data[:, 7] = rng.uniform(0.0, 1.0e9, n)                # HRR [W/m3]

    os.makedirs(dtb_folder)
    with h5py.File(os.path.join(dtb_folder, "solutions.h5"), "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=data).attrs["cols"] = cols
        grp.create_dataset("Y", data=data).attrs["cols"] = cols

    return data


def test_kmeans_features_constant_dt(tmp_path):
    os.chdir(tmp_path)
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    dtb_folder = tmp_path / "STOCH_DTB_KMEANS_TEST"
    data = write_synthetic_database(dtb_folder)

    database = LearningDatabase()
    database.clusterize_dataset()

    # k-means features must be [T, log(Yk)] without N2, as built in NNTesting.attribute_cluster
    Xscaler = joblib.load(dtb_folder / "dtb_kmeans" / "Xscaler_kmeans.pkl")
    expected = np.column_stack([data[:, 0], np.log(data[:, 2:5])])

    assert Xscaler.n_features_in_ == expected.shape[1]
    assert np.allclose(Xscaler.mean_, expected.mean(axis=0))
