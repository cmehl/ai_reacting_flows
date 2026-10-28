"""
Test 21: processing parameters stored in training_data.h5.

Processes a small synthetic database with LearningDatabase (Box-Cox, outputs as differences,
one species kept on a linear scale).
Check:
- process_database writes the processing parameters in training_data.h5, read back with the
  right values and types by read_processing_parameters (used by training and testing).
- for an older database without these parameters, the same values are read from the YAML files.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import oyaml as yaml
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from ai_reacting_flows.databases_processing.processing_parameters import read_processing_parameters

# Mechanism file (only copied by LearningDatabase)
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

dtb_processing_parameters = {
    "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb",
                        "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
    "data_processing": {"log_transform_X": 2, "log_transform_Y": 2, "lambda_bct": 0.2, "threshold": 1.0e-10,
                        "T_threshold": 200.0, "output_omegas": True, "with_N_chemistry": False,
                        "log_excluded_species": ["H2O"]},
    "data_clustering": {"clustering_method": "kmeans", "nb_clusters": 1, "clusterize_on": "phys"},
    "train_set_size": 0.75,
}

expected = {"dt_var": False, "fuel": "H2", "log_transform_X": 2, "log_transform_Y": 2, "lambda_bct": 0.2,
            "threshold": 1.0e-10, "output_omegas": True, "with_N_chemistry": False, "log_excluded_species": ["H2O"],
            "clustering_method": "kmeans", "clusterize_on": "phys", "nb_clusters": 1, "pv_species": ["H2O"]}


def write_synthetic_database(dtb_folder):
    """Small constant-dt database with the same columns as a stochastic reactors database."""

    rng = np.random.default_rng(0)
    n = 100
    cols = ["Temperature", "Pressure", "H2", "O2", "H2O", "N2", "Prog_var", "HRR"]

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

    # Generation parameters (only the progress variable species are read)
    with open(os.path.join(dtb_folder, "dtb_params.yaml"), "w") as file:
        yaml.dump({"pv_species": ["H2O"]}, file, default_flow_style=False)


def test_processing_parameters(tmp_path):
    os.chdir(tmp_path)
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    write_synthetic_database(tmp_path / "STOCH_DTB_TEST")
    LearningDatabase().process_database()

    dataset_path = tmp_path / "STOCH_DTB_TEST" / "dtb"

    # Parameters stored in training_data.h5
    params = read_processing_parameters(dataset_path, tmp_path)
    assert params == expected
    assert type(params["output_omegas"]) is bool and type(params["nb_clusters"]) is int

    # Older database: parameters removed from training_data.h5, read from the YAML files
    with h5py.File(dataset_path / "training_data.h5", "a") as f:
        for key in list(f.attrs.keys()):
            del f.attrs[key]

    assert read_processing_parameters(dataset_path, tmp_path) == expected
