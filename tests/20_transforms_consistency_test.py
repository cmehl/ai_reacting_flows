"""
Test 20: consistency of the transforms between database processing and inference.

Builds a small synthetic database where Y_k(t+dt) = c_k Y_k(t), processes it with LearningDatabase
for every allowed combination of transforms (none, log, Box-Cox), outputs (differences or not)
and with or without a species kept on a linear scale (log_excluded_species).
Check: from the training data written by process_database, the shared functions of
tools/transforms.py used at inference (inverse_species, output_to_Yk) give back Y_k(t) and Y_k(t+dt).

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import oyaml as yaml
import pytest
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
import ai_reacting_flows.tools.transforms as transforms

# Mechanism file (only copied by LearningDatabase)
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

species = ["H2", "O2", "H2O", "N2"]
factors = np.array([0.5, 1.5, 2.0, 1.0])   # Y_k(t+dt) / Y_k(t) [-]


def write_synthetic_database(dtb_folder):
    """Small constant-dt database with the same columns as a stochastic reactors database."""

    rng = np.random.default_rng(0)
    n = 100
    cols = ["Temperature", "Pressure"] + species + ["Prog_var", "HRR"]

    X = np.empty((n, len(cols)))
    X[:, 0] = rng.uniform(300.0, 2000.0, n)            # T [K]
    X[:, 1] = 101325.0                                  # P [Pa]
    X[:, 2:6] = 10.0**rng.uniform(-8.0, -0.5, (n, 4))   # Yk [-], above the threshold
    X[:, 6] = rng.uniform(0.0, 1.0, n)                  # Prog_var [-]
    X[:, 7] = rng.uniform(0.0, 1.0e9, n)                # HRR [W/m3]

    Y = X.copy()
    Y[:, 2:6] = factors * X[:, 2:6]

    os.makedirs(dtb_folder)
    with h5py.File(os.path.join(dtb_folder, "solutions.h5"), "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=X).attrs["cols"] = cols
        grp.create_dataset("Y", data=Y).attrs["cols"] = cols


@pytest.mark.parametrize("log_transform_X, log_transform_Y", [(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)])
@pytest.mark.parametrize("output_omegas", [True, False])
@pytest.mark.parametrize("log_excluded_species", [[], ["H2O"]])
def test_transforms_consistency(tmp_path, log_transform_X, log_transform_Y, output_omegas, log_excluded_species):
    os.chdir(tmp_path)

    params = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb",
                            "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": log_transform_X, "log_transform_Y": log_transform_Y, "lambda_bct": 0.2,
                            "threshold": 1.0e-10, "T_threshold": 200.0, "output_omegas": output_omegas,
                            "with_N_chemistry": False, "log_excluded_species": log_excluded_species},
        "data_clustering": {"clustering_method": "kmeans", "nb_clusters": 1, "clusterize_on": "phys"},
        "train_set_size": 0.75,
    }
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)

    write_synthetic_database(tmp_path / "STOCH_DTB_TEST")
    LearningDatabase().process_database()

    with h5py.File(tmp_path / "STOCH_DTB_TEST" / "dtb" / "training_data.h5", "r") as f:
        grp = f["CLUSTER_0"]
        X_scaled = np.vstack([grp["X_train"][()], grp["X_val"][()]])
        Y_scaled = np.vstack([grp["Y_train"][()], grp["Y_val"][()]])
        X_mean, X_var = grp["Xscaler"][:, 0], grp["Xscaler"][:, 1]
        Y_mean, Y_var = grp["Yscaler"][:, 0], grp["Yscaler"][:, 1]
        species_ann = [str(col)[:-2] for col in grp["X_train"].attrs["cols"][1:]]

    assert species_ann == ["H2", "O2", "H2O"]   # N2 removed

    # Unscaled network input (species) and output, as at inference
    x_species_t = transforms.unscale(X_scaled, X_mean, transforms.scaler_std(X_var))[:, 1:]
    y_t = transforms.unscale(Y_scaled, Y_mean, transforms.scaler_std(Y_var))

    Yk_in = transforms.inverse_species(x_species_t, species_ann, log_transform_X, 0.2, log_excluded_species)
    Yk_out = transforms.output_to_Yk(x_species_t, y_t, species_ann, output_omegas, log_transform_X, log_transform_Y,
                                     0.2, log_excluded_species)

    assert np.allclose(Yk_out, factors[:3] * Yk_in, rtol=1.0e-8, atol=1.0e-14)
