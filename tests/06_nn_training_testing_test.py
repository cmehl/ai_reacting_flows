"""
Test 06: ANN training and inference on a small synthetic database.

Builds a tiny H2 learning database (training_data.h5, N2 removed, log transforms, output omegas)
in a temporary folder, trains one MLP with NN_manager for a few epochs, then advances one state
with NNTesting.advance_state_NN. Run for both database types (stoch and flamelets).
Check: training runs with epochs not a multiple of val_every (val_every omitted -> default),
and inference runs and keeps N2 unchanged.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import oyaml as yaml
import cantera as ct
import pytest
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")


def write_learning_database(dtb_folder, database_type):
    """Synthetic learning database with the layout written by LearningDatabase.process_database."""

    gas = ct.Solution(mech_file)
    species = [spec for spec in gas.species_names if spec != "N2"]
    X_cols = ["Temperature_X"] + [f"{spec}_X" for spec in species]
    Y_cols = [f"{spec}_Y" for spec in species]

    rng = np.random.default_rng(0)
    n = 64
    X = rng.standard_normal((n, len(X_cols)))
    Y = rng.standard_normal((n, len(Y_cols)))
    scaler_X = np.column_stack([np.zeros(len(X_cols)), np.ones(len(X_cols))])
    scaler_Y = np.column_stack([np.zeros(len(Y_cols)), np.ones(len(Y_cols))])

    os.makedirs(os.path.join(dtb_folder, "dtb"))
    with h5py.File(os.path.join(dtb_folder, "dtb", "training_data.h5"), "w") as f:
        grp = f.create_group("CLUSTER_0")
        grp.create_dataset("Xscaler", data=scaler_X)
        grp.create_dataset("Yscaler", data=scaler_Y)
        grp.create_dataset("X_train", data=X).attrs["cols"] = np.array(X_cols, dtype=object)
        grp.create_dataset("Y_train", data=Y).attrs["cols"] = np.array(Y_cols, dtype=object)
        grp.create_dataset("X_val", data=X).attrs["cols"] = np.array(X_cols, dtype=object)
        grp.create_dataset("Y_val", data=Y).attrs["cols"] = np.array(Y_cols, dtype=object)

    dtb_processing = {
        "database_params": {"database_type": database_type, "dtb_folder_suffix": "TEST", "dtb_file": "solutions.h5",
                            "database_name": "dtb", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": 1, "log_transform_Y": 1, "threshold": 1.0e-10, "T_threshold": 200.0,
                            "output_omegas": True, "with_N_chemistry": False},
        "data_clustering": {"clusterize_on": "phys", "clustering_method": "kmeans", "nb_clusters": 1},
        "train_set_size": 0.75,
    }
    with open(os.path.join(dtb_folder, "dtb", "dtb_processing.yaml"), "w") as file:
        yaml.dump(dtb_processing, file, default_flow_style=False)

    # Database generation parameters (only fuel, mechanism and progress variable are read)
    dtb_params_file = "dtb_params.yaml" if database_type == "stoch" else "dtb_params_flmts.yaml"
    with open(os.path.join(dtb_folder, dtb_params_file), "w") as file:
        yaml.dump({"fuel": ["H2"], "mech_file": mech_file, "pv_species": ["H2O"]}, file, default_flow_style=False)


@pytest.mark.parametrize("database_type, prefix", [("stoch", "STOCH"), ("flamelets", "FLAMELETS")])
def test_nn_training_and_inference(tmp_path, database_type, prefix):
    os.chdir(tmp_path)
    write_learning_database(tmp_path / f"{prefix}_DTB_TEST", database_type)

    # Training: 11 epochs with val_every omitted (default 10) -> not a multiple of val_every
    networks_params = {
        "database_path": f"{prefix}_DTB_TEST/dtb",
        "model_name_suffix": "TEST",
        "new_model_folder": True,
        "networks_types": ["MLP"],
        "networks_def": ["cluster0"],
        "clusters": {"cluster0": {"nb_units_in_layers_list": [8], "layers_activation_list": ["tanh", "Id"],
                                  "layers_type": ["dense", "dense"]}},
        "learning": {"batch_size": 16, "epochs_list": [11]},
    }
    with open("networks_params.yaml", "w") as file:
        yaml.dump(networks_params, file, default_flow_style=False)

    NN_manager().train_all_clusters()

    # Inference on one premixed H2/air state
    testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": False,
                          "hybrid_ann_cvode": False, "hybrid_ann_cvode_tol": 5.0e-05}
    test = NNTesting(testing_parameters)
    test.cluster = 0

    gas = ct.Solution(mech_file)
    gas.TP = 1200.0, 101325.0
    gas.set_equivalence_ratio(0.6, "H2", {"O2": 1.0, "N2": 3.76})
    T_new, Y_new = test.advance_state_NN(gas.T, gas.Y, gas.P, 1.0e-6)

    i_N2 = gas.species_index("N2")
    assert np.isfinite(T_new)
    assert np.all(np.isfinite(Y_new))
    assert np.isclose(np.ravel(Y_new)[i_N2], gas.Y[i_N2])
