"""
Helper for tests: small H2 learning database built from a Cantera ignition trajectory.

The database has the layout written by LearningDatabase.process_database (training_data.h5 with
one cluster, dtb_processing.yaml) with: N2 removed, log transform (threshold 1e-10), outputs as
differences of transformed mass fractions (output_omegas), optional log_excluded_species.
It also writes the run folder inputs needed by NN_manager (networks_params.yaml, dtb_params.yaml).
"""
import os
import h5py
import numpy as np
import oyaml as yaml
import cantera as ct

mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

threshold = 1.0e-10    # Clipping value before log [-]
dt = 2.0e-6            # Time step between X and Y [s]
pressure = 101325.0    # [Pa]


def ignition_states(n=64):
    """States (T [K], Y [-]) along a lean H2/air ignition at 1200 K, every dt."""

    gas = ct.Solution(mech_file)
    gas.TP = 1200.0, pressure
    gas.set_equivalence_ratio(0.6, "H2", {"O2": 1.0, "N2": 3.76})

    reactor = ct.IdealGasConstPressureReactor(gas)
    sim = ct.ReactorNet([reactor])

    T = np.empty(n + 1)
    Y = np.empty((n + 1, gas.n_species))
    for i in range(n + 1):
        sim.advance(i * dt)
        T[i] = reactor.thermo.T
        Y[i] = reactor.thermo.Y

    return gas.species_names, T, Y


def write_h2_learning_database(run_folder, log_excluded_species=[]):
    """Write the learning database and run inputs in run_folder. Returns physical (T, Y) at t and t+dt."""

    species, T, Y = ignition_states()
    T_in, Y_in, Y_out = T[:-1], Y[:-1], Y[1:]

    # Transformed mass fractions (N2 removed), as in LearningDatabase.process_database
    i_ann = [i for i, spec in enumerate(species) if spec != "N2"]
    species_ann = [species[i] for i in i_ann]

    def transform(y):
        y_t = y[:, i_ann].copy()
        for j, spec in enumerate(species_ann):
            if spec not in log_excluded_species:
                y_t[:, j] = np.log(np.maximum(y_t[:, j], threshold))
        return y_t

    X = np.column_stack([T_in, transform(Y_in)])
    Y_diff = transform(Y_out) - transform(Y_in)

    # Standardization
    X_mean, X_var = X.mean(axis=0), X.var(axis=0)
    Y_mean, Y_var = Y_diff.mean(axis=0), Y_diff.var(axis=0)
    X_scaled = (X - X_mean) / np.sqrt(X_var)
    Y_scaled = (Y_diff - Y_mean) / np.sqrt(Y_var)

    X_cols = np.array(["Temperature_X"] + [f"{spec}_X" for spec in species_ann], dtype=object)
    Y_cols = np.array([f"{spec}_Y" for spec in species_ann], dtype=object)

    dtb_folder = os.path.join(run_folder, "STOCH_DTB_TEST", "dtb")
    os.makedirs(dtb_folder)
    with h5py.File(os.path.join(dtb_folder, "training_data.h5"), "w") as f:
        grp = f.create_group("CLUSTER_0")
        grp.create_dataset("Xscaler", data=np.column_stack([X_mean, X_var]))
        grp.create_dataset("Yscaler", data=np.column_stack([Y_mean, Y_var]))
        for name, data, cols in [("X_train", X_scaled, X_cols), ("X_val", X_scaled, X_cols),
                                 ("Y_train", Y_scaled, Y_cols), ("Y_val", Y_scaled, Y_cols)]:
            grp.create_dataset(name, data=data).attrs["cols"] = cols

    dtb_processing = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "dtb_file": "solutions.h5",
                            "database_name": "dtb", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": 1, "log_transform_Y": 1, "threshold": threshold, "T_threshold": 200.0,
                            "output_omegas": True, "with_N_chemistry": False,
                            "log_excluded_species": list(log_excluded_species)},
        "data_clustering": {"clusterize_on": "phys", "clustering_method": "kmeans", "nb_clusters": 1},
        "train_set_size": 0.75,
    }
    with open(os.path.join(dtb_folder, "dtb_processing.yaml"), "w") as file:
        yaml.dump(dtb_processing, file, default_flow_style=False)

    with open(os.path.join(run_folder, "STOCH_DTB_TEST", "dtb_params.yaml"), "w") as file:
        yaml.dump({"fuel": ["H2"], "mech_file": mech_file, "pv_species": ["H2O"]}, file, default_flow_style=False)

    networks_params = {
        "database_path": "STOCH_DTB_TEST/dtb",
        "model_name_suffix": "TEST",
        "new_model_folder": True,
        "networks_types": ["MLP"],
        "networks_def": ["cluster0"],
        "clusters": {"cluster0": {"nb_units_in_layers_list": [8], "layers_activation_list": ["tanh", "Id"],
                                  "layers_type": ["dense", "dense"]}},
        "learning": {"batch_size": 16, "epochs_list": [1]},
    }
    with open(os.path.join(run_folder, "networks_params.yaml"), "w") as file:
        yaml.dump(networks_params, file, default_flow_style=False)

    return T_in, Y_in, Y_out
