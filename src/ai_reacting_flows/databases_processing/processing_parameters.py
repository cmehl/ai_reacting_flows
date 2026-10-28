"""
Processing parameters of a learning database, stored as attributes of training_data.h5.

LearningDatabase.process_database writes them, so that training (NN_manager) and testing (NNTesting)
read the parameters actually used to build the training data, without re-reading the YAML files
of database processing and generation. Databases processed before this was introduced have no
such attributes: their parameters are then read from the YAML files, as before.

Parameters (keys of the dictionary):
    dt_var, fuel                                 database_params of dtb_processing.yaml
    log_transform_X, log_transform_Y, lambda_bct,
    threshold, output_omegas, with_N_chemistry,
    log_excluded_species                         data_processing of dtb_processing.yaml
    clustering_method, clusterize_on, nb_clusters data_clustering of dtb_processing.yaml
    pv_species                                   generation parameters (dtb_params.yaml or dtb_params_flmts.yaml)

The mechanism is the copy mech_detailed.yaml written by LearningDatabase in the database folder.
"""

import os

import h5py
import numpy as np
import oyaml as yaml

# Name of the mechanism copied in the database folder by LearningDatabase
MECHANISM_FILE = "mech_detailed.yaml"


def write_processing_parameters(h5file, params):
    """Write the parameters (dict) as attributes of the open h5 file h5file."""

    for key, value in params.items():

        # Lists of names (possibly empty): variable-length strings
        if isinstance(value, list):
            h5file.attrs[key] = np.array(value, dtype=h5py.string_dtype())

        else:
            h5file.attrs[key] = value


def read_processing_parameters(dataset_path, run_folder):
    """Parameters of the learning database in dataset_path (folder of training_data.h5).

    Read from the attributes of training_data.h5, or, for databases processed before they were
    written, from dataset_path/dtb_processing.yaml and the generation parameters of the
    stochastic/flamelets database in run_folder.
    """

    with h5py.File(os.path.join(dataset_path, "training_data.h5"), "r") as h5file:

        # Parameters stored at database processing
        if "log_transform_X" in h5file.attrs:
            return {key: _from_attribute(value) for key, value in h5file.attrs.items()}

    # Older databases: YAML files
    return _read_yaml_parameters(dataset_path, run_folder)


def _from_attribute(value):
    """Python value (bool, int, float, str, list of str) from an h5 attribute."""

    # Lists of names
    if isinstance(value, np.ndarray):
        return [v.decode() if isinstance(v, bytes) else str(v) for v in value]

    # Strings and scalars
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()

    return value


def _read_yaml_parameters(dataset_path, run_folder):
    """Parameters from dtb_processing.yaml (copied in dataset_path) and the generation parameters."""

    with open(os.path.join(dataset_path, "dtb_processing.yaml"), "r") as file:
        dtb_processing_params = yaml.safe_load(file)

    database_params = dtb_processing_params["database_params"]
    data_processing = dtb_processing_params["data_processing"]
    data_clustering = dtb_processing_params["data_clustering"]

    # Generation parameters: stochastic reactors or flamelets database
    if database_params["database_type"] == "stoch":
        params_file = os.path.join(f"STOCH_DTB_{database_params['dtb_folder_suffix']}", "dtb_params.yaml")
    elif database_params["database_type"] == "flamelets":
        params_file = os.path.join(f"FLAMELETS_DTB_{database_params['dtb_folder_suffix']}", "dtb_params_flmts.yaml")

    with open(os.path.join(run_folder, params_file), "r") as file:
        dtb_parameters = yaml.safe_load(file)

    return {
        "dt_var": database_params["dt_var"],
        "fuel": database_params["fuel"],
        "log_transform_X": data_processing["log_transform_X"],
        "log_transform_Y": data_processing["log_transform_Y"],
        "lambda_bct": data_processing.get("lambda_bct", 0.1),
        "threshold": data_processing["threshold"],
        "output_omegas": data_processing["output_omegas"],
        "with_N_chemistry": data_processing["with_N_chemistry"],
        "log_excluded_species": data_processing.get("log_excluded_species", []),
        "clustering_method": data_clustering["clustering_method"],
        "clusterize_on": data_clustering.get("clusterize_on", "phys"),
        "nb_clusters": data_clustering["nb_clusters"],
        "pv_species": dtb_parameters["pv_species"],
    }
