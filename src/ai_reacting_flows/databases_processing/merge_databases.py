"""Merge several databases (stochastic reactors, canonical flames, or already merged) into one.

The (X, Y) pairs of the iterations of all the input solution files are copied, one file after the other, in
MERGED_DTB_<results_folder_suffix>/<dtb_file>, with iterations renumbered ITERATION_00000, ITERATION_00001, ...
The merged database is then processed like a stochastic one, with database_type: merged (multi-dt
generation and LearningDatabase). Only X and Y are copied: not all_states (stochastic post-processing).

Parameters in dtb_merging.yaml, read from the current working directory:
    input_databases: list of solution files, e.g. ../H2_IGN_KERNEL/STOCH_DTB_H2_POF_2023/solutions.h5
    results_folder_suffix: suffix of the merged database folder
    dtb_file: name of the merged solution file
    pv_species: species of the progress variable, which must be the ones of all input databases
"""

import os
import shutil

import h5py
import numpy as np
import oyaml as yaml


# Generation parameters copied in a database folder, by type of database
GENERATION_YAML_FILES = ["dtb_params.yaml", "dtb_params_flmts.yaml", "dtb_merging.yaml"]


def _generation_parameters(dtb_folder):
    """Generation parameters (dict) of a database, from the yaml file copied in its folder."""

    for file_name in GENERATION_YAML_FILES:
        file_path = os.path.join(dtb_folder, file_name)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                return yaml.safe_load(file)

    raise FileNotFoundError(f"No generation yaml file ({', '.join(GENERATION_YAML_FILES)}) in {dtb_folder}")


def merge_databases():
    """Merge the databases listed in dtb_merging.yaml. Returns the path of the merged solution file."""

    run_folder = os.getcwd()
    with open(os.path.join(run_folder, "dtb_merging.yaml"), "r") as file:
        params = yaml.safe_load(file)

    print("WARNING: the time steps of the input databases are not checked. They must be the same,")
    print("         unless a multi-dt database is generated from the merged database (only X values are used in this case).\n")

    # Output folder (deleted if it exists, as for database generation)
    results_folder = os.path.join(run_folder, f"MERGED_DTB_{params['results_folder_suffix']}")
    if os.path.isdir(results_folder):
        print(f">> Merged database folder {results_folder} already exists => deleting")
        shutil.rmtree(results_folder)
    os.mkdir(results_folder)
    shutil.copy(os.path.join(run_folder, "dtb_merging.yaml"), results_folder)

    output_file = os.path.join(results_folder, params["dtb_file"])
    i_out = 0
    with h5py.File(output_file, "w") as f_out:

        for i_dtb, input_file in enumerate(params["input_databases"]):

            # Same progress variable as the merged database (its definition is in the copied generation yaml)
            input_folder = os.path.dirname(os.path.abspath(input_file))
            pv_species = _generation_parameters(input_folder)["pv_species"]
            if pv_species != params["pv_species"]:
                raise ValueError(f"pv_species of {input_file} ({pv_species}) differ from dtb_merging.yaml ({params['pv_species']})")

            # Copy of the generation yaml, to document the merged database
            input_yaml_dir = os.path.join(results_folder, f"input_{i_dtb}")
            os.mkdir(input_yaml_dir)
            for file_name in GENERATION_YAML_FILES:
                if os.path.isfile(os.path.join(input_folder, file_name)):
                    shutil.copy(os.path.join(input_folder, file_name), input_yaml_dir)

            with h5py.File(input_file, "r") as f_in:

                nb_iterations = len(f_in.keys())
                print(f">> {input_file}: {nb_iterations} iterations")

                # Checks on the first iteration: all iterations of a file have the same layout
                grp_first = f_in["ITERATION_00000"]

                # Only single-dt databases: the dt of a multi-dt database (DT) would be lost
                if "DT" in grp_first:
                    raise ValueError(f"{input_file} is a multi-dt database: merge the databases before the multi-dt generation")

                # Same columns (same mechanism and species order) as the first database
                cols_X = grp_first["X"].attrs["cols"]
                cols_Y = grp_first["Y"].attrs["cols"]
                if i_dtb == 0:
                    cols_X_ref, cols_Y_ref = cols_X, cols_Y
                if not (np.array_equal(cols_X, cols_X_ref) and np.array_equal(cols_Y, cols_Y_ref)):
                    raise ValueError(f"Columns of {input_file} differ from the ones of the first database")

                # Copy of the iterations, numbered after the ones of the previous databases
                for i in range(nb_iterations):
                    grp_in = f_in[f"ITERATION_{i:05d}"]

                    grp_out = f_out.create_group(f"ITERATION_{i_out:05d}")
                    grp_out.create_dataset("X", data=grp_in["X"][()]).attrs["cols"] = cols_X
                    grp_out.create_dataset("Y", data=grp_in["Y"][()]).attrs["cols"] = cols_Y
                    i_out += 1

    print(f"\n>> Merged database: {i_out} iterations in {output_file}")

    return output_file
