"""
Test 30: merging of databases (merge_databases.py) and processing of the merged database.

STOCH_DTB_TEST/solutions.h5 (session fixture h2_stoch_dtb) is merged with itself in MERGED_DTB_TEST.
Checks: iterations copied one file after the other, without all_states; LearningDatabase with
database_type: merged reads all of them; errors on different pv_species and on a multi-dt input.
"""
import os

import h5py
import numpy as np
import oyaml as yaml
import pytest

from ai_reacting_flows.databases_processing.merge_databases import merge_databases
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from h2_stoch_database import mech_file


def write_merging_yaml(input_databases, pv_species=["H2O"]):
    params = {"input_databases": input_databases, "results_folder_suffix": "TEST", "dtb_file": "solutions.h5",
              "pv_species": pv_species}
    with open("dtb_merging.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def test_merge_and_process(h2_stoch_dtb, tmp_path):
    os.chdir(tmp_path)
    input_file = str(h2_stoch_dtb / "STOCH_DTB_TEST" / "solutions.h5")
    write_merging_yaml([input_file, input_file])

    output_file = merge_databases()

    # Iterations of the first file, then of the second one, X and Y only
    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "r") as f_out:
        n = len(f_in.keys())
        assert len(f_out.keys()) == 2 * n
        for i in [0, n - 1]:
            for i_out in [i, n + i]:
                grp_out = f_out[f"ITERATION_{i_out:05d}"]
                assert set(grp_out.keys()) == {"X", "Y"}
                np.testing.assert_array_equal(grp_out["X"][()], f_in[f"ITERATION_{i:05d}/X"][()])
                np.testing.assert_array_equal(grp_out["Y"][()], f_in[f"ITERATION_{i:05d}/Y"][()])
        nb_points = sum(f_in[k]["X"].shape[0] for k in f_in.keys())

    assert os.path.isfile(tmp_path / "MERGED_DTB_TEST" / "dtb_merging.yaml")
    assert os.path.isfile(tmp_path / "MERGED_DTB_TEST" / "input_1" / "dtb_params.yaml")

    # Processing of the merged database: all iterations are read
    dtb_processing_parameters = {
        "database_params": {"database_type": "merged", "dtb_folder_suffix": "TEST", "database_name": "dtb",
                            "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": 1, "log_transform_Y": 1, "threshold": 1.0e-10, "T_threshold": 200.0,
                            "output_omegas": False, "with_N_chemistry": False},
        "data_clustering": {"clustering_method": "kmeans", "nb_clusters": 1, "clusterize_on": "phys"},
        "train_set_size": 0.75}
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    database = LearningDatabase()
    assert len(database.X) == 2 * nb_points

    database.process_database()
    assert os.path.isfile(tmp_path / "MERGED_DTB_TEST" / "dtb" / "training_data.h5")


def test_merge_errors(h2_stoch_dtb, tmp_path):
    os.chdir(tmp_path)
    input_file = str(h2_stoch_dtb / "STOCH_DTB_TEST" / "solutions.h5")

    # Progress variable species different from the input database
    write_merging_yaml([input_file], pv_species=["H2O", "OH"])
    with pytest.raises(ValueError, match="pv_species"):
        merge_databases()

    # Multi-dt input database (group with DT)
    multi_dt_folder = tmp_path / "STOCH_DTB_MULTI_DT"
    multi_dt_folder.mkdir()
    with open(multi_dt_folder / "dtb_params.yaml", "w") as file:
        yaml.dump({"pv_species": ["H2O"]}, file)
    with h5py.File(multi_dt_folder / "solutions_5dt.h5", "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=np.zeros((2, 3))).attrs["cols"] = ["Temperature", "Pressure", "H2"]
        grp.create_dataset("Y", data=np.zeros((2, 3, 5))).attrs["cols"] = ["Temperature", "Pressure", "H2"]
        grp.create_dataset("DT", data=np.zeros((2, 5)))

    write_merging_yaml([str(multi_dt_folder / "solutions_5dt.h5")])
    with pytest.raises(ValueError, match="multi-dt"):
        merge_databases()
