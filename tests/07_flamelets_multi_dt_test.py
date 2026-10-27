"""
Test 07: multi-dt database generation from a flamelets database.

Writes a small flamelets database (FLAMELETS_DTB_TEST/solutions.h5, same layout as
DatabaseFlamelets.generate_XY_h5) in a temporary folder, then re-integrates its states with
3 random time steps using GenerateVariable_dt.
Check: the multi-dt file contains X, Y and DT with consistent shapes.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import cantera as ct
from ai_reacting_flows.databases_processing.generate_var_dt_dtb import GenerateVariable_dt

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

params = {}
params["mech_file"] = mech_file
params["results_folder_suffix"] = "TEST"
params["dtb_file"] = "solutions.h5"
params["time_step_type"] = "random"
params["time_step_range"] = [1.0e-7, 1.0e-6]
params["nb_dt"] = 3
params["new_file_name"] = "solutions_3dt.h5"


def write_flamelets_database(dtb_folder):
    """Small flamelets database: premixed H2/air states at several temperatures [K]."""

    gas = ct.Solution(mech_file)
    cols = ["Temperature", "Pressure"] + gas.species_names + ["Prog_var", "HRR"]

    states = []
    for T in [1000.0, 1200.0, 1400.0, 1600.0]:
        gas.TP = T, 101325.0
        gas.set_equivalence_ratio(0.6, "H2", {"O2": 1.0, "N2": 3.76})
        states.append(np.concatenate([[gas.T, gas.P], gas.Y, [0.0, 0.0]]))
    data = np.array(states)

    os.makedirs(dtb_folder)
    with h5py.File(os.path.join(dtb_folder, "solutions.h5"), "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=data).attrs["cols"] = cols
        grp.create_dataset("Y", data=data).attrs["cols"] = cols

    return data


def test_flamelets_multi_dt(tmp_path):
    os.chdir(tmp_path)
    data = write_flamelets_database(tmp_path / "FLAMELETS_DTB_TEST")

    GenerateVariable_dt("flamelets", params, dc.DummyComm())

    with h5py.File(tmp_path / "FLAMELETS_DTB_TEST" / "solutions_3dt.h5", "r") as f:
        X = f["ITERATION_00000/X"][()]
        Y = f["ITERATION_00000/Y"][()]
        DT = f["ITERATION_00000/DT"][()]

    assert X.shape == data.shape
    assert Y.shape == (data.shape[0], data.shape[1], 3)
    assert DT.shape == (data.shape[0], 3)
