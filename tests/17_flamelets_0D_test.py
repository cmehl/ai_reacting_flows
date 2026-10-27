"""
Test 17: 0D reactors of the flamelets database with CVODE time steps.

Computes a few H2/air 0D reactors with DatabaseFlamelets.compute_0d_reactors and
solve_mode = dt_cvode, storing one CVODE step out of 5 (cvode_stride).
Check: the reactors are computed without error and the stored states have a progress
variable in [0, 1].

Independent of the other tests.
"""
import os
import numpy as np
from ai_reacting_flows.flmts_reactors_data_gen.database_flmts import DatabaseFlamelets

import DummyComm as dc

# Mechanism file
mech_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/chemical_mechanisms/mech_H2.yaml")

params = {"mech_file": mech_file, "fuel": ["H2"], "time_step": 1.0e-6, "include_zerod": True,
          "include_oned_prem": False, "include_oned_diff": False, "augment_dataset": False,
          "results_folder_suffix": "TEST", "dtb_file": "solutions.h5", "pressure": 101325.0, "pv_species": ["H2O"]}

zerod_params = {"phi_bounds": [0.6, 1.0], "T0_bounds": [1200.0, 1400.0], "n_samples": 5,
                "max_sim_time": 1.0e-4, "solve_mode": "dt_cvode", "cvode_stride": 5}


def test_flamelets_0D_dt_cvode(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("FLAMELETS_DTB_TEST")

    dtb = DatabaseFlamelets(params, dc.DummyComm())
    dtb.compute_0d_reactors(zerod_params)

    progvar = dtb.df["Progress_variable"].to_numpy()
    assert progvar.size > 0
    assert np.all((progvar >= 0.0) & (progvar <= 1.0 + 1.0e-6))
