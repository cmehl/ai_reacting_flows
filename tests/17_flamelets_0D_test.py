"""
Test 17: 0D reactors of the flamelets database with CVODE time steps.

Computes a few H2/air 0D reactors with DatabaseFlamelets.compute_0d_reactors and
solve_mode = dt_cvode, storing one CVODE step out of 5 (cvode_stride).
Check: the reactors are computed without error and the stored states have a progress
variable in [0, 1]. With T_eq_tolerance [K], each reactor ends when its temperature is within
T_eq_tolerance of the HP equilibrium temperature of its initial mixture.

Independent of the other tests.
"""
import os
import numpy as np
import cantera as ct
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


def test_flamelets_0D_T_eq_tolerance(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("FLAMELETS_DTB_TEST")

    T_eq_tolerance = 20.0
    dtb = DatabaseFlamelets(params, dc.DummyComm())
    # Every CVODE step stored, so that the last stored state is the one that ends the reactor
    dtb.compute_0d_reactors(dict(zerod_params, max_sim_time=1.0e-1, cvode_stride=1, T_eq_tolerance=T_eq_tolerance))

    # Last state of each reactor within T_eq_tolerance of the equilibrium of its initial state, not the one before
    gas = ct.Solution(mech_file)
    for _, sim in dtb.df.groupby("Simulation number"):
        gas.TPY = sim["Temperature"].iloc[0], 101325.0, sim[gas.species_names].iloc[0].to_numpy()
        gas.equilibrate("HP")
        assert abs(sim["Temperature"].iloc[-1] - gas.T) <= T_eq_tolerance
        assert abs(sim["Temperature"].iloc[-2] - gas.T) > T_eq_tolerance
