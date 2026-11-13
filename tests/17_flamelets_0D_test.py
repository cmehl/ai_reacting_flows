"""
Test 17: 0D reactors of the flamelets database with CVODE time steps.

Computes a few H2/air 0D reactors with DatabaseFlamelets.compute_0d_reactors and
solve_mode = dt_cvode, storing one CVODE step out of 5 (cvode_stride).
Check: the reactors are computed without error and the stored states have a progress
variable in [0, 1]. With T_eq_tolerance [K], each reactor ends when its temperature is within
T_eq_tolerance of the HP equilibrium temperature of its initial mixture, and rich NH3 reactors, which
cross the equilibrium temperature on the ignition front, end after their temperature overshoot.
With dT_tolerance [K], they end only when their temperature changes by less than dT_tolerance over time_step, after
they have cooled down through the equilibrium temperature.
With skip_initial_state, the initial mixture of each reactor is not stored, the other states are unchanged.

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


def test_flamelets_0D_rich_overshoot(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("FLAMELETS_DTB_TEST")

    # Rich NH3/air: T rises about 50 K per step on the ignition front through T_eq, up to an overshoot of about 100 K.
    # A tolerance of 30 K alone would end all reactors on the front
    mech_nh3 = os.path.join(os.path.dirname(mech_file), "mech_nh3_noArHe.yaml")
    params_nh3 = dict(params, mech_file=mech_nh3, fuel=["NH3"])
    zerod_rich = {"phi_bounds": [1.2, 1.3], "T0_bounds": [1900.0, 2000.0], "n_samples": 4, "max_sim_time": 2.0e-3,
                  "solve_mode": "dt_cfd", "T_eq_tolerance": 30.0}
    dtb = DatabaseFlamelets(params_nh3, dc.DummyComm())
    dtb.compute_0d_reactors(zerod_rich)

    for _, sim in dtb.df.groupby("Simulation number"):
        T = sim["Temperature"].to_numpy()
        assert np.argmax(T) < len(T) - 1
    n_states = dtb.df.groupby("Simulation number").size()

    # With T_eq_tolerance alone the reactors end while cooling down through T_eq (about 0.2 K per step); with
    # dT_tolerance they end later, at rest
    dT_tolerance = 0.01
    dtb = DatabaseFlamelets(params_nh3, dc.DummyComm())
    dtb.compute_0d_reactors(dict(zerod_rich, max_sim_time=5.0e-3, T_eq_tolerance=2.0, dT_tolerance=dT_tolerance))

    for number, sim in dtb.df.groupby("Simulation number"):
        T = sim["Temperature"].to_numpy()
        assert abs(T[-1] - T[-2]) <= dT_tolerance
        assert len(T) > 2 * n_states[number]


def test_flamelets_0D_skip_initial_state(tmp_path):
    os.chdir(tmp_path)
    os.mkdir("FLAMELETS_DTB_TEST")

    zerod_cfd = dict(zerod_params, solve_mode="dt_cfd")

    dtb = DatabaseFlamelets(params, dc.DummyComm())
    dtb.compute_0d_reactors(zerod_cfd)
    df_all = dtb.df

    dtb = DatabaseFlamelets(params, dc.DummyComm())
    dtb.compute_0d_reactors(dict(zerod_cfd, skip_initial_state=True))
    df_skip = dtb.df

    # Same states without the first one of each reactor (the initial mixture, at T0 of sim_train_0D.csv)
    for number, sim in df_all.groupby("Simulation number"):
        sim_skip = df_skip[df_skip["Simulation number"] == number]
        assert np.array_equal(sim_skip.to_numpy(), sim.iloc[1:].to_numpy())
        assert sim_skip["Temperature"].iloc[0] != dtb.df_ODE_0D["T0"].iloc[int(number)]
