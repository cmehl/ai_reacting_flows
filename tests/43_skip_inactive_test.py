"""
Test 43: inactive states (tools/skip_inactive.py, criterion "log" of the skip_inactive of OneDCombustion).

Check:
- active_states: fresh H2/air at 300 K and burnt gas at equilibrium are inactive; radical-free H2/air at 1000 K
  (induction period of autoignition) is active.
- LearningDatabase.skip_inactive_points on the H2 stochastic database of the session (fixture h2_stoch_dtb): inactive
  points (fresh gas) are removed, X and Y keep the same points, and all remaining points are active.
- NNTesting with skip_inactive: a 0D case of fresh gas at 300 K is kept unchanged by the ANN (steps skipped,
  cluster -1), the CVODE reference is not skipped.

Independent of the other tests.
"""
import os

import numpy as np
import cantera as ct
import oyaml as yaml

from ai_reacting_flows.tools.skip_inactive import active_states
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from h2_learning_database import write_h2_learning_database, mech_file, dt, pressure

dt_stoch = 5.0e-6   # Time step of the H2 stochastic database of the session [s]


def test_active_states():
    gas = ct.Solution(mech_file)
    T, Y = [], []

    # Fresh gas at 300 K, radical-free gas at 1000 K
    for T0 in [300.0, 1000.0]:
        gas.TPX = T0, pressure, "H2:2, O2:1, N2:3.76"
        T.append(gas.T)
        Y.append(gas.Y)

    # Burnt gas at equilibrium: end of a 1 s reactor (gas.equilibrate is not the steady state of the kinetics,
    # some reactions of the mechanism have no reverse rate)
    reactor = ct.IdealGasConstPressureReactor(gas)
    ct.ReactorNet([reactor]).advance(1.0)
    T.append(gas.T)
    Y.append(gas.Y)

    active = active_states(gas, np.array(T), np.array(Y), pressure, 1.0e-6)
    assert list(active) == [False, True, False]


def test_skip_inactive_points(h2_stoch_dtb, tmp_path):
    # Own run folder linked to the database (as test 32)
    os.chdir(tmp_path)
    os.symlink(h2_stoch_dtb / "STOCH_DTB_TEST", tmp_path / "STOCH_DTB_TEST")
    dtb_processing_parameters = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb_skip",
                            "dtb_file": "solutions.h5", "dt_var": False, "fuel": "H2", "mech_file": mech_file},
        "data_processing": {"log_transform_X": 1, "log_transform_Y": 1, "threshold": 1.0e-10, "T_threshold": 200.0,
                            "output_omegas": True, "with_N_chemistry": False},
        "data_clustering": {"clustering_method": "kmeans", "nb_clusters": 1, "clusterize_on": "phys"},
        "train_set_size": 0.75}
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing_parameters, file, default_flow_style=False)

    database = LearningDatabase()
    n_points = len(database.X)
    T_min = database.X["Temperature"].min()

    database.skip_inactive_points(dt=dt_stoch)

    # Fresh gas removed, same points in X and Y
    assert len(database.X) < n_points
    assert len(database.Y) == len(database.X)
    assert database.X["Temperature"].min() > T_min

    # All remaining points active
    gas = ct.Solution(mech_file)
    active = active_states(gas, database.X["Temperature"].values, database.X[gas.species_names].values,
                           database.X["Pressure"].values, dt_stoch)
    assert active.all()


def test_nn_testing_skip_inactive(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    NN_manager().train_all_clusters()

    testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": 0,
                          "hybrid_ann_cvode": 0, "hybrid_ann_cvode_tol": 5.0e-05, "skip_inactive": True}
    test = NNTesting(testing_parameters)

    # Fresh gas: all ANN steps skipped, state unchanged
    nb_steps = 3
    result = test.run_0D_ignition_case(phi=0.6, T0=300.0, pressure=pressure, dt=dt, max_simulation_time=nb_steps * dt)
    assert result["skipped_calls"] == nb_steps
    assert result["ann_calls"] == 0
    assert np.all(result["clusters"] == -1)
    assert np.all(result["states_ann"] == result["states_ann"][0])
