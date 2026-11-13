"""
Test 13: NNTesting 0D ignition and progress variable.

Builds a small H2 learning database (N2 removed) and trains a model for one epoch.
Check:
- compute_progvar gives c = (Yc - Yc_u) / (Yc_eq - Yc_u): 0 for the unburnt mixture (here
  containing H2O, a progress variable species) and 1 at equilibrium.
- run_0D_ignition_case runs with N2 removed from the network species.
- with progress variable clustering, the 0D case attributes clusters from the progress variable
  of the current state (states advanced with CVODE through the hybrid mode, so that it ignites).
- the 0D case stops at the first step where the CVODE temperature is within T_eq_tolerance of the
  equilibrium temperature, with the same number of ANN steps.
- the cluster of each step is recorded (-1 for CVODE steps of the hybrid mode, with the criterion on the
  sum of mass fractions or on the element mass fractions), and the ANN can be
  continued after equilibrium (drifts of T and of the sum of Y_k).
- with dT_tolerance, the 0D case stops later, when the CVODE temperature changes by less than dT_tolerance per step.
- with cvode_initial_steps, the first steps are done by CVODE (cluster -1, same states as the reference) and the
  ANN does the other ones.

Independent of the other tests.
"""
import os
import numpy as np
import cantera as ct
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from h2_learning_database import write_h2_learning_database, mech_file, dt, pressure

testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": 0,
                      "hybrid_ann_cvode": 0, "hybrid_ann_cvode_tol": 5.0e-05}


def test_nn_testing_0D(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    NN_manager().train_all_clusters()
    test = NNTesting(testing_parameters)

    # Progress variable (pv_species: H2O) of an unburnt mixture containing H2O, and of its equilibrium
    gas = ct.Solution(mech_file)
    gas.TPX = 900.0, pressure, "H2:0.2, O2:0.15, N2:0.55, H2O:0.1"
    state_u = np.append(gas.T, gas.Y)
    Yc_u = gas["H2O"].Y[0]
    gas.equilibrate("HP")
    state_eq = np.append(gas.T, gas.Y)

    assert np.isclose(test.compute_progvar(state_u, pressure, Yc_u), 0.0, atol=1.0e-10)
    assert np.isclose(test.compute_progvar(state_eq, pressure, Yc_u), 1.0, rtol=1.0e-6)

    # 0D ignition with N2 removed from the network species
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=3 * dt)
    assert result["states_ann"].shape == (4, 1 + gas.n_species)

    # Norm Omega of the scaled increments of each step: increments of ln Y of the network species over the standard
    # deviations of the outputs (one cluster)
    species = test.inference.spec_list_ANN
    cols = [gas.species_names.index(spec) + 1 for spec in species]
    ln_Y = np.log(np.maximum(result["states_ref"][:, cols], test.inference.threshold))
    std = test.inference.Yscaler_list[0][:, 1]
    assert np.allclose(result["Omega_norm_ref"], np.sqrt(np.sum((np.diff(ln_Y, axis=0) / std)**2, axis=1)))
    assert len(result["Omega_norm_ann"]) == 3

    # Progress variable clustering (2 clusters with the same network), states advanced with CVODE
    test.inference.nb_clusters = 2
    test.inference.clustering_method = "progvar"
    test.inference.c_bounds = [0.0, 0.5, 1.0]
    test.inference.models_list = 2 * test.inference.models_list
    test.inference.Xscaler_list = 2 * test.inference.Xscaler_list
    test.inference.Yscaler_list = 2 * test.inference.Yscaler_list
    test.hybrid_ann_cvode = 1
    test.hybrid_ann_cvode_tol = -1.0

    clusters = []
    attribute_cluster = test.attribute_cluster
    def spy(state, dt, progvar):
        attribute_cluster(state, dt, progvar)
        clusters.append(test.cluster)
    test.attribute_cluster = spy

    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=64 * dt)
    progvar = result["progvar"][:-1]   # progress variable of the state before each step

    assert progvar.max() > 0.9
    assert clusters[1:] == [0 if c < 0.5 else 1 for c in progvar]   # First call: ANN warm-up
    assert np.all(result["clusters"] == -1)   # All steps done by CVODE (hybrid with negative tolerance)

    # Same with the hybrid criterion on the element mass fractions
    test.hybrid_ann_cvode = 2
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=4 * dt)
    assert np.all(result["clusters"] == -1)
    test.hybrid_ann_cvode = 1

    # Stop at CVODE equilibrium (max_simulation_time not reached), same number of ANN steps
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=1.0, T_eq_tolerance=1.0)
    T_ref = result["states_ref"][:, 0]

    assert result["equilibrium_reached"]
    assert abs(T_ref[-1] - result["T_eq"]) <= 1.0 and abs(T_ref[-2] - result["T_eq"]) > 1.0
    assert result["states_ann"].shape == result["states_ref"].shape
    assert len(T_ref) < 1.0 / dt

    # Stop only when the CVODE temperature changes by less than dT_tolerance over a step: later
    dT_tolerance = 1.0e-3
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=1.0, T_eq_tolerance=1.0,
                                       dT_tolerance=dT_tolerance)
    T_ref_rest = result["states_ref"][:, 0]

    assert result["equilibrium_reached"]
    assert len(T_ref_rest) > len(T_ref)
    assert abs(T_ref_rest[-1] - T_ref_rest[-2]) <= dT_tolerance and abs(T_ref_rest[-2] - T_ref_rest[-3]) > dT_tolerance

    # Stability after equilibrium: ANN continued, drifts of T and sum(Y_k)
    test.hybrid_ann_cvode = 0
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=4 * dt, stability_factor=2.0)
    assert result["T_drift"] >= 0.0 and result["sumYk_drift"] >= 0.0
    assert len(result["clusters"]) == 4 and np.all(result["clusters"] >= 0)

    # First step done by CVODE (database without the initial mixtures), the other ones by the ANN
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=4 * dt, cvode_initial_steps=1)
    assert result["clusters"][0] == -1 and np.all(result["clusters"][1:] >= 0)
    assert np.allclose(result["states_ann"][:2], result["states_ref"][:2], rtol=1.0e-10, atol=0.0)   # Same up to round-off
    assert not np.allclose(result["states_ann"][2], result["states_ref"][2], rtol=1.0e-10, atol=0.0)
    assert result["cvode_calls"] == 1 and result["ann_calls"] == 3
