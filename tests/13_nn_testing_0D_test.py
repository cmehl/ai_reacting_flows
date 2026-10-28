"""
Test 13: NNTesting 0D ignition and progress variable.

Builds a small H2 learning database (N2 removed) and trains a model for one epoch.
Check:
- compute_progvar gives c = (Yc - Yc_u) / (Yc_eq - Yc_u): 0 for the unburnt mixture (here
  containing H2O, a progress variable species) and 1 at equilibrium.
- run_0D_ignition_case runs with N2 removed from the network species.
- with progress variable clustering, the 0D case attributes clusters from the progress variable
  of the current state (states advanced with CVODE through the hybrid mode, so that it ignites).

Independent of the other tests.
"""
import os
import numpy as np
import cantera as ct
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from h2_learning_database import write_h2_learning_database, mech_file, dt, pressure

testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": False,
                      "hybrid_ann_cvode": False, "hybrid_ann_cvode_tol": 5.0e-05}


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
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, simulation_time=3 * dt)
    assert result["states_ann"].shape == (4, 1 + gas.n_species)

    # Progress variable clustering (2 clusters with the same network), states advanced with CVODE
    test.inference.nb_clusters = 2
    test.inference.clustering_method = "progvar"
    test.inference.c_bounds = [0.0, 0.5, 1.0]
    test.inference.models_list = 2 * test.inference.models_list
    test.inference.Xscaler_list = 2 * test.inference.Xscaler_list
    test.inference.Yscaler_list = 2 * test.inference.Yscaler_list
    test.hybrid_ann_cvode = True
    test.hybrid_ann_cvode_tol = -1.0

    clusters = []
    attribute_cluster = test.attribute_cluster
    def spy(state, dt, progvar):
        attribute_cluster(state, dt, progvar)
        clusters.append(test.cluster)
    test.attribute_cluster = spy

    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, simulation_time=64 * dt)
    progvar = result["progvar"][:-1]   # progress variable of the state before each step

    assert progvar.max() > 0.9
    assert clusters == [0 if c < 0.5 else 1 for c in progvar]
