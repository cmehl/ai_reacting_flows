"""
Test 09: NNTesting inference with log_excluded_species and output omegas.

Builds a small H2 learning database from a Cantera ignition trajectory (log transform except for
H2O, outputs as differences), trains a model for one epoch to create the model folder, then
replaces the network by one returning the exact (scaled) target of a chosen state.
Check: NNTesting.advance_state_NN gives back the Cantera mass fractions at t+dt, including
the species kept on a linear scale (H2O) and N2.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import torch
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from h2_learning_database import write_h2_learning_database, dt, pressure


def test_inference_log_excluded_species(tmp_path):
    os.chdir(tmp_path)
    T_in, Y_in, Y_out = write_h2_learning_database(tmp_path, log_excluded_species=["H2O"])

    NN_manager().train_all_clusters()

    testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": False,
                          "hybrid_ann_cvode": False, "hybrid_ann_cvode_tol": 5.0e-05}
    test = NNTesting(testing_parameters)
    test.cluster = 0

    # Network replaced by the exact scaled target of state i (during ignition)
    i = 40
    with h5py.File(tmp_path / "STOCH_DTB_TEST" / "dtb" / "training_data.h5", "r") as f:
        target = torch.tensor(f["CLUSTER_0/Y_val"][i:i+1])
    test.models_list[0] = lambda x: target

    T_new, Y_new = test.advance_state_NN(T_in[i], Y_in[i], pressure, dt)

    assert np.allclose(np.ravel(Y_new), Y_out[i], rtol=1.0e-5, atol=1.0e-9)
