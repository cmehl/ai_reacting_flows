"""
Test 08: physical mass fractions reconstructed in NN_manager training diagnostics.

Builds a small H2 learning database from a Cantera ignition trajectory (log transform, outputs as
differences of log mass fractions) and gives the exact targets to NN_manager._physical_Yk, which
is used for the mass and elements conservation diagnostics during training.
Check: reconstructed Yk at t and t+dt match the Cantera mass fractions.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import torch
import cantera as ct
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from h2_learning_database import write_h2_learning_database, mech_file


def test_physical_Yk_from_targets(tmp_path):
    os.chdir(tmp_path)
    T_in, Y_in, Y_out = write_h2_learning_database(tmp_path)

    manager = NN_manager()

    with h5py.File(tmp_path / "STOCH_DTB_TEST" / "dtb" / "training_data.h5", "r") as f:
        X_val = torch.tensor(f["CLUSTER_0/X_val"][()])
        Y_val = torch.tensor(f["CLUSTER_0/Y_val"][()])
        Xscaler = torch.tensor(f["CLUSTER_0/Xscaler"][()])
        Yscaler = torch.tensor(f["CLUSTER_0/Yscaler"][()])

    Yk_in, Yk_out = manager._physical_Yk(X_val, Y_val, Xscaler[:, 0], torch.sqrt(Xscaler[:, 1]),
                                         Yscaler[:, 0], torch.sqrt(Yscaler[:, 1]))

    # Compare with Cantera mass fractions without N2
    i_N2 = ct.Solution(mech_file).species_index("N2")
    assert np.allclose(Yk_in.numpy(), np.delete(Y_in, i_N2, axis=1), rtol=1.0e-5, atol=1.0e-9)
    assert np.allclose(Yk_out.numpy(), np.delete(Y_out, i_N2, axis=1), rtol=1.0e-5, atol=1.0e-9)
