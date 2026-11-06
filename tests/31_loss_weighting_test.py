"""
Test 31: increment weights of the training loss (learning.loss_weighting_p, loss_weighting_eps).

Check:
- increment_weights: p = 0 gives weights 1; weights (omega^2 + eps^2)^(-p) normalized to a mean of 1 per species on the
  points that are not clipped; weight 1 for clipped points and species without ln transform; normalization given for
  validation points;
- NN_manager._loss_weights on a small H2 learning database: increments computed from the Cantera mass fractions;
- the weighted loss is the mean of w (y_pred - y)^2, and the usual loss with weights 1; training runs and logs the
  weights and the unweighted validation loss;
- errors for outputs that are not increments of ln Y, and for another loss than MSE.

Independent of the other tests.
"""
import os
import glob

import numpy as np
import oyaml as yaml
import pytest
import torch
import torch.nn as nn
import cantera as ct

from ai_reacting_flows.tools.loss_weighting import increment_weights, effective_fraction
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from h2_learning_database import write_h2_learning_database, mech_file, threshold

p = 0.5
eps = 1.0e-3


def set_learning(run_folder, **learning):
    """Add learning parameters to networks_params.yaml."""

    with open(os.path.join(run_folder, "networks_params.yaml"), "r") as file:
        params = yaml.safe_load(file)
    params["learning"].update(learning)
    with open(os.path.join(run_folder, "networks_params.yaml"), "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def test_increment_weights():
    log_thr = np.log(threshold)

    # Species 0: weighted, last point clipped at t and t+dt. Species 1: without ln transform
    x = np.array([[-1.0, 0.1], [-2.0, 0.2], [-3.0, 0.3], [log_thr, 0.4]])
    omega = np.array([[1.0e-1, 1.0], [1.0e-3, 2.0], [1.0e-5, 3.0], [0.0, 4.0]])

    # p = 0: usual loss
    w, _ = increment_weights(x, omega, log_thr, [0], 0.0, eps)
    assert np.allclose(w, 1.0)

    w, mean_ref = increment_weights(x, omega, log_thr, [0], p, eps)
    w_ref = (omega[:3, 0]**2 + eps**2) ** (-p)
    assert np.allclose(w[:3, 0], w_ref / w_ref.mean())
    assert np.isclose(w[:3, 0].mean(), 1.0)
    assert w[3, 0] == 1.0 and np.all(w[:, 1] == 1.0)

    # Bounded ratio, and smaller weights for larger increments
    assert w[0, 0] < w[1, 0] < w[2, 0] and w[2, 0] / w[0, 0] <= (omega[0, 0] / eps) ** (2 * p) * (1 + 1e-12)

    # Validation points with the normalization of training
    w_val, _ = increment_weights(x[:1], omega[:1], log_thr, [0], p, eps, mean_ref)
    assert np.isclose(w_val[0, 0], w[0, 0])

    assert effective_fraction(np.ones(10)) == 1.0 and effective_fraction(w) < 1.0


def test_loss_weighting_training(tmp_path):
    os.chdir(tmp_path)
    T_in, Y_in, Y_out = write_h2_learning_database(tmp_path)
    set_learning(tmp_path, loss_weighting_p=p, loss_weighting_eps=eps)

    manager = NN_manager()
    X_train, X_val, Y_train, Y_val = manager.read_training_data(0)
    Xm, Xv, Ym, Yv = manager.get_scalers_stats(0)
    W, _ = manager._loss_weights(X_train, Y_train, Xm, np.sqrt(Xv), Ym, np.sqrt(Yv))

    # Increments of ln Y from the Cantera mass fractions (N2 removed)
    i_N2 = ct.Solution(mech_file).species_index("N2")
    ln_in = np.log(np.maximum(np.delete(Y_in, i_N2, axis=1), threshold))
    ln_out = np.log(np.maximum(np.delete(Y_out, i_N2, axis=1), threshold))
    W_ref, _ = increment_weights(ln_in, ln_out - ln_in, np.log(threshold), range(ln_in.shape[1]), p, eps)
    assert np.allclose(W, W_ref, rtol=1.0e-6)

    # Weighted MSE, and usual loss with weights 1
    y_pred, y = torch.randn(5, 3), torch.randn(5, 3)
    w = torch.rand(5, 3)
    loss_fn = nn.MSELoss()
    assert torch.isclose(manager._loss(loss_fn, y_pred, y, w), torch.mean(w * (y_pred - y)**2))
    assert torch.isclose(manager._loss(loss_fn, y_pred, y, torch.ones(5, 3)), loss_fn(y_pred, y))

    # Training with the weights
    manager.train_all_clusters()
    log = open(glob.glob(str(tmp_path / "nn_training_*.log"))[0]).read()
    assert "LOSS_WEIGHTS cluster=0" in log and "val_loss_unweighted=" in log


def test_loss_weighting_errors(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)

    # Loss other than MSE
    set_learning(tmp_path, loss_weighting_p=p, loss_weighting_eps=eps, loss="L1")
    with pytest.raises(ValueError, match="MSE"):
        NN_manager()

    # Outputs that are not increments of ln Y
    set_learning(tmp_path, loss="MSE")
    dtb_yaml = tmp_path / "STOCH_DTB_TEST" / "dtb" / "dtb_processing.yaml"
    params = yaml.safe_load(open(dtb_yaml))
    params["data_processing"]["output_omegas"] = False
    yaml.dump(params, open(dtb_yaml, "w"), default_flow_style=False)
    with pytest.raises(ValueError, match="output_omegas"):
        NN_manager()
