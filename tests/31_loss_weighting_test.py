"""
Test 31: increment weights of the training loss (learning.loss_weighting_p, loss_weighting_eps, loss_weighting_q,
loss_weighting_min_count).

Check:
- increment_norm and increment_weights: p = q = 0 gives weights 1; weights (Omega + eps)^(-p), one per point,
  normalized to a mean of 1, with Omega^2 = sum_k (omega_k / sigma_k)^2; reference given for validation points;
- density factor: with q = 1, all bins of log10(Omega) have the same total weight; min_count bounds the weights of
  rare bins;
- NN_manager._loss_weights on a small H2 learning database: Omega computed from the Cantera mass fractions;
- the weighted loss is the mean of w (y_pred - y)^2, and the usual loss with weights 1; training runs and logs the
  weights and the unweighted validation loss;
- error_per_decade, and its log and plots after training, for outputs that are increments only;
- gradient shares of the decades (norm and projection), checked against the gradient of the whole loss;
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

from ai_reacting_flows.tools.loss_weighting import increment_norm, increment_weights, error_per_decade, effective_fraction
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from h2_learning_database import write_h2_learning_database, mech_file, threshold

p = 1.0
eps = 1.0e-2


def set_learning(run_folder, **learning):
    """Add learning parameters to networks_params.yaml."""

    with open(os.path.join(run_folder, "networks_params.yaml"), "r") as file:
        params = yaml.safe_load(file)
    params["learning"].update(learning)
    with open(os.path.join(run_folder, "networks_params.yaml"), "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def test_increment_weights():
    omega = np.array([[1.0e-1, -2.0e-1], [1.0e-3, 0.0], [1.0e-5, 1.0e-5]])
    sigma = np.array([0.1, 0.2])

    Omega = increment_norm(omega, sigma)
    assert np.allclose(Omega, np.sqrt(np.sum((omega / sigma)**2, axis=1)))
    assert np.isclose(Omega[0], np.sqrt(2.0))

    # p = 0: usual loss
    w, _ = increment_weights(Omega, 0.0, eps)
    assert np.allclose(w, 1.0)

    w, ref = increment_weights(Omega, p, eps)
    w_ref = (Omega + eps) ** (-p)
    assert np.allclose(w, w_ref / w_ref.mean()) and np.isclose(w.mean(), 1.0)

    # Bounded ratio, and smaller weights for larger increments
    assert w[0] < w[1] < w[2] and w[2] / w[0] <= ((Omega[0] + eps) / eps) ** p * (1 + 1e-12)

    # Validation points with the normalization of training
    w_val, _ = increment_weights(Omega[:1], p, eps, ref=ref)
    assert np.isclose(w_val[0], w[0])

    assert effective_fraction(np.ones(10)) == 1.0 and effective_fraction(w) < 1.0


def test_density_weights():
    rng = np.random.default_rng(0)

    # Many points with large increments, few with small ones
    Omega = np.concatenate([10.0**rng.uniform(-0.5, 0.5, 9000), 10.0**rng.uniform(-3.0, -2.0, 300)])

    w, ref = increment_weights(Omega, 0.0, 1.0e-4, q=1.0, min_count=1)
    i_bin = np.clip(np.digitize(np.log10(Omega), ref["edges"][1:-1]), 0, len(ref["counts"]) - 1)

    # Same total weight in all occupied bins
    weight_per_bin = np.array([w[i_bin == b].sum() for b in range(len(ref["counts"])) if ref["counts"][b] > 0])
    assert np.allclose(weight_per_bin, weight_per_bin[0])
    assert np.isclose(w.mean(), 1.0)

    # Floor of the numbers of points: bins with fewer than min_count points have the weights of min_count points
    w_floor, _ = increment_weights(Omega, 0.0, 1.0e-4, q=1.0, min_count=1000)
    counts = ref["counts"][i_bin]
    assert np.allclose(w_floor[counts < 1000], w_floor[counts < 1000][0])
    assert w_floor.max() / w_floor.min() <= w.max() / w.min()

    # Validation point in a bin of training: weight of the training points of this bin
    w_val, _ = increment_weights(Omega[:1], 0.0, 1.0e-4, q=1.0, min_count=1, ref=ref)
    assert np.isclose(w_val[0], w[0])


def test_loss_weighting_training(tmp_path):
    os.chdir(tmp_path)
    T_in, Y_in, Y_out = write_h2_learning_database(tmp_path)
    set_learning(tmp_path, loss_weighting_p=p, loss_weighting_eps=eps, loss_weighting_q=0.5, loss_weighting_min_count=5)

    manager = NN_manager()
    _, _, Y_train, _ = manager.read_training_data(0)
    _, _, Ym, Yv = manager.get_scalers_stats(0)
    W, _, Omega = manager._loss_weights(Y_train, Ym, np.sqrt(Yv))
    assert W.shape == (len(Y_train), 1)

    # Increments of ln Y from the Cantera mass fractions (N2 removed), scaled by their standard deviations
    i_N2 = ct.Solution(mech_file).species_index("N2")
    ln_in = np.log(np.maximum(np.delete(Y_in, i_N2, axis=1), threshold))
    ln_out = np.log(np.maximum(np.delete(Y_out, i_N2, axis=1), threshold))
    omega = ln_out - ln_in
    Omega_ref = np.sqrt(np.sum((omega / omega.std(axis=0))**2, axis=1))
    assert np.allclose(Omega, Omega_ref, rtol=1.0e-6)
    W_ref, _ = increment_weights(Omega_ref, p, eps, q=0.5, min_count=5)
    assert np.allclose(W[:, 0], W_ref, rtol=1.0e-6)

    # Weighted MSE, and usual loss with weights 1
    y_pred, y = torch.randn(5, 3), torch.randn(5, 3)
    w = torch.rand(5, 1)
    loss_fn = nn.MSELoss()
    assert torch.isclose(manager._loss(loss_fn, y_pred, y, w), torch.mean(w * (y_pred - y)**2))
    assert torch.isclose(manager._loss(loss_fn, y_pred, y, torch.ones(5, 1)), loss_fn(y_pred, y))

    # Training with the weights
    manager.train_all_clusters()
    log = open(glob.glob(str(tmp_path / "nn_training_*.log"))[0]).read()
    assert "LOSS_WEIGHTS cluster=0" in log and "val_loss_unweighted=" in log


def test_error_per_decade():
    Omega = np.array([2.0e-3, 5.0e-3, 2.0e-1, 0.0])
    err = np.array([1.0e-2, 3.0e-2, 4.0e-2, 1.0e-2])

    rows = error_per_decade(Omega, err)

    # Decades -12 (Omega = 0, floored), -3 and -1
    assert [row[0] for row in rows] == [-12, -3, -1] and [row[1] for row in rows] == [1, 2, 1]
    assert np.isclose(rows[1][2], 2.0e-2) and np.isclose(rows[1][3], np.median(err[:2] / Omega[:2]))
    assert np.isclose(rows[2][3], 0.2)
    assert np.isclose(rows[1][6], np.sqrt(np.sum(err[:2]**2) / np.sum(Omega[:2]**2)))

    # Shares of the loss (w err^2) and of the gradient drive (w err)
    w = np.array([1.0, 2.0, 4.0, 8.0])
    rows = error_per_decade(Omega, err, w)
    assert np.isclose(sum(row[4] for row in rows), 1.0) and np.isclose(sum(row[5] for row in rows), 1.0)
    assert np.isclose(rows[2][4], 4.0 * err[2]**2 / np.sum(w * err**2))
    assert np.isclose(rows[2][5], 4.0 * err[2] / np.sum(w * err))


def test_error_vs_Omega_log(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)

    # Without weights: the error is also logged, at the selected epochs and for the best model
    set_learning(tmp_path, epochs_list=[3], val_every=1, omega_diagnostics_epochs=[1])
    NN_manager().train_all_clusters()
    log = open(glob.glob(str(tmp_path / "nn_training_*.log"))[0]).read()
    assert "gradient_projection_share=" in log
    folder = tmp_path / "MODELS" / "MODEL_TEST" / "training" / "cluster_0"
    best = [f for f in os.listdir(folder) if f.startswith("Omega_points_best_epoch")]
    assert len(best) == 1 and "ERROR_VS_OMEGA cluster=0, epoch00001, median_error=" in log
    for name in ["Omega_points", "Omega_loss_share", "Omega_gradient_shares", "Omega_relative_error"]:
        assert os.path.isfile(folder / f"{name}_epoch00001.png")
        assert os.path.isfile(folder / best[0].replace("Omega_points", name))
    assert sorted(f for f in os.listdir(folder) if f.startswith("Omega")) == sorted(
        [f"{name}_{tag}.png" for name in ["Omega_points", "Omega_loss_share", "Omega_gradient_shares", "Omega_relative_error"]
         for tag in ["epoch00001", best[0][len("Omega_points_"):-4]]])
    for name in glob.glob(str(tmp_path / "nn_training_*.log")):
        os.remove(name)

    # Outputs that are states: no increments, nothing logged
    dtb_yaml = tmp_path / "STOCH_DTB_TEST" / "dtb" / "dtb_processing.yaml"
    params = yaml.safe_load(open(dtb_yaml))
    params["data_processing"]["output_omegas"] = False
    yaml.dump(params, open(dtb_yaml, "w"), default_flow_style=False)
    NN_manager().train_all_clusters()
    assert "ERROR_VS_OMEGA" not in open(glob.glob(str(tmp_path / "nn_training_*.log"))[0]).read()
    assert not any(f.startswith("Omega") for f in os.listdir(tmp_path / "MODELS" / "MODEL_TEST" / "training" / "cluster_0"))


def test_gradient_shares(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    manager = NN_manager()

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 2)).double()
    X = torch.randn(40, 3)
    decades = np.repeat([-2, 0], 20)

    # No error in decade -2: no gradient from it
    with torch.no_grad():
        Y = model(X)
    Y[20:] += torch.randn(20, 2)
    norm_share, projection_share = manager._gradient_shares(model, X, Y, None, decades)
    assert np.allclose(norm_share, [0.0, 1.0]) and np.allclose(projection_share, [0.0, 1.0])

    # Errors everywhere: norm shares and projection shares sum to 1, projections from the gradient of the whole loss
    Y[:20] += torch.randn(20, 2)
    w = torch.rand(40, 1)
    norm_share, projection_share = manager._gradient_shares(model, X, Y, w, decades)
    assert np.isclose(norm_share.sum(), 1.0) and np.isclose(projection_share.sum(), 1.0)

    params = list(model.parameters())
    loss = lambda i: torch.sum(w[i, 0] * torch.sum((model(X[i]) - Y[i])**2, dim=1)) / (40 * 2)
    flat = lambda i: torch.cat([g.flatten() for g in torch.autograd.grad(loss(i), params)])
    g_low, g_all = flat(slice(0, 20)), flat(slice(0, 40))
    assert np.isclose(projection_share[0], float(g_low @ g_all / (g_all @ g_all)))


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
