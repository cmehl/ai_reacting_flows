"""
Test 22: NNTesting error metrics and plots.

Builds a small H2 learning database (N2 removed) and trains a model for one epoch.
Check:
- error metrics (NN_testing_metrics.py): relative error, L2 error over a window, validation errors
  in unscaled space equal to the difference of the unscaled outputs.
- test_0D_ignition and test_0D_pyrolysis write their plots, including N2 when it is removed
  from the network species (the pyrolysis plots used to fail in this case).
- run_0D_series summary and evaluate_on_validation_set statistics and plots.

Independent of the other tests.
"""
import os
import numpy as np
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
import ai_reacting_flows.ann_model_generation.NN_testing_metrics as metrics
import ai_reacting_flows.tools.transforms as transforms
from h2_learning_database import write_h2_learning_database, dt, pressure

testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2", "N2"], "yk_renormalization": False,
                      "hybrid_ann_cvode": False, "hybrid_ann_cvode_tol": 5.0e-05}


def test_metrics():
    assert np.isclose(metrics.relative_error_percent(2000.0, 1980.0), 1.0)

    # Constant difference 2 over [0.2, 0.6] m: 41 points of weight dx = 0.01 m
    x = np.linspace(0.0, 1.0, 101)
    assert np.isclose(metrics.l2_error(x, np.full(101, 3.0), np.full(101, 1.0), (0.2, 0.6)), np.sqrt(4.0 * 0.41))

    assert metrics.mean_std([]) == (0.0, 0.0)
    assert np.allclose(metrics.mean_std([1.0, 3.0]), (2.0, 1.0))

    # Unscaled errors: difference of the unscaled outputs
    rng = np.random.default_rng(0)
    Y_pred, Y_val = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    mean_Y, std_Y = np.array([1.0, -2.0, 0.5]), np.array([0.1, 3.0, 1.0])
    diff = transforms.unscale(Y_pred, mean_Y, std_Y) - transforms.unscale(Y_val, mean_Y, std_Y)

    errors = metrics.validation_errors(Y_pred - Y_val, std_Y)
    assert np.allclose(errors["mse_phys"], np.mean(diff**2, axis=0))
    assert np.allclose(errors["mae_phys"], np.mean(np.abs(diff), axis=0))
    assert np.allclose(errors["rmse_scaled"], np.sqrt(np.mean((Y_pred - Y_val)**2, axis=0)))


def test_nn_testing_outputs(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    NN_manager().train_all_clusters()
    test = NNTesting(testing_parameters)

    # 0D plots, N2 plotted while removed from the network species
    test.test_0D_ignition(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, nb_ite=3)
    test.test_0D_pyrolysis(T0=1200.0, pressure=pressure, dt=dt, nb_ite=3, diluent="N2", X_diluent=0.5)

    for folder in [f"plots_0D_T0#1200.0_phi#0.6", f"plots_0D_pyro_P#{pressure}_T0#1200.0_dt#{dt}"]:
        for name in ["Temperature", "YH2", "YN2", "SumYs", "AtomCons"]:
            assert os.path.isfile(os.path.join(folder, f"{name}.png"))

    # Series of 0D cases
    cases = [{"phi": phi, "T0": 1200.0, "pressure": pressure, "dt": dt, "nb_ite": 3} for phi in [0.5, 0.7]]
    results, summary = test.run_0D_series(cases)
    errors = [metrics.relative_error_percent(r["Teq_ref"], r["Teq_ann"]) for r in results]

    assert summary["n_cases"] == 2
    assert np.isclose(summary["Teq_error_mean_percent"], np.mean(errors))

    # Validation set: statistics per output and scatter plot per cluster
    stats = test.evaluate_on_validation_set(scatter_folder="plots_validation")

    assert len(stats) == test.nb_clusters
    assert stats[0]["mse_phys"].shape == (test.nb_species_ANN,)
    assert os.path.isfile("plots_validation/cluster_0_errors.png")
