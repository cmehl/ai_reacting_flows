"""
Test 22: NNTesting error metrics and plots.

Builds a small H2 learning database (N2 removed) and trains a model for one epoch.
Check:
- error metrics (NN_testing_metrics.py): relative error, L2 error over a window, validation errors
  in unscaled space equal to the difference of the unscaled outputs, pooled statistics of groups,
  normalized simulation error (eq. 6 of Mehl & Aubagnac), for 0D cases and 1D flames (a priori).
- test_0D_ignition and test_0D_pyrolysis write their plots in TESTING_<model>/, including N2 when
  it is removed from the network species (the pyrolysis plots used to fail in this case).
- series_cases gives all combinations of the parameters given as lists, and with a conditions file
  (sim_test_*.csv of a flamelets database) one case per row of the file and per value of the other lists.
- run_testing (sections of nn_testing.yaml, networks on CPU with 1 thread) runs the selected series, including 1D premixed and
  diffusion flames, and writes their summaries (errors of each case, mean); evaluate_on_validation_set
  statistics and plots; the 0D simulation errors E_sim per case and per dt in CSV tables, with the
  training standard deviations as normalization, the contribution of each variable and the speed-up,
  the drifts after equilibrium, the errors per cluster, and the speed-up benchmark on batches.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import pandas as pd
import oyaml as yaml
import torch
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting, series_cases, read_conditions_file
import ai_reacting_flows.ann_model_generation.NN_testing_metrics as metrics
import ai_reacting_flows.tools.transforms as transforms
from h2_learning_database import write_h2_learning_database, dt, pressure

testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2", "N2"], "yk_renormalization": False,
                      "hybrid_ann_cvode": False, "hybrid_ann_cvode_tol": 5.0e-05}

# Series as in nn_testing.yaml: 2 ignition cases, 1 premixed flame, 1 diffusion flame, validation set
series_parameters = {
    "0D_ignition": {"run": True, "plots": False, "phi": [0.5, 0.7], "T0": [1200.0], "pressure": [pressure],
                    "dt": [dt], "max_simulation_time": 3 * dt, "T_eq_tolerance": 1.0, "stability_factor": 1.0},
    "0D_pyrolysis": {"run": False},
    "1D_premixed": {"run": True, "plots": False, "phi": [0.6], "T0": [300.0], "pressure": [pressure], "dt": [dt],
                    "T_threshold": 600.0, "x_window": None},
    "1D_diffusion": {"run": True, "plots": True, "strain": [1000.0], "T_ox": [1000.0], "T_fuel": [300.0],
                     "pressure": [pressure], "dt": [dt], "width": 0.02, "T_threshold": 600.0, "x_window": None},
    "validation_set": {"run": True, "max_points_scatter": 50000},
    "speedup_benchmark": {"run": True, "batch_sizes": [1, 50], "dt": dt, "pressure": pressure, "nb_states_cvode": 5},
}


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

    # Pooled statistics equal to the statistics of the union
    groups = [rng.normal(loc=i, scale=i + 1.0, size=(n, 2)) for i, n in enumerate([30, 50, 20])]
    mean, var = metrics.pooled_mean_var(np.array([g.mean(axis=0) for g in groups]),
                                        np.array([g.var(axis=0) for g in groups]), [len(g) for g in groups])
    assert np.allclose(mean, np.concatenate(groups).mean(axis=0))
    assert np.allclose(var, np.concatenate(groups).var(axis=0))

    # Simulation error of each variable: mean over steps of the normalized absolute differences
    x_ref = np.zeros((2, 3))
    x_ann = np.array([[2.0, 0.0, 1.0], [0.0, 4.0, 2.0]])
    assert np.allclose(metrics.normalized_errors_per_variable(x_ann, x_ref, np.array([2.0, 4.0, 1.0])), [0.5, 0.5, 1.5])

    errors = metrics.validation_errors(Y_pred - Y_val, std_Y)
    assert np.allclose(errors["mse_phys"], np.mean(diff**2, axis=0))
    assert np.allclose(errors["mae_phys"], np.mean(np.abs(diff), axis=0))
    assert np.allclose(errors["rmse_scaled"], np.sqrt(np.mean((Y_pred - Y_val)**2, axis=0)))


def test_series_cases():
    cases = series_cases({"run": True, "plots": False, "x_window": None, "phi": [0.5, 0.7], "T0": [1000.0, 1200.0],
                          "max_simulation_time": 1.0e-3, "diluent": None})

    assert len(cases) == 4
    assert {"phi": 0.7, "T0": 1000.0, "max_simulation_time": 1.0e-3, "diluent": None} in cases


def test_conditions_file(tmp_path):
    # Test simulations of a flamelets database (as written by database_flmts.py): one case per row and dt
    path = tmp_path / "sim_test_1D_diff.csv"
    pd.DataFrame({"Strain": [100.0, 500.0], "T0": [300.0, 400.0], "sim_number": [3, 7]}).to_csv(path, index=False)
    conditions = read_conditions_file(path, "1D_diffusion")

    assert conditions == [{"strain": 100.0, "T_ox": 300.0}, {"strain": 500.0, "T_ox": 400.0}]

    cases = series_cases({"run": True, "conditions_file": str(path), "strain": [1.0, 2.0, 3.0], "T_ox": [300.0],
                          "T_fuel": [300.0], "dt": [1.0e-6, 1.0e-5]}, conditions)

    assert len(cases) == 4
    assert cases[3] == {"strain": 500.0, "T_ox": 400.0, "T_fuel": 300.0, "dt": 1.0e-5}


def test_nn_testing_outputs(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    NN_manager().train_all_clusters()
    test = NNTesting(testing_parameters)

    # 0D plots, N2 plotted while removed from the network species
    test.test_0D_ignition(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=3 * dt)
    test.test_0D_pyrolysis(T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=3 * dt, diluent="N2", X_diluent=0.5)

    output = "TESTING_MODEL_TEST"
    for folder in [f"0D_ignition_phi#0.6_T0#1200.0_pressure#{pressure}_dt#{dt}_max_simulation_time#{3 * dt}_T_eq_tolerance#1.0_stability_factor#0.0",
                   f"0D_pyrolysis_T0#1200.0_pressure#{pressure}_dt#{dt}_max_simulation_time#{3 * dt}_T_eq_tolerance#1.0_stability_factor#0.0_diluent#N2_X_diluent#0.5"]:
        for name in ["Temperature", "YH2", "YN2", "SumYs", "AtomCons"]:
            assert os.path.isfile(os.path.join(output, folder, f"{name}.png"))

    # Series of nn_testing.yaml
    torch_threads = torch.get_num_threads()
    test = NNTesting({**testing_parameters, **series_parameters, "device": "cpu", "torch_threads": 1})
    assert test.inference.device.type == "cpu" and torch.get_num_threads() == 1
    torch.set_num_threads(torch_threads)   # Back to the default for the other tests
    summaries = test.run_testing()

    assert set(summaries) == {"0D_ignition", "1D_premixed", "1D_diffusion"}

    with open(os.path.join(output, "summary_0D_ignition.yaml"), "r") as file:
        summary = yaml.safe_load(file)
    errors = [case["errors"]["T_final_error_percent"] for case in summary["cases"]]

    assert summary["n_cases"] == 2
    assert [case["params"]["phi"] for case in summary["cases"]] == [0.5, 0.7]
    assert np.isclose(summary["mean"]["T_final_error_percent"], np.mean(errors))

    # Simulation errors: one row per case, and E(dt) aggregated over the 2 cases of the single dt
    table = pd.read_csv(os.path.join(output, "simulation_errors_0D_ignition.csv"))
    by_dt = pd.read_csv(os.path.join(output, "simulation_errors_0D_ignition_by_dt.csv"))

    assert list(table["phi"]) == [0.5, 0.7]
    assert np.all(table["E_sim"] >= 0.0) and np.all(table["equilibrium_steps"] == 3)
    assert not table["equilibrium_reached"].any()   # stopped by max_simulation_time
    assert by_dt["nb_simulations"][0] == 2
    assert np.isclose(by_dt["E_mean"][0], table["E_sim"].mean())
    assert os.path.isfile(os.path.join(output, "simulation_errors_0D_ignition_by_dt.png"))

    # Contributions of the variables (E_sim is their mean) and speed-up
    variables = ["E_T"] + [f"E_{spec}" for spec in test.inference.spec_list_ANN]
    assert np.allclose(table[variables].mean(axis=1), table["E_sim"])
    assert np.allclose(table["speedup"], table["time_cvode"] / table["time_ann"]) and np.all(table["speedup"] > 0.0)
    assert np.isclose(by_dt["E_T"][0], table["E_T"].mean()) and np.isclose(by_dt["speedup_mean"][0], table["speedup"].mean())
    assert os.path.isfile(os.path.join(output, "simulation_errors_0D_ignition_by_variable.png"))

    # Drifts after equilibrium (ANN continued for as many steps)
    assert np.all(table["T_drift"] >= 0.0) and np.all(table["sumYk_drift"] >= 0.0)

    # Errors per cluster: all the steps of the series, and their mean is the mean of E_sim (same number of steps per case)
    by_cluster = pd.read_csv(os.path.join(output, "simulation_errors_0D_ignition_by_cluster.csv"))
    assert by_cluster["nb_points"].sum() == table["equilibrium_steps"].sum()
    assert np.isclose(np.average(by_cluster["E_mean"], weights=by_cluster["nb_points"]), table["E_sim"].mean())

    # Speed-up of the ANN on batches of validation states
    benchmark = pd.read_csv(os.path.join(output, "speedup_benchmark.csv"))
    assert list(benchmark["batch_size"]) == [1, 50] and np.all(benchmark["speedup"] > 0.0)

    # Normalization: standard deviations of the training inputs (one cluster)
    with h5py.File("STOCH_DTB_TEST/dtb/training_data.h5", "r") as f:
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in f["CLUSTER_0/X_train"].attrs["cols"]]
        idx = [cols.index("Temperature_X")] + [cols.index(f"{spec}_X") for spec in test.inference.spec_list_ANN]
        assert np.allclose(test.std_train, np.sqrt(f["CLUSTER_0/Xscaler"][idx, 1]))

    # 1D flames: errors on the reaction rates, and plots
    assert summaries["1D_premixed"]["cases"][0]["errors"]["L2_error_omega_H2"] >= 0.0

    errors = summaries["1D_diffusion"]["cases"][0]["errors"]
    assert errors["L2_error_omega_H2"] >= 0.0
    assert errors["ann_calls"] > 0
    folder = f"1D_diffusion_strain#1000.0_T_ox#1000.0_T_fuel#300.0_pressure#{pressure}_dt#{dt}_width#0.02_T_threshold#600.0"
    assert os.path.isfile(os.path.join(output, folder, "YH2.png"))

    # 1D a priori error E_sim and its aggregation per dt (one flame)
    assert errors["E_sim"] >= 0.0
    by_dt = pd.read_csv(os.path.join(output, "simulation_errors_1D_diffusion_by_dt.csv"))
    assert by_dt["nb_simulations"][0] == 1 and np.isclose(by_dt["E_mean"][0], errors["E_sim"])
    assert os.path.isfile(os.path.join(output, "simulation_errors_1D_diffusion_by_dt.png"))
    assert errors["speedup"] > 0.0 and "E_H2" in errors

    # Errors per cluster: points of the fuel side below T_threshold are not advanced by a network (cluster -1)
    by_cluster = pd.read_csv(os.path.join(output, "simulation_errors_1D_diffusion_by_cluster.csv"))
    assert -1 in list(by_cluster["cluster"]) and 0 in list(by_cluster["cluster"])

    # Validation set: statistics per output and scatter plot per cluster
    stats = test.evaluate_on_validation_set()

    assert len(stats) == test.inference.nb_clusters
    assert stats[0]["mse_phys"].shape == (test.nb_species_ANN,)
    assert os.path.isfile(os.path.join(output, "plots_validation", "cluster_0_errors.png"))
