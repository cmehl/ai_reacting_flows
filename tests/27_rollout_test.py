"""
Test 27: rollout training (time_step_type: rollout, database_params.rollout).

Builds a small H2 database from an ignition trajectory, generates its rollout database (3 steps of dt) with
GenerateVariable_dt, processes it with LearningDatabase and trains a network for one epoch, for outputs as
mass fractions or differences, with log, Box-Cox or no output transform.
Check:
- the states of the rollout database are successive steps: step k+1 is step k advanced by dt with Cantera;
- rows stay aligned through temperature threshold, undersampling and train/validation split: the first step
  of Y_rollout_val gives back the usual outputs Y_val;
- with one step, the rollout loss is the usual loss;
- with 3 steps, it is the loss on the absolute states of a chain computed separately with NNInference.advance
  (same network inputs, including the temperature update from enthalpy conservation, N2 unchanged);
- the validation losses per step written after training have this loss as mean, and the usual loss as first step.
- with loss weights (loss_weighting_p, loss_weighting_q), one per point or per point and species
  (loss_weighting_per_species): all the steps of a rollout have the weights of the first step of their initial state,
  weights 1 give the unweighted loss, and training logs the weights and the unweighted loss.

Independent of the other tests.
"""
import os
import glob
import h5py
import numpy as np
import oyaml as yaml
import cantera as ct
import pytest
import torch
import ai_reacting_flows.tools.transforms as transforms
from ai_reacting_flows.ann_model_generation.NN_inference import NNInference
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from ai_reacting_flows.databases_processing.generate_var_dt_dtb import GenerateVariable_dt
from h2_learning_database import ignition_states, mech_file, dt, pressure, threshold
import DummyComm as dc

rollout_steps = 3

generation_parameters = {"results_folder_suffix": "TEST", "dtb_file": "solutions.h5", "mech_file": mech_file,
                         "time_step_type": "rollout", "rollout_dt": dt, "rollout_steps": rollout_steps,
                         "new_file_name": "solutions_rollout.h5"}


def write_rollout_database(folder):
    """Stochastic-like database from an ignition trajectory, and its rollout database. Returns the species."""

    species, T, Y = ignition_states(n=200)
    n = len(T)

    cols = ["Temperature", "Pressure"] + species + ["Prog_var", "HRR"]
    states = np.column_stack([T, np.full(n, pressure), Y, np.linspace(0.0, 1.0, n), np.linspace(1.0e8, 1.0e9, n)])

    os.makedirs(folder)
    with h5py.File(os.path.join(folder, "solutions.h5"), "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=states).attrs["cols"] = cols
        grp.create_dataset("Y", data=states).attrs["cols"] = cols

    with open(os.path.join(folder, "dtb_params.yaml"), "w") as file:
        yaml.dump({"pv_species": ["H2O"]}, file, default_flow_style=False)

    GenerateVariable_dt("stoch", generation_parameters, dc.DummyComm())

    return species


def write_inputs(output_omegas, log_transform_Y, log_transform_X):
    """dtb_processing.yaml and networks_params.yaml of the rollout case."""

    dtb_processing = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb",
                            "dtb_file": "solutions_rollout.h5", "dt_var": False, "rollout": True, "fuel": "H2",
                            "mech_file": mech_file},
        "data_processing": {"log_transform_X": log_transform_X, "log_transform_Y": log_transform_Y, "threshold": threshold,
                            "T_threshold": 1205.0, "output_omegas": output_omegas, "with_N_chemistry": False},
        "data_clustering": {"clusterize_on": "phys", "clustering_method": "kmeans", "nb_clusters": 1},
        "train_set_size": 0.75,
    }
    with open("dtb_processing.yaml", "w") as file:
        yaml.dump(dtb_processing, file, default_flow_style=False)

    networks_params = {
        "database_path": "STOCH_DTB_TEST/dtb", "model_name_suffix": "TEST", "new_model_folder": True,
        "networks_types": ["MLP"], "networks_def": ["cluster0"],
        "clusters": {"cluster0": {"nb_units_in_layers_list": [8], "layers_activation_list": ["tanh", "Id"],
                                  "layers_type": ["dense", "dense"]}},
        "learning": {"batch_size": 16, "epochs_list": [1]},
    }
    with open("networks_params.yaml", "w") as file:
        yaml.dump(networks_params, file, default_flow_style=False)


def test_rollout_states_are_successive_steps(tmp_path):
    os.chdir(tmp_path)
    species = write_rollout_database(tmp_path / "STOCH_DTB_TEST")

    with h5py.File("STOCH_DTB_TEST/solutions_rollout.h5", "r") as f:
        grp = f["ITERATION_00000"]
        Y = grp["Y"][()]   # (n, n_var, n_steps)
        assert Y.shape[2] == rollout_steps
        assert np.allclose(grp["DT"][()], dt * np.arange(1, rollout_steps + 1))

    gas = ct.Solution(mech_file)
    for state in Y[::40]:
        for k in range(rollout_steps - 1):
            gas.TPY = state[0, k], state[1, k], state[2:2 + len(species), k]
            sim = ct.ReactorNet([ct.IdealGasConstPressureReactor(gas)])
            sim.advance(dt)
            assert np.isclose(gas.T, state[0, k + 1], rtol=1.0e-5)


# output_omegas, log_transform_Y, log_transform_X
cases = [(False, 1, 1), (True, 1, 1), (True, 0, 1), (False, 2, 2), (True, 0, 0)]

@pytest.mark.parametrize("output_omegas, log_transform_Y, log_transform_X", cases)
def test_rollout_processing_training(tmp_path, output_omegas, log_transform_Y, log_transform_X):
    os.chdir(tmp_path)
    species = write_rollout_database(tmp_path / "STOCH_DTB_TEST")
    write_inputs(output_omegas, log_transform_Y, log_transform_X)

    # Processing, with filters of the database
    database = LearningDatabase()
    database.apply_temperature_threshold()
    database.undersample_1D("Temperature")
    n_selected = database.X.shape[0]
    assert database.Y_rollout.shape == (n_selected, rollout_steps, database.Y.shape[1])
    database.process_database()

    with h5py.File("STOCH_DTB_TEST/dtb/training_data.h5", "r") as f:
        assert f.attrs["rollout"]
        grp = f["CLUSTER_0"]
        X_val, Y_val = grp["X_val"][()], grp["Y_val"][()]
        Y_rollout_val, Y_N2_val = grp["Y_rollout_val"][()], grp["Y_N2_val"][()]
        X_mean, X_var = grp["Xscaler"][:, 0], grp["Xscaler"][:, 1]
        Y_mean, Y_var = grp["Yscaler"][:, 0], grp["Yscaler"][:, 1]
        assert grp["Y_rollout_train"].shape[0] + Y_rollout_val.shape[0] == n_selected

    species_ANN = [spec for spec in species if spec != "N2"]
    X_std, Y_std = transforms.scaler_std(X_var), transforms.scaler_std(Y_var)
    x_t = transforms.unscale(X_val, X_mean, X_std)
    y_t = transforms.unscale(Y_val, Y_mean, Y_std)

    # First step of the rollout targets: usual outputs, row by row
    if not output_omegas:
        base = 0.0
    elif log_transform_Y > 0:
        base = x_t[:, 1:]
    else:
        base = transforms.inverse_species(x_t[:, 1:], species_ANN, log_transform_X, 0.1, [])
    assert np.allclose(Y_rollout_val[:, 0] - base, y_t, rtol=1.0e-10, atol=1.0e-12)

    # Training
    NN_manager().train_all_clusters()

    params = yaml.safe_load(open("networks_params.yaml"))
    params["new_model_folder"] = False
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)
    manager = NN_manager()

    device = manager.device
    model = torch.load("MODELS/MODEL_TEST/cluster0_model.pth", map_location=device, weights_only=False)
    assert all(torch.isfinite(p).all() for p in model.parameters())

    def tensor(a):
        return torch.tensor(a, dtype=torch.float64, device=device)

    def rollout_loss(steps):
        with torch.no_grad():
            return manager._rollout_loss(model, torch.nn.MSELoss(), tensor(X_val), tensor(Y_rollout_val[:, :steps]), tensor(Y_N2_val),
                                         tensor(X_mean), tensor(X_std), tensor(Y_mean), tensor(Y_std)).item()

    # One step: usual loss
    with torch.no_grad():
        usual_loss = torch.nn.MSELoss()(model(tensor(X_val)), tensor(Y_val)).item()
    assert np.isclose(rollout_loss(1), usual_loss, rtol=1.0e-10)

    # Three steps: loss on the absolute states of a chain computed with the inference
    inference = NNInference("MODELS/MODEL_TEST", str(tmp_path), device="cpu")
    i_N2 = species.index("N2")
    cols_ANN = [species.index(spec) for spec in species_ANN]

    T = x_t[:, 0]
    Y = np.empty((len(T), len(species)))
    Y[:, cols_ANN] = transforms.inverse_species(x_t[:, 1:], species_ANN, log_transform_X, 0.1, [])
    Y[:, i_N2] = Y_N2_val

    clusters = np.zeros(len(T), dtype=int)
    expected = 0.0
    for k in range(rollout_steps):
        T, Y, _, _ = inference.advance(T, Y, pressure, dt, clusters, 0)
        Y_state = transforms.forward(Y[:, cols_ANN], log_transform_Y, 0.1)
        expected += np.mean(((Y_state - Y_rollout_val[:, k]) / Y_std)**2) / rollout_steps

    assert np.isclose(rollout_loss(rollout_steps), expected, rtol=1.0e-6)

    # Validation losses per step written after training, by the first NN_manager (with the saved best model)
    step_losses = np.loadtxt("MODELS/MODEL_TEST/training/cluster_0/rollout_val_loss_per_step.csv", delimiter=",", skiprows=1)
    assert np.array_equal(step_losses[:, 0], np.arange(1, rollout_steps + 1))
    assert np.isclose(step_losses[0, 1], usual_loss, rtol=1.0e-10)
    assert np.isclose(step_losses[:, 1].mean(), rollout_loss(rollout_steps), rtol=1.0e-10)


@pytest.mark.parametrize("per_species", [False, True])
def test_rollout_loss_weighting(tmp_path, per_species):
    os.chdir(tmp_path)
    write_rollout_database(tmp_path / "STOCH_DTB_TEST")
    write_inputs(True, 1, 1)
    LearningDatabase().process_database()

    params = yaml.safe_load(open("networks_params.yaml"))
    params["learning"].update({"loss_weighting_p": 1.0, "loss_weighting_eps": 1.0e-2, "loss_weighting_q": 0.5,
                               "loss_weighting_min_count": 5, "loss_weighting_per_species": per_species, "val_every": 1})
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)

    # Training with the weights
    manager = NN_manager()
    manager.train_all_clusters()
    log = open(glob.glob(str(tmp_path / "nn_training_*.log"))[0]).read()
    assert "LOSS_WEIGHTS cluster=0" in log and "val_loss_unweighted=" in log

    device = manager.device
    model = torch.load("MODELS/MODEL_TEST/cluster0_model.pth", map_location=device, weights_only=False)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float64, device=device)

    # Weights of the validation points, with the bins and normalization of the training points
    _, X_val, Y_train, Y_val = manager.read_training_data(0)
    _, Y_rollout_val, _, Y_N2_val = manager.read_rollout_data(0)
    Xm, Xv, Ym, Yv = manager.get_scalers_stats(0)
    X_std, Y_std = np.sqrt(Xv), np.sqrt(Yv)
    _, ref, _ = manager._loss_weights(Y_train, Ym, Y_std)
    W_val, _, _ = manager._loss_weights(Y_val, Ym, Y_std, ref)
    assert W_val.max() / W_val.min() > 2.0
    assert W_val.shape == ((len(X_val), Y_val.shape[1]) if per_species else (len(X_val), 1))

    def step_losses(weights):
        with torch.no_grad():
            return manager._rollout_step_losses(model, torch.nn.MSELoss(), tensor(X_val), tensor(Y_rollout_val), tensor(Y_N2_val),
                                                tensor(Xm), tensor(X_std), tensor(Ym), tensor(Y_std), weights=weights).cpu().numpy()

    weighted, unweighted = step_losses(tensor(W_val)), step_losses(None)

    # Weights 1: unweighted loss. First step: weighted usual loss
    assert np.allclose(step_losses(torch.ones(len(X_val), 1, dtype=torch.float64, device=device)), unweighted, rtol=1.0e-10)
    with torch.no_grad():
        first_step = torch.mean(tensor(W_val) * (model(tensor(X_val)) - tensor(Y_val))**2).item()
    assert np.isclose(weighted[0], first_step, rtol=1.0e-10)
    assert not np.allclose(weighted, unweighted, rtol=1.0e-3)

    # One point with a weight: only the errors of its chain are multiplied, at all steps
    w_one = np.ones((len(X_val), 1))
    w_one[0] = 3.0
    only_first = np.zeros((len(X_val), 1))
    only_first[0] = 1.0
    assert np.allclose(step_losses(tensor(w_one)), unweighted + 2.0 * step_losses(tensor(only_first)), rtol=1.0e-10)

    # Validation losses per step written after training: weighted
    written = np.loadtxt("MODELS/MODEL_TEST/training/cluster_0/rollout_val_loss_per_step.csv", delimiter=",", skiprows=1)
    assert np.allclose(written[:, 1], weighted, rtol=1.0e-10)
