"""
Test 28: element projection in the training loss (learning.project_elements).

Builds a small H2 learning database from an ignition trajectory (with its rollout version), trains a network for
one epoch with the projection, for outputs as mass fractions or differences, with log, Box-Cox or no output transform.
Check:
- project_elements gives the same result on numpy arrays and tensors, conserves the elements, and has finite gradients;
- the training loss on the projected state is the loss on the state of NNInference.advance with yk_renormalization: 3
  (same equations, independent code path), and raw_state_loss_weight adds the usual loss on the network output;
- with 3 rollout steps, the loss is the one of a chain computed separately with NNInference.advance, with the correction;
- NNInference warns once if these networks are used with another correction.

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import oyaml as yaml
import cantera as ct
import pytest
import torch
import ai_reacting_flows.tools.transforms as transforms
from ai_reacting_flows.tools.element_projection import project_elements
import ai_reacting_flows.tools.utilities as utils
from ai_reacting_flows.ann_model_generation.NN_inference import NNInference
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from ai_reacting_flows.databases_processing.generate_var_dt_dtb import GenerateVariable_dt
from h2_learning_database import ignition_states, mech_file, dt, pressure, threshold
import DummyComm as dc

rollout_steps = 3
raw_weight = 0.3
lambda_bct = 0.1

generation_parameters = {"results_folder_suffix": "TEST", "dtb_file": "solutions.h5", "mech_file": mech_file,
                         "time_step_type": "rollout", "rollout_dt": dt, "rollout_steps": rollout_steps,
                         "new_file_name": "solutions_rollout.h5"}


def write_databases(folder):
    """Stochastic-like database from an ignition trajectory (X at t, Y at t+dt), and its rollout database."""

    species, T, Y = ignition_states(n=200)
    n = len(T)

    cols = ["Temperature", "Pressure"] + species + ["Prog_var", "HRR"]
    states = np.column_stack([T, np.full(n, pressure), Y, np.linspace(0.0, 1.0, n), np.linspace(1.0e8, 1.0e9, n)])

    os.makedirs(folder)
    with h5py.File(os.path.join(folder, "solutions.h5"), "w") as f:
        grp = f.create_group("ITERATION_00000")
        grp.create_dataset("X", data=states[:-1]).attrs["cols"] = cols
        grp.create_dataset("Y", data=states[1:]).attrs["cols"] = cols

    with open(os.path.join(folder, "dtb_params.yaml"), "w") as file:
        yaml.dump({"pv_species": ["H2O"]}, file, default_flow_style=False)

    GenerateVariable_dt("stoch", generation_parameters, dc.DummyComm())

    return species


def write_inputs(rollout, output_omegas, log_transform_Y, log_transform_X):
    """dtb_processing.yaml and networks_params.yaml of the case."""

    dtb_processing = {
        "database_params": {"database_type": "stoch", "dtb_folder_suffix": "TEST", "database_name": "dtb",
                            "dtb_file": "solutions_rollout.h5" if rollout else "solutions.h5", "dt_var": False,
                            "rollout": rollout, "fuel": "H2", "mech_file": mech_file},
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
        "learning": {"batch_size": 16, "epochs_list": [1], "project_elements": True, "raw_state_loss_weight": raw_weight},
    }
    with open("networks_params.yaml", "w") as file:
        yaml.dump(networks_params, file, default_flow_style=False)


def prepare_case(tmp_path, rollout, output_omegas, log_transform_Y, log_transform_X):
    """Process the database and train one epoch. Returns the validation data and a manager on the trained model."""

    os.chdir(tmp_path)
    species = write_databases(tmp_path / "STOCH_DTB_TEST")
    write_inputs(rollout, output_omegas, log_transform_Y, log_transform_X)

    database = LearningDatabase()
    database.apply_temperature_threshold()
    database.process_database()

    NN_manager().train_all_clusters()

    params = yaml.safe_load(open("networks_params.yaml"))
    params["new_model_folder"] = False
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)
    manager = NN_manager()

    model = torch.load("MODELS/MODEL_TEST/cluster0_model.pth", map_location=manager.device, weights_only=False)
    assert all(torch.isfinite(p).all() for p in model.parameters())

    data = {}
    with h5py.File("STOCH_DTB_TEST/dtb/training_data.h5", "r") as f:
        grp = f["CLUSTER_0"]
        data["X"], data["Y"] = grp["X_val"][()], grp["Y_val"][()]
        data["X_mean"], data["X_std"] = grp["Xscaler"][:, 0], transforms.scaler_std(grp["Xscaler"][:, 1])
        data["Y_mean"], data["Y_std"] = grp["Yscaler"][:, 0], transforms.scaler_std(grp["Yscaler"][:, 1])
        if rollout:
            data["Y_rollout"], data["Y_N2"] = grp["Y_rollout_val"][()], grp["Y_N2_val"][()]

    return species, data, manager, model


def initial_state(species, data, log_transform_X, Y_N2):
    """T [K] and Y [-] of all species at t, from the network inputs of the validation set."""

    species_ANN = [spec for spec in species if spec != "N2"]
    cols_ANN = [species.index(spec) for spec in species_ANN]

    x_t = transforms.unscale(data["X"], data["X_mean"], data["X_std"])

    Y = np.empty((len(x_t), len(species)))
    Y[:, cols_ANN] = transforms.inverse_species(x_t[:, 1:], species_ANN, log_transform_X, lambda_bct, [])
    Y[:, species.index("N2")] = Y_N2

    return x_t[:, 0], Y, species_ANN, cols_ANN


def test_project_elements_numpy_torch():
    gas = ct.Solution(mech_file)
    atomic_array = utils.get_atomic_array(gas)
    a = utils.get_atomic_weights()[:, None] * atomic_array / gas.molecular_weights
    c = (atomic_array > 0).astype(float)

    _, _, Y = ignition_states(n=20)
    rng = np.random.default_rng(1)
    Y_new = Y * (1.0 + 0.01 * rng.standard_normal(Y.shape))

    Y_numpy = project_elements(Y, Y_new, a, c)
    assert np.allclose(Y_numpy @ a.T, Y @ a.T, rtol=1.0e-12, atol=1.0e-14)

    Y_new_t = torch.tensor(Y_new, requires_grad=True)
    Y_torch = project_elements(torch.tensor(Y), Y_new_t, torch.tensor(a), torch.tensor(c))
    assert np.allclose(Y_torch.detach().numpy(), Y_numpy, rtol=1.0e-14, atol=0.0)

    Y_torch.sum().backward()
    assert torch.isfinite(Y_new_t.grad).all()


# output_omegas, log_transform_Y, log_transform_X
cases = [(False, 1, 1), (True, 1, 1), (True, 0, 1), (False, 2, 2), (True, 0, 0)]

@pytest.mark.parametrize("output_omegas, log_transform_Y, log_transform_X", cases)
def test_one_step_loss(tmp_path, capsys, output_omegas, log_transform_Y, log_transform_X):
    species, data, manager, model = prepare_case(tmp_path, False, output_omegas, log_transform_Y, log_transform_X)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float64, device=manager.device)

    scalers = [tensor(data[key]) for key in ["X_mean", "X_std", "Y_mean", "Y_std"]]
    with torch.no_grad():
        y_pred = model(tensor(data["X"]))
        loss_projected = manager._one_step_loss(torch.nn.MSELoss(), tensor(data["X"]), y_pred, tensor(data["Y"]), *scalers).item()
        loss_total = manager._one_step_loss(torch.nn.MSELoss(), tensor(data["X"]), y_pred, tensor(data["Y"]), *scalers, raw_weight).item()
        loss_usual = torch.nn.MSELoss()(y_pred, tensor(data["Y"])).item()

    # raw_state_loss_weight adds the usual loss
    assert np.isclose(loss_total, loss_projected + raw_weight * loss_usual, rtol=1.0e-12)

    # Loss on the projected state: state of the inference with yk_renormalization: 3
    inference = NNInference("MODELS/MODEL_TEST", str(tmp_path), device="cpu")
    T, Y, species_ANN, cols_ANN = initial_state(species, data, log_transform_X, 0.7)
    _, Y_new, _, _ = inference.advance(T, Y, pressure, dt, np.zeros(len(T), dtype=int), 3)

    x_species_t = transforms.unscale(data["X"], data["X_mean"], data["X_std"])[:, 1:]
    y_t = transforms.Yk_to_output(x_species_t, Y_new[:, cols_ANN], species_ANN, output_omegas, log_transform_X, log_transform_Y,
                                  threshold, lambda_bct, [])
    expected = np.mean((transforms.scale(y_t, data["Y_mean"], data["Y_std"]) - data["Y"])**2)
    assert np.isclose(loss_projected, expected, rtol=1.0e-6)

    # Warning once if used with another correction
    capsys.readouterr()
    inference.advance(T[:2], Y[:2], pressure, dt, np.zeros(2, dtype=int), 0)
    assert "project_elements" in capsys.readouterr().out
    inference.advance(T[:2], Y[:2], pressure, dt, np.zeros(2, dtype=int), 0)
    assert "project_elements" not in capsys.readouterr().out


@pytest.mark.parametrize("output_omegas, log_transform_Y, log_transform_X", cases)
def test_rollout_loss(tmp_path, output_omegas, log_transform_Y, log_transform_X):
    species, data, manager, model = prepare_case(tmp_path, True, output_omegas, log_transform_Y, log_transform_X)

    def tensor(a):
        return torch.tensor(a, dtype=torch.float64, device=manager.device)

    scalers = [tensor(data[key]) for key in ["X_mean", "X_std", "Y_mean", "Y_std"]]
    with torch.no_grad():
        loss = manager._rollout_loss(model, torch.nn.MSELoss(), tensor(data["X"]), tensor(data["Y_rollout"]), tensor(data["Y_N2"]), *scalers).item()

    # Chain computed with the inference, with the correction at each step
    inference = NNInference("MODELS/MODEL_TEST", str(tmp_path), device="cpu")
    T, Y, species_ANN, cols_ANN = initial_state(species, data, log_transform_X, data["Y_N2"])

    expected = 0.0
    for k in range(rollout_steps):
        T, Y, _, _ = inference.advance(T, Y, pressure, dt, np.zeros(len(T), dtype=int), 3)
        Y_state = transforms.transform_species(Y[:, cols_ANN], species_ANN, log_transform_Y, threshold, lambda_bct, [])
        expected += np.mean(((Y_state - data["Y_rollout"][:, k]) / data["Y_std"])**2) / rollout_steps

    assert np.isclose(loss, expected, rtol=1.0e-6)
