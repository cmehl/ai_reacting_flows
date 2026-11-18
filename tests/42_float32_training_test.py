"""
Test 42: training in float32 (learning.precision: float32).

Trains a network for one epoch in float32 and in float64, on a one-step database (MLP and PerSpeciesMLP) and on a
rollout database (3 steps) with the element projection and the loss weights, so that all the tensors of the losses
are used.
Check:
- in float32, every floating-point tensor computed during training is float32 (a float64 tensor anywhere, e.g. a
  constant created with the default type, would make PyTorch compute in float64 without error);
- the saved networks (.pth and h5 for NNICE) are in float64, as in float64 training;
- the losses of the two trainings are close (same initial weights, rounded to float32).

Independent of the other tests.
"""
import os
import glob
import importlib
import h5py
import numpy as np
import oyaml as yaml
import pytest
import torch
from torch.overrides import TorchFunctionMode
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.databases_processing.database_processing import LearningDatabase
from h2_learning_database import write_h2_learning_database

rollout_test = importlib.import_module("27_rollout_test")


class FloatTypes(TorchFunctionMode):
    """Types of the floating-point tensors (not scalars) returned by the torch functions called in the mode.

    Scalars (0-d tensors) are skipped: they do not change the type of an operation with a tensor of dimension >= 1
    (type promotion of PyTorch), and Adam keeps its step counter as a scalar of the default type (float64 here)."""

    def __init__(self):
        super().__init__()
        self.dtypes = set()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for out in (result if isinstance(result, (tuple, list)) else [result]):
            if isinstance(out, torch.Tensor) and out.is_floating_point() and out.dim() > 0:
                self.dtypes.add(out.dtype)
        return result


def set_learning(learning, network_type=None):
    """Learning parameters, and network type of the cluster if given, in networks_params.yaml."""

    params = yaml.safe_load(open("networks_params.yaml"))
    if network_type is not None:
        params["networks_types"] = [network_type]
    params["learning"].update(learning)
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)


def train(precision):
    """Training of one epoch with the precision, from the same initial weights. Returns the types of the tensors
    computed during training, and the training and validation losses."""

    set_learning({"precision": precision})

    manager = NN_manager()
    types = FloatTypes()
    losses = {}

    # Training only in the mode (not the creation of the network, in float64, nor its conversion for saving)
    train_model = manager.train_model
    def spy(*args):
        with types:
            result = train_model(*args)
        losses["train"], losses["val"] = result[2][-1], result[3][-1]
        return result
    manager.train_model = spy

    torch.manual_seed(0)
    manager.train_all_clusters()

    return types.dtypes, losses


def check_float32_training():
    """float32 then float64 trainings in the current folder."""

    dtypes_32, losses_32 = train("float32")

    # Saved networks in float64
    model = torch.load("MODELS/MODEL_TEST/cluster0_model.pth", weights_only=False)
    assert all(p.dtype == torch.float64 for p in model.parameters())
    with h5py.File("MODELS/MODEL_TEST/cluster0_model.h5", "r") as f:
        h5_dtypes = set()
        f.visititems(lambda name, obj: h5_dtypes.add(obj.dtype) if isinstance(obj, h5py.Dataset) and obj.dtype.kind == "f" else None)
    assert h5_dtypes == {np.dtype("float64")}

    dtypes_64, losses_64 = train("float64")

    assert dtypes_32 == {torch.float32}
    assert dtypes_64 == {torch.float64}
    for key in losses_32:
        assert np.isclose(losses_32[key], losses_64[key], rtol=1.0e-3), key


@pytest.mark.parametrize("network_type", ["MLP", "PerSpeciesMLP"])
def test_float32_one_step(tmp_path, network_type):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    set_learning({"val_every": 1}, network_type)

    check_float32_training()


def test_float32_rollout(tmp_path):
    os.chdir(tmp_path)
    rollout_test.write_rollout_database(tmp_path / "STOCH_DTB_TEST")
    rollout_test.write_inputs(True, 1, 1)
    LearningDatabase().process_database()
    set_learning({"val_every": 1, "project_elements": True, "raw_state_loss_weight": 0.5,
                  "loss_weighting_p": 1.0, "loss_weighting_eps": 1.0e-2, "loss_weighting_min_count": 5})

    check_float32_training()


def test_precision_error(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)
    set_learning({"precision": "float16"})

    with pytest.raises(ValueError, match="precision"):
        NN_manager()
