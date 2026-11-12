"""
Test 34: DeepONet networks (networks_types: DeepONet, DeepONetShift).

Check:
- the networks are created, and give one output per species for a batch of inputs [state, dt];
- NN_manager refuses them with a constant time step database (dt_var: false): their trunk takes the last
  input column as dt, which would be a species.

The h5 export of these networks is not implemented (future work): their training is not tested.

Independent of the other tests.
"""
import os

import oyaml as yaml
import pytest
import torch

from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_models import DeepONet, DeepONet_shift
from h2_learning_database import write_h2_learning_database

# Inputs [T, 8 species, dt], 8 outputs, 4 neurons per output in the last layers of branch and trunk
n_in, n_out, n_neurons = 10, 8, 4


@pytest.mark.parametrize("network", [DeepONet, DeepONet_shift])
def test_deeponet_outputs(network):

    # Branch: state without dt, trunk: dt, shift (DeepONet_shift): state without dt to a shift of dt
    hidden_layers = {"branch": [n_in - 1, 6, n_neurons * n_out], "trunk": [1, 6, n_neurons * n_out], "shift": [n_in - 1, 6, 1]}
    layers_type = {key: ["dense", "dense"] for key in hidden_layers}
    activations = {key: [torch.nn.Tanh, torch.nn.Identity] for key in hidden_layers}

    model = network("cpu", hidden_layers, layers_type, activations, n_out, n_neurons)

    y = model(torch.rand(5, n_in, dtype=torch.float64))
    assert y.shape == (5, n_out)


@pytest.mark.parametrize("network_type", ["DeepONet", "DeepONetShift"])
def test_deeponet_needs_multiple_dt(tmp_path, network_type):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)   # Constant time step database

    with open("networks_params.yaml") as file:
        params = yaml.safe_load(file)
    params["networks_types"] = [network_type]
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)

    with pytest.raises(ValueError, match="dt_var: true"):
        NN_manager()
