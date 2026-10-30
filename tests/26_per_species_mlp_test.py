"""
Test 26: networks with one MLP per species (networks_types: PerSpeciesMLP).

Builds a small H2 learning database (N2 removed: 8 network species) and trains a PerSpeciesMLP for one
epoch (each species network: 2 hidden layers of 6 neurons).
Check:
- the model has one MLP per network species, each with one output, and its outputs are those of the
  species networks in the order of the species;
- the h5 export has one group species_<i> per species, each with the layers of its MLP (kernel:0
  transposed as in the export of MLP, bias:0, activation);
- the model is used for inference like an MLP (0D ignition with NNTesting).

Independent of the other tests.
"""
import os
import h5py
import numpy as np
import oyaml as yaml
import torch
from ai_reacting_flows.ann_model_generation.NN_manager import NN_manager
from ai_reacting_flows.ann_model_generation.NN_models import PerSpeciesMLP
from ai_reacting_flows.ann_model_generation.NN_testing import NNTesting
from h2_learning_database import write_h2_learning_database, dt, pressure

testing_parameters = {"models_folder": "MODEL_TEST", "spec_to_plot": ["H2"], "yk_renormalization": 0,
                      "hybrid_ann_cvode": 0, "hybrid_ann_cvode_tol": 5.0e-05}


def test_per_species_mlp(tmp_path):
    os.chdir(tmp_path)
    write_h2_learning_database(tmp_path)

    # One MLP per species, 2 hidden layers of 6 neurons
    with open("networks_params.yaml") as file:
        params = yaml.safe_load(file)
    params["networks_types"] = ["PerSpeciesMLP"]
    params["clusters"]["cluster0"] = {"nb_units_in_layers_list": [6, 6], "layers_activation_list": ["tanh", "tanh", "Id"],
                                      "layers_type": ["dense", "dense", "dense"]}
    with open("networks_params.yaml", "w") as file:
        yaml.dump(params, file, default_flow_style=False)

    NN_manager().train_all_clusters()

    # One MLP per network species (8 without N2), each with one output, concatenated in the species order
    model = torch.load("MODELS/MODEL_TEST/cluster0_model.pth", map_location="cpu", weights_only=False)
    n_in, n_species = 1 + 8, 8
    assert isinstance(model, PerSpeciesMLP) and len(model.species_models) == n_species

    x = torch.rand(5, n_in, dtype=torch.float64)
    with torch.no_grad():
        y = model(x)
        y_species = torch.cat([species_model(x) for species_model in model.species_models], dim=1)
    assert y.shape == (5, n_species) and torch.equal(y, y_species)

    # h5 export: one group per species, layers of its MLP
    with h5py.File("MODELS/MODEL_TEST/cluster0_model.h5", "r") as f:
        assert sorted(f.keys()) == sorted(f"species_{i}" for i in range(n_species))
        for i, species_model in enumerate(model.species_models):
            grp = f[f"species_{i}"]
            assert [grp[layer]["kernel:0"].shape for layer in ["dense_layer_1", "dense_layer_2", "output_layer"]] == [(n_in, 6), (6, 6), (6, 1)]
            assert np.array_equal(grp["output_layer/kernel:0"][()], species_model.model.output_layer.weight.detach().numpy().T)
            assert np.array_equal(grp["output_layer/bias:0"][()], species_model.model.output_layer.bias.detach().numpy())
            assert grp["dense_layer_1"].attrs["activation"] == "Tanh"

    # Inference as for an MLP
    test = NNTesting(testing_parameters)
    result = test.run_0D_ignition_case(phi=0.6, T0=1200.0, pressure=pressure, dt=dt, max_simulation_time=3 * dt)
    assert result["states_ann"].shape == result["states_ref"].shape
    assert np.all(np.isfinite(result["states_ann"]))
