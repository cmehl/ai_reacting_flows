import sys

from collections import OrderedDict

import torch
import torch.nn as nn

import ai_reacting_flows.ann_model_generation.Custom_layers as CLayers

torch.set_default_dtype(torch.float64)

class MLPModel(nn.Module):
    def __init__(self, device, hidden_layers: list[int], layers_type: list[str], activations: list):
        super().__init__()
        self.activation_map = {} 

        layers_dict = OrderedDict()

        for i in range(1, len(hidden_layers)):

            # Setting layers names properly
            if i==len(hidden_layers)-1:
                layer_name = "output_layer"
            else:
                if layers_type[i-1] == "dense":
                    layer_name=f"dense_layer_{i}"
                elif layers_type[i-1] == "resnet":
                    layer_name=f"hidden_residual_block_{i}"
                else:
                    sys.exit(f"ERROR: layer type \"{layers_type[i-1]}\" does not exist")
            
            act_name = f"activation_{i}"

            if layers_type[i-1] == "dense":
                layers_dict[layer_name] = nn.Linear(hidden_layers[i-1], hidden_layers[i], dtype=torch.float64)
                layers_dict[act_name] = activations[i-1]()
                self.activation_map[layer_name] = activations[i-1].__name__
            elif layers_type[i-1] == "resnet":
                layers_dict[layer_name] = CLayers.ResidualBlock(hidden_layers[i-1], hidden_layers[i], activations[i-1])
                self.activation_map[layer_name] = activations[i-1].__name__
            else:
                sys.exit(f"ERROR: layer type \"{layers_type[i-1]}\" does not exist")

        self.model = nn.Sequential(layers_dict).to(device).to(torch.float64)
        
    def forward(self, x):
        return self.model(x)

class PerSpeciesMLP(nn.Module):
    """One MLP per output (network species), all with the same architecture of dense layers: each takes the
    whole input state [T, Y_k, (dt)] and predicts a single output. The outputs are in the order of the species,
    so that the model is used as MLPModel in training, inference and testing.

    The MLPs of all species are computed together: the weights of a layer are stacked in one tensor
    (n_species, n_in, n_out), and the layer is one batched matrix product for all species, instead of one
    small product per species (much faster to train), as in the inference of NNICE.

    hidden_layers: [n_in, hidden layers..., 1], layers_type: "dense" only, activations: one per layer
    (including the output layer), n_out: number of outputs (one MLP each).
    """

    def __init__(self, device, hidden_layers: list[int], layers_type: list[str], activations: list, n_out: int):
        super().__init__()

        if any(layer_type != "dense" for layer_type in layers_type):
            sys.exit(f"ERROR: PerSpeciesMLP only has dense layers, got {layers_type}")

        self.n_species = n_out

        # Layers named as in MLPModel (h5 export), activations
        n_layers = len(hidden_layers) - 1
        self.layer_names = [f"dense_layer_{i}" for i in range(1, n_layers)] + ["output_layer"]
        self.activations = nn.ModuleList([activation() for activation in activations])

        # Weights (n_species, n_in, n_out) and biases (n_species, 1, n_out) of each layer, initialized as nn.Linear:
        # uniform in [-1/sqrt(n_in), 1/sqrt(n_in)]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for n_in, n_out_layer in zip(hidden_layers[:-1], hidden_layers[1:]):
            bound = 1.0 / n_in**0.5
            self.weights.append(nn.Parameter(torch.empty(n_out, n_in, n_out_layer, device=device).uniform_(-bound, bound)))
            self.biases.append(nn.Parameter(torch.empty(n_out, 1, n_out_layer, device=device).uniform_(-bound, bound)))

    def forward(self, x):
        # Same input for all species: (n_species, batch, n_in)
        h = x.unsqueeze(0).expand(self.n_species, -1, -1)

        # Each layer: h W + b for all species at once, then activation
        for weight, bias, activation in zip(self.weights, self.biases, self.activations):
            h = activation(torch.baddbmm(bias, h, weight))

        # (n_species, batch, 1) -> (batch, n_species)
        return h[:, :, 0].T

    def species_layers(self, i_species):
        """Layers of the MLP of species i_species: list of (name, activation name, kernel (n_in, n_out), bias (n_out)),
        as numpy arrays (h5 export, with the layout of MLPModel)."""

        return [(name, type(activation).__name__, weight[i_species].detach().cpu().numpy(), bias[i_species, 0].detach().cpu().numpy())
                for name, activation, weight, bias in zip(self.layer_names, self.activations, self.weights, self.biases)]

class DeepONet(nn.Module):
    
    def __init__(self, device, hidden_layers : dict[str,list[int]], layers_type : dict[str,list[str]], activations : dict[str,list], n_out, n_neuron):
        super(MLPModel, self).__init__()

        self.branch = MLPModel(device, hidden_layers["branch"], layers_type["branch"] ,activations["branch"])
        self.trunk = MLPModel(device, hidden_layers["trunk"], layers_type["trunk"] ,activations["trunk"])

        self.n_neurons = n_neuron
        self.n_out = n_out

    def forward(self, x):
        dt = x[:,-1]
        y = x[:,:-1]

        dt = dt.reshape((x.shape[0], 1))

        b = self.branch(y)
        t = self.trunk(dt)

        y_dt = torch.zeros((x.shape[0], self.n_out))

        b = b.reshape((x.shape[0], self.n_out, self.n_neurons))
        t = t.reshape((x.shape[0], self.n_out, self.n_neurons))

        y_dt = torch.sum(b * t, dim=2) 

        return y_dt

class DeepONet_shift(nn.Module):
    
    def __init__(self, device, hidden_layers : dict[str,list[int]], layers_type : dict[str,list[str]], activations : dict[str,list], n_out, n_neuron):
        super(MLPModel, self).__init__()

        self.branch = MLPModel(device, hidden_layers["branch"], layers_type["branch"] ,activations["branch"])
        self.trunk = MLPModel(device, hidden_layers["trunk"], layers_type["trunk"] ,activations["trunk"])
        self.shift = MLPModel(device, hidden_layers["shift"], layers_type["shift"] ,activations["shift"])

        self.n_neurons = n_neuron
        self.n_out = n_out

    def forward(self, x):
        dt = x[:,-1]
        y = x[:,:-1]

        dt = dt.reshape((x.shape[0], 1))

        s = self.shift(y)
        b = self.branch(y)
        dt_s = dt + s
        t = self.trunk(dt_s)

        y_dt = torch.zeros((x.shape[0], self.n_out))

        b = b.reshape((x.shape[0], self.n_out, self.n_neurons))
        t = t.reshape((x.shape[0], self.n_out, self.n_neurons))

        y_dt = torch.sum(b * t, dim=2) 

        return y_dt