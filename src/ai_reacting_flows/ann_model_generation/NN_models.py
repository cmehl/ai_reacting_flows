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