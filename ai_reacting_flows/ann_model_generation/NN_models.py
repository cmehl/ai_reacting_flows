import sys

import torch
import torch.nn as nn

import ai_reacting_flows.ann_model_generation.Custom_layers as CLayers

torch.set_default_dtype(torch.float64)

class MLPModel(nn.Module):
    
    def __init__(self, device, hidden_layers : list[int], layers_type : list[str], activations : list):
        super(MLPModel, self).__init__()
            
        layers = []        

        # Add hidden layers
        for i in range(1, len(hidden_layers)):
            if layers_type[i-1] == "dense":
                layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i],dtype=torch.float64))
                layers.append(activations[i-1]())
            elif layers_type[i-1] == "resnet":
                layers.append(CLayers.ResidualBlock(hidden_layers[i - 1], hidden_layers[i], activations[i-1]))
            else:
                sys.exit(f"ERROR: layer type \"{layers_type[i]}\" does not exist")

        # Combine all layers
        self.model = nn.Sequential(*layers).to(device).to(torch.float64)

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