import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(ResidualBlock, self).__init__()

        self.hidden_unit_0 = nn.Linear(in_features, out_features, dtype=torch.float64)
        self.activation0 = activation()

        self.hidden_unit_1 = nn.Linear(out_features, out_features, dtype=torch.float64)
        self.activation1 = activation()

        # Si les dimensions d'entrée et de sortie diffèrent, ajuster avec une couche linéaire
        self.shortcut = nn.Linear(in_features, out_features, dtype=torch.float64) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.hidden_unit_0(x)
        out = self.activation0(out)
        out = self.hidden_unit_1(out)
        out += identity

        return self.activation1(out)