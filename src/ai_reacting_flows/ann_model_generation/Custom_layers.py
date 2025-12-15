import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(in_features, out_features, dtype=torch.float64)
        self.activation1 = activation()

        self.linear2 = nn.Linear(out_features, out_features, dtype=torch.float64)
        self.activation2 = activation()

        # Si les dimensions d'entrée et de sortie diffèrent, ajuster avec une couche linéaire
        self.shortcut = nn.Linear(in_features, out_features, dtype=torch.float64) if in_features != out_features else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.linear1(x)
        out = self.activation1(out)
        out = self.linear2(out)
        out += identity

        return self.activation2(out)