from dataclasses import dataclass
from typing import List

import torch


@dataclass
class MLPConfig:
    """Configuration for the MLP model."""

    # Number of input features
    input_size: int

    # Number of hidden units
    hidden_size: List[int]

    # Number of output units
    output_size: int

    # Dropout rate
    dropout: float = 0.1


class MLP(torch.nn.Module):
    """Multi-layer perceptron model."""

    def __init__(self, cfg: MLPConfig):
        super(MLP, self).__init__()
        self.cfg = cfg
        self.layers = torch.nn.ModuleList()
        for i, hidden_size in enumerate(cfg.hidden_size):
            if i == 0:
                self.layers.append(torch.nn.Linear(cfg.input_size, hidden_size))
            else:
                self.layers.append(torch.nn.Linear(cfg.hidden_size[i - 1], hidden_size))
            self.layers.append(torch.nn.SiLU())
            if cfg.dropout > 0:
                self.layers.append(torch.nn.Dropout(cfg.dropout))
        self.layers.append(torch.nn.Linear(cfg.hidden_size[-1], cfg.output_size))

        self._init_weights()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
