import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig
from typing import Tuple

from src.model.components import RectangularMLP, ResidualMLP

class Encoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.input_dim, self.input_len = cfg.io_format.token_cnt, cfg.io_format.seq_len
        self.hidden_size = cfg.encoder.size_hidden

        if cfg.encoder.architecture == 'mlp-parameterized':
            self.mlp = RectangularMLP(cfg.encoder.depth, cfg.encoder.width, self.input_len*self.input_dim, self.hidden_size)
        elif cfg.encoder.architecture == 'residual-parameterized':
            self.mlp = ResidualMLP(cfg.encoder.depth, cfg.encoder.width, self.input_len*self.input_dim, self.hidden_size)
        else:
            raise ValueError(f'Invalid value for `architecture`: {cfg.encoder.architecture}.'
                             ' Must be in [conv-small, conv-large, conv-extra-large, mlp-parameterized, residual-parameterized]')

        self.mu = nn.Linear(self.hidden_size, cfg.z_size)
        self.sigma = nn.Linear(self.hidden_size, cfg.z_size)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode x into a mean and variance of a Normal.

        Input size is size is [batch_cnt, seq_len, token_cnt] where token_cnt is the number of categories plus one for the value.
        """
        x = x.flatten(1)  # [batch_size, seq_len*token_cnt]
        h = self.mlp(x)

        mean = self.mu(h)
        ln_var = self.softplus(self.sigma(h))

        return mean, ln_var