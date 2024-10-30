import torch.nn as nn
from omegaconf.dictconfig import DictConfig

from src.model.util import calc_zslice
from src.model.components import build_rectengular_mlp, build_residual_mlp

class ValueDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Calculate input size
        self.z_slice, self.input_size = calc_zslice(cfg.value_decoder.z_slice, cfg.z_size)
        self.out_dim = cfg.io_format.val_points
        self.architecture = cfg.value_decoder.architecture
        
        # Define the layers
        if self.architecture == 'mlp-small':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, self.out_dim)
            )
        elif self.architecture == 'mlp-medium':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
        elif self.architecture == 'mlp-parameterized':
            self.lin = build_rectengular_mlp(cfg.value_decoder.depth, cfg.value_decoder.width, self.input_size, self.out_dim)
        elif self.architecture == 'residual-parameterized':
            self.lin = build_residual_mlp(cfg.value_decoder.depth, cfg.value_decoder.width, self.input_size, self.out_dim)
        else:
            raise ValueError(f'Invalid value for `architecture`: {self.architecture}.'
                             ' Must be in [small, medium, large]')
    
    def forward(self, z):
        # Get relevant part of latent space
        u = z[:, self.z_slice[0]:self.z_slice[1]]

        out = self.lin(u)
        return out
