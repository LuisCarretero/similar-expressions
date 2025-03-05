import torch
import torch.nn as nn
from omegaconf.dictconfig import DictConfig

from src.utils.training import calc_zslice
from src.model.components import build_rectengular_mlp, build_residual_mlp

class ValueDecoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Calculate input size
        self.z_slice, self.input_size = calc_zslice(cfg.value_decoder.z_slice, cfg.z_size)
        self.out_dim = cfg.io_format.val_points
        self.architecture = cfg.value_decoder.architecture
        
        # Define the layers
        if self.architecture == 'mlp-parameterized':
            self.lin = build_rectengular_mlp(cfg.value_decoder.depth, cfg.value_decoder.width, self.input_size, self.out_dim)
        elif self.architecture == 'residual-parameterized':
            self.lin = build_residual_mlp(cfg.value_decoder.depth, cfg.value_decoder.width, self.input_size, self.out_dim)
        else:
            raise ValueError(f'Invalid value for `architecture`: {self.architecture}.'
                             ' Must be in [mlp-parameterized, residual-parameterized]')
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Get relevant part of latent space
        u = z[:, self.z_slice[0]:self.z_slice[1]]

        out = self.lin(u)
        return out
