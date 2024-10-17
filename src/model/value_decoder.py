import torch.nn as nn
from torch.nn import functional as F
from config_util import ModelConfig
from util import calc_zslice, build_rectengular_mlp

class ValueDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Calculate input size
        self.z_slice, self.input_size = calc_zslice(cfg.value_decoder.z_slice, cfg.z_size)
        self.out_dim = cfg.io_format.val_points
        
        # Define the layers
        if cfg.value_decoder.conv_size == 'mlp-small':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, self.out_dim)
            )
        elif cfg.value_decoder.conv_size == 'mlp-medium':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, self.out_dim)
            )
        elif cfg.value_decoder.conv_size == 'mlp':
            self.lin = build_rectengular_mlp(cfg.value_decoder.depth, cfg.value_decoder.width, self.input_size, self.out_dim)
        elif cfg.value_decoder.conv_size == 'mlp-medium-wide':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, self.out_dim)
            )
        elif cfg.value_decoder.conv_size == 'mlp-large':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, self.out_dim)
            )
        elif cfg.value_decoder.conv_size == 'mlp-extra-large':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 512), nn.ReLU(),
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, self.out_dim)
            )
        else:
            raise ValueError(f'Invalid value for `conv_size`: {cfg.value_decoder.conv_size}.'
                             ' Must be in [small, medium, large]')
    
    def forward(self, z):
        # Get relevant part of latent space
        u = z[:, self.z_slice[0]:self.z_slice[1]]

        out = self.lin(u)
        return out
