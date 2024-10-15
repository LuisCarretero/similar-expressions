import torch.nn as nn
from torch.nn import functional as F
from config_util import ModelConfig

class ValueDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()

        # Calculate input size
        self.z_slice = cfg.value_decoder.z_slice.copy()
        if self.z_slice[1] == -1:
            self.z_slice[1] = cfg.z_size
        self.input_size = self.z_slice[1] - self.z_slice[0]
        
        # Define the layers
        if cfg.value_decoder.conv_size == 'small':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, cfg.io_format.val_points)
            )
        elif cfg.value_decoder.conv_size == 'medium':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, cfg.io_format.val_points)
            )
        elif cfg.value_decoder.conv_size == 'medium-new':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, cfg.io_format.val_points)
            )
        elif cfg.value_decoder.conv_size == 'large':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, cfg.io_format.val_points)
            )
        elif cfg.value_decoder.conv_size == 'extra_large':
            self.lin = nn.Sequential(
                nn.Linear(self.input_size, 512), nn.ReLU(),
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, 2048), nn.ReLU(),
                nn.Linear(2048, cfg.io_format.val_points)
            )
        else:
            raise ValueError(f'Invalid value for `conv_size`: {cfg.value_decoder.conv_size}.'
                             ' Must be in [small, medium, large]')
    
    def forward(self, z):
        # Get relevant part of latent space
        u = z[:, self.z_slice[0]:self.z_slice[1]]

        out = self.lin(u)
        return out
