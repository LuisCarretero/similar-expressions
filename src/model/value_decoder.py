import torch.nn as nn
from torch.nn import functional as F
from config_util import ModelConfig

class ValueDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        
        # Define the layers
        if cfg.value_decoder.conv_size == 'small':
            self.fc1 = nn.Linear(cfg.z_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.final_linear = nn.Linear(256, cfg.io_format.val_points)
            self.fc4 = None
            self.fc5 = None
        elif cfg.value_decoder.conv_size == 'medium':
            self.fc1 = nn.Linear(cfg.z_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 512)
            self.fc4 = nn.Linear(512, 512)
            self.final_linear = nn.Linear(512, cfg.io_format.val_points)
        elif cfg.value_decoder.conv_size == 'large':
            self.fc1 = nn.Linear(cfg.z_size, 256)
            self.fc2 = nn.Linear(256, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, 1024)
            self.fc5 = nn.Linear(1024, 1024)
            self.final_linear = nn.Linear(1024, cfg.io_format.val_points)
        else:
            raise ValueError(f'Invalid value for `conv_size`: {cfg.value_decoder.conv_>>size}.'
                             ' Must be in [small, medium, large]')
    
    def forward(self, x):
        # Forward pass through the layers with ReLU activations
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        if self.fc3 is not None:
            h = F.relu(self.fc3(h))
        if self.fc4 is not None:
            h = F.relu(self.fc4(h))
        
        # Output layer with linear activation
        val_y = self.final_linear(h)
        
        return val_y
