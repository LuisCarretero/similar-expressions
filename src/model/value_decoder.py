import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from configs import ArchitectureConfig

from encoder import Encoder

class ValueDecoder(nn.Module):
    def __init__(self, cfg: ArchitectureConfig):
        super().__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(cfg.z_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc2_5 = nn.Linear(128, 256)
        self.fc2_6 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, cfg.io_format.val_points)
        print("Using updated value decoder 2.0!")
    
    def forward(self, x):
        # Forward pass through the layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_5(x))
        x = F.relu(self.fc2_6(x))
        
        # Output layer with linear activation
        x = self.fc3(x)
        
        return x


if __name__ == '__main__':
    # Run tests? See other decoder for reference
    pass
