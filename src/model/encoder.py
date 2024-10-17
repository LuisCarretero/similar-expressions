import torch
import torch.nn as nn
from config_util import ModelConfig
from util import build_rectengular_mlp
class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_dim, self.input_len = cfg.io_format.token_cnt, cfg.io_format.seq_len
        self.hidden_size = cfg.encoder.size_hidden
        
        self.mlp, self.conv = None, None

        if cfg.encoder.architecture == 'small':
            self.conv = nn.Sequential(
                nn.Conv1d(self.input_dim, 2, kernel_size=2), nn.ReLU(),
                nn.Conv1d(2, 3, kernel_size=3), nn.ReLU(),
                nn.Conv1d(3, 4, kernel_size=4), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(36, cfg.encoder.size_hidden), nn.ReLU()
            )
        elif cfg.encoder.architecture == 'large':
            self.conv = nn.Sequential(
                nn.Conv1d(self.input_dim, self.input_dim*2, kernel_size=2), nn.ReLU(),
                nn.Conv1d(self.input_dim*2, self.input_dim, kernel_size=3), nn.ReLU(),
                nn.Conv1d(self.input_dim, self.input_dim, kernel_size=4), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.input_dim*9, self.hidden_size), nn.ReLU()
            )
        elif cfg.encoder.architecture == 'extra_large':
            self.conv = nn.Sequential(
                nn.Conv1d(self.input_dim, self.input_dim*4, kernel_size=2), nn.ReLU(),
                nn.Conv1d(self.input_dim*4, self.input_dim*2, kernel_size=3), nn.ReLU(),
                nn.Conv1d(self.input_dim*2, self.input_dim*2, kernel_size=4), nn.ReLU(),
                nn.Conv1d(self.input_dim*2, self.input_dim, kernel_size=5), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self.input_dim*5, self.hidden_size), nn.ReLU()
            )
        elif cfg.encoder.architecture == 'mlp-large':
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_len*self.input_dim, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 2048), nn.ReLU(),
                nn.Linear(2048, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, self.hidden_size), nn.ReLU()
            )
        elif cfg.encoder.architecture == 'mlp':
            self.mlp = build_rectengular_mlp(cfg.encoder.depth, cfg.encoder.width, self.input_len*self.input_dim, self.hidden_size)
        elif cfg.encoder.architecture == 'mlp-wide':
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_len*self.input_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, self.hidden_size), nn.ReLU()
            )
        else:
            raise ValueError(f'Invalid value for `architecture`: {cfg.encoder.architecture}.'
                             ' Must be in [small, large, extra_large, mlp-large, mlp, mlp-wide]')

        self.mu = nn.Linear(self.hidden_size, cfg.z_size)
        self.sigma = nn.Linear(self.hidden_size, cfg.z_size)

        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Encode x into a mean and variance of a Normal.

        Input size is size is [batch_cnt, seq_len, token_cnt] where token_cnt is the number of categories plus one for the value.
        """
        if self.conv is not None:
            x = x.permute(0, 2, 1)  # conv1d expects [batch, in_channels, seq_len]
            h = self.conv(x)
        elif self.mlp is not None:
            h = self.mlp(x)

        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))

        return mu, sigma