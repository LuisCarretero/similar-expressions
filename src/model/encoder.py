import torch
import torch.nn as nn
from config_util import ModelConfig

class Encoder(nn.Module):
    """Convolutional encoder for Grammar VAE.

    Applies a series of one-dimensional convolutions to a batch
    of one-hot encodings of the sequence of rules that generate
    an artithmetic expression.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        input_dim = cfg.io_format.token_cnt
        seq_length = cfg.io_format.seq_len
        
        self.mlp, self.conv = None, None

        if cfg.encoder.conv_size == 'small':
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, 2, kernel_size=2), nn.ReLU(),
                nn.Conv1d(2, 3, kernel_size=3), nn.ReLU(),
                nn.Conv1d(3, 4, kernel_size=4), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(36, cfg.encoder.size_hidden), nn.ReLU()
            )
        elif cfg.encoder.conv_size == 'large':
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim*2, kernel_size=2), nn.ReLU(),
                nn.Conv1d(input_dim*2, input_dim, kernel_size=3), nn.ReLU(),
                nn.Conv1d(input_dim, input_dim, kernel_size=4), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(input_dim*9, cfg.encoder.size_hidden), nn.ReLU()
            )
        elif cfg.encoder.conv_size == 'extra_large':
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, input_dim*4, kernel_size=2), nn.ReLU(),
                nn.Conv1d(input_dim*4, input_dim*2, kernel_size=3), nn.ReLU(),
                nn.Conv1d(input_dim*2, input_dim*2, kernel_size=4), nn.ReLU(),
                nn.Conv1d(input_dim*2, input_dim, kernel_size=5), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(input_dim*5, cfg.encoder.size_hidden), nn.ReLU()
            )
        elif cfg.encoder.conv_size == 'mlp-large':
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_length*input_dim, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 1024), nn.ReLU(),
                nn.Linear(1024, 2048), nn.ReLU(),
                nn.Linear(2048, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, cfg.encoder.size_hidden), nn.ReLU()
            )
        elif cfg.encoder.conv_size == 'mlp':
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_length*input_dim, 512), nn.ReLU(),
                nn.Linear(512, 512), nn.ReLU(),
                nn.Linear(512, cfg.encoder.size_hidden), nn.ReLU()
            )
        elif cfg.encoder.conv_size == 'mlp-wide':
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(seq_length*input_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 1024), nn.ReLU(),
                nn.Linear(1024, cfg.encoder.size_hidden), nn.ReLU()
            )
        else:
            raise ValueError(f'Invalid value for `conv_size`: {cfg.encoder.conv_size}.'
                             ' Must be in [small, large, extra_large]')

        self.mu = nn.Linear(cfg.encoder.size_hidden, cfg.z_size)
        self.sigma = nn.Linear(cfg.encoder.size_hidden, cfg.z_size)

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