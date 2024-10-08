import torch
import torch.nn as nn
import h5py
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
        # input_dim, hidden_dim=20, z_dim=2, conv_size='large'
        if cfg.encoder.conv_size == 'small':
            # 12 rules, so 12 input channels
            self.l1 = nn.Conv1d(input_dim, 2)
            self.l2 = nn.Conv1d(2, 3)
            self.l3 = nn.Conv1d(3, 4)
            self.l_out = nn.Linear(36, cfg.encoder.size_hidden)
            self.l4 = None
        elif cfg.encoder.conv_size == 'large':
            self.l1 = nn.Conv1d(input_dim, input_dim*2)
            self.l2 = nn.Conv1d(input_dim*2, input_dim)
            self.l3 = nn.Conv1d(input_dim, input_dim)
            self.l_out = nn.Linear(input_dim*9, cfg.encoder.size_hidden)  # 15+(-2+1)+(-3+1)+(-4+1)=9 from sequence length + conv sizes
            self.l4 = None
        elif cfg.encoder.conv_size == 'extra_large':
            self.l1 = nn.Conv1d(input_dim, input_dim*4, kernel_size=2)
            self.l2 = nn.Conv1d(input_dim*4, input_dim*2, kernel_size=3)
            self.l3 = nn.Conv1d(input_dim*2, input_dim*2, kernel_size=4)
            self.l4 = nn.Conv1d(input_dim*2, input_dim, kernel_size=5)
            self.l_out = nn.Linear(input_dim*5, cfg.encoder.size_hidden)  # 15+(-2+1)+(-3+1)+(-4+1)+(-5+1)=5 from sequence length + conv sizes
        elif cfg.encoder.conv_size == 'large_mlp':
            self.l1 = nn.Linear(input_dim, 256)
            self.l2 = nn.Linear(256, 512)
            self.l3 = nn.Linear(512, 1024)
            self.l4 = nn.Linear(1024, 2048)
            self.l_out = nn.Linear(2048, cfg.encoder.size_hidden)
        else:
            raise ValueError(f'Invalid value for `conv_size`: {cfg.encoder.conv_size}.'
                             ' Must be in [small, large, extra_large]')

        self.mu = nn.Linear(cfg.encoder.size_hidden, cfg.z_size)
        self.sigma = nn.Linear(cfg.encoder.size_hidden, cfg.z_size)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Encode x into a mean and variance of a Normal.

        Input size is size is [batch, max_length, token_cnt]
        """
        x = x.permute(0, 2, 1)  # conv1d expects [batch, in_channels, seq_len]

        h = self.relu(self.l1(x))
        h = self.relu(self.l2(h))
        h = self.relu(self.l3(h))
        if self.l4 is not None:
            h = self.relu(self.l4(h))

        h = h.view(x.size(0), -1) # flatten
        h = self.relu(self.l_out(h))

        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))

        return mu, sigma
