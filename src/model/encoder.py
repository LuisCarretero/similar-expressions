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
            self.conv1 = nn.Conv1d(input_dim, 2, kernel_size=2)
            self.conv2 = nn.Conv1d(2, 3, kernel_size=3)
            self.conv3 = nn.Conv1d(3, 4, kernel_size=4)
            self.linear = nn.Linear(36, cfg.encoder.size_hidden)
        elif cfg.encoder.conv_size == 'large':
            self.conv1 = nn.Conv1d(input_dim, input_dim*2, kernel_size=2)
            self.conv2 = nn.Conv1d(input_dim*2, input_dim, kernel_size=3)
            self.conv3 = nn.Conv1d(input_dim, input_dim, kernel_size=4)
            self.linear = nn.Linear(input_dim*9, cfg.encoder.size_hidden)  # 15+(-2+1)+(-3+1)+(-4+1)=9 from sequence length + conv sizes
        else:
            raise ValueError('Invallid value for `conv_size`: {}.'
                             ' Must be in [small, large]'.format(cfg.encoder.conv_size))

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

        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = h.view(x.size(0), -1) # flatten
        h = self.relu(self.linear(h))
        mu = self.mu(h)
        sigma = self.softplus(self.sigma(h))
        return mu, sigma

if __name__ == '__main__':
    # Load data
    data_path = '../data/eq2_grammar_dataset.h5'
    f = h5py.File(data_path, 'r')
    data = f['data']

    # Create encoder
    encoder = Encoder(20, 2)

    # Pass through some data
    x = torch.from_numpy(data[:100]).transpose(-2, -1).float() # shape [batch, 12, 15]
    mu, sigma = encoder(x)

    print(x)
    print(mu)
