import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig

from src.utils.training import calc_zslice
from src.model.components import RectangularMLP, ResidualMLP

class Decoder(nn.Module):
    """Decoder that reconstructs the sequence of rules from laten z"""
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.z_slice, self.input_size = calc_zslice(cfg.value_decoder.z_slice, cfg.z_size)
        self.hidden_size = cfg.decoder.size_hidden
        self.architecture = cfg.decoder.architecture
        self.out_len, self.out_width = cfg.io_format.seq_len, cfg.io_format.token_cnt

        if self.architecture == 'mlp-parameterized':
            self.mlp = RectangularMLP(cfg.decoder.depth, cfg.decoder.width, self.input_size, self.out_len * self.out_width)
        elif self.architecture == 'residual-parameterized':
            self.mlp = ResidualMLP(cfg.decoder.depth, cfg.decoder.width, self.input_size, self.out_len * self.out_width)
        else:
            raise ValueError('Select architecture from [mlp-parameterized, residual-parameterized]')

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """The forward pass used for training the Grammar VAE.
        Output size is [batch, max_length, token_cnt]  where token_cnt includes const category
        """
        # Get relevant part of latent space
        z = z[:, self.z_slice[0]:self.z_slice[1]]
        batch_size = z.shape[0]

        x = self.mlp(z)
        x = x.view(batch_size, self.out_len, self.out_width)

        return x
