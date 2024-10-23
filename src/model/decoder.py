import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig

from src.model.util import calc_zslice, build_rectengular_mlp

class Decoder(nn.Module):
    """Decoder that reconstructs the sequence of rules from laten z"""
    def __init__(self, cfg: DictConfig):
        super().__init__()  # Decoder, self

        self.z_slice, self.input_size = calc_zslice(cfg.value_decoder.z_slice, cfg.z_size)
        self.hidden_size = cfg.decoder.size_hidden
        self.architecture = cfg.decoder.architecture
        self.out_len, self.out_width = cfg.io_format.seq_len, cfg.io_format.token_cnt

        self.rnn, self.lin = None, None

        if self.architecture == 'lstm':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.architecture == 'lstm-large':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=5, batch_first=True)
        elif self.architecture == 'gru':
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.architecture == 'mlp-parameterized':
            out_dim = self.out_len * self.out_width
            self.lin = build_rectengular_mlp(cfg.decoder.depth, cfg.decoder.width, self.input_size, out_dim)
        else:
            raise ValueError('Select architecture from [lstm, lstm-large, gru, mlp-parameterized]')

        if self.rnn is not None:
            self.linear_in = nn.Linear(self.input_size, self.hidden_size)
            self.linear_out = nn.Linear(self.hidden_size, self.out_width)
    
    def forward(self, z, max_length=None):  # FIXME: Rm max_length
        """The forward pass used for training the Grammar VAE.
        TODO: Does it make sense to have max_length as parameter?

        For the rnn we follow the same convention as the official keras
        implementaion: the latent z is the input to the rnn at each timestep.
        See line 138 of
            https://github.com/mkusner/grammarVAE/blob/master/models/model_eq.py
        for reference.

        Output size is [batch, max_length, token_cnt]  where token_cnt includes const category
        """
        # Get relevant part of latent space
        z = z[:, self.z_slice[0]:self.z_slice[1]]
        batch_size = z.size(0)

        if self.rnn is not None:
            x = F.relu(self.linear_in(z))

            # The input to the rnn is the same for each timestep: it is z.
            x = x.unsqueeze(1).expand(-1, self.out_len, -1)

            if self.architecture == 'lstm' or self.architecture == 'lstm-large':
                # Init hidden and cell states
                h0 = torch.zeros(1, batch_size, self.hidden_size).to(z.device)
                c0 = torch.zeros(1, batch_size, self.hidden_size).to(z.device)
                hx = (h0, c0)
            else:  # for GRU
                hx = torch.zeros(1, batch_size, self.hidden_size).to(z.device)

            x, _ = self.rnn(x, hx)
            x = self.linear_out(F.relu(x))
        elif self.lin is not None:
            x = self.lin(z)
            x = x.view(batch_size, self.out_len, self.out_width)
        else:
            raise ValueError('Invalid architecture')

        return x
