import torch
import torch.nn as nn
from config_util import ModelConfig

class Decoder(nn.Module):
    """RNN decoder that reconstructs the sequence of rules from laten z"""
    def __init__(self, cfg: ModelConfig):
        super().__init__()  # Decoder, self

        # Calculate input size
        self.z_slice = cfg.value_decoder.z_slice.copy()
        if self.z_slice[1] == -1:
            self.z_slice[1] = cfg.z_size
        self.input_size = self.z_slice[1] - self.z_slice[0]
        
        self.hidden_size = cfg.decoder.size_hidden
        self.rnn_type = cfg.decoder.rnn_type

        self.linear_in = nn.Linear(self.input_size, self.hidden_size)
        self.linear_out = nn.Linear(self.hidden_size, cfg.io_format.token_cnt)

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.rnn_type == 'lstm-large':
            self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=5, batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.rnn_type == 'mlp':
            self.lin = nn.Sequential(
                nn.Linear(self.hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.hidden_size*cfg.io_format.seq_len),
                nn.ReLU()
            )
        else:
            raise ValueError('Select rnn_type from [lstm, gru]')

        self.relu = nn.ReLU()
    
    def forward(self, z, max_length):
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

        if self.rnn_type != 'mlp':
            x = self.linear_in(z)
            x = self.relu(x)

            # The input to the rnn is the same for each timestep: it is z.
            x = x.unsqueeze(1).expand(-1, max_length, -1)

            if self.rnn_type == 'lstm' or self.rnn_type == 'lstm-large':
                # Init hidden and cell states
                h0 = torch.zeros(1, batch_size, self.hidden_size).to(z.device)
                c0 = torch.zeros(1, batch_size, self.hidden_size).to(z.device)
                hx = (h0, c0)
            else:  # for GRU
                hx = torch.zeros(1, batch_size, self.hidden_size).to(z.device)

            x, _ = self.rnn(x, hx)
        else:
            x = self.linear_in(z)
            x = self.lin(x)
            x = x.view(batch_size, max_length, self.hidden_size)

        x = self.relu(x)
        x = self.linear_out(x)
        return x
