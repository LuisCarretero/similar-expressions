import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from encoder import Encoder
from decoder import Decoder
from value_decoder import ValueDecoder
from grammar import GCFG
from util import logits_to_prods

class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder"""
    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, token_cnt, seq_len, rnn_type, val_points, device=None):
        super(GrammarVAE, self).__init__()
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu") 
            print(f'Using device: {self.device}')
        else:
            self.device = device

        self.encoder = Encoder(token_cnt, hidden_encoder_size, z_dim).to(self.device)
        self.decoder = Decoder(z_dim, hidden_decoder_size, token_cnt, rnn_type).to(self.device)
        self.value_decoder = ValueDecoder(z_dim, val_points).to(self.device)
        self.to(self.device)

    def sample(self, mu, sigma):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        normal = Normal(torch.zeros(mu.shape).to(self.device), torch.ones(sigma.shape).to(self.device))
        eps = Variable(normal.sample())
        z = mu + eps*torch.sqrt(sigma)
        return z

    def kl(self, mu, sigma):
        """KL divergence between two normal distributions"""
        return torch.mean(-0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp(), 1))

    def forward(self, x, max_length=15):
        mu, sigma = self.encoder(x)
        z = self.sample(mu, sigma)
        logits = self.decoder(z, max_length=max_length)
        values = self.value_decoder(z)
        return logits, values

    def generate(self, z, sample=False, max_length=15):
        """Generate a valid expression from z using the decoder and grammar to create a set of rules that can 
        be parsed into an expression tree. Note that it only works on single equations at a time."""

        # Decoder works with general batch size. Only allow batch size 1 for now
        logits = self.decoder(z, max_length=max_length)
        assert logits.shape[0] == 1, "Batch size must be 1"
        logits = logits.squeeze()  # Only considering 1st batch

        return logits_to_prods(logits, GCFG, sample=sample, max_length=max_length)
        