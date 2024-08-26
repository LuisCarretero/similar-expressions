import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal
from encoder import Encoder
from decoder import Decoder
from value_decoder import ValueDecoder
from parsing import logits_to_prods
from grammar import GCFG
from config_util import Config

class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder"""
    def __init__(self, cfg: Config):
        super(GrammarVAE, self).__init__()
        if cfg.training.device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu") 
            print(f'Using device: {self.device}')
        else:
            self.device = cfg.training.device

        self.encoder = Encoder(cfg.model).to(self.device)
        self.decoder = Decoder(cfg.model).to(self.device)
        self.value_decoder = ValueDecoder(cfg.model).to(self.device)
        self.to(self.device)

    def sample(self, mean, ln_var):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        normal = Normal(torch.zeros(mean.shape).to(self.device), torch.ones(ln_var.shape).to(self.device)*0.01)
        eps = normal.sample()
        z = mean + eps * torch.exp(ln_var/2)
        return z

    def kl(self, mean, ln_var):
        """KL divergence between two normal distributions"""
        return -0.5 * torch.mean(torch.sum(1 + ln_var - mean.pow(2) - ln_var.exp(), 1))

    def forward(self, x, max_length=15):
        mean, ln_var = self.encoder(x)
        z = self.sample(mean, ln_var)
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
        