import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Categorical

from nltk import Nonterminal
from nltk.grammar import Production

from encoder import Encoder
from decoder import Decoder
from stack import Stack
from grammar import GCFG, S, get_mask

class GrammarVAE(nn.Module):
    """Grammar Variational Autoencoder"""
    def __init__(self, hidden_encoder_size, z_dim, hidden_decoder_size, output_size, rnn_type, device=None):
        super(GrammarVAE, self).__init__()
        if device is None:
            self.device = (
                "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu") 
            print(f'Using device: {self.device}')
        else:
            self.device = device

        self.encoder = Encoder(hidden_encoder_size, z_dim).to(self.device)
        self.decoder = Decoder(z_dim, hidden_decoder_size, output_size, rnn_type).to(self.device)
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
        return logits

    def generate(self, z, sample=False, max_length=15):
        """Generate a valid expression from z using the decoder and grammar to create a set of rules that can 
        be parsed into an expression tree. Note that it only works on single equations at a time."""
        stack = Stack(grammar=GCFG, start_symbol=S)

        # Decoder works with general batch size. Only allow batch size 1 for now
        logits = self.decoder(z, max_length=max_length)
        assert logits.shape[0] == 1, "Batch size must be 1"
        logits = logits.squeeze()  # Only considering 1st batch

        # print(f'{logits.shape = }')

        logits_prods = logits[:, :-1]
        constants = logits[:, -1]

        prods = []
        t = 0  # "Time step" in sequence
        while stack.nonempty:
            alpha = stack.pop()  # Alpha is notation in paper.
            mask = get_mask(alpha, stack.grammar, as_variable=True)
            # print(f'{mask.shape = }')
            probs = mask * logits_prods[t].exp()
            probs = probs / probs.sum()
            
            if sample:
                m = Categorical(probs)
                i = m.sample()
            else:
                _, i = probs.max(-1) # argmax

            # select rule i
            rule = stack.grammar.productions()[i.item()]

            # If rule has -> [CONST] add const
            if rule.rhs()[0] == '[CONST]':
                rule = Production(lhs=rule.lhs(), rhs=(str(constants[t].item()),))

            prods.append(rule)
            # add rhs nonterminals to stack in reversed order
            for symbol in reversed(rule.rhs()):
                if isinstance(symbol, Nonterminal):
                    stack.push(symbol)
            t += 1
            if t == max_length:
                break
        return prods
    