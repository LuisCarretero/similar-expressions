import math
from config_util import Config
from typing import Dict

class Stack:
    """A simple first in last out stack.

    Args:
        grammar: an instance of nltk.CFG
        start_symbol: an instance of nltk.Nonterminal that is the
            start symbol the grammar
    """
    def __init__(self, grammar, start_symbol):
        self.grammar = grammar
        self._stack = [start_symbol]

    def pop(self):
        return self._stack.pop()

    def push(self, symbol):
        self._stack.append(symbol)

    def __str__(self):
        return str(self._stack)

    @property
    def nonempty(self):
        return bool(self._stack)

class AnnealKLSigmoid:
    """Anneal the KL for VAE based training using a sigmoid schedule"""
    def __init__(self, cfg: Config):
        self.total_epochs = cfg.training.epochs
        self.midpoint = cfg.training.anneal.midpoint
        self.steepness = cfg.training.anneal.steepness

    def alpha(self, epoch):
        """
        Calculate the annealing factor using a sigmoid function.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
        
        Returns:
            float: Annealing factor between 0 and 1
        """
        x = (epoch / self.total_epochs - self.midpoint) * self.steepness
        return 1 / (1 + math.exp(-x))

def criterion_factory(cfg: Config, priors: Dict):
    SYNTAX_LOSS_WEIGHT = cfg.training.criterion.syntax_loss_weight
    CONSTS_LOSS_WEIGHT = cfg.training.criterion.consts_loss_weight
    VALUES_LOSS_WEIGHT = cfg.training.criterion.values_loss_weight

    SYNTAX_PRIOR = priors['syntax_prior']
    CONSTS_PRIOR = priors['consts_prior']
    VALUES_PRIOR = priors['values_prior']

    cross_entropy = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    def criterion(logits, values, y_rule_idx, y_consts, y_val):
        
        logits_onehot = logits[:, :, :-1]
        loss_syntax = cross_entropy(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))/SYNTAX_PRIOR
        loss_consts = mse(logits[:, :, -1], y_consts)/CONSTS_PRIOR

        loss_values = mse(values, y_val)/VALUES_PRIOR

        loss = loss_syntax*SYNTAX_LOSS_WEIGHT + loss_consts*CONSTS_LOSS_WEIGHT + loss_values*VALUES_LOSS_WEIGHT
        loss = loss / (SYNTAX_LOSS_WEIGHT + CONSTS_LOSS_WEIGHT + VALUES_LOSS_WEIGHT)

        return loss_syntax, loss_consts, loss_values, loss
    return criterion

def calc_syntax_accuracy(logits, y_rule_idx):
    y_hat = logits.argmax(-1)
    a = (y_hat == y_rule_idx).float().mean()
    return 100 * a.item()
