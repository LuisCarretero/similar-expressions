import math
from config_util import Config
from typing import Dict
import torch

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
    """Anneal the KL for VAE based training using a sigmoid schedule. No overall weighting so this return float between 0 and 1."""
    def __init__(self, cfg: Config):
        self.total_epochs = cfg.training.epochs
        self.midpoint = cfg.training.kl_anneal.midpoint
        self.steepness = cfg.training.kl_anneal.steepness

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
    """
    Factory function to create the criterion for the VAE.
    """
    AE_WEIGHT = cfg.training.criterion.ae_weight
    KL_WEIGHT = cfg.training.criterion.kl_weight
    SYNTAX_WEIGHT = cfg.training.criterion.syntax_weight

    assert 0 <= AE_WEIGHT <= 1, "AE_WEIGHT must be between 0 and 1"
    assert 0 <= SYNTAX_WEIGHT <= 1, "SYNTAX_WEIGHT must be between 0 and 1"
    assert KL_WEIGHT > 0, "KL_WEIGHT must be greater than 0"

    SYNTAX_PRIOR = priors['syntax_prior']
    CONSTS_PRIOR = priors['consts_prior']
    VALUES_PRIOR = priors['values_prior']

    cross_entropy = torch.nn.CrossEntropyLoss()  # default reduction is mean over batch and time steps
    mse = torch.nn.MSELoss()

    def criterion(logits_pred: torch.Tensor, values_pred: torch.Tensor, y_rule_idx: torch.Tensor, y_consts: torch.Tensor, y_val: torch.Tensor, kl: float, alpha: float):
        """
        logits: syntax and consts prediction of the model
        values: value prediction of the model
        y_rule_idx: true one-hot encoded syntax indices
        y_consts: true real-valued consts
        y_val: true values
        
        kl: kl divergence of samples (single scalar summed over all dimensions of latent space and mean over batch)

        """
        # VAE reconstruction loss
        logits_onehot = logits_pred[:, :, :-1]
        loss_syntax = cross_entropy(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))/SYNTAX_PRIOR
        loss_consts = mse(logits_pred[:, :, -1], y_consts)/CONSTS_PRIOR
        loss_recon_ae = SYNTAX_WEIGHT*loss_syntax + (1-SYNTAX_WEIGHT)*loss_consts

        # VAE total loss (loss_ae = -ELBO = -log p(x|z) + KL_WEIGHT*KL(q(z|x)||p(z)) where KL_WEIGHT is usually denoted as beta)
        loss_ae = loss_recon_ae + KL_WEIGHT*alpha*kl

        # Value prediction loss
        loss_values = mse(values_pred, y_val)/VALUES_PRIOR

        # Total loss
        loss = AE_WEIGHT*loss_ae + (1-AE_WEIGHT)*loss_values

        partial_losses = {
            'loss_syntax': loss_syntax,
            'loss_consts': loss_consts,
            'loss_recon_ae': loss_recon_ae,
            'kl': kl,
            'alpha': alpha,
            'loss_ae': loss_ae,   # -ELBO but with KL_WEIGHT*alpha so really only some distant cousing of ELBO
            'loss_values': loss_values,
            'loss': loss
        }

        return loss, partial_losses
    return criterion

def compute_latent_metrics(mean, ln_var):
    """
    Compute the metrics for the latent space.
    """
    mean_norm = torch.norm(mean, dim=1).mean().item()
    ln_var_norm = torch.norm(ln_var, dim=1).mean().item()

    metrics = {
        'mean_norm': mean_norm,
        'ln_var_norm': ln_var_norm
    }

    return metrics


def calc_syntax_accuracy(logits, y_rule_idx):
    y_hat = logits.argmax(-1)
    a = (y_hat == y_rule_idx).float().mean()
    return 100 * a.item()
