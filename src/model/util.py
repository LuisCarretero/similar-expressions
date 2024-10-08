import math
from config_util import Config
from typing import Dict
import torch
import torch.nn.functional as F

class Stack:
    # TODO: Use built-in
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

    def alpha(self, epoch: int) -> float:
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
    CONTRASTIVE_WEIGHT = cfg.training.criterion.contrastive_weight
    KL_WEIGHT = cfg.training.criterion.kl_weight
    SYNTAX_WEIGHT = cfg.training.criterion.syntax_weight

    assert 0 <= AE_WEIGHT <= 1, "AE_WEIGHT must be between 0 and 1"
    assert 0 <= SYNTAX_WEIGHT <= 1, "SYNTAX_WEIGHT must be between 0 and 1"
    assert KL_WEIGHT >= 0, "KL_WEIGHT must be nonnegative"
    assert CONTRASTIVE_WEIGHT >= 0, "CONTRASTIVE_WEIGHT must be nonnegative"

    SYNTAX_PRIOR = priors['syntax_prior']
    CONSTS_PRIOR = priors['consts_prior']
    VALUES_PRIOR = priors['values_prior']

    cross_entropy = torch.nn.CrossEntropyLoss()  # default reduction is mean over batch and time steps
    mse = torch.nn.MSELoss()

    # Contrastive loss setup
    l2_dist = torch.nn.PairwiseDistance(p=2)
    a = 40  # Sharpness
    b = 5  # Shift
    gamma_func = lambda x: 1/(1+ torch.exp(a * x - b))

    def criterion(expr_pred: torch.Tensor, values_pred: torch.Tensor, y_rule_idx: torch.Tensor, y_consts: torch.Tensor, y_val: torch.Tensor, kl: float, alpha: float, z: torch.Tensor):
        """
        expr_pred: expression prediction of the model
        values_pred: value prediction of the model
        y_rule_idx: true one-hot encoded syntax indices
        y_consts: true real-valued consts
        y_val: true values
        z: latent samples

        kl: kl divergence of samples (single scalar summed over all dimensions of latent space and mean over batch)

        """
        # VAE reconstruction loss
        logits_syntax = expr_pred[:, :, :-1]
        loss_syntax = cross_entropy(logits_syntax.reshape(-1, logits_syntax.size(-1)), y_rule_idx.reshape(-1))/SYNTAX_PRIOR
        loss_consts = mse(expr_pred[:, :, -1], y_consts)/CONSTS_PRIOR
        loss_recon_ae = SYNTAX_WEIGHT*loss_syntax + (1-SYNTAX_WEIGHT)*loss_consts

        # VAE total loss (loss_ae = -ELBO = -log p(x|z) + KL_WEIGHT*KL(q(z|x)||p(z)) where KL_WEIGHT is usually denoted as beta)
        loss_vae = loss_recon_ae + KL_WEIGHT*alpha*kl

        # Value prediction loss
        loss_values = mse(values_pred, y_val)/VALUES_PRIOR

        # Contrastive loss (batch-wise)
        y_val = y_val / y_val.norm(dim=1, keepdim=True)
        values_dist = l2_dist(y_val.unsqueeze(1), y_val.unsqueeze(0))  # Make this indep of z_size, add some scaling factor
        gamma = gamma_func(values_dist.square())  # Scaling factor: Only if graphs are similar should large z_dissim be penalized

        z_sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # -> 1 if similar, 0 if unrelated, -> -1 if different
        z_dissim_loss = -torch.log((1+z_sim)/2)  # -> 1 if different, 0 if similar, -> 2 if unrelated
        loss_contrastive = torch.sum(gamma * z_dissim_loss)

        # Total loss
        loss = AE_WEIGHT*loss_vae + (1-AE_WEIGHT)*loss_values + CONTRASTIVE_WEIGHT*loss_contrastive

        partial_losses = {
            'loss_syntax': loss_syntax.item(),
            'loss_consts': loss_consts.item(),
            'loss_recon_ae': loss_recon_ae.item(),
            'kl': kl.item(),
            'alpha': alpha,
            'loss_vae': loss_vae.item(),   # -ELBO but with KL_WEIGHT*alpha so really only some distant cousing of ELBO
            'loss_values': loss_values.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss': loss.item()
        }

        return loss, partial_losses
    return criterion

def compute_latent_metrics(mean: torch.Tensor, ln_var: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for the latent space.
    """
    mean_norm = torch.norm(mean, dim=1).mean().item()
    std_mean = ln_var.exp().sqrt().mean().item()

    metrics = {
        'mean_norm': mean_norm,
        'std_mean': std_mean
    }

    return metrics

def calc_syntax_accuracy(logits: torch.Tensor, y_rule_idx: torch.Tensor) -> float:
    y_hat = logits.argmax(-1)
    a = (y_hat == y_rule_idx).float().mean()
    return 100 * a.item()
