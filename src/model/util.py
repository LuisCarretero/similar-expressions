import math
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
import os
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def load_config(file_path: str, use_fallback: bool = False, fallback_cfg_path: str = 'config.yaml') -> DictConfig:
    # TODO: Add error handling, fallback values, etc.
    cfg = OmegaConf.load(file_path)

    if use_fallback:
        fallback_cfg = OmegaConf.load(fallback_cfg_path)
        cfg = OmegaConf.merge(fallback_cfg, cfg)  # second arg overrides first

    return cfg

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
    def __init__(self, cfg: DictConfig):
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

def criterion_factory(cfg: DictConfig, priors: Dict):
    """
    Factory function to create the criterion for the VAE.
    """
    AE_WEIGHT = cfg.training.criterion.ae_weight
    CONTRASTIVE_WEIGHT = cfg.training.criterion.contrastive_weight
    KL_WEIGHT = cfg.training.criterion.kl_weight
    SYNTAX_WEIGHT = cfg.training.criterion.syntax_weight
    CONTRASTIVE_SCALE = cfg.training.criterion.contrastive_scale
    SIMILARITY_THRESHOLD = cfg.training.criterion.similarity_threshold

    assert 0 <= AE_WEIGHT <= 1, "AE_WEIGHT must be between 0 and 1"
    assert 0 <= SYNTAX_WEIGHT <= 1, "SYNTAX_WEIGHT must be between 0 and 1"
    assert KL_WEIGHT >= 0, "KL_WEIGHT must be nonnegative"
    assert CONTRASTIVE_WEIGHT >= 0, "CONTRASTIVE_WEIGHT must be nonnegative"
    assert CONTRASTIVE_SCALE >= 0, "CONTRASTIVE_SCALE must be nonnegative"
    assert SIMILARITY_THRESHOLD >= 0, "SIMILARITY_THRESHOLD must be nonnegative"

    SYNTAX_PRIOR = priors['syntax_prior']
    CONSTS_PRIOR = priors['consts_prior']
    VALUES_PRIOR = priors['values_prior']

    cross_entropy = torch.nn.CrossEntropyLoss()  # default reduction is mean over batch and time steps
    mse = torch.nn.MSELoss()

    # Contrastive loss setup
    z_slice = cfg.model.value_decoder.z_slice.copy()
    if z_slice[1] == -1:
        z_slice[1] = cfg.model.z_size
    input_size = z_slice[1] - z_slice[0]

    l2_dist_fn = torch.nn.PairwiseDistance(p=2)
    # m = 1e-1
    # a = 40  # Sharpness
    # b = 5  # Shift
    # gamma_func = lambda x: 1/(1+ torch.exp(a * x - b))

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
        kl = torch.tensor(0.0, device=z.device) if KL_WEIGHT == 0 else kl
        loss_vae = loss_recon_ae + KL_WEIGHT*alpha*kl

        # Value prediction loss
        loss_values = mse(values_pred, y_val)/VALUES_PRIOR

        if CONTRASTIVE_WEIGHT > 0:  # Expensive, so only calculate if necessary

            values_dist = torch.mean((y_val.unsqueeze(1) - y_val.unsqueeze(0))**2, dim=2)
            similarity_mask = values_dist < SIMILARITY_THRESHOLD
            
            u = z[:, z_slice[0]:z_slice[1]]
            u_dist = l2_dist_fn(u.unsqueeze(1), u.unsqueeze(0))
            loss_contrastive = torch.sum(similarity_mask * u_dist**2 + (~similarity_mask) * torch.maximum(torch.tensor(0.0, device=z.device), CONTRASTIVE_SCALE - u_dist)**2)
              # FIXME: Should this be sum or mean? Might get averages later? <- Normalisation should make this invariant under batch size, u-dim and u scale
              # /torch.sum(u**2)**2

            # Stats for debug
            # sim_ratio = similarity_mask.float().mean()
            # sim_count = similarity_mask.sum() - similarity_mask.shape[0]
            sim_notsame = similarity_mask & (True^torch.eye(similarity_mask.shape[0], device=similarity_mask.device, dtype=torch.bool))
            u_dist_simnotsame = u_dist[sim_notsame]
            mean_dist_sim = torch.mean(u_dist_simnotsame)
            if u_dist_simnotsame.numel() > 0:
                mean_dist_top_quartile_sim = torch.quantile(u_dist_simnotsame, 0.75)
                mean_dist_bottom_quartile_sim = torch.quantile(u_dist_simnotsame, 0.25)
            else:
                mean_dist_top_quartile_sim = torch.tensor(float('nan'), device=z.device)
                mean_dist_bottom_quartile_sim = torch.tensor(float('nan'), device=z.device)
            mean_dist_dissim = torch.mean(u_dist[~similarity_mask])

        else:
            loss_contrastive = torch.tensor(0.0, device=z.device)

        # Total loss
        # loss_vae = torch.tensor(0.0, device=z.device) if AE_WEIGHT == 0 else loss_vae  # FIXME: This is a hack to avoid NaNs in the loss if AE_WEIGHT is 0 (and hence the loss may grow without bound)
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
            'loss': loss.item(),
            # 'sim_ratio': sim_ratio.item(),
            # 'sim_count': sim_count.item(),
            'mean_dist_sim': mean_dist_sim.item(),
            'mean_dist_dissim': mean_dist_dissim.item(),
            'mean_dist_top_quartile_sim': mean_dist_top_quartile_sim.item(),
            'mean_dist_bottom_quartile_sim': mean_dist_bottom_quartile_sim.item(),
            'dist_ratio': mean_dist_dissim.item() / mean_dist_sim.item()
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


def calc_zslice(z_slice: List[int], z_size: int) -> Tuple[List[int], int]:
    assert len(z_slice) == 2, f"z_slice has to be a list of two integers ({z_slice = })"
    assert z_slice[0] >= 0, f"z_slice has to be a valid slice of z ({z_slice[0] = })"

    z_slice = z_slice.copy()
    if z_slice[1] == -1:
        z_slice[1] = z_size
    input_size = z_slice[1] - z_slice[0]

    assert z_slice[0] < z_slice[1], f'z_slice has to be a valid slice of z ({z_slice[0] = }, {z_slice[1] = })'
    assert z_slice[1] <= z_size, f"z_slice has to be subset of z: z_slice[1]: {z_slice[1]}, z_size: {z_size}"

    return z_slice, input_size

class MiscCallback(Callback):
    """
    Custom callback to access the WandB run data. This cannot be accessed during setup as Logger is initialised only when trainer.fit() is called.

    From Docs:
    trainer.logger.experiment: Actual wandb object. To use wandb features in your :class:`~lightning.pytorch.core.LightningModule` do the
    following. self.logger.experiment.some_wandb_function()

    # Only available in rank0 process, others have _DummyExperiment
    """
    def on_train_start(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger) and trainer.is_global_zero:
            # Dynamically set the checkpoint directory in ModelCheckpoint
            print(f"Checkpoints will be saved in: {trainer.logger.experiment.dir}")
            trainer.checkpoint_callback.dirpath = trainer.logger.experiment.dir

        print(f'Node rank: {trainer.node_rank}, Global rank: {trainer.global_rank}, Local rank: {trainer.local_rank}')
        print(f'Trainer strategy: {trainer.strategy}')

    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger) and trainer.is_global_zero:
            # print(f'Files in wandb dir: {os.listdir(trainer.logger.experiment.dir)}')
            # FIXME: Quickfix to make sure last checkpoint is saved.
            trainer.logger.experiment.save(os.path.join(trainer.logger.experiment.dir, 'last.ckpt'),
                                           base_path=trainer.logger.experiment.dir)

def set_wandb_cache_dir(dir: str):
    """
    Not sure which ones are needed but better safe than sorry.
    """
    os.environ['WANDB_CACHE_DIR'] = dir
    os.environ['WANDB_DATA_DIR'] = dir
    os.environ['WANDB_CONFIG_DIR'] = dir
    os.environ['WANDB_ARTIFACT_DIR'] = dir


def create_callbacks(cfg: DictConfig) -> List[Callback]:
    """
    Create callbacks for the trainer.
    """
    callbacks = [MiscCallback()]

    if cfg.training.early_stopping.enabled:
        callbacks.append(EarlyStopping(
            monitor=cfg.training.performance_metric, 
            min_delta=cfg.training.early_stopping.min_delta, 
            patience=cfg.training.early_stopping.patience, 
            verbose=False, 
            mode="min"
        ))

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}', 
        monitor=cfg.training.performance_metric, 
        mode='min', 
        save_top_k=1, # If this is used, need to specify correct dirpath
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    return callbacks