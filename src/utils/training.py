import math
import torch
from torch.nn import functional as F
import os
from typing import Dict, List, Tuple, Callable
from omegaconf.dictconfig import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger


class AnnealKLSigmoid:
    """Anneal the KL for VAE based training using a sigmoid schedule. No overall weighting so this return float between 0 and 1."""
    def __init__(self, cfg: DictConfig):
        self.total_epochs = cfg.training.epochs
        self.midpoint = cfg.training.kl_anneal.midpoint
        self.steepness = cfg.training.kl_anneal.steepness

    def alpha(self, epoch: int) -> float:
        """
        Calculate the annealing factor using a sigmoid function.
        """
        x = (epoch / self.total_epochs - self.midpoint) * self.steepness
        return 1 / (1 + math.exp(-x))

def criterion_factory(cfg: DictConfig, priors: Dict) -> Callable:
    """
    Factory function to create the criterion for the VAE.
    """
    AE_WEIGHT = cfg.training.loss.ae_weight
    KL_WEIGHT = cfg.training.loss.kl_weight
    SYNTAX_WEIGHT = cfg.training.loss.syntax_weight
    ctx = cfg.training.loss.contrastive
    CONTRASTIVE_WEIGHT, CONTRASTIVE_SCALE, CONTRASTIVE_MODE, CONTRASTIVE_DIMS, SIMILARITY_THRESHOLD = \
        ctx.weight, ctx.scale, ctx.mode, ctx.dimensions, ctx.similarity_threshold

    assert 0 <= AE_WEIGHT <= 1, "AE_WEIGHT must be between 0 and 1"
    assert 0 <= SYNTAX_WEIGHT <= 1, "SYNTAX_WEIGHT must be between 0 and 1"
    assert KL_WEIGHT >= 0, "KL_WEIGHT must be nonnegative"
    assert CONTRASTIVE_WEIGHT >= 0, "CONTRASTIVE_WEIGHT must be nonnegative"
    assert CONTRASTIVE_SCALE >= 0, "CONTRASTIVE_SCALE must be nonnegative"
    assert SIMILARITY_THRESHOLD >= 0, "SIMILARITY_THRESHOLD must be nonnegative"

    assert cfg.model.io_format.val_points % (CONTRASTIVE_DIMS[1] - CONTRASTIVE_DIMS[0]) == 0, "Value count must be divisible by contrastive dimensions."

    assert CONTRASTIVE_MODE in [None, 'total', 'piecewise'], "CONTRASTIVE_MODE must be one of [None, 'total', 'piecewise']"
    assert CONTRASTIVE_DIMS is None or len(CONTRASTIVE_DIMS) == 2, "CONTRASTIVE_DIMENSIONS must be a list of two integers"

    SYNTAX_PRIOR, CONSTS_PRIOR, VALUES_PRIOR = priors['syntax_prior'], priors['consts_prior'], priors['values_prior']

    cross_entropy = torch.nn.CrossEntropyLoss()  # default reduction is mean over batch and time steps
    mse = torch.nn.MSELoss()
    z_slice, input_size = calc_zslice(cfg.model.value_decoder.z_slice, cfg.model.z_size)

    def criterion(
        expr_pred: torch.Tensor, 
        values_pred: torch.Tensor, 
        y_rule_idx: torch.Tensor, 
        y_consts: torch.Tensor, 
        y_val: torch.Tensor, 
        kl: float, 
        alpha: float, 
        z: torch.Tensor
    ):
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
        loss_reconstruction = SYNTAX_WEIGHT*loss_syntax + (1-SYNTAX_WEIGHT)*loss_consts

        # VAE total loss (loss_ae = -ELBO = -log p(x|z) + KL_WEIGHT*KL(q(z|x)||p(z)) where KL_WEIGHT is usually denoted as beta)
        kl = torch.tensor(0.0, device=z.device) if KL_WEIGHT == 0 else kl
        loss_vae = loss_reconstruction + KL_WEIGHT*alpha*kl

        # Value prediction loss
        loss_values = mse(values_pred, y_val)/VALUES_PRIOR

        # Contrastive loss
        if CONTRASTIVE_MODE is None or CONTRASTIVE_WEIGHT == 0:
            loss_contrastive, cl_stats = torch.tensor(0.0, device=z.device), {}
        elif CONTRASTIVE_MODE == 'total':
            loss_contrastive, cl_stats = contrastive_loss_total(z[:, z_slice[0]:z_slice[1]], y_val, SIMILARITY_THRESHOLD, CONTRASTIVE_SCALE)
        elif CONTRASTIVE_MODE == 'piecewise':
            loss_contrastive, cl_stats = contrastive_loss_piecewise(z[:, CONTRASTIVE_DIMS[0]:CONTRASTIVE_DIMS[1]], y_val, 
                                                                    SIMILARITY_THRESHOLD, CONTRASTIVE_SCALE, CONTRASTIVE_DIMS)

        # Total loss
        loss = AE_WEIGHT*loss_vae + (1-AE_WEIGHT)*loss_values + CONTRASTIVE_WEIGHT*loss_contrastive

        stats = {
            'loss_syntax': loss_syntax.item(),
            'loss_consts': loss_consts.item(),
            'loss_recon_ae': loss_reconstruction.item(),
            'kl': kl.item(),
            'alpha': alpha,
            'loss_vae': loss_vae.item(),   # -ELBO but with KL_WEIGHT*alpha so really only some distant cousing of ELBO
            'loss_values': loss_values.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss': loss.item(),
            **cl_stats
        }

        return loss, stats
    return criterion

def contrastive_loss_total(u: torch.Tensor, y_val: torch.Tensor, similarity_threshold: float, contrastive_scale: float) -> torch.Tensor:
    values_dist = torch.mean((y_val.unsqueeze(1) - y_val.unsqueeze(0))**2, dim=2)
    similarity_mask = values_dist < similarity_threshold

    u_dist_L2 = F.pairwise_distance(u.unsqueeze(1), u.unsqueeze(0), p=2)  # L2 distance
    u_dist_squared = u_dist_L2**2
    loss_contrastive = torch.mean(similarity_mask * u_dist_squared + (~similarity_mask) * (torch.clamp(contrastive_scale - u_dist_L2, min=0))**2)/torch.mean(u_dist_squared)

    stats = calc_contrastive_stats(similarity_mask, u_dist_L2)
    return loss_contrastive, stats
    
def contrastive_loss_piecewise(
    u: torch.Tensor, 
    y_val: torch.Tensor, 
    similarity_threshold: float, 
    contrastive_scale: float, 
    dimensions: List[int]
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Split y_val into ndim intervals and calculate similarity within each interval. 
    Then apply contrastive loss to each interval and its corresponding latent space slice.
    """
    ndim = dimensions[1] - dimensions[0]
    y_val = y_val.reshape(y_val.shape[0], ndim, -1)  # [batch_size, intervals, interval_size]
    values_dist = torch.mean((y_val.unsqueeze(1) - y_val.unsqueeze(0))**2, dim=-1)  # Piecewise MSE between expressions
    similarity_mask = values_dist < similarity_threshold
    
    # Each distance only along one dimension so L2 reduces to scalar difference
    u_dist = torch.abs(u.unsqueeze(1) - u.unsqueeze(0))  # [batch_size, batch_size, udim]
    u_dist_L2, u_dist_squared = torch.norm(u_dist, p=2, dim=-1), u_dist**2

    loss_contrastive_intervals = torch.mean(
        similarity_mask * u_dist_squared + (~similarity_mask) * (torch.clamp(contrastive_scale - u_dist, min=0))**2, 
        dim=(0, 1)
    ) / torch.mean(u_dist_squared)
    loss_contrastive = torch.mean(loss_contrastive_intervals)  # Mean over intervals
    
    stats = calc_contrastive_stats(similarity_mask, u_dist_L2, u_dist, loss_contrastive_intervals)
    
    return loss_contrastive, stats

def calc_contrastive_stats(
    similarity_mask: torch.Tensor, 
    u_dist_L2: torch.Tensor, 
    u_dist: torch.Tensor = None, 
    loss_contrastive_intervals: torch.Tensor = None,
    interval_stats_cnt: int = 4
) -> Dict[str, float]:
    stats = {}

    if u_dist is not None:  # Pairwise distance for each dimension seperately
        is_simnotsame = similarity_mask & \
            (True^torch.eye(similarity_mask.shape[0], device=similarity_mask.device, dtype=torch.bool)).unsqueeze(-1)
        u_dist_simnotsame = u_dist[is_simnotsame][::int(is_simnotsame.numel()/1e5)]  # Only using about 1e5 elements to reduce resources

        # Stats per dimension. 
        for i in torch.linspace(0, u_dist.shape[2]-1, interval_stats_cnt, dtype=int):
            u_dist_sim = (u_dist[:, :, i][is_simnotsame[:, :, i]]).mean()
            u_dist_dissim = (u_dist[:, :, i][~is_simnotsame[:, :, i]]).mean()
            u_dist_ratio = u_dist_dissim / u_dist_sim
            stats.update({f'u_dist_sim_{i}': u_dist_sim.item(), 
                          f'u_dist_dissim_{i}': u_dist_dissim.item(), 
                          f'u_dist_ratio_{i}': u_dist_ratio.item()})
    else:
        is_simnotsame = similarity_mask & \
            (True^torch.eye(similarity_mask.shape[0], device=similarity_mask.device, dtype=torch.bool))
        u_dist_simnotsame = u_dist_L2[is_simnotsame][::int(is_simnotsame.numel()/1e5)]  # Only using about 1e5 elements to reduce resources

    if loss_contrastive_intervals is not None:
        for i in torch.linspace(0, loss_contrastive_intervals.shape[0]-1, interval_stats_cnt, dtype=int):
            stats.update({f'loss_contrastive_intervals_{i}': loss_contrastive_intervals[i].item()})

    mean_dist_sim = torch.mean(u_dist_simnotsame)
    if u_dist_simnotsame.numel() > 0:
        mean_dist_top_quartile_sim = torch.quantile(u_dist_simnotsame, 0.75)
        mean_dist_bottom_quartile_sim = torch.quantile(u_dist_simnotsame, 0.25)
    else:
        mean_dist_top_quartile_sim = torch.tensor(float('nan'))
        mean_dist_bottom_quartile_sim = torch.tensor(float('nan'))
    mean_dist_dissim = torch.mean(u_dist[~similarity_mask])

    stats.update({
        'L2_dist_sim': mean_dist_sim.item(),
        'L2_dist_dissim': mean_dist_dissim.item(),
        'L2_dist_ratio': (mean_dist_dissim / mean_dist_sim).item(),
        'L2_dist_top_quartile_sim': mean_dist_top_quartile_sim.item(),
        'L2_dist_bottom_quartile_sim': mean_dist_bottom_quartile_sim.item(),
        'sim_ratio': (is_simnotsame.sum()/is_simnotsame.numel()).item(),
    })
    return stats

def compute_latent_metrics(mean: torch.Tensor, ln_var: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics for the latent space.
    """
    return {
        'mean_norm': torch.norm(mean, dim=1).mean().item(),
        'std_mean': ln_var.exp().sqrt().mean().item()
    }

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
            for file in os.listdir(trainer.logger.experiment.dir):
                if file.endswith('.ckpt'):
                    trainer.logger.experiment.save(os.path.join(trainer.logger.experiment.dir, file),
                                                    base_path=trainer.logger.experiment.dir)

def set_wandb_cache_dir(dir: str) -> None:
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
        save_top_k=1,
        save_last=True,
        save_weights_only=True
    )
    callbacks.append(checkpoint_callback)

    return callbacks