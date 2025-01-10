import torch
import math
import lightning as L
from typing import Dict, List
from omegaconf.dictconfig import DictConfig
from torch.distributions import Normal

from src.model.encoder import Encoder
from src.model.decoder import Decoder
from src.model.value_decoder import ValueDecoder
from src.model.grammar import calc_grammar_mask, GRAMMAR_STR, create_masks_and_allowed_prod_idx
from src.model.util import criterion_factory, AnnealKLSigmoid, compute_latent_metrics, calc_syntax_accuracy


class LitGVAE(L.LightningModule):
    def __init__(self, cfg: DictConfig, priors: Dict[str, float]):
        super().__init__()

        self.cfg = cfg
        self.prior_std = cfg.training.sampling.prior_std
        self.prior_var = self.prior_std**2
        self.ln_prior_var = math.log(self.prior_var)
        self.sampling_eps = cfg.training.sampling.eps
        self.mode = cfg.training.mode  # value_prediction, autoencoding, mixed, encoding

        self.encoder = Encoder(cfg.model)
        self.decoder = Decoder(cfg.model) if self.mode in ['autoencoding', 'mixed'] else None
        self.value_decoder = ValueDecoder(cfg.model) if self.mode in ['value_prediction', 'mixed'] else None

        self.use_grammar_mask = cfg.training.use_grammar_mask
        
        self.priors = priors
        self.criterion = criterion_factory(cfg, self.priors)
        self.kl_anneal = AnnealKLSigmoid(cfg)

        self.train_step_metrics_buffer = []
        self.valid_step_metrics_buffer = []
        
        masks, allowed_prod_idx = create_masks_and_allowed_prod_idx(GRAMMAR_STR)
        self.masks = masks.to(self.device)
        self.allowed_prod_idx = allowed_prod_idx.to(self.device)
    
    def sample(self, mean, ln_var):
        """Reparametrized sample from a N(mu, sigma) distribution"""
        normal = Normal(torch.zeros(mean.shape).to(self.device), torch.ones(ln_var.shape).to(self.device))
        eps = normal.sample() * self.sampling_eps  # Sample from N(0, self.sampling_eps^2), i.e. std is self.sampling_eps
        z = mean + eps * torch.exp(ln_var/2)  # Reparametrization trick. Effectively sample from N(mean, var*sampling_eps^2)
        return z

    def calc_kl(self, mean: torch.Tensor, ln_var: torch.Tensor) -> torch.Tensor:
        """KL divergence between N(mean, exp(ln_var)) and N(0, prior_std^2). Returns a positive definite scalar."""
        kl_per_sample = 0.5 * torch.sum(  # Sum over all dimensions
            -ln_var + self.ln_prior_var -1 + (mean**2 + ln_var.exp())/self.prior_var,
            dim=1
        )
        return torch.mean(kl_per_sample)  # Average over samples

    def forward(self, x):
        mean, ln_var = self.encoder(x)
        z = self.sample(mean, ln_var)

        if self.mode == 'mixed':
            logits = self.decoder(z)
            values = self.value_decoder(z)
        elif self.mode == 'value_prediction':
            logits = torch.zeros(x.shape[0], self.cfg.model.io_format.seq_len, self.cfg.model.io_format.token_cnt).to(z.device)
            values = self.value_decoder(z)
        elif self.mode == 'autoencoding':
            logits = self.decoder(z)
            values = torch.zeros(x.shape[0], self.cfg.model.io_format.val_points).to(z.device)
        elif self.mode == 'encoding':
            logits = torch.zeros(x.shape[0], self.cfg.model.io_format.seq_len, self.cfg.model.io_format.token_cnt).to(z.device)
            values = torch.zeros(x.shape[0], self.cfg.model.io_format.val_points).to(z.device)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        return mean, ln_var, z, logits, values
    
    # def __call__(self, x, sample_eps):
    #     """
    #     Used for final ONNX inference version.
    #     """
    #     mean, ln_var = self.encoder(x)
    #     self.sampling_eps = sample_eps
    #     z = self.sample(mean, torch.zeros_like(ln_var))
    #     logits = self.decoder(z)
    #     return logits
    
    def training_step(self, batch, batch_idx):
        x, y_syntax, y_consts, y_values = batch

        # Forward pass
        mean, ln_var, z, logits, values = self.forward(x)
        if self.use_grammar_mask:
            logits = logits * calc_grammar_mask(y_syntax, self.masks, self.allowed_prod_idx)

        # Compute losses
        kl = self.calc_kl(mean, ln_var)  # Positive definite scalar, aim to minimize
        alpha = self.kl_anneal.alpha(self.current_epoch)
        loss, partial_losses = self.criterion(logits, values, y_syntax, y_consts, y_values, kl, alpha, z)

        # Compute metrics   
        latent_metrics = compute_latent_metrics(mean, ln_var)
        syntax_accuracy = calc_syntax_accuracy(logits, y_syntax)

        # Merge and sum up metrics
        step_metrics = {f'train/{k}': v for k, v in {**partial_losses, **latent_metrics, 'syntax_accuracy': syntax_accuracy, 'lr': self.lr_schedulers().get_last_lr()[0]}.items()}

        return {'loss': loss, 'step_metrics': step_metrics}
    
    def validation_step(self, batch, batch_idx):
        x, y_syntax, y_consts, y_values = batch

        # Forward pass
        mean, ln_var, z, logits, values = self.forward(x)
        if self.use_grammar_mask:
            logits = logits * calc_grammar_mask(y_syntax, self.masks, self.allowed_prod_idx)

        # Compute losses
        kl = self.calc_kl(mean, ln_var)  # Positive definite scalar, aim to minimize
        alpha = self.kl_anneal.alpha(self.current_epoch)
        loss, partial_losses = self.criterion(logits, values, y_syntax, y_consts, y_values, kl, alpha, z)

        # Compute metrics   
        latent_metrics = compute_latent_metrics(mean, ln_var)
        syntax_accuracy = calc_syntax_accuracy(logits, y_syntax)

        # Merge and sum up metrics
        step_metrics = {f'valid/{k}': v for k, v in {**partial_losses, **latent_metrics, 'syntax_accuracy': syntax_accuracy, 'lr': self.lr_schedulers().get_last_lr()[0]}.items()}

        return {'loss': loss, 'step_metrics': step_metrics}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(  # AdamW uses weight_decay=0.01 by default
            self.parameters(), 
            lr=self.cfg.training.optimizer.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.cfg.training.optimizer.scheduler_factor,
            patience=self.cfg.training.optimizer.scheduler_patience,
            threshold=self.cfg.training.optimizer.scheduler_threshold,
            min_lr=self.cfg.training.optimizer.scheduler_min_lr
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.cfg.training.performance_metric,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if torch.isnan(outputs['loss']):
            print(f"NaN detected in training loss at batch {batch_idx}")
            raise ValueError("NaN detected in training loss")

        # Store and log metrics every N steps
        self.train_step_metrics_buffer.append(outputs['step_metrics'])
        
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self._log_from_buffer(self.train_step_metrics_buffer)

    def on_validation_batch_end(self, outputs, batch, batch_idx):
        # Store and log metrics every N steps
        self.valid_step_metrics_buffer.append(outputs['step_metrics'])
        
        if batch_idx % self.trainer.log_every_n_steps == 0:
            self._log_from_buffer(self.valid_step_metrics_buffer)

    def on_validation_epoch_end(self):
        self._log_from_buffer(self.valid_step_metrics_buffer)

    def on_train_epoch_end(self) -> None:
        self._log_from_buffer(self.train_step_metrics_buffer)
        
    def _log_from_buffer(self, buffer: List):
        if len(buffer) == 0:
            return

        # Average the metrics over the last N steps
        avg_metrics = {k: sum(step_dict[k] for step_dict in buffer) / len(buffer) for k in buffer[0]}

        # Log the averaged metrics and clear the buffer
        self.log_dict(avg_metrics, sync_dist=True)
        buffer.clear()
