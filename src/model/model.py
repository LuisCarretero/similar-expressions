import torch
from torch.distributions import Normal
from encoder import Encoder
from decoder import Decoder
from value_decoder import ValueDecoder
from parsing import logits_to_prods
from grammar import GCFG, calc_grammar_mask
from config_util import Config
import math
import lightning as L
from util import criterion_factory, AnnealKLSigmoid, compute_latent_metrics, calc_syntax_accuracy
from typing import Dict, List


class LitGVAE(L.LightningModule):
    """Grammar Variational Autoencoder"""
    def __init__(self, cfg: Config, priors: Dict[str, float]):
        super().__init__()

        self.encoder = Encoder(cfg.model)
        self.decoder = Decoder(cfg.model)
        self.value_decoder = ValueDecoder(cfg.model)

        self.cfg = cfg
        self.prior_std = cfg.training.sampling.prior_std
        self.prior_var = self.prior_std**2
        self.ln_prior_var = math.log(self.prior_var)
        self.sampling_eps = cfg.training.sampling.eps

        self.use_grammar_mask = cfg.training.use_grammar_mask
        self.max_length = cfg.model.io_format.seq_len
        
        self.priors = priors
        self.criterion = criterion_factory(cfg, self.priors)
        self.kl_anneal = AnnealKLSigmoid(cfg)

        self.train_step_metrics_buffer = []
        self.valid_step_metrics_buffer = []
    
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
    
    def training_step(self, batch, batch_idx):
        x, y_syntax, y_consts, y_values = batch

        # Forward pass
        mean, ln_var = self.encoder(x)
        z = self.sample(mean, ln_var)
        logits = self.decoder(z, max_length=self.max_length)
        if self.use_grammar_mask:
            logits = logits * calc_grammar_mask(y_syntax)
        values = self.value_decoder(z)
        
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
        mean, ln_var = self.encoder(x)
        z = self.sample(mean, ln_var)
        logits = self.decoder(z, max_length=self.max_length)
        if self.use_grammar_mask:
            logits = logits * calc_grammar_mask(y_syntax)
        values = self.value_decoder(z)

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
        optimizer = torch.optim.Adam(
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
                "monitor": "valid/loss",
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
        # Average the metrics over the last N steps
        if len(buffer) == 0:
            return

        # avg_metrics = {k: sum(d[k] for d in buffer) / len(buffer) for k in buffer[0]}
        avg_metrics = {k: sum(step_dict[k] for step_dict in buffer) / len(buffer) for k in buffer[0]}

        # Log the averaged metrics
        self.log_dict(avg_metrics)
        
        # Clear the buffer
        buffer.clear()
