#!/usr/bin/env python3

from model import LitGVAE
from config_util import load_config
import lightning as L
from data_util import create_dataloader, calc_priors_and_means, summarize_dataloaders
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import Callback
import pickle
import os

seed_everything(42, workers=True, verbose=False)


class SetupModelCheckpointCallback(Callback):
    """
    Custom callback to access the WandB run data. Cannot be called during setup as Logger is initialised only during trainer.fit().

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


def main(cfg_path, data_path, dataset_name):
    # Load config
    cfg_dict, cfg = load_config(cfg_path)

    # Load data
    train_loader, valid_loader, info = create_dataloader(data_path, dataset_name, cfg, num_workers=1)
    priors, means = calc_priors_and_means(train_loader)  # TODO: Introduce bias initialization
    summarize_dataloaders(train_loader, valid_loader)

    # Setup model
    gvae = LitGVAE(cfg, priors)

    # Setup logger
    logger = WandbLogger(project='similar-expressions-01')  # Disable automatic syncing
    cfg_dict['dataset_hashes'] = info['hashes']
    cfg_dict['dataset_name'] = info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    checkpoint_callback = ModelCheckpoint(
        # dirpath='/store/DAMTP/lc865/workspace/checkpoints/similar-expressions-01/', 
        filename='{epoch:02d}', 
        monitor='valid/loss', 
        mode='min', 
        save_top_k=0, 
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", 
        min_delta=0.00, 
        patience=4, 
        verbose=False, 
        mode="min"
    )

    # Train model
    trainer = L.Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=[checkpoint_callback, early_stopping_callback, SetupModelCheckpointCallback()],
        # profiler=AdvancedProfiler(dirpath='.', filename='profile.txt'),
        log_every_n_steps=100,
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    trainer.fit(gvae, train_loader, valid_loader)
    wandb.finish()


if __name__ == '__main__':
    cfg_path = '/home/lc865/workspace/similar-expressions/src/model/config.json'
    data_path = '/store/DAMTP/lc865/workspace/data'
    main(cfg_path, data_path, dataset_name='dataset_240910_2')  # dataset_240910_1, dataset_240822_1, dataset_240817_2




