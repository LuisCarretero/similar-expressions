#!/usr/bin/env python3

from model import LitGVAE
from config_util import load_config
import lightning as L
from data_util import create_dataloader, calc_priors_and_means, summarize_dataloaders
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import Callback
import torch.distributed as dist
import wandb
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

    # Determine the number of workers for data loading
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        n = int(os.environ['SLURM_CPUS_PER_TASK'])
    else:
        n = min(os.cpu_count(), 8)

    # Load data
    print(f"Using {n} workers for data loading.")
    train_loader, valid_loader, data_info = create_dataloader(data_path, dataset_name, cfg, num_workers=n)
    priors, _ = calc_priors_and_means(train_loader)  # TODO: Introduce bias initialization
    summarize_dataloaders(train_loader, valid_loader)

    # Setup model
    gvae = LitGVAE(cfg, priors)

    # Setup logger
    logger = WandbLogger(project='similar-expressions-01')  # Disable automatic syncing
    cfg_dict['dataset_hashes'] = data_info['hashes']
    cfg_dict['dataset_name'] = data_info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    checkpoint_callback = ModelCheckpoint(
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

    # Determine appropriate strategy
    if 'SLURM_JOB_ID' in os.environ:  # Running distributed via SLURM
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"
    print(f"Using strategy: {strategy}")

    # Setup trainer and train model
    trainer = L.Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=[checkpoint_callback, early_stopping_callback, SetupModelCheckpointCallback()],
        # profiler=AdvancedProfiler(dirpath='.', filename='profile.txt'),
        log_every_n_steps=100,
        strategy=strategy
    )
    trainer.fit(gvae, train_loader, valid_loader)

    # Finish logging
    wandb.finish()


if __name__ == '__main__':
    cfg_path = 'src/model/config.json'  # /home/lc865/workspace/similar-expressions/src/model
    data_path = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'  # /Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'  #  
    main(cfg_path, data_path, dataset_name='dataset_241008_1')  # dataset_240910_1, dataset_240822_1, dataset_240817_2




