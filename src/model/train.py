#!/usr/bin/env python3

from model import LitGVAE
from config_util import load_config
import lightning as L
from data_util import create_dataloader, calc_priors_and_means, summarize_dataloaders
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
import wandb
import os
import torch

seed_everything(42, workers=True, verbose=False)


class MiscCallback(Callback):
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

    def on_train_end(self, trainer, pl_module):
        if isinstance(trainer.logger, WandbLogger) and trainer.is_global_zero:
            # print(f'Files in wandb dir: {os.listdir(trainer.logger.experiment.dir)}')
            # FIXME: Quickfix to make sure last checkpoint is saved.
            trainer.logger.experiment.save(os.path.join(trainer.logger.experiment.dir, 'last.ckpt'),
                                           base_path=trainer.logger.experiment.dir)


def train_model(cfg_dict, cfg, data_path, dataset_name):
    # Determine the number of workers for data loading
    if 'SLURM_JOB_ID' in os.environ:
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
    logger = WandbLogger(project='similar-expressions-01')  # , log_model=True, Disable automatic syncing
    cfg_dict['dataset_hashes'] = data_info['hashes']
    cfg_dict['dataset_name'] = data_info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}', 
        monitor=cfg.training.performance_metric, 
        mode='min', 
        save_top_k=0, # If this is used, need to specify correct dirpath
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.performance_metric, 
        min_delta=0.001, 
        patience=5, 
        verbose=False, 
        mode="min"
    )

    if 'SLURM_JOB_ID' in os.environ:  # Running distributed via SLURM
        strategy = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
    else:
        strategy = "auto"
    print(f"Using strategy: {strategy}")

    # Get device count from SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        devices = int(os.environ['SLURM_NNODES']) * int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
        devices = 1  # Default to 1 if not running on SLURM or GPU count not specified
    print(f"Using {devices} device(s)")

    # Setup trainer and train model
    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=[checkpoint_callback, early_stopping_callback, MiscCallback()],
        log_every_n_steps=100,
        devices=devices,
        strategy=strategy
    )
    trainer.fit(gvae, train_loader, valid_loader)

    if trainer.is_global_zero:
        wandb.finish()


if __name__ == '__main__':
    data_path = ['/store/DAMTP/lc865/workspace/data', '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'][0]

    cfg_dict, cfg = load_config('src/model/config.json')
    train_model(cfg_dict, cfg, data_path, dataset_name='dataset_241008_1')  # dataset_240910_1, dataset_240822_1, dataset_240817_2




