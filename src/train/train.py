#!/usr/bin/env python3
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.strategies import DDPStrategy
from lightning import Trainer
import wandb
import os
import torch
from omegaconf import OmegaConf

from src.model.model import LitGVAE
from src.utils.dataset import create_dataloader, calc_priors_and_means, summarize_dataloaders
from src.utils.training import set_wandb_cache_dir, create_callbacks
from src.utils.config import load_config

seed_everything(42, workers=True, verbose=False)


def train_model(cfg, data_path, overwrite_device_count=None, overwrite_strategy=None):
    # Set wandb logging dir
    fpath = '/cephfs/store/gr-mc2473/lc865/workspace/wandb-cache'
    set_wandb_cache_dir(fpath)

    # Dataloader and device setup
    if 'SLURM_JOB_ID' in os.environ:  # Running distributed via SLURM
        num_workers = int(os.environ['SLURM_CPUS_PER_TASK'])
        devices = int(os.environ['SLURM_NNODES']) * int(os.environ['SLURM_NTASKS_PER_NODE'])
        if devices > 1:
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        else:
            strategy = "auto"  # SingleDeviceStrategy()
    else:
        num_workers = min(os.cpu_count(), 8)
        strategy = "auto"
        devices = 1  # Default to 1 if not running on SLURM or GPU count not specified
    if overwrite_device_count is not None:
        devices = overwrite_device_count
    if overwrite_strategy is not None:
        strategy = overwrite_strategy
    print(f"Using strategy: {strategy} and {devices} device(s)")

    # Load data
    print(f"Using {num_workers} workers for data loading.")
    train_loader, valid_loader, data_info = create_dataloader(data_path, cfg.training.dataset_name, cfg, num_workers=num_workers)
    priors, _ = calc_priors_and_means(train_loader)  # TODO: Introduce bias initialization
    summarize_dataloaders(train_loader, valid_loader)

    # Setup model
    gvae = LitGVAE(cfg, priors)

    # Setup logger
    logger = WandbLogger(project=cfg.training.wandb_settings.project, save_dir=fpath)
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['dataset_hashes'] = data_info['hashes']
    logger.log_hyperparams(cfg_dict)

    # Setup trainer and train model
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=create_callbacks(cfg),
        log_every_n_steps=100,
        devices=devices,
        strategy=strategy
    )
    trainer.fit(gvae, train_loader, valid_loader)

    if trainer.is_global_zero:
        wandb.finish()


if __name__ == '__main__':
    data_path = '/cephfs/store/gr-mc2473/lc865/workspace/data'
    cfg = load_config('src/train/config.yaml')
    train_model(cfg, data_path)
