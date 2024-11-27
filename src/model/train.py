#!/usr/bin/env python3

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning import Trainer
import wandb
import os
import torch
from omegaconf import OmegaConf

from src.model.model import LitGVAE
from src.model.util import load_config, MiscCallback, set_wandb_cache_dir
from src.model.data_util import create_dataloader, calc_priors_and_means, summarize_dataloaders

seed_everything(42, workers=True, verbose=False)

def train_model(cfg, data_path, dataset_name, project_name=None, overwrite_device_count=None, overwrite_strategy=None):
    # Set wandb logging dir
    fpath = '/store/DAMTP/lc865/workspace/wandb-cache'
    set_wandb_cache_dir(fpath)

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
    logger = WandbLogger(project=project_name, save_dir=fpath)
    cfg_dict = OmegaConf.to_container(cfg)
    cfg_dict['dataset_hashes'] = data_info['hashes']
    cfg_dict['dataset_name'] = data_info['dataset_name']
    logger.log_hyperparams(cfg_dict)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}', 
        monitor=cfg.training.performance_metric, 
        mode='min', 
        save_top_k=1, # If this is used, need to specify correct dirpath
        save_last=True
    )

    early_stopping_callback = EarlyStopping(
        monitor=cfg.training.performance_metric, 
        min_delta=0, 
        patience=15, 
        verbose=False, 
        mode="min"
    )

    if 'SLURM_JOB_ID' in os.environ:  # Running distributed via SLURM
        devices = int(os.environ['SLURM_NNODES']) * int(os.environ['SLURM_NTASKS_PER_NODE'])
        if devices > 1:
            strategy = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        else:
            strategy = "auto"  # SingleDeviceStrategy()
    else:
        strategy = "auto"
        devices = 1  # Default to 1 if not running on SLURM or GPU count not specified
    if overwrite_device_count is not None:
        devices = overwrite_device_count
    if overwrite_strategy is not None:
        strategy = overwrite_strategy
    print(f"Using strategy: {strategy} and {devices} device(s)")

    # Setup trainer and train model
    torch.set_float32_matmul_precision('medium')
    trainer = Trainer(
        logger=logger, 
        max_epochs=cfg.training.epochs, 
        gradient_clip_val=cfg.training.optimizer.clip,
        callbacks=[checkpoint_callback, MiscCallback(), early_stopping_callback],
        log_every_n_steps=100,
        devices=devices,
        strategy=strategy
    )
    trainer.fit(gvae, train_loader, valid_loader)

    if trainer.is_global_zero:
        wandb.finish()


if __name__ == '__main__':
    data_path = ['/store/DAMTP/lc865/workspace/data', '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data'][0]

    cfg = load_config('src/model/config.yaml')
    train_model(cfg, data_path, dataset_name='dataset_241127_2', project_name='simexp-03')  # dataset_241008_1, dataset_240910_1, dataset_240822_1, dataset_240817_2




