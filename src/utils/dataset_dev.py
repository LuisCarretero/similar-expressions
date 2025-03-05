from typing import Tuple, Dict
import wandb
import yaml
from omegaconf import OmegaConf
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader
import torch

from src.model.model import LitGVAE
from src.utils.dataset import create_dataloader, summarize_dataloaders
from src.utils.config import dict_lit2num


def get_empty_priors():
    return {
        'syntax_prior': 1,
        'consts_prior': 1,
        'values_prior': 1
    }

def load_wandb_model(
    run: str, 
    name: str = 'model.pth', 
    device: str = 'cpu', 
    wandb_cache_path: str = '/cephfs/store/gr-mc2473/lc865/wandb_cache', 
    project: str = 'similar-expressions-01', 
    replace: bool = True, 
    fallback_cfg_path: str = '/src/train/config.yaml'
) -> Tuple[LitGVAE, DictConfig]:
    
    with wandb.restore(name, run_path=f"luis-carretero-eth-zurich/{project}/runs/{run}", root=wandb_cache_path, replace=replace) as io:
        model_fname = io.name

    # Read the model parameters from the WandB config.yaml file
    with wandb.restore('config.yaml', run_path=f"luis-carretero-eth-zurich/{project}/runs/{run}", root=wandb_cache_path, replace=True) as config_file:
        cfg_dict = yaml.safe_load(config_file)

    cfg = OmegaConf.create({k: dict_lit2num(v['value']) for k, v in list(cfg_dict.items()) if k not in ['wandb_version', '_wandb']})
    fallback_cfg = OmegaConf.load(fallback_cfg_path)
    cfg = OmegaConf.merge(fallback_cfg, cfg)
    
    # Load the Lightning checkpoint
    if True:  # Quickfix while transitioning to new model architecture
        checkpoint = torch.load(model_fname, map_location=device)
        # Rename keys to match new model structure
        new_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if 'encoder.mlp' in k or 'decoder.lin' in k:
                parts = k.split('.')
                new_key = f"{parts[0]}.mlp.net.{'.'.join(parts[2:])}"
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        checkpoint['state_dict'] = new_state_dict
        print(checkpoint['state_dict'].keys())
        
        vae_model = LitGVAE(cfg=cfg, priors=get_empty_priors()).to(device)
        vae_model.load_state_dict(checkpoint['state_dict'])
    else:
        LitGVAE.load_from_checkpoint(model_fname, cfg=cfg, priors=get_empty_priors(), map_location=device)
    vae_model.eval()

    summary = ModelSummary(vae_model, max_depth=1)
    print(f'Imported model from run "{run}".')
    print(summary)

    return vae_model, cfg

def dataloader_from_wandb_cfg(
    cfg: DictConfig, 
    value_transform=None, 
    datapath: str = '/cephfs/store/gr-mc2473/lc865/workspace/data', 
    allow_different_dataset_hash: bool = False,
    max_length: int = None
) -> Tuple[DataLoader, DataLoader, Dict]:
    
    if max_length is not None:
        assert allow_different_dataset_hash, \
            "Error: Cannot limit max length of dataset if allow_different_dataset_hash is False."
        cfg.training.dataset_len_limit = max_length
    
    train_loader, valid_loader, info = create_dataloader(
        datapath, 
        name=cfg.training.dataset_name, 
        cfg=cfg, 
        value_transform=value_transform, 
        shuffle_train=False
    )

    if not allow_different_dataset_hash:
        assert all([cfg.dataset_hashes[key] == info['hashes'][key] for key in cfg.dataset_hashes.keys()]), \
            "Error: Using different dataset than used for training."
    elif not all([cfg.dataset_hashes[key] == info['hashes'][key] for key in cfg.dataset_hashes.keys()]):
        print("Warning: Using different dataset than used for training.")

    print(f'Using dataset "{cfg.training.dataset_name}" of size {len(train_loader.dataset)}')
    summarize_dataloaders(train_loader, valid_loader)

    return train_loader, valid_loader, info

def data_from_loader(
    data_loader: DataLoader, 
    data_type: str, 
    idx=None, 
    max_length=None, 
    batch_size=None
) -> torch.Tensor:
    """
    Used for debugging and latent space analysis.
    """
    data_idx = {'x': 0, 'syntax': 1, 'consts': 2, 'values': 3}[data_type]
    subset_idx = data_loader.dataset.indices

    # Param cleansing
    if isinstance(idx, range):
        idx = slice(idx.start, idx.stop, idx.step)

    if idx is not None:  # rows indexed by idx
        overall_idx = subset_idx[idx]
        res = data_loader.dataset.dataset[overall_idx][data_idx]
    elif batch_size is not None:  # Generator
        if max_length is not None:
            subset_idx = subset_idx[:max_length]
        res = (data_loader.dataset.dataset[subset_idx[i:i+batch_size]][data_idx] for i in range(0, len(subset_idx), batch_size))
    elif max_length is not None:  # rows until max_length
        subset_idx = subset_idx[:max_length]
        res = data_loader.dataset.dataset[subset_idx][data_idx]
    else:
        raise ValueError("No data selected")

    # Add batch dimension if not present
    if isinstance(idx, int) or (batch_size is None and (max_length == 1)):
        res = res.unsqueeze(0)

    return res
