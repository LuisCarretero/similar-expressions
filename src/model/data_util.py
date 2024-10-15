import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import hashlib
from config_util import Config
from typing import Tuple
import wandb
import yaml
from config_util import dict_to_config
from model import LitGVAE
from matplotlib import pyplot as plt
from scipy.special import softmax
from lightning.pytorch.utilities.model_summary import ModelSummary

class CustomTorchDataset(Dataset):
    """
    
    Data dimensions:
    - data_syntax: (n_samples, seq_len, n_tokens+1)
    - data_values: (n_samples, n_values)

    """
    def __init__(self, data_syntax: np.ndarray, data_values: np.ndarray, value_transform=None):
        assert data_syntax.shape[0] == data_values.shape[0]

        # Calculate hashes  FIXME: Might want to only take hash of parts? Check performance impact.
        md5 = hashlib.md5()
        md5.update(data_syntax.tobytes())
        md5.update(data_values.tobytes())
        self.hash = md5.hexdigest()

        self.data_syntax = torch.tensor(data_syntax, dtype=torch.float32)
        self.values_transformed = value_transform(torch.tensor(data_values, dtype=torch.float32))
        self.x_shape = data_syntax.shape

    def __len__(self):
        return len(self.data_syntax)

    def __getitem__(self, idx):
        """
        Careful: idx can be int, list of int of slice. Make sure that beheviour is the same in all cases. When calling DataLoader, an idx is passed for each item in the batch.
        For custom access in analysis script (e.g. data_loader.dataset.dataset[data_loader.dataset.indices]), a list of indices is passed instead.

        Use -ve indexing to make it independent of shape (i.e. with or without batch_size dimension.)
        """
        x = self.data_syntax[idx]  # Shape: (1, seq_len, n_tokens+1)
        
        y_rule_idx = self.data_syntax[idx, :, :-1].argmax(axis=-1) # The rule index (argmax over onehot part, excluding consts) 
        y_consts = self.data_syntax[idx, :, -1]
        y_values = self.values_transformed[idx]
        return x, y_rule_idx, y_consts, y_values


def calc_priors_and_means(dataloader: torch.utils.data.DataLoader):
    # Extract data from DataLoader FIXME: Calculate on GPU?
    x = dataloader.dataset.dataset[dataloader.dataset.indices][0]
    syntax = x[:, :-1, :].detach().cpu().numpy().transpose(0, 2, 1)
    consts = x[:, -1, :].squeeze().detach().cpu().numpy()
    values = dataloader.dataset.dataset[dataloader.dataset.indices][3]  # Already transformed

    # Calculate priors and means
    prod_counts = np.bincount(syntax.argmax(axis=-1).flatten())
    p = prod_counts / np.sum(prod_counts)
    syntax_prior_xent = -np.sum(p * np.log(p), where=p!=0).astype(np.float32)

    consts_prior_mse = consts.var()
    values_prior_mse = values.var()

    priors = {
        'syntax_prior': syntax_prior_xent,
        'consts_prior': consts_prior_mse,
        'values_prior': values_prior_mse
    }

    consts_bias = consts.mean(axis=0)
    values_bias = values.mean(axis=0)

    means = {
        'consts_mean': consts_bias,
        'values_mean': values_bias
    }
    return priors, means

def get_empty_priors():
    return {
        'syntax_prior': 1,
        'consts_prior': 1,
        'values_prior': 1
    }

def load_dataset(datapath, name):
    with h5py.File(os.path.join(datapath, f'{name}.h5'), 'r') as f:
        # Extract onehot, values (eval_y), and consts
        syntax = f['onehot'][:].astype(np.float32).transpose([2, 1, 0])  # Shape: (n_samples, seq_len, n_tokens)
        consts = f['consts'][:].astype(np.float32).T  # Shape: (n_samples, seq_len)
        values = f['eval_y'][:].astype(np.float32).T  # Shape: (n_samples, n_values)
        val_x = f['eval_x'][:].astype(np.float32)  # Shape: (n_values, 1)
        syntax_cats = list(map(lambda x: x.decode('utf-8'), f['onehot_legend'][:]))

    return syntax, consts, values, val_x, syntax_cats

def create_dataloader(datapath: str, name: str, cfg: Config, random_seed=0, shuffle_train=True, value_transform=None, num_workers=4) -> Tuple[DataLoader, DataLoader, dict]:
    gen = torch.Generator()
    gen.manual_seed(random_seed)

    syntax, consts, values, val_x, syntax_cats = load_dataset(datapath, name)
    data_syntax = np.concatenate([syntax, consts[:, :, np.newaxis]], axis=-1)  # Shape: (n_samples, seq_len, n_tokens+1)

    if cfg.training.dataset_len_limit is not None:
        data_syntax = data_syntax[:cfg.training.dataset_len_limit]
        values = values[:cfg.training.dataset_len_limit]

    # Create value transform
    min_, max_ = np.arcsinh(values.min()), np.arcsinh(values.max())
    if value_transform is None:
        value_transform = lambda x: 2 * (torch.arcsinh(x) - min_) / (max_ - min_) - 1  # Center in range

    # Create the full dataset
    full_dataset = CustomTorchDataset(data_syntax, values, value_transform=value_transform)

    # Split the dataset
    valid_size = int(cfg.training.valid_split * len(full_dataset))
    train_size = len(full_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=gen)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    # Create hashes
    assert id(full_dataset) == id(train_loader.dataset.dataset) == id(valid_loader.dataset.dataset), "Datasets are not the same"
    hashes = {
        'dataset': full_dataset.hash,
        'train_idx': hashlib.md5(str(train_loader.dataset.indices).encode()).hexdigest(),
        'valid_idx': hashlib.md5(str(valid_loader.dataset.indices).encode()).hexdigest(),
        'random_seed': random_seed
    }
    info = {
        'hashes': hashes,
        'min_value': min_,
        'max_value': max_,
        'value_transform': value_transform,
        'dataset_name': name,
        'datapath': datapath, 
        'syntax_cats': syntax_cats,
        'val_x': val_x
    }
    return train_loader, valid_loader, info

def load_wandb_model(run: str, name:str = 'model.pth', device='cpu', wandb_cache_path='/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/wandb_cache'):
    # Load model
    with wandb.restore(name, run_path=f"luis-carretero-eth-zurich/similar-expressions-01/runs/{run}", root=wandb_cache_path, replace=True) as io:
        name = io.name

    # Read the model parameters from the WandB config.yaml file
    with wandb.restore('config.yaml', run_path=f"luis-carretero-eth-zurich/similar-expressions-01/runs/{run}", root=wandb_cache_path, replace=True) as config_file:
        cfg_dict = yaml.safe_load(config_file)
        cfg = {k: v['value'] for k, v in list(cfg_dict.items()) if k not in ['wandb_version', '_wandb']}
        cfg = dict_to_config(cfg)

    # vae_model = LitGVAE(cfg, get_empty_priors())
    
    # Load the Lightning checkpoint
    vae_model = LitGVAE.load_from_checkpoint(name, cfg=cfg, priors=get_empty_priors(), map_location=device)
    vae_model.eval()

    summary = ModelSummary(vae_model, max_depth=1)
    print(f'Imported model from run "{run}".')
    print(summary)

    return vae_model, cfg_dict, cfg

def create_dataloader_from_wandb(cfg_dict, cfg, value_transform=None, datapath='/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'):
    # FIXME: Was quick fix, can be removed?
    try:
        name = cfg_dict['dataset_name']['value']
    except KeyError:
        name = cfg_dict['dataset']['value']

    train_loader, valid_loader, info = create_dataloader(datapath, name=name, cfg=cfg, value_transform=value_transform, shuffle_train=False)
    assert all([cfg_dict['dataset_hashes']['value'][key] == info['hashes'][key] for key in cfg_dict['dataset_hashes']['value']]), "Error: Using different dataset than used for training."

    print(f'Using dataset "{name}" of size {len(train_loader.dataset)}')
    summarize_dataloaders(train_loader, valid_loader)

    return train_loader, valid_loader, info

def data_from_loader(data_loader: DataLoader, data_type: str, idx=None, max_length=None, batch_size=None):
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

    # Add batch dimension if not present
    if isinstance(idx, int) or (batch_size is None and (max_length == 1)):
        res = res.unsqueeze(0)

    return res

def summarize_dataloaders(train_loader, valid_loader, val_loader=None):
    def loader_info(loader, name):
        dataset_size = len(loader.dataset)
        batch_size = loader.batch_size
        num_batches = len(loader)
        
        print(f"  | {name:<12} | Size: {dataset_size:<7} | Batch: {batch_size:<5} | Batches: {num_batches:<5}")

    print("DataLoader Summary")
    print("-"*69)
    loader_info(train_loader, "Train")
    loader_info(valid_loader, "valid")
    if val_loader:
        loader_info(val_loader, "Validation")
    print("-"*69)