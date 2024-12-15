import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import hashlib
from omegaconf.dictconfig import DictConfig
from typing import Tuple, Dict
import wandb
import yaml
from omegaconf import OmegaConf
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf.dictconfig import DictConfig

from src.model.model import LitGVAE

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

def create_value_transform(transform_cfg: DictConfig, values: torch.Tensor):
    """
    Parts:
    - nonlinear mapping (e.g. arcsinh)
    - normalization (after mapping)
        - bias/mean (wrt. whole dataset or per sample)
        - scale/std (wrt. whole dataset or per sample)

    Build function dynamically.
    """ 

    mapping = {
        'arcsinh': lambda x: torch.arcsinh(x),
        None: lambda x: x
    }[transform_cfg.mapping]

    if transform_cfg.bias == 'dataset' or transform_cfg.scale in ['dataset-std', 'dataset-range']:
        assert values is not None, "Values are needed to calculate bias and scale over whole dataset."
        values_mapped = mapping(values)
    else:
        values_mapped = None

    bias = {
        'dataset': lambda _: values_mapped.mean(),
        'sample': lambda x: x.mean(dim=1).unsqueeze(1),
        None: lambda _: 0
    }[transform_cfg.bias]

    def replace_with(values_mapped, value, replacement):
        values_mapped[values_mapped == value] = replacement # Dont change the scale if all values are the same
        return values_mapped

    scale = {
        'dataset-std': lambda _: values_mapped.std().item(),
        'sample-std': lambda x: replace_with(x.std(dim=1), 0, 1).unsqueeze(1),
        'dataset-range': lambda _: (values_mapped.max() - values_mapped.min()).item(),
        'sample-range': lambda x: replace_with(x.max(dim=1)[0] - x.min(dim=1)[0], 0, 1).unsqueeze(1),
        None: lambda _: 1
    }[transform_cfg.scale]
    
    def combined_transform(x):
        x_mapped = mapping(x)
        x_scaled = x_mapped / scale(x_mapped)
        return x_scaled - bias(x_scaled)
    
    return combined_transform

def create_dataloader(datapath: str, name: str, cfg: DictConfig, random_seed=0, shuffle_train=True, value_transform=None, num_workers=4, drop_last=True) -> Tuple[DataLoader, DataLoader, dict]:
    gen = torch.Generator()
    gen.manual_seed(random_seed)

    syntax, consts, values, val_x, syntax_cats = load_dataset(datapath, name)
    data_syntax = np.concatenate([syntax, consts[:, :, np.newaxis]], axis=-1)  # Shape: (n_samples, seq_len, n_tokens+1)

    if cfg.training.dataset_len_limit is not None:
        data_syntax = data_syntax[:cfg.training.dataset_len_limit]
        values = values[:cfg.training.dataset_len_limit]

    # Create value transform
    value_transform = create_value_transform(cfg.training.value_transform, torch.tensor(values))

    # Create the full dataset
    full_dataset = CustomTorchDataset(data_syntax, values, value_transform=value_transform)

    # Split the dataset
    valid_size = int(cfg.training.valid_split * len(full_dataset))
    train_size = len(full_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=gen)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=drop_last)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=drop_last)

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
        'min_value': values.min(),  # FIXME: Still needed? Esp if we use individual normalisation
        'max_value': values.max(),
        'value_transform': value_transform,
        'dataset_name': name,
        'datapath': datapath, 
        'syntax_cats': syntax_cats,
        'val_x': val_x
    }
    return train_loader, valid_loader, info

def dict_lit2num(d: Dict, verbose=False):
    """
    Convert all literal values in a dictionary to numbers (int or float). Needed as WandB stores all values as strings?
    """
    def _convert(x):
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        else:  # Leaf node
            try:
                tmp = float(x)
                if tmp.is_integer():
                    tmp = int(tmp)
            except:
                tmp = x
            if verbose:
                print(f'Leaf node:{x} -> {tmp}; type: {type(tmp)}')
            return tmp
    return _convert(d)


def load_wandb_model(run: str, name:str = 'model.pth', device='cpu', wandb_cache_path='/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/wandb_cache', project='similar-expressions-01', replace=True, fallback_cfg_path='config.yaml'):
    # Load model
    with wandb.restore(name, run_path=f"luis-carretero-eth-zurich/{project}/runs/{run}", root=wandb_cache_path, replace=replace) as io:
        name = io.name

    # Read the model parameters from the WandB config.yaml file
    with wandb.restore('config.yaml', run_path=f"luis-carretero-eth-zurich/{project}/runs/{run}", root=wandb_cache_path, replace=True) as config_file:
        cfg_dict = yaml.safe_load(config_file)

    cfg = OmegaConf.create({k: dict_lit2num(v['value']) for k, v in list(cfg_dict.items()) if k not in ['wandb_version', '_wandb']})
    fallback_cfg = OmegaConf.load(fallback_cfg_path)  # FIXME: Combine into one method with load_config?
    cfg = OmegaConf.merge(fallback_cfg, cfg)
    # vae_model = LitGVAE(cfg, get_empty_priors())
    
    # Load the Lightning checkpoint
    vae_model = LitGVAE.load_from_checkpoint(name, cfg=cfg, priors=get_empty_priors(), map_location=device)
    vae_model.eval()

    summary = ModelSummary(vae_model, max_depth=1)
    print(f'Imported model from run "{run}".')
    print(summary)

    return vae_model, cfg

def create_dataloader_from_wandb(cfg: DictConfig, value_transform=None, datapath='/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/data', old_x_format=False):
    train_loader, valid_loader, info = create_dataloader(datapath, name=cfg.training.dataset_name, cfg=cfg, value_transform=value_transform, shuffle_train=False)
    assert all([cfg.dataset_hashes[key] == info['hashes'][key] for key in cfg.dataset_hashes.keys()]), "Error: Using different dataset than used for training."

    print(f'Using dataset "{cfg.training.dataset_name}" of size {len(train_loader.dataset)}')
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
    else:
        raise ValueError("No data selected")

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