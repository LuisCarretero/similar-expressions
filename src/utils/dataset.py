import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import hashlib
from typing import Tuple, Dict, Callable
from omegaconf.dictconfig import DictConfig


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

def load_dataset(datapath, name, max_length=-1):
    with h5py.File(os.path.join(datapath, f'{name}.h5'), 'r') as f:
        # Extract onehot, values (eval_y), and consts
        syntax = f['onehot'][:, :, :max_length].transpose([2, 1, 0]).astype(np.float32)  # Shape: (n_samples, seq_len, n_tokens)
        consts = f['consts'][:, :max_length].T.astype(np.float32)  # Shape: (n_samples, seq_len)
        values = f['eval_y'][:, :max_length].T.astype(np.float32)  # Shape: (n_samples, n_values)

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

def create_dataloader(
    datapath: str, 
    name: str, 
    cfg: DictConfig, 
    random_seed: int = 0, 
    shuffle_train: bool = True, 
    value_transform: Callable = None, 
    num_workers: int = 4, 
    drop_last: bool = True
) -> Tuple[DataLoader, DataLoader, Dict]:

    gen = torch.Generator()
    gen.manual_seed(random_seed)

    syntax, consts, values, val_x, syntax_cats = load_dataset(datapath, name, max_length=cfg.training.dataset_len_limit)
    data_syntax = np.concatenate([syntax, consts[:, :, np.newaxis]], axis=-1)  # Shape: (n_samples, seq_len, n_tokens+1)

    # Create value transform
    value_transform = create_value_transform(cfg.training.value_transform, torch.tensor(values))

    # Create the full dataset
    full_dataset = CustomTorchDataset(data_syntax, values, value_transform=value_transform)

    # Split the dataset
    valid_size = int(cfg.training.valid_split * len(full_dataset))
    train_size = len(full_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size], generator=gen)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=shuffle_train, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        drop_last=drop_last
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        drop_last=drop_last
    )

    # Calculate hashes
    assert id(full_dataset) == id(train_loader.dataset.dataset) == id(valid_loader.dataset.dataset), "Datasets are not the same"
    hashes = {
        'dataset': full_dataset.hash,
        'train_idx': hashlib.md5(str(train_loader.dataset.indices).encode()).hexdigest(),
        'valid_idx': hashlib.md5(str(valid_loader.dataset.indices).encode()).hexdigest(),
        'random_seed': random_seed
    }
    info = {
        'hashes': hashes,
        'min_value': values.min(),
        'max_value': values.max(),
        'value_transform': value_transform,
        'dataset_name': name,
        'datapath': datapath, 
        'syntax_cats': syntax_cats,
        'val_x': val_x
    }
    return train_loader, valid_loader, info

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
