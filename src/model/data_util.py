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
from model import GrammarVAE

class CustomTorchDataset(Dataset):
    def __init__(self, data_syntax: np.ndarray, data_values: np.ndarray, value_transform=None, device='cpu'):
        assert data_syntax.shape[0] == data_values.shape[0]

        # Calculate hashes  FIXME: Might want to only take hash of parts? Check performance impact.
        md5 = hashlib.md5()
        md5.update(data_syntax.tobytes())
        md5.update(data_values.tobytes())
        self.hash = md5.hexdigest()

        self.data_syntax = torch.tensor(data_syntax, dtype=torch.float32).to(device)
        self.values_transformed = value_transform(torch.tensor(data_values, dtype=torch.float32).to(device))
        self.value_transform = value_transform

    def __len__(self):
        return len(self.data_syntax)

    def __getitem__(self, idx):
        x = self.data_syntax[idx].transpose(-2, -1)
        y_rule_idx = self.data_syntax[idx, :, :-1].argmax(axis=1) # The rule index (argmax over onehot part, excluding consts)
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

def load_dataset(datapath, name):
    with h5py.File(os.path.join(datapath, f'{name}.h5'), 'r') as f:
        # Extract onehot, values (eval_y), and consts
        syntax = f['onehot'][:].astype(np.float32).transpose([2, 1, 0])
        consts = f['consts'][:].astype(np.float32).T
        values = f['eval_y'][:].astype(np.float32).T
        val_x = f['eval_x'][:].astype(np.float32)
        syntax_cats = list(map(lambda x: x.decode('utf-8'), f['onehot_legend'][:]))

    return syntax, consts, values, val_x, syntax_cats

def create_dataloader(datapath: str, name: str, cfg: Config, random_seed=0, shuffle_train=True) -> Tuple[DataLoader, DataLoader, dict]:
    gen = torch.Generator()
    gen.manual_seed(random_seed)

    syntax, consts, values, val_x, syntax_cats = load_dataset(datapath, name)
    data_syntax = np.concatenate([syntax, consts[:, :, np.newaxis]], axis=-1)

    if cfg.training.dataset_len_limit is not None:
        data_syntax = data_syntax[:cfg.training.dataset_len_limit]
        values = values[:cfg.training.dataset_len_limit]

    # Create value transform
    min_, max_ = np.arcsinh(values.min()), np.arcsinh(values.max())
    value_transform = lambda x: (torch.arcsinh(x)-min_)/(max_-min_)

    # Create the full dataset
    full_dataset = CustomTorchDataset(data_syntax, values, value_transform=value_transform, device=cfg.training.device)

    # Split the dataset
    test_size = int(cfg.training.test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=gen)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=shuffle_train)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Create hashes
    assert id(full_dataset) == id(train_loader.dataset.dataset) == id(test_loader.dataset.dataset), "Datasets are not the same"
    hashes = {
        'dataset': full_dataset.hash,
        'train_idx': hashlib.md5(str(train_loader.dataset.indices).encode()).hexdigest(),
        'test_idx': hashlib.md5(str(test_loader.dataset.indices).encode()).hexdigest(),
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
    return train_loader, test_loader, info

def data2input(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().unsqueeze(0).transpose(-2, -1)

def load_wandb_model(run: str, device='cpu', wandb_cache_path='/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/wandb_cache'):
    # Load model
    with wandb.restore('model.pth', run_path=f"luis-carretero-eth-zurich/similar-expressions-01/runs/{run}", root=wandb_cache_path, replace=True) as io:
        name = io.name
    checkpoint = torch.load(name, map_location=device)

    # Read the model parameters from the WandB config.yaml file
    with wandb.restore('config.yaml', run_path=f"luis-carretero-eth-zurich/similar-expressions-01/runs/{run}", root=wandb_cache_path, replace=True) as config_file:
        cfg_dict = yaml.safe_load(config_file)
        cfg = {k: v['value'] for k, v in list(cfg_dict.items()) if k not in ['wandb_version', '_wandb']}
        cfg = dict_to_config(cfg)

    cfg.training.device = device
    vae_model = GrammarVAE(cfg)
    vae_model.load_state_dict(checkpoint['model_state_dict'])
    return vae_model, cfg_dict, cfg

def create_dataloader_from_wandb(cfg_dict, cfg, datapath='/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'):
    train_loader, test_loader, info = create_dataloader(datapath, name=cfg_dict['dataset']['value'], cfg=cfg)
    assert all([cfg_dict['dataset_hashes']['value'][key] == info['hashes'][key] for key in cfg_dict['dataset_hashes']['value']]), "Error: Using different dataset than used for training."


    return train_loader, test_loader, info

def data_from_loader(data_loader, data, idx=None, max_length=None, batch_size=None):
    data_idx = {'x': 0, 'syntax': 1, 'consts': 2, 'values': 3}[data]
    dataset = data_loader.dataset.dataset[data_loader.dataset.indices][data_idx]
    
    if idx is not None:
        res = dataset[idx, ...]
    elif max_length is not None:
        dataset = dataset[:max_length, ...]
        if batch_size is not None:
            res = [dataset[i:i+batch_size, ...] for i in range(0, len(dataset), batch_size)]
        else:
            res = dataset
    elif batch_size is not None:
        res = [dataset[i:i+batch_size, ...] for i in range(0, len(dataset), batch_size)]
    else:
        res = dataset

    # Add batch dimension if not present
    if len(res.shape) == 2 and data in ['x', 'syntax'] or len(res.shape) == 1 and data in ['values', 'consts']:
        res = res.unsqueeze(0)
    return res