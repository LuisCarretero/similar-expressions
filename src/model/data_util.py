import os
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import hashlib
from config_util import Config
from typing import Tuple


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
    # Extract data from DataLoader
    x = dataloader.dataset.dataset[dataloader.dataset.indices][0]
    syntax = x[:, :-1, :].detach().numpy().transpose(0, 2, 1)
    consts = x[:, -1, :].squeeze().detach().numpy()
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
        val_x = f['eval_x'][:].astype(np.float32)
        val = f['eval_y'][:].astype(np.float32).T
        syntax_cats = list(map(lambda x: x.decode('utf-8'), f['onehot_legend'][:]))

    return syntax, consts, val_x, val, syntax_cats

def create_dataloader(datapath: str, name: str, cfg: Config, value_transform=None, random_seed=0) -> Tuple[DataLoader, DataLoader, dict]:
    gen = torch.Generator()
    gen.manual_seed(random_seed)

    syntax, consts, _, values, _ = load_dataset(datapath, name)
    data_syntax = np.concatenate([syntax, consts[:, :, np.newaxis]], axis=-1)

    if cfg.training.dataset_len_limit is not None:
        data_syntax = data_syntax[:cfg.training.dataset_len_limit]
        values = values[:cfg.training.dataset_len_limit]

    # Create the full dataset
    full_dataset = CustomTorchDataset(data_syntax, values, value_transform=value_transform, device=cfg.training.device)

    # Split the dataset
    test_size = int(cfg.training.test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size], generator=gen)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Create hashes
    assert id(full_dataset) == id(train_loader.dataset.dataset) == id(test_loader.dataset.dataset), "Datasets are not the same"
    hashes = {
        'dataset': full_dataset.hash,
        'train_idx': hashlib.md5(str(train_loader.dataset.indices).encode()).hexdigest(),
        'test_idx': hashlib.md5(str(test_loader.dataset.indices).encode()).hexdigest(),
        'random_seed': random_seed
    }

    return train_loader, test_loader, hashes

def data2input(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().unsqueeze(0).transpose(-2, -1)