import os
import sys
project_root = os.path.dirname(os.path.abspath(''))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'model'))
sys.path.insert(0, os.path.join(project_root, 'dataset_generation'))

import torch
from util import load_data

BATCH_SIZE = 32

# Load data
def value_transform(x):
    eps = 1e-10
    return torch.log(torch.abs(x) + eps)/10  # Example transformation
datapath = '/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data'
train_loader, test_loader = load_data(datapath, 'dataset_240816_2', test_split=0.1, batch_size=BATCH_SIZE, value_transform=value_transform)


import h5py
import os
import torch
import numpy as np

name = 'dataset_240816_2'

# Load the HDF5 file
with h5py.File(os.path.join(datapath, f'{name}.h5'), 'r') as f:
    # Extract onehot, values (eval_y), and consts
    onehot = f['onehot'][:].astype(np.float32)
    values = f['eval_y'][:].astype(np.float32)
    consts = f['consts'][:].astype(np.float32)

syntax_data = np.concatenate([onehot.transpose([2, 1, 0]), consts.T[:, :, np.newaxis]], axis=-1)
value_data = values.T
