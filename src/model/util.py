import time
import torch
from torch.autograd import Variable
import h5py
import matplotlib.pyplot as plt
import numpy as np

class Timer:
	"""A simple timer to use during training"""
	def __init__(self):
		self.time0 = time.time()

	def elapsed(self):
		time1 = time.time()
		elapsed = time1 - self.time0
		self.time0 = time1
		return elapsed

class AnnealKL:
	"""Anneal the KL for VAE based training"""
	def __init__(self, step=1e-3, rate=500):
		self.rate = rate
		self.step = step

	def alpha(self, update):
		n, _ = divmod(update, self.rate)
		return min(1., n*self.step)

def load_data(data_path):
	"""Returns the h5 dataset as numpy array"""

	with h5py.File(data_path, 'r') as f:
		data = f['data'][:]
	return data

def data2input(x):
    x = torch.from_numpy(x).float().unsqueeze(0).transpose(-2, -1)
    return Variable(x)

def prods_to_eq(prods, verbose=False):
    """Takes a list of productions and a list of constants and returns a string representation of the equation."""
    seq = [prods[0].lhs()]  # Start with LHS of first rule (always nonterminal start)
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':  # Padding rule. Reached end.
            break
        for ix, s in enumerate(seq):  # Applying rule to first element in seq that matches lhs
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]  # Replace LHS with RHS
                break
    try:
        return ' '.join(seq)
    except TypeError:
        if verbose:
            print(f'Nonterminal found. {seq = }')
        return ''
    
def plot_onehot(onehot_matrix, grammar):
    plt.imshow(onehot_matrix)

    plt.ylabel('Sequence')
    plt.xlabel('Rule')

    xticks = grammar.productions() + ['[CONST]']
    plt.xticks(range(len(xticks)), xticks, rotation='vertical')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
def load_onehot_data(path: str):
    """Loads data as Numpy array."""
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
    return data

def batch_iter(data: np.ndarray, batch_size: int):
    """A simple iterator over batches of data"""

    n = data.shape[0]
    for i in range(0, n, batch_size):
        x = data[i:i+batch_size, ...].transpose(-2, -1) # new shape [batch, RULE_COUNT+1, SEQ_LEN] as required by model
        x = Variable(x)
        y_rule_idx = x[:, :-1, ...].argmax(axis=1) # The rule index (argmax over onehot part, excluding consts)
        y_consts = x[:, -1, ...]
        yield x, y_rule_idx, y_consts