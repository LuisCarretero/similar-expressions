import time
from collections import defaultdict

import h5py
from nltk import Tree, Nonterminal

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


def prods_to_eq(prod, numericals, verbose=False):
    seq = [prod[0].lhs()]  # Start with LHS of first rule (always nonterminal start)
    for prod_idx, prod in enumerate(prod):
        if str(prod.lhs()) == 'Nothing':  # Padding rule. Reached end.
            break
        for ix, s in enumerate(seq):  # Applying rule to each element in seq
            if s == prod.lhs():
                print(f'{prod.rhs() = }')
                if not prod.rhs()[0] == '[CONST]':
                    seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]  # Replace LHS with RHS
                else:
                    seq = seq[:ix] + [str(numericals[prod_idx, ix].item())] + seq[ix+1:]  # Replace LHS with RHS but use numerical values for each [CONST] placeholder
                break
    try:
        return ''.join(seq)
    except TypeError:
        if verbose:
            print(f'Nonterminal found. {seq = }')
        return ''