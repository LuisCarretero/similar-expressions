import time
import torch
from torch.autograd import Variable
import h5py
import matplotlib.pyplot as plt
import numpy as np
from nltk import Nonterminal
from torch.distributions import Categorical
from nltk.grammar import Production
from stack import Stack
from grammar import get_mask
from scipy.special import softmax

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
    
def plot_onehot(onehot_matrix, grammar, apply_softmax=False, figsize=(10, 5)):
    onehot_matrix = onehot_matrix.copy()
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    if apply_softmax:
        onehot_matrix[:, :-1] = softmax(onehot_matrix[:, :-1], axis=-1)
    im1 = ax1.imshow(onehot_matrix[:, :-1])
    im2 = ax2.imshow(np.expand_dims(onehot_matrix[:, -1], axis=1))

    ax1.set_ylabel('Sequence')
    ax1.set_xlabel('Rule')

    xticks = grammar.productions()
    ax1.set_xticks(range(len(xticks)), xticks, rotation='vertical')
    ax2.set_xticks([0], ['[CONST]'], rotation='vertical')
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
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

def load_raw_parsed_data(datapath: str, name: str):
    # Load raw data
    with open(f'{datapath}/{name}.txt', 'r') as f:
        eqs = f.readlines()
        eqs = [eq.strip('\n') for eq in eqs]
        eqs = np.array(eqs)

    # Load parsed dataset
    parsed_path = f'{datapath}/{name}-parsed.h5'
    with h5py.File(parsed_path, 'r') as f:
        onehot = f['data'][:]
        invalid_idx = f['invalid_indices'][:]

    mask = np.ones(len(eqs), dtype=bool)
    mask[invalid_idx] = False

    return eqs[mask], onehot

def logits_to_prods(logits, grammar, start_symbol, sample=False, max_length=15):
    stack = Stack(grammar=grammar, start_symbol=start_symbol)

    logits_prods = logits[:, :-1]
    constants = logits[:, -1]

    prods = []
    t = 0  # "Time step" in sequence
    while stack.nonempty:
        alpha = stack.pop()  # Alpha is notation in paper.
        mask = get_mask(alpha, stack.grammar, as_variable=True)
        probs = mask * logits_prods[t].exp()
        probs = probs / probs.sum()
        
        if sample:
            m = Categorical(probs)
            i = m.sample()
        else:
            _, i = probs.max(-1) # argmax

        # select rule i
        rule = stack.grammar.productions()[i.item()]

        # If rule has -> [CONST] add const
        if rule.rhs()[0] == '[CONST]':
            rule = Production(lhs=rule.lhs(), rhs=(str(constants[t].item()),))

        prods.append(rule)
        # add rhs nonterminals to stack in reversed order
        for symbol in reversed(rule.rhs()):
            if isinstance(symbol, Nonterminal):
                stack.push(symbol)
        t += 1
        if t == max_length:
            break
    return prods