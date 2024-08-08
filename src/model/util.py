import os
import torch
from torch.autograd import Variable
import h5py
import matplotlib.pyplot as plt
import numpy as np
from nltk import Nonterminal
from torch.distributions import Categorical
from nltk.grammar import Production
from grammar import get_mask
from scipy.special import softmax
from torch.utils.data import DataLoader, random_split, Dataset


class Stack:
    """A simple first in last out stack.

    Args:
        grammar: an instance of nltk.CFG
        start_symbol: an instance of nltk.Nonterminal that is the
            start symbol the grammar
    """
    def __init__(self, grammar, start_symbol):
        self.grammar = grammar
        self._stack = [start_symbol]

    def pop(self):
        return self._stack.pop()

    def push(self, symbol):
        self._stack.append(symbol)

    def __str__(self):
        return str(self._stack)

    @property
    def nonempty(self):
        return bool(self._stack)

class AnnealKL:
	"""Anneal the KL for VAE based training"""
	def __init__(self, step=1e-3, rate=500):
		self.rate = rate
		self.step = step

	def alpha(self, update):
		n, _ = divmod(update, self.rate)
		return min(1., n*self.step)

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

def save_model(model, file_name: str):
    checkpoint_path = os.path.abspath('./checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)

    while os.path.exists(f'{checkpoint_path}/{file_name}.pt'):
        file_name = f'{file_name}_copy'
    torch.save(model, f'{checkpoint_path}/{file_name}.pt')

class CustomDataset(Dataset):
    def __init__(self, data_syntax, values):
        assert data_syntax.shape[0] == values.shape[0]
        self.data_syntax = torch.tensor(data_syntax, dtype=torch.float32)
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.data_syntax)

    def __getitem__(self, idx):
        # x_syn = self.data_syntax[idx, ...].transpose(-2, -1) # new shape [batch, RULE_COUNT+1, SEQ_LEN] as required by model
        # x_syn = Variable(x_syn)
        y_rule_idx = self.data_syntax[idx, :, :-1].argmax(axis=1) # The rule index (argmax over onehot part, excluding consts)
        y_consts = self.data_syntax[idx, :, -1]

        return self.data_syntax[idx].transpose(-2, -1), y_rule_idx, y_consts, self.values[idx]

def load_data(datapath: str, name: str, test_split: float = 0.2, batch_size: int = 32):
    eqs, data_syntax, values = load_raw_parsed_value_data(datapath, name)
    # Create the full dataset
    full_dataset = CustomDataset(data_syntax, values)

    # Split the dataset
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_onehot_data(path: str):
    """Loads data as Numpy array."""
    with h5py.File(path, 'r') as f:
        data = f['data'][:]
    return data

def batch_iter(data_syntax: np.ndarray, data_value: np.ndarray, batch_size: int):
    """A simple iterator over batches of data"""
    assert (n := data_syntax.shape[0]) == data_value.shape[0]

    for i in range(0, n, batch_size):
        x_syn = data_syntax[i:i+batch_size, ...].transpose(-2, -1) # new shape [batch, RULE_COUNT+1, SEQ_LEN] as required by model
        x_syn = Variable(x_syn)
        y_rule_idx = x_syn[:, :-1, ...].argmax(axis=1) # The rule index (argmax over onehot part, excluding consts)
        y_consts = x_syn[:, -1, ...]

        y_val = data_value[i:i+batch_size, ...]

        yield x_syn, y_rule_idx, y_consts, y_val

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

def load_raw_parsed_value_data(datapath: str, name: str):
    eqs, onehot = load_raw_parsed_data(datapath, name)


    # Load parsed dataset
    parsed_path = f'{datapath}/{name}-values.h5'
    with h5py.File(parsed_path, 'r') as f:
        values = f['data'][:]
        invalid = f['invalid_mask'][:]

    eqs = eqs[~invalid]
    onehot = onehot[~invalid]

    assert len(eqs) == onehot.shape[0] == len(values)
    
    return eqs, onehot, values

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