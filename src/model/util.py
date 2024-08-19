import os
import torch
from torch.autograd import Variable
import h5py
import matplotlib.pyplot as plt
import numpy as np
from nltk import Nonterminal
from torch.distributions import Categorical
from nltk.grammar import Production
from grammar import get_mask, S
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

def prods_to_eq(prods, verbose=False, to_string=False):
    """Takes a list of productions and a list of constants and returns a string representation of the equation. Only works with infix CFG."""
    seq = [prods[0].lhs()]  # Start with LHS of first rule (always nonterminal start)
    for prod in prods:
        if str(prod.lhs()) == 'Nothing':  # Padding rule. Reached end.
            break
        for ix, s in enumerate(seq):  # Applying rule to first element in seq that matches lhs
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]  # Replace LHS with RHS
                break
    if to_string:
        try:
            return ' '.join(seq)
        except TypeError:
            print(f'Error. Could not create equation from {seq = }; {prods = }')
            return ''
    else:
        return seq
    
def plot_onehot(onehot_matrix, xticks, apply_softmax=False, figsize=(10, 5)):
    onehot_matrix = onehot_matrix.copy()
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    if apply_softmax:
        onehot_matrix[:, :-1] = softmax(onehot_matrix[:, :-1], axis=-1)
    im1 = ax1.imshow(onehot_matrix[:, :-1])
    im2 = ax2.imshow(np.expand_dims(onehot_matrix[:, -1], axis=1))

    ax1.set_ylabel('Sequence')
    ax1.set_xlabel('Rule')

    ax1.set_xticks(range(len(xticks)), xticks, rotation='vertical')
    ax2.set_xticks([0], ['[CON]'], rotation='vertical')
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    plt.show()

class CustomTorchDataset(Dataset):
    def __init__(self, data_syntax, data_values, value_transform=None, device='cpu'):
        assert data_syntax.shape[0] == data_values.shape[0]
        self.data_syntax = torch.tensor(data_syntax, dtype=torch.float32).to(device)
        self.values = torch.tensor(data_values, dtype=torch.float32).to(device)

        self.value_transform = value_transform

    def __len__(self):
        return len(self.data_syntax)

    def __getitem__(self, idx):  # FIXME: Why would we call this on each single element and not batchwise?!
        # x_syn = self.data_syntax[idx, ...].transpose(-2, -1) # new shape [batch, RULE_COUNT+1, SEQ_LEN] as required by model
        # x_syn = Variable(x_syn)
        y_rule_idx = self.data_syntax[idx, :, :-1].argmax(axis=1) # The rule index (argmax over onehot part, excluding consts)
        y_consts = self.data_syntax[idx, :, -1]

        if self.value_transform is not None:
            y_val = self.value_transform(self.values[idx])
        else:
            y_val = self.values[idx]

        return self.data_syntax[idx].transpose(-2, -1), y_rule_idx, y_consts, y_val

def load_dataset(datapath, name):
    with h5py.File(os.path.join(datapath, f'{name}.h5'), 'r') as f:
        # Extract onehot, values (eval_y), and consts
        syntax = f['onehot'][:].astype(np.float32).transpose([2, 1, 0])
        consts = f['consts'][:].astype(np.float32).T
        val_x = f['eval_x'][:].astype(np.float32)
        val = f['eval_y'][:].astype(np.float32).T
        syntax_cats = list(map(lambda x: x.decode('utf-8'), f['onehot_legend'][:]))

    return syntax, consts, val_x, val, syntax_cats

def create_dataloader(datapath: str, name: str, test_split: float = 0.2, batch_size: int = 32, max_length: int = None, value_transform=None, device='cpu'):
    syntax, consts, _, values, _ = load_dataset(datapath, name)
    data_syntax = np.concatenate([syntax, consts[:, :, np.newaxis]], axis=-1)

    if max_length is not None:
        data_syntax = data_syntax[:max_length]
        values = values[:max_length]

    # Create the full dataset
    full_dataset = CustomTorchDataset(data_syntax, values, value_transform=value_transform, device=device)

    # Split the dataset
    test_size = int(test_split * len(full_dataset))
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

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


def logits_to_prods(logits, grammar, start_symbol: Nonterminal = S, sample=False, max_length=15, insert_const=True, const_token='CON'):
    stack = Stack(grammar=grammar, start_symbol=start_symbol)

    logits_prods = logits[:, :-1]
    constants = logits[:, -1]

    prods = []
    t = 0  # "Time step" in sequence
    while stack.nonempty:
        alpha = stack.pop()  # Alpha is notation in paper: current LHS token
        mask = get_mask(alpha, grammar, as_variable=True)
        probs = mask * logits_prods[t].exp()
        assert (tot := probs.sum()) > 0, f"Sum of probs is 0 at t={t}. Probably due to bad mask or invalid logits?"
        probs = probs / tot

        if sample:
            m = Categorical(probs)
            i = m.sample()
        else:
            _, i = probs.max(-1) # argmax

        # select rule i
        rule = grammar.productions()[i.item()]

        # If rule has -> [CONST] add const
        if insert_const and (rule.rhs()[0] == const_token):
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

def logits_to_prefix(logits, syntax_cats: list[str], sample=False, max_length=15):


    consts = logits[:, -1]  # FIXME: Replace const placeholders
    syntax = logits[:, :-1]

    token_idx = []
    probs = torch.softmax(syntax, dim=-1)  # Convert logits to probabilities, excluding the last column (constants)
    
    for t in range(max_length):
        if sample:
            dist = torch.distributions.Categorical(probs[t])
            idx = dist.sample()
        else:
            idx = torch.argmax(probs[t])
        
        token_idx.append(idx.item())
        
        # Break if we've reached the end of the sequence
        if idx == len(syntax_cats) - 1:  # Assuming the last category is an end token
            break
    
    # Convert token indices to actual tokens
    tokens = [syntax_cats[idx] for idx in token_idx]
    
    return tokens
