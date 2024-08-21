import torch
from nltk import Nonterminal
from torch.distributions import Categorical
from nltk.grammar import Production
from grammar import get_mask, S
import math
from config_util import CriterionConfig
from typing import Dict

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

class AnnealKLSigmoid:
    """Anneal the KL for VAE based training using a sigmoid schedule"""
    def __init__(self, total_epochs, midpoint=0.5, steepness=10):
        self.total_epochs = total_epochs
        self.midpoint = midpoint
        self.steepness = steepness

    def alpha(self, epoch):
        """
        Calculate the annealing factor using a sigmoid function.
        
        Args:
            epoch (int): Current epoch number (0-indexed)
        
        Returns:
            float: Annealing factor between 0 and 1
        """
        x = (epoch / self.total_epochs - self.midpoint) * self.steepness
        return 1 / (1 + math.exp(-x))

def prods_to_eq(prods, to_string=False):
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

def criterion_factory(cfg: CriterionConfig, priors: Dict):
    SYNTAX_LOSS_WEIGHT = cfg.syntax_loss_weight
    CONSTS_LOSS_WEIGHT = cfg.consts_loss_weight
    VALUES_LOSS_WEIGHT = cfg.values_loss_weight

    SYNTAX_PRIOR = priors['syntax_prior']
    CONSTS_PRIOR = priors['consts_prior']
    VALUES_PRIOR = priors['values_prior']

    cross_entropy = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    def criterion(logits, values, y_rule_idx, y_consts, y_val):
        
        logits_onehot = logits[:, :, :-1]
        loss_syntax = cross_entropy(logits_onehot.reshape(-1, logits_onehot.size(-1)), y_rule_idx.reshape(-1))/SYNTAX_PRIOR
        loss_consts = mse(logits[:, :, -1], y_consts)/CONSTS_PRIOR

        loss_values = mse(values, y_val)/VALUES_PRIOR

        loss = loss_syntax*SYNTAX_LOSS_WEIGHT + loss_consts*CONSTS_LOSS_WEIGHT + loss_values*VALUES_LOSS_WEIGHT
        loss = loss / (SYNTAX_LOSS_WEIGHT + CONSTS_LOSS_WEIGHT + VALUES_LOSS_WEIGHT)

        return loss_syntax, loss_consts, loss_values, loss
    return criterion

def calc_syntax_accuracy(logits, y_rule_idx):
    y_hat = logits.argmax(-1)
    a = (y_hat == y_rule_idx).float().mean()
    return 100 * a.item()
