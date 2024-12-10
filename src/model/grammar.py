import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal
import numpy as np

grammar = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S
S -> 'CON'
S -> 'x1' 
END -> 'END'
"""

# grammar = """
# S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
# S -> 'COS' S | 'EXP' S
# S -> 'CON'
# S -> 'x1' 
# END -> 'END'
# """

GCFG = CFG.fromstring(grammar)
S = Nonterminal('S')
MAX_LEN, DIM = 15, 9  # FIXME: use config

# Collect all lhs symbols, and the unique set of them
all_lhs = [prod.lhs().symbol() for prod in GCFG.productions()]
unique_lhs = []
[unique_lhs.append(x) for x in all_lhs if x not in unique_lhs]

# Rhs symbol indices for each production rule
rhs_map = []
for prod in GCFG.productions():  # [S -> 'ADD' S S]
    tmp = []
    for prod_target in prod.rhs():  # ['ADD', S, S]
        if not isinstance(prod_target, str):  # S
            s = prod_target.symbol()  # 'S'
            tmp.extend(list(np.where(np.array(unique_lhs) == s)[0]))  # [1]
    rhs_map.append(tmp)

# For each lhs symbol which productions rules should be masked
masks = np.array([np.array([lhs == symbol for lhs in all_lhs], dtype=bool).reshape(1, -1) for symbol in unique_lhs]).squeeze()
allowed_prod_idx = np.where(masks)[0]
masks = torch.tensor(masks)
allowed_prod_idx = torch.tensor(allowed_prod_idx)


def get_mask(nonterminal, grammar=GCFG, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')
    

def calc_grammar_mask(y_syntax: torch.Tensor):
    """
    Use true indices to mask predictions. Spefically, only the LHS of the true rule at step t is allowed to be applied at step t. Consequently, all other productions with different LHS are set to zero.

    """
    true_prod_idx = y_syntax.reshape(-1)  # True indices but whole batch flattened
    true_lhs_idx = torch.gather(allowed_prod_idx, 0, true_prod_idx) # LHS rule idx (here 0-S or 1-END)
    allowed_prods_mask = masks[true_lhs_idx]  # get slices of masks with indices FIXME: Use gather?
    allowed_prods_mask = allowed_prods_mask.reshape(-1, MAX_LEN, DIM)  # reshape them to have masks as rows
    return allowed_prods_mask