import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal
import numpy as np

grammar = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'EXP' S
S -> 'CON'
S -> 'x1' 
END -> 'END'
"""

GCFG = CFG.fromstring(grammar)
S = Nonterminal('S')

def get_mask(nonterminal, grammar=GCFG, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

# Copied from grammarVAE-keras. Refractor, this looks like a mess.

# collect all lhs symbols, and the unique set of them
all_lhs = [a.lhs().symbol() for a in GCFG.productions()]
lhs_list = []
for a in all_lhs:
    if a not in lhs_list:
        lhs_list.append(a)

D = len(GCFG.productions())

# this map tells us the rhs symbol indices for each production rule
rhs_map = [None]*D
count = 0
for a in GCFG.productions():
    rhs_map[count] = []
    for b in a.rhs():
        if not isinstance(b, str):
            s = b.symbol()
            rhs_map[count].extend(list(np.where(np.array(lhs_list) == s)[0]))
    count = count + 1

masks = np.zeros((len(lhs_list),D))
count = 0

# this tells us for each lhs symbol which productions rules should be masked
for symbol in lhs_list:
    is_in = np.array([a == symbol for a in all_lhs], dtype=int).reshape(1,-1)
    masks[count] = is_in
    count = count + 1

# this tells us the indices where the masks are equal to 1
ind_of_ind = np.array([np.where(masks[:, i] == 1)[0][0] for i in range(masks.shape[1])])

max_rhs = max([len(l) for l in rhs_map])