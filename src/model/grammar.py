import torch
from torch.autograd import Variable
from nltk import CFG, Nonterminal

grammar = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'EXP' S
S -> 'CON'
S -> 'x1' 
END -> 'END'
"""

GCFG = CFG.fromstring(grammar)
S = Nonterminal('S')
GCFG.productions()

def get_mask(nonterminal, grammar, as_variable=False):
    if isinstance(nonterminal, Nonterminal):
        mask = [rule.lhs() == nonterminal for rule in grammar.productions()]
        mask = Variable(torch.FloatTensor(mask)) if as_variable else mask
        return mask
    else:
        raise ValueError('Input must be instance of nltk.Nonterminal')

if __name__ == '__main__':
    # Usage:
    GCFG = CFG.fromstring(grammar)

    print(get_mask(S, GCFG))
