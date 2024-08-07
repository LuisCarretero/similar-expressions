import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from typing import List, Tuple
import re
import nltk
import numpy as np
from model.grammar import GCFG
from nltk.grammar import Nonterminal


S, T = Nonterminal('S'), Nonterminal('T')

SEQ_LEN = 15

def tokenize(s):
    """Tokenize an equation string s into a list of tokens."""
    
    # Replace numerical constants with placeholder and store them seperately
    pattern = r'-?\b(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\b'
    consts = []

    def replace_and_collect(match):
        consts.append(match.group())
        return '[CONST]'

    s = re.sub(pattern, replace_and_collect, s)
    
    unaops = ['sin', 'exp']
    binops = ['+', '-', '*', '/']
    for fn in unaops: 
        s = s.replace(fn+'(', ' '+fn)  # Temporarily remove opening parentheses after unary operator
    s = s.replace('(', ' ( ')
    for fn in unaops: 
        s = s.replace(fn, fn+'( ') 

    for fn in binops: 
        s = s.replace(fn, ' '+fn+' ')
    s = s.replace(')', ' ) ')
    tokens = s.split()
    
    return tokens, consts

# Tokenization and parsing functions
_productions = GCFG.productions()
_parser = nltk.ChartParser(GCFG)
_prod_map = {prod: idx for idx, prod in enumerate(_productions)}


def parse_dataset(equations: List[str]) -> Tuple[np.ndarray, List[int], List[List[str]]]:
    tokens, consts = zip(*map(tokenize, equations))
    parse_trees = map(lambda x: next(_parser.parse(x)), tokens)
    prod_seqs = list(map(lambda parse_tree: parse_tree.productions(), parse_trees))  # Sequence of productions

    # Filter out too long productions
    invalid = [i for i, seq in enumerate(prod_seqs) if len(seq) > SEQ_LEN]
    prod_seqs = [seq for i, seq in enumerate(prod_seqs) if i not in invalid]
    consts = [consts[i] for i in range(len(consts)) if i not in invalid]

    indices = map(lambda productions: np.array([_prod_map[prod] for prod in productions], dtype=int), prod_seqs)
    indices = list(indices)

    onehot = np.zeros([len(indices), SEQ_LEN, len(GCFG.productions())+1], dtype=float)
    additional_row_idx = np.arange(len(indices))* (len(GCFG.productions())+1)

    for eq_i, (idxs, const) in enumerate(zip(indices, consts)):
        indices_total = idxs + additional_row_idx[:len(idxs)]
        tmp = np.zeros(np.prod([SEQ_LEN, len(GCFG.productions())+1]))
        tmp[indices_total] = 1
        onehot[eq_i, ...] = tmp.reshape([SEQ_LEN, len(GCFG.productions())+1])

        # Add constants (whereever the production -> [CONST] is invoked). TODO: Check order.
        onehot[eq_i, np.where(onehot[eq_i, :, -3] == 1), -1] = const  # Implicit conversion to float

        # Pad with Nothing -> None prod
        onehot[eq_i, np.where(onehot[eq_i, ...].sum(axis=1) == 0), -2] = 1

    return onehot, invalid, consts