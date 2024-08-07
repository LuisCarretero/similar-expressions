import re
import nltk
import grammar
import numpy as np
import h5py
import os


MAX_LEN = 15

def tokenize(s):
    funcs = ['sin', 'exp']
    for fn in funcs: s = s.replace(fn+'(', fn+' ')
    s = re.sub(r'([^a-z ])', r' \1 ', s)
    for fn in funcs: s = s.replace(fn, fn+'(')
    return s.split()

# Tokenization and parsing functions
_productions = grammar.GCFG.productions()
_tokenize = tokenize
_parser = nltk.ChartParser(grammar.GCFG)
_n_chars = len(_productions)
_prod_map = {}
for ix, prod in enumerate(_productions):
    _prod_map[prod] = ix


def onehot_encode(eqs):
    """
    Taken a list of expressions as string (e.g. "sin(exp(sin(x1) * 0.4609526466213322))") and parses them into a one-hot encoded representation.
    """
    assert type(eqs) == list
    print('Starting tokenization...')
    tokens = map(_tokenize, eqs)
    print('Starting parsing...')
    parse_trees = [next(_parser.parse(t)) for t in tokens]
    print('Starting production extraction...')
    productions_seq = [tree.productions() for tree in parse_trees]

    indices = [np.array([_prod_map[prod] for prod in entry], dtype=int) for entry in productions_seq]
    print('Starting one-hot encoding...')
    one_hot = np.zeros((len(indices), MAX_LEN, _n_chars), dtype=np.float32)
    for i in range(len(indices)):
        num_productions = len(indices[i])
        one_hot[i][np.arange(num_productions),indices[i]] = 1.
        one_hot[i][np.arange(num_productions, MAX_LEN),-1] = 1.
    return one_hot


if __name__ == '__main__':
    datapath = r'/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data/expr_240807_1.txt'
    parsed_path = r'/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/grammar-vae/data/equation2_15_dataset_parsed.h5'

    with open(datapath, 'r') as f:
        eqs = f.readlines()
        eqs = [eq.strip('\n') for eq in eqs]

    one_hot = onehot_encode(eqs)

    if os.path.exists(parsed_path):
        print('File already exists. Exiting.')
        exit()
    print('Saving to h5...')
    with h5py.File(parsed_path, 'w') as f:
        f.create_dataset('data', data=one_hot)



