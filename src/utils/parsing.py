import torch
import numpy as np
from nltk import Nonterminal
from nltk.grammar import Production
from torch.distributions import Categorical
from typing import List, Tuple, Literal
import sympy as sp
from sympy.utilities.lambdify import lambdify

from src.utils.grammar import get_mask, NT, GCFG

CUSTOM_FUNCTIONS = {
    'zero_sqrt': lambda x: np.sqrt(np.maximum(x, 0)),  # Safe sqrt that returns 0 for negative inputs
}

OPERATOR_ARITY = {
    # Elementary functions
    "ADD": 2,
    "SUB": 2,
    "MUL": 2,
    "DIV": 2,
    "POW": 2,
    "INV": 1,
    "SQRT": 1,
    "ZERO_SQRT": 1,  # Custom safe sqrt
    "EXP": 1,
    "LN": 1,
    "ABS": 1,

    # Trigonometric Functions
    "SIN": 1,
    "COS": 1,
    "TAN": 1,

    # Trigonometric Inverses
    "ASIN": 1,
    "ACOS": 1,
    "ATAN": 1,

    # Hyperbolic Functions
    "SINH": 1,
    "COSH": 1,
    "TANH": 1,
    "COTH": 1,
}

OPERATORS = OPERATOR_ARITY.keys()
UNARY_OP_TOKENS  = [k for k, v in OPERATOR_ARITY.items() if v == 1]

def prods_to_prefix(prods, to_string=False, require_no_nonterminals=True):
    """
    Takes a list of productions and a list of constants and returns a string representation of the equation.
    """
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
        if require_no_nonterminals:
            assert not any(isinstance(s, Nonterminal) for s in seq), f'Found nonterminals in sequence: {seq}'
        return seq

def logits_to_prods(
    logits, 
    start_symbol: Nonterminal = NT, 
    sample=False, 
    max_length=15, 
    insert_const=True, 
    const_token='CON', 
    replace_const: Literal['numerical', 'placeholder', 'nothing', 'numerical_rounded'] = 'numerical', 
    round_const_decimals=2
) -> List[Production]:
    stack = [start_symbol]

    logits_prods = logits[:, :-1]
    constants = logits[:, -1]

    prods = []
    t = 0  # "Time step" in sequence
    j = 0  # Index of constant
    while len(stack) > 0:
        alpha = stack.pop()  # Alpha is notation in paper: current LHS token
        mask = get_mask(alpha)
        probs = mask * logits_prods[t].exp()
        assert (tot := probs.sum()) > 0, f"Sum of probs is 0 at t={t}. Probably due to bad mask or invalid logits?"
        probs = probs / tot

        if sample:
            m = Categorical(probs)
            i = m.sample()
        else:
            _, i = probs.max(-1) # argmax

        # select rule i
        rule = GCFG.productions()[i.item()]

        # If rule has -> [CONST] add const
        if insert_const and (rule.rhs()[0] == const_token):
            if replace_const == 'numerical':
                rule = Production(lhs=rule.lhs(), rhs=(str(constants[t].item()),))
            elif replace_const == 'numerical_rounded':
                rule = Production(lhs=rule.lhs(), rhs=(str(round(constants[t].item(), round_const_decimals)),))
            elif replace_const == 'placeholder':
                placeholder_name = f'CON_{j}'
                rule = Production(lhs=rule.lhs(), rhs=(placeholder_name,))
                j += 1
            elif replace_const == 'nothing':
                pass
            else:
                raise ValueError(f'Unknown replace_const mode: {replace_const}')

        prods.append(rule)
        # add rhs nonterminals to stack in reversed order
        for symbol in reversed(rule.rhs()):
            if isinstance(symbol, Nonterminal):
                stack.append(symbol)
        t += 1
        if t == max_length:
            break
    return prods

def logits_to_prefix(logits, syntax_cats: list[str], sample=False, max_length=15):
    print('Warning: Not using constants in syntax probs.')

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
        if idx == (len(syntax_cats) - 1):  # Assuming the last category is an end token
            break
    
    # Convert token indices to actual tokens
    tokens = [syntax_cats[idx] for idx in token_idx]
    
    return tokens

def _is_leaf_token(token: str) -> bool:
    """
    A leaf node does not need paranthesis around it.

    TODO: Check if this are all the cases that need parenthesis.
    """
    return not any(op in token for op in ['*', '+', '-', '/', '**'])

def _write_infix(token: str, args: List[str]) -> str:
    """
    Convert prefix operator with arguments to a infix string that SymPy can parse.
    FIXME: Remove unnecessary parenthesis in case argument is already enclosed by unary operator.
    """
    
    if token in UNARY_OP_TOKENS:  # FIXME: Check parenthesis
        return f"{token}({args[0]})"
    else:
        args = ['(' + arg + ')' if not _is_leaf_token(arg) else arg for arg in args]
        if token == "ADD":
            return f"{args[0]}+{args[1]}"
        elif token == "SUB":
            return f"{args[0]}-{args[1]}"
        elif token == "MUL":
            return f"{args[0]}*{args[1]}"
        elif token == "DIV":
            return f"{args[0]}/{args[1]}"
        elif token == "POW":
            return f"{args[0]}**{args[1]}"
        elif token == "ABS":
            return f"Abs({args[0]})"
        elif token == "INV":
            return f"1/{args[0]}"
        else:
            raise Exception(f"Unknown token in prefix expression: {token}, with arguments {args}")

def _prefix_to_infix(expr: List[str], variables=None) -> Tuple[str, List[str]]:
    """
    Parse an expression in prefix mode, and output it in either:
        - infix mode (returns human readable string)
        - develop mode (returns a dictionary with the simplified expression)
    """
    if len(expr) == 0:
        raise Exception("Empty prefix list.")
    t = expr[0]
    if t in OPERATORS:
        args = []
        l1 = expr[1:]
        for _ in range(OPERATOR_ARITY[t]):
            i1, l1 = _prefix_to_infix(l1, variables)
            args.append(i1)
        return _write_infix(t, args), l1
    elif t in variables:
        return t, expr[1:]
    else: # Constant
        val = expr[0]
        return str(val), expr[1:]
    
def prefix_to_infix(expr: List[str], variables=['x1']) -> List[str]:
    infix_str, remainder = _prefix_to_infix(expr, variables)
    if len(remainder) > 0:
        raise Exception(f'Invalid prefix expression "{expr}". Successfully parsed "{infix_str}" but "{remainder}" is still remaining.')
    return infix_str

def logits_to_infix(logits, sample=False, replace_const='numerical', round_const_decimals=2):
    # FIXME: Add variables, GCFG pass-through

    assert len(logits.shape) == 2, "Logits should be 2D, no batch dimension"
    prods = logits_to_prods(logits, sample=sample, replace_const=replace_const, round_const_decimals=round_const_decimals)
    prefix = prods_to_prefix(prods, require_no_nonterminals=True)
    infix = prefix_to_infix(prefix, variables=['x1'])
    return infix

def eval_from_logits(logits, val_x):
    assert len(val_x.shape) == 1, "val_x should be 1D"
    x1 = sp.Symbol('x1')

    infix = logits_to_infix(logits)
    expr = sp.sympify(infix.lower())

    # Pass custom_functions as third argument
    # Check for infinity or similar in expression
    assert not expr.has(sp.zoo), f"Expression {expr} contains infinity"
    func = lambdify(x1, expr, ('numpy', CUSTOM_FUNCTIONS))

    with np.errstate(all='ignore'):
        val_y = func(val_x)

    if not isinstance(val_y, np.ndarray):
        # sp.simpify simplifies expressions which may result in `expr` being a constant. 
        # In this case calling func() simply returns this constant instead of a full numpy array
        val_y = np.repeat(val_y, val_x.shape[0])

    return val_y
