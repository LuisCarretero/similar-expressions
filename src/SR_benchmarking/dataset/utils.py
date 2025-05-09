import numpy as np
from typing import Tuple, Dict, List, Iterable
from dataclasses import dataclass
import csv
import os
import pandas as pd


@dataclass
class SyntheticDataset:
    idx: int
    equation: str  # Including 'y = '
    X: np.ndarray
    y: np.ndarray
    var_order: Dict[str, str]

    def __repr__(self):
        return f'SyntheticDataset(idx={self.idx}, equation="{self.equation}", X={self.X.shape}, y={self.y.shape}, var_order={self.var_order})'


# Useful constants. Required for sample_equation -> eval(lambda ...) to work.
pi = np.pi
cos = np.cos
sin = np.sin
sqrt = np.sqrt
exp = np.exp
arcsin = np.arcsin
arccos = np.arccos
log = np.log
ln = np.log
tanh = np.tanh
sinh = np.sinh
cosh = np.cosh


def sample_equation(equation: str, bounds: Dict[str, Tuple[str, Tuple[float, float]]], num_samples: int, noise: float, add_extra_vars: bool):
    """
    From LaSR codebase.
    """
    out = []
    for var in bounds:
        if bounds[var] is None:  # goal
            continue
        out.append((var, sample(*bounds[var], num_samples=num_samples)))

    expr = equation.split(" = ")[1].replace("^", "**")
    expr_as_func = eval(f"lambda {','.join([x[0] for x in out])}: {expr}")  # TODO: use sympy?

    y = list()
    X_temp = np.transpose(np.stack([x[1] for x in out]))
    for i in range(num_samples):
        y.append(expr_as_func(*list(X_temp[i])))
    y = np.array(y)
    y = y + np.random.normal(0, np.sqrt(np.square(y).mean()) * noise, y.shape)

    if add_extra_vars:
        total_vars = len(["x", "y", "z", "k", "j", "l", "m", "n", "p", "a", "b"])
        extra_vars = {
            chr(ord("A") + c): ("uniform", (1, 20))
            for c in range(total_vars - len(bounds) + 1)
        }
        for var in extra_vars:
            out.append((var, sample(*extra_vars[var], num_samples=num_samples)))


    np.random.shuffle(out)
    var_order = {"x" + str(i): out[i][0] for i in range(len(out))}
    X = np.transpose(np.stack([x[1] for x in out]))

    return X, y, var_order

def sample_datasets(
    equations: List[Tuple[int, Tuple[str, Dict[str, Tuple[str, List[float]]]]]], 
    num_samples: int, noise: float, add_extra_vars: bool
) -> List[SyntheticDataset]:
    """
    Dataset: [(idx, (equation, X, Y, var_order))]
    """
    datasets = []
    for (idx, (eq, bounds)) in equations:
        X, Y, var_order = sample_equation(eq, bounds, num_samples, noise, add_extra_vars=add_extra_vars)
        datasets.append(SyntheticDataset(idx, eq, X, Y, var_order))
    return datasets

def sample(method: str, b: Tuple[float, float], num_samples: int):
    """
    Samples x values from specified distribution.
    """
    if method == "constant":  # const
        return np.full(num_samples, b)
    elif method == "uniform":  # (low, high)
        return np.random.uniform(low=b[0], high=b[1], size=num_samples)
    elif method == "normal":  # (mean, std)
        return np.random.normal(loc=b[0], scale=b[1], size=num_samples)
    elif method == "loguniform":  # logU(a, b) ~ exp(U(log(a), log(b))
        return np.exp(
            np.random.uniform(low=np.log(b[0]), high=np.log(b[1]), size=num_samples)
        )
    elif method == "lognormal":  # ln of var is normally distributed
        return np.random.lognormal(mean=b[0], sigma=b[1], size=num_samples)
    else:
        print("Invalid method")

def load_datasets(
    which: str, # ['synthetic', 'feynman', 'pysr-difficult']
    num_samples: int, 
    noise: float, 
    equation_indices: Iterable[int] = None, 
    add_extra_vars: bool = False, 
    fpath: str = None, 
    forbid_ops: Iterable[str] = None
) -> List[SyntheticDataset]:

    if isinstance(equation_indices, int):
        equation_indices = [equation_indices]
    elif not isinstance(equation_indices, Iterable):
        raise ValueError(f"Invalid equation_indices: {equation_indices}")

    if which == "synthetic":
        if equation_indices is None:
            equation_indices = range(0, 41)
        else:
            if max(equation_indices) > 40 or min(equation_indices) < 0:
                raise ValueError("Synthetic dataset numbering starts at 0 and goes up to 40.")
        if fpath is None:
            fpath = os.path.join(os.path.dirname(__file__), 'LaSR-SyntheticEquations.csv')
        equations = load_synthetic_equations(fpath, equation_indices, forbid_ops)
    elif which == "feynman":
        if fpath is None:
            fpath = os.path.join(os.path.dirname(__file__), 'LaSR-FeynmanEquations.csv')
        if max(equation_indices) > 100 or min(equation_indices) < 1:
            raise ValueError("Feynman dataset numbering starts at 1 and goes up to 100.")
        equation_indices = set() if equation_indices is None else set(equation_indices)
        equations = load_feynman_equations(fpath, skip_equations=set(range(1, 101)) - equation_indices)
    elif which == "pysr-difficult":
        if fpath is None:
            fpath = os.path.join(os.path.dirname(__file__), 'PySR-difficultEquations.csv')
        if max(equation_indices) > 3360 or min(equation_indices) < -3360:
            raise ValueError("PySR-difficult dataset numbering starts at 0 and goes up to 3360.")
        equations = load_pysr_equations(fpath, equation_indices, forbid_ops)
    else:
        raise ValueError(f"Invalid dataset type: {which}")

    return sample_datasets(equations, num_samples, noise, add_extra_vars)

def load_synthetic_equations(
    fpath: str, idx: Iterable[int], forbid_ops: Iterable[str] | None = None
) -> List[Tuple[int, Tuple[str, Dict[str, Tuple[str, List[float]]]]]]:
    EXPRESSIONS_RAW = pd.read_csv(fpath)['equation'].to_list()
    variable_distr = {f'y{j}': ('uniform', (1, 10)) for j in range(1, 6)}
    return [
        (i, (f'y = {EXPRESSIONS_RAW[i]}', variable_distr)) 
        for i in sorted(idx) 
        if forbid_ops is None or all(op not in EXPRESSIONS_RAW[i] for op in forbid_ops)
    ]

def load_pysr_equations(
    fpath: str, idx: Iterable[int], forbid_ops: Iterable[str] | None = None
) -> List[Tuple[int, Tuple[str, Dict[str, Tuple[str, List[float]]]]]]:
    EXPRESSIONS_RAW = pd.read_csv(fpath)['true_equation'].to_list()
    variable_distr = {f'x{j}': ('uniform', (1, 10)) for j in range(1, 6)}
    return [
        (i, (f'y = {EXPRESSIONS_RAW[i]}', variable_distr)) 
        for i in sorted(idx) 
        if forbid_ops is None or all(op not in EXPRESSIONS_RAW[i] for op in forbid_ops)
    ]

def load_feynman_equations(
    fpath: str, skip_equations: Iterable[int] = None
) -> List[Tuple[int, Tuple[str, Dict[str, Tuple[str, List[float]]]]]]:
    """
    From LaSR codebase. TODO: Rework and use Pandas instead of CSV.
    """
    dataset = []
    skip_equations = set() if skip_equations is None else set(skip_equations)
    with open(fpath) as file_obj:
        heading = next(file_obj)
        reader_obj = csv.reader(file_obj)
        for row in reader_obj:
            if row[1] == "":
                break

            id = int(row[1])
            if id in skip_equations:
                continue

            output = row[2]
            formula = row[3]
            num_vars = (row.index("") - 5) // 3

            bounds = {output: None}
            for i in range(num_vars):
                var_name = row[5 + (3 * i)]
                var_bounds = (int(row[6 + (3 * i)]), int(row[7 + (3 * i)]))
                var_sample = "uniform"
                bounds[var_name] = (var_sample, var_bounds)

            dataset.append((id, (output + " = " + formula, bounds)))

    dataset.sort()
    return dataset

def create_dataset_from_expression(
    expr: str, num_samples: int, noise: float
) -> SyntheticDataset:
    eq_str = f'y = {expr}'

    # Extract variables x0-x9 from expression
    variable_distr = {f'x{i}': ('uniform', (1, 10)) for i in range(10) if f'x{i}' in expr}
    X, y, var_order = sample_equation(eq_str, variable_distr, num_samples, noise, add_extra_vars=False)
    return SyntheticDataset(idx=0, equation=eq_str, X=X, y=y, var_order=var_order)
