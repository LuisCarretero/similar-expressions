"""
Dataset utilities. Some of the code is from the LaSR codebase.
"""

import numpy as np
from typing import Tuple, Dict, List, Iterable, Literal
from dataclasses import dataclass
import os
import pandas as pd
import re
import sympy as sp

# Dataset configuration constants
DATASET_CONFIGS = {
    'synthetic': {
        'filename': 'LaSR-SyntheticEquations.csv',
        'max_index': 40,  # 0-based: 0-40 (41 equations)
        'equation_col': 'equation',
        'equation_format': 'y = {expression}',
        'extract_vars': True
    },
    'feynman': {
        'filename': 'LaSR-FeynmanEquations.csv',
        'max_index': 99,  # 0-based: 0-99 (100 equations)
        'equation_col': 'Formula',
        'output_col': 'Output',
        'equation_format': '{output} = {expression}',
        'extract_vars': False  # Use v1_name/v1_low/v1_high columns
    },
    'pysr-difficult': {
        'filename': 'PySR-difficultEquations.csv',
        'max_index': 3360,  # 0-based: 0-3359 (3360 equations)
        'equation_col': 'true_equation',
        'equation_format': 'y = {expression}',
        'extract_vars': True
    },
    'pysr-univariate': {
        'filename': 'PySR-univariate.csv',
        'max_index': 2015,  # 0-based: 0-2015 (2016 equations)
        'equation_col': 'true_equation',
        'equation_format': 'y = {expression}',
        'extract_vars': True
    }
}
DEFAULT_VARIABLE_BOUNDS = (1, 10)


@dataclass
class SyntheticDataset:
    eq_idx: int
    equation: str
    X: np.ndarray
    y: np.ndarray
    var_order: Dict[str, str]
    expression: str | None = None

    def __repr__(self):
        return f'SyntheticDataset(eq_idx={self.eq_idx}, equation="{self.equation}", X={self.X.shape}, y={self.y.shape}, var_order={self.var_order})'


@dataclass
class VariableDistribution:
    """Represents a variable's sampling distribution and parameters."""
    method: Literal['uniform', 'normal', 'loguniform', 'lognormal', 'constant']
    params: Tuple[float, float]  # (min, max) for uniform, (mean, std) for normal, etc.

    def __repr__(self):
        return f'VariableDistribution(method="{self.method}", params={self.params})'


def _sample_equation(
    equation: str, 
    var_distributions: Dict[str, VariableDistribution], 
    num_samples: int, 
    rel_noise_magn: float, 
    add_extra_vars: bool
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Sample values for variables according to their distributions and evaluate the equation.
    From LaSR codebase, updated to use VariableDistribution objects.
    """
    # Sample values for all variables
    var_names = list(var_distributions.keys())
    var_data = np.column_stack([
        _sample_from_distribution(var_dist.method, var_dist.params, num_samples=num_samples)
        for var_dist in var_distributions.values()
    ])

    # Parse and evaluate expression using sympy
    expr = equation.split(" = ")[1].replace("^", "**")
    var_symbols = [sp.Symbol(name) for name in var_names]
    sympy_expr = sp.sympify(expr)
    expr_as_func = sp.lambdify(var_symbols, sympy_expr, modules=['numpy'])

    # Evaluate expression vectorized
    y = expr_as_func(*var_data.T)

    # Handle case where expression evaluates to a constant (returns scalar instead of array)
    if np.isscalar(y):
        y = np.full(num_samples, y)

    # Noise according to LaSR, which itself cites https://pmc.ncbi.nlm.nih.gov/articles/PMC11074949/
    abs_noise_magn = np.sqrt(np.square(y).mean()) * rel_noise_magn
    y += np.random.normal(0, abs_noise_magn, y.shape)

    # Handle extra variables if needed
    if add_extra_vars:
        num_extra = 11 - len(var_distributions)
        extra_var_names = [chr(ord("A") + c) for c in range(num_extra)]
        extra_var_data = np.column_stack([
            _sample_from_distribution("uniform", (1, 20), num_samples=num_samples)
            for _ in range(num_extra)
        ])
        
        # Combine original and extra variables
        all_var_names = var_names + extra_var_names
        all_var_data = np.column_stack([var_data, extra_var_data])
        
        # Shuffle the order
        shuffle_indices = np.random.permutation(len(all_var_names))
        all_var_names = [all_var_names[i] for i in shuffle_indices]
        all_var_data = all_var_data[:, shuffle_indices]
    else:
        all_var_names = var_names
        all_var_data = var_data

    # Create final output
    var_order = {"x" + str(i): all_var_names[i] for i in range(len(all_var_names))}
    X = all_var_data

    return X, y, var_order

def _sample_from_distribution(
    distr: Literal['uniform', 'normal', 'loguniform', 'lognormal', 'constant', 'linspace'], 
    params: Tuple[float, float], 
    num_samples: int
) -> np.ndarray:
    """
    Samples x values from specified distribution.
    """
    if distr == "constant":  # const
        return np.full(num_samples, params[0])
    elif distr == "uniform":  # (low, high)
        return np.random.uniform(low=params[0], high=params[1], size=num_samples)
    elif distr == "normal":  # (mean, std)
        return np.random.normal(loc=params[0], scale=params[1], size=num_samples)
    elif distr == "loguniform":  # logU(a, b) ~ exp(U(log(a), log(b))
        return np.exp(
            np.random.uniform(low=np.log(params[0]), high=np.log(params[1]), size=num_samples)
        )
    elif distr == "lognormal":  # ln of var is normally distributed
        return np.random.lognormal(mean=params[0], sigma=params[1], size=num_samples)
    elif distr == "linspace":  # (start, stop)
        return np.linspace(start=params[0], stop=params[1], num=num_samples)
    else:
        raise ValueError(f"Invalid distribution: {distr}")

def _extract_variable_names(equation_text: str) -> set:
    """Extract variable names from equation text, excluding mathematical functions."""
    var_pattern = r'\b[a-zA-Z][a-zA-Z0-9_]*\b'
    potential_vars = set(re.findall(var_pattern, equation_text))
    
    math_functions = {
        'sin', 'cos', 'tan', 'exp', 'log', 'ln', 'sqrt', 'abs',
        'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan',
        'pi', 'e', 'inf', 'nan'
    }
    
    return potential_vars - math_functions

def _extract_feynman_distributions(row: pd.Series) -> Dict[str, VariableDistribution]:
    """
    Extract variables from Feynman dataset row using v1_name/v1_low/v1_high columns.

    Args:
        row: Pandas Series representing a row from Feynman dataset

    Returns:
        Dictionary of variable names to VariableDistribution objects
    """
    variables = {}
    for i in range(1, 11):  # v1 through v10
        name_col = f'v{i}_name'
        low_col = f'v{i}_low'
        high_col = f'v{i}_high'

        if pd.notna(row[name_col]):  # Variable exists
            var_name = row[name_col]
            var_low = float(row[low_col])
            var_high = float(row[high_col])
            variables[var_name] = VariableDistribution("uniform", (var_low, var_high))

    return variables

def _format_equation_text(row: pd.Series, config: dict) -> Tuple[str, str]:
    """
    Format equation text according to dataset configuration.

    Args:
        row: Pandas Series representing a dataset row
        config: Dataset configuration dictionary

    Returns:
        Tuple of (formatted equation string, raw expression)
    """
    expression = row[config['equation_col']]

    if 'output_col' in config:  # Feynman case
        output = row[config['output_col']]
        formatted_equation = config['equation_format'].format(output=output, expression=expression)
        return formatted_equation, expression
    else:
        formatted_equation = config['equation_format'].format(expression=expression)
        return formatted_equation, expression

def load_datasets(
    which: Literal['synthetic', 'feynman', 'pysr-difficult', 'pysr-univariate'],
    num_samples: int,
    rel_noise_magn: float,
    equation_indices: Iterable[int] | int | None = None,
    add_extra_vars: bool = False,
    fpath: str | None = None,
    remove_op_equations: Iterable[str] | None = None,
) -> List[SyntheticDataset]:
    """Load datasets and generate synthetic data from equation specifications."""

    # Get dataset configuration
    config = DATASET_CONFIGS[which]

    # Normalize equation_indices to list
    if isinstance(equation_indices, int):
        equation_indices = [equation_indices]
    elif equation_indices is not None and not isinstance(equation_indices, Iterable):
        raise ValueError(f"Invalid equation_indices: {equation_indices}")

    # Validate equation_indices bounds (0-based for all datasets)
    if equation_indices is not None:
        max_allowed = config['max_index']
        if max(equation_indices) > max_allowed or min(equation_indices) < 0:
            raise ValueError(f"{which} indices must be 0-{max_allowed}")
    else:
        equation_indices = list(range(config['max_index'] + 1))

    # Load CSV file
    if fpath is None:
        fpath = os.path.join(os.path.dirname(__file__), config['filename'])
    df = pd.read_csv(fpath)

    # Filter by equation_indices
    df = df.iloc[equation_indices]

    # Apply remove_op_equations filter if specified
    if remove_op_equations is not None:
        eq_col = config['equation_col']
        for op in remove_op_equations:
            df = df[~df[eq_col].str.contains(op, na=False)]

    # Sort by index
    df = df.sort_index()

    # Create EquationSpecs using unified processing
    datasets = []
    for eq_idx, row in df.iterrows():
        equation, expression = _format_equation_text(row, config)

        if config['extract_vars']:
            # Extract variables from equation text
            variables = _extract_variable_names(expression)
            var_distributions = {var: VariableDistribution('uniform', DEFAULT_VARIABLE_BOUNDS) for var in variables}
        else:
            # Use Feynman-specific variable extraction
            var_distributions = _extract_feynman_distributions(row)

        X, Y, var_order = _sample_equation(equation, var_distributions, num_samples, rel_noise_magn, add_extra_vars=add_extra_vars)
        datasets.append(SyntheticDataset(eq_idx, equation, X, Y, var_order, expression))

    return datasets

def create_dataset_from_expression(expr: str, num_samples: int, rel_noise_magn: float) -> SyntheticDataset:
    eq_str = f'y = {expr}'
    variables = {
        f'x{i}': VariableDistribution('uniform', DEFAULT_VARIABLE_BOUNDS) 
        for i in range(10) if f'x{i}' in expr
    }
    X, y, var_order = _sample_equation(eq_str, variables, num_samples, rel_noise_magn, add_extra_vars=False)
    return SyntheticDataset(eq_idx=0, equation=eq_str, X=X, y=y, var_order=var_order, expression=expr)

# ----------------

def calculate_equation_complexity(equation: str) -> int:
    """
    Calculate the complexity of an equation by counting nodes in the unsimplified equation tree.
    Complexity = number of operators + number of constants + number of variables

    Args:
        equation: Mathematical equation as a string

    Returns:
        Integer representing the complexity (total number of nodes)
    """
    # Clean the equation string
    equation = equation.strip()

    # First, replace variables with placeholders to avoid confusion with numbers
    # Find all variables and replace them temporarily
    variables = re.findall(r'\bx\d+\b', equation)
    temp_equation = equation
    for i, var in enumerate(set(variables)):
        temp_equation = temp_equation.replace(var, f'VAR{i}')

    # Find and replace negative constants first
    constants_with_neg = re.findall(r'(?<![a-zA-Z\d])-\d*\.?\d+(?:[eE][+-]?\d+)?(?![a-zA-Z])', temp_equation)

    # Replace each negative constant with a placeholder to avoid double counting
    for i, const in enumerate(constants_with_neg):
        temp_equation = temp_equation.replace(const, f'NEGCONST{i}', 1)  # Replace only first occurrence

    # Now find positive constants in the modified equation
    constants_pos = re.findall(r'(?<![a-zA-Z\d])\d*\.?\d+(?:[eE][+-]?\d+)?(?![a-zA-Z])', temp_equation)

    # Count binary operators: +, -, *, / (after removing negative number signs)
    binary_ops = len(re.findall(r'[+\-*/]', temp_equation))

    # Count unary functions: exp, cos, sqrt, log
    unary_ops = len(re.findall(r'\b(exp|cos|sqrt|log)\b', equation))

    # Count all variable occurrences (each usage counts as a node)
    variable_count = len(variables)

    # Count constants (positive and negative)
    constants_count = len(constants_with_neg) + len(constants_pos)

    total_complexity = binary_ops + unary_ops + variable_count + constants_count

    return total_complexity

def make_equation_univariate(equation: str) -> str:
    """
    Replace all variables in the equation with 'x' to make it univariate.
    
    Handles different variable naming conventions:
    - x_i variables (e.g., x1, x2, x3, ...) -> x
    - y_i variables (e.g., y1, y2, y3, ...) -> x  
    - Feynman variables (any single letter or letter+digit) -> x
    """
    # Split equation at '=' to get the expression part
    if ' = ' in equation:
        lhs, rhs = equation.split(' = ', 1)
        expr = rhs
    else:
        expr = equation
        lhs = 'y'
    
    # Find all variables in the expression
    variables = _extract_variable_names(expr)
    
    # Replace all variables with 'x'
    expr_univariate = expr
    for var in variables:
        # Use word boundaries to avoid partial replacements
        expr_univariate = re.sub(r'\b' + re.escape(var) + r'\b', 'x', expr_univariate)
    
    return f'{lhs} = {expr_univariate}'

def calc_equation_eval_stats(equation: str, grid_bounds: Tuple[float, float], num_points: int = 1000) -> dict:
    """
    Calculate statistics (mean, std, min, max) for an equation evaluated over a grid.

    Args:
        equation: Equation string
        grid_bounds: Tuple of (start, end) of interval
        num_points: Number of evaluation points

    Returns:
        Dictionary with statistics and metadata
    """
    try:
        # Check if equation has variables other than 'x'
        expr = equation.split(" = ")[1] if " = " in equation else equation
        variables = _extract_variable_names(expr)
        
        if len(variables) == 0:
            raise ValueError("No variables found in equation")
        
        if len(variables) > 1 or 'x' not in variables:
            raise ValueError(f"Expected univariate equation with variable 'x', found variables: {variables}")
        
        # Use single variable 'x' for univariate equations
        var_distributions = {'x': VariableDistribution('linspace', grid_bounds)}
        
        # Use _sample_equation with linspace distribution and no noise
        X, y_values, var_order = _sample_equation(
            equation, 
            var_distributions, 
            num_samples=num_points, 
            rel_noise_magn=0.0,  # No noise for grid evaluation
            add_extra_vars=False
        )
        
        # Check for invalid values (NaN, inf)
        if np.any(~np.isfinite(y_values)):
            finite_mask = np.isfinite(y_values)
            if np.sum(finite_mask) < len(y_values) * 0.5:  # Less than 50% valid values
                return {
                    'equation': equation,
                    'evaluation_error': f'Too many invalid values: {np.sum(~finite_mask)}/{len(y_values)}',
                    'std': np.nan,
                    'mean': np.nan,
                    'min': np.nan,
                    'max': np.nan
                }
            else:
                y_values = y_values[finite_mask]
                print(f"Removed {np.sum(~finite_mask)} invalid values from evaluation")
        
        return {
            'equation': equation,
            'evaluation_error': None,
            'std': float(np.std(y_values)),
            'mean': float(np.mean(y_values)),
            'min': float(np.min(y_values)),
            'max': float(np.max(y_values))
        }
        
    except Exception as e:
        return {
            'equation': equation,
            'evaluation_error': f'Evaluation error: {str(e)}',
            'std': np.nan,
            'mean': np.nan,
            'min': np.nan,
            'max': np.nan
        }