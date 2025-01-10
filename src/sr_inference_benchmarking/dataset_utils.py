import numpy as np
from typing import Tuple, Dict, List, Iterable
from dataclasses import dataclass
import csv
import os

@dataclass
class SyntheticDataset:
    idx: int
    equation: str  # Including 'y = '
    X: np.ndarray
    Y: np.ndarray
    var_order: Dict[str, str]

    def __repr__(self):
        return f'SyntheticDataset(idx={self.idx}, equation="{self.equation}", X={self.X.shape}, Y={self.Y.shape}, var_order={self.var_order})'


# useful constants. Required for sample_equation -> eval(lambda ...) to work.
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

    Y = list()
    X_temp = np.transpose(np.stack([x[1] for x in out]))
    for i in range(num_samples):
        Y.append(expr_as_func(*list(X_temp[i])))
    Y = np.array(Y)
    Y = Y + np.random.normal(0, np.sqrt(np.square(Y).mean()) * noise, Y.shape)

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

    return X, Y, var_order

def sample_datasets(equations: List[Tuple[int, Tuple[str, Dict[str, Tuple[str, List[float]]]]]], num_samples: int, noise: float, add_extra_vars: bool) -> List[SyntheticDataset]:
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

def get_synthetic_equations(idx: Iterable[int]):
    EXPRESSIONS_RAW = [
        "exp(((1.485035504085099 - log(y2)) / (-0.5917667741788188 - y4)) + (sqrt(y1 + y4) + sqrt(y2)))",
        "((y1 + -0.2281497263470263) + (sqrt(y3) * (y1 * ((0.6069324861405777 - y3) * y3)))) / cos(cos(y1))",
        "sqrt(exp(y1) * y1) * (cos(y1) + (y1 + y1))",
        "(((exp(y3) + -0.12616306381744816) - (y4 + 0.030015044238449563)) * (-0.10809170430638036 * cos(sqrt(y2) / exp(y4)))) - 1.3429435542734653",
        "exp(log(y4 + (y2 * y1))) * ((cos(y2) - sqrt(y1)) + ((y3 / -1.3646587706285505) - 0.1494997716589463))",
        "(y3 + y2) * ((((y3 + 1.2909882620958377) * y2) + (y1 + y3)) - 0.4061091838418741)",
        "(sqrt(y2) + exp(-1.3938382096701456 + sqrt(y2))) * ((y4 * y3) * log(y3))",
        "((0.3850755754421437 * y3) * exp(sqrt(y2) - y1)) / (-0.09373867256287108 / (y2 + y1))",
        "(((y2 + 0.8837196826797421) * sqrt(y1)) / -1.5836242846038588) * sqrt(exp(y1) / y1)",
        "y1 * (((y1 * y1) / (2.189911201366985 / cos((1.2114819663272414 - y4) + -0.20111570724898717))) / exp(-0.08661496242802426 * y5))",
        "exp(log(y3) * sqrt(y2)) - exp(sqrt((y3 + y4) + (y1 * y2)) * 0.22969901281819288)",
        "exp(sqrt((y3 + y2) + (y1 / 1.796444089382479)))",
        "log(0.2530649444334089 * y2) * (y1 - sqrt(exp(y3 - y1)))",
        "(((((sqrt(y1) / exp(y2)) + 1.1081588306633263) - (y1 * y2)) * log(y1)) + y2) * y1",
        "(y1 * exp(sqrt(y1) + cos(y1 / 1.1226221361137747))) * cos((0.849151055078886 - y1) * (0.9750675999994569 * y1))",
        "(exp(y1) + (y4 - 0.6051935187513243)) / sqrt(exp(y1 * 0.35644852479483746) + y3)",
        "((-0.9527141152352505 * y1) * (y2 + y3)) - ((y3 / 0.28681680743642923) * (exp(cos(y2 - 0.3200766662509227)) - 1.0430778756174919))",
        "(y2 + (y5 + -0.6496448318659299)) + ((((y2 - y1) + y3) * exp(sqrt(y4))) + 0.4531634210300153)",
        "((y1 + y3) / cos(sqrt(y1 + y4))) / exp(1.0829032955388451 + y1)",
        "(1.2493094004268066 - y1) - (log(sqrt(y3)) * (y4 * ((-0.2586753997576587 * y4) * (y3 - cos(y4)))))",
        "(y1 * (y1 - -0.2997757954427672)) * ((1.615015710213039 * (y1 / -1.259234825417202)) - cos(cos(y1 / 0.7034932537870883)))",
        "log((sqrt(y3) + -0.6783545650051788) + exp(y2)) / (-0.13494117998537 / y4)",
        "(((-0.7020083837396962 * (y1 * y1)) / (y1 - (log(y5) + y5))) / exp(sqrt(y5))) * 0.26068741801247797",
        "(-0.8531552969749455 * (y2 * (-0.9826822517958028 * log(y2)))) * exp(sqrt((y3 + 1.4730564073819723) - -0.7225654101129367))",
        "(-0.781987400765949 / exp(y1)) * (((-0.6868296552024491 - y3) / (log(y2) - (exp(cos(y1)) * y2))) / y2)",
        "(((y2 * y5) - -0.04091973891240853) + exp(y5 * 0.8492536663999203)) - (y3 / cos(cos(y5 * y4)))",
        "(cos(log(y2)) * exp(y2)) / ((y2 * 0.8866496129486751) + (-0.12233894135460263 + y1))",
        "(sqrt(exp(y1) - 0.4324725952426049) + cos(y1)) * (sqrt(y1) * y1)",
        "((sqrt(y2) * y2) * y4) + (sqrt(sqrt(y3)) * (((y2 / 4.378755308349022) - y3) / 0.1201673737199319))",
        "((y2 - y1) * cos(sqrt(y2))) * ((-1.5290360532523042 / exp((-1.5532664019564584 - y2) * 0.5494791521253727)) - 1.8329895143477763)",
        "(cos(y1 * -1.487605784281181) - (0.4644107533597082 - (y1 * y1))) * (y2 / sqrt(y1))",
        "((y3 + (0.6558304543577124 + (0.7621321253168721 + y2))) * (0.06492075219036358 * (y2 - (y3 - y2)))) * (y2 + -2.208176675173205)",
        "(cos(0.9728047253922968 + y5) * (exp(y5) / ((y5 * -0.9321235805389282) - y3))) * (1.131234209598357 + (0.5756313995888583 * y3))",
        "sqrt(y3 * y2) / ((-0.9232383232272274 / ((1.2349525541432025 / log(y1)) + y2)) - (y3 / y4))",
        "(((sqrt(exp(y1)) * cos(y2)) / -0.333327335830982) * cos(y2)) - exp(-2.0178108091651836 * (y3 / 1.9978366796712463))",
        "sqrt(exp(y1) - -0.7391706051004766) - (y1 * y1)",
        "exp(cos(y3)) * ((y3 + (0.689367154622428 / exp(y4))) * (y2 / 0.23835709388480572))",
        "exp(0.5468331960693928 * (((0.829867349578025 + (y1 - (y1 * -2.702140567522345))) + sqrt(y1)) / exp(y1 + -1.6079113070964928)))",
        "exp(((y1 * 1.3985029137652467) / 0.5126851665088263) / (cos(y1) + sqrt(y1 * y1)))",
        "(y3 * (cos(cos(y1 + y4)) * y3)) * (-0.03533937466533261 * (y3 + exp(cos(y1))))",
        "(((y1 * exp(sqrt(y2))) + log(y1)) * cos(sqrt(y2) * -0.9577782191948018)) - sqrt(exp(y2))",
    ]
    variable_distr = {f'y{j}': ('uniform', (1, 10)) for j in range(1, 6)}

    return [(i, (f'y = {EXPRESSIONS_RAW[i]}', variable_distr)) for i in sorted(idx)]

def load_datasets(which: str, num_samples: int, noise: float, equation_indices: Iterable[int] = None, add_extra_vars: bool = False, feynman_fpath: str = None):
    if which == "synthetic":
        if equation_indices is None:
            equation_indices = range(0, 41)
        else:
            if max(equation_indices) > 40 or min(equation_indices) < 0:
                raise ValueError("Synthetic dataset numbering starts at 0 and goes up to 40.")
        equations = get_synthetic_equations(equation_indices)
    elif which == "feynman":
        if feynman_fpath is None:
            feynman_fpath = os.path.join(os.path.dirname(__file__), "data", "FeynmanEquations.csv")
        if max(equation_indices) > 100 or min(equation_indices) < 1:
            raise ValueError("Feynman dataset numbering starts at 1 and goes up to 100.")
        equation_indices = set() if equation_indices is None else set(equation_indices)
        equations = load_feynman_equations(feynman_fpath, skip_equations=set(range(1, 101)) - equation_indices)
    else:
        raise ValueError(f"Invalid dataset type: {which}")

    return sample_datasets(equations, num_samples, noise, add_extra_vars)

def load_feynman_equations(fpath: str, skip_equations: Iterable[int] = None):
    """
    From LaSR codebase.
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

def create_dataset_from_expression(expr: str, num_samples: int, noise: float):
    eq_str = f'y = {expr}'

    # Extract variables x0-x9 from expression
    variable_distr = {f'x{i}': ('uniform', (1, 10)) for i in range(10) if f'x{i}' in expr}
    X, Y, var_order = sample_equation(eq_str, variable_distr, num_samples, noise, add_extra_vars=False)
    return SyntheticDataset(idx=0, equation=eq_str, X=X, Y=Y, var_order=var_order)