import pysr
import numpy as np
from typing import Dict, Union

def get_mutation_stats():
    raw = pysr.julia_import.SymbolicRegression.NeuralMutationsModule.get_mutation_stats()
    stats = {}
    for k in filter(lambda x: not x.startswith("_"), dir(raw)):
        val = getattr(raw, k)
        if isinstance(val, pysr.julia_import.VectorValue):
            stats[k] = val.to_numpy()
        else:
            stats[k] = val
    return stats

def reset_mutation_stats():
    pysr.julia_import.SymbolicRegression.NeuralMutationsModule.reset_mutation_stats_b()

def print_summary_stats(stats: Dict[str, Union[int, float, np.ndarray]]):
    print("Summary statistics:")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v}")
        else:  # numpy array
            if len(v) > 0:
                if np.issubdtype(v.dtype, np.number):
                    valid_mask = np.isfinite(v)
                    mean = v[valid_mask].mean()
                    std = v[valid_mask].std()
                    print(f"{k}_mean: {mean:.1f}")
                    print(f"{k}_std: {std:.1f}")
                    print(f"{k}_ninvalid: {len(v) - valid_mask.sum()}")
                else:  # Probably string
                    print(f"{k}: {v[:max(len(v), 10000)]}")
            else:
                print(f"{k}: {v}")
