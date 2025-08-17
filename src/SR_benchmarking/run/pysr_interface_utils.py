import pysr
import numpy as np
from typing import Dict, Union
from numbers import Number


def get_neural_mutation_stats() -> Dict[str, Union[Number, np.ndarray]]:
    """
    Get the mutation stats from the neural mutation module.
    """
    raw = pysr.julia_import.SymbolicRegression.NeuralMutationsModule.get_mutation_stats()
    stats = {}
    for k in filter(lambda x: not x.startswith("_"), dir(raw)):
        val = getattr(raw, k)
        if isinstance(val, pysr.julia_import.VectorValue):
            stats[k] = val.to_numpy()
        else:
            stats[k] = val
    return stats

def reset_neural_mutation_stats() -> None:
    """
    Reset the mutation stats in the neural mutation module.
    """
    pysr.julia_import.SymbolicRegression.NeuralMutationsModule.reset_mutation_stats_b()

def summarize_stats_dict(stats: Dict[str, Union[Number, np.ndarray]]) -> Dict[str, Number]:
    """
    Summarize a general stats dictionary to a set of summary statistics by taking
    the mean and standard deviation of the vector values.
    """
    summary = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            try:
                if len(v) > 0:
                    tmp = {
                        f'{k}_mean': float(v.mean()),
                        f'{k}_std': float(v.std()),
                        f'{k}_ninvalid': int(len(v) - np.isfinite(v).sum())
                    }
                else:
                    tmp = {
                        f'{k}_mean': float('nan'),
                        f'{k}_std': float('nan'),
                        f'{k}_ninvalid': 0
                    }
            except Exception as e:
                print(f"Error summarizing key {k}: {e}")
            summary.update(tmp)
        elif isinstance(v, Number):
            summary[k] = v
        else:  # String?
            print(f"Warning: Key {k} is not a number or numpy array. Skipping.")
    return summary

def print_summary_stats(stats: Dict[str, Union[Number, np.ndarray]]) -> None:
    print("Summary statistics:")
    summary = summarize_stats_dict(stats)
    for k, v in summary.items():
        print(f"{k}: {v}")

def init_mutation_logger(log_dir: str, prefix: str = 'mutations') -> None:
    """
    Initialize the mutation logger which logs various statistics about each mutation
    happening during the course of the model fitting. Data is stored in memory and 
    regularly flushed to a csv file in the given directory.

    If mutation logger is not initialized, no data will be logged.

    Todo: Turn this into a context manager/object.
    """
    assert isinstance(log_dir, str), "log_dir must be a string"
    assert isinstance(prefix, str), "prefix must be a string"
    pysr.julia_import.SymbolicRegression.MutationLoggingModule.init_logger(log_dir, prefix)

def close_mutation_logger() -> None:
    """
    Close the mutation logger. Flushes any remaining data to the csv file. No data 
    will be logged after this function is called.
    """
    pysr.julia_import.SymbolicRegression.MutationLoggingModule.close_global_logger_b()
