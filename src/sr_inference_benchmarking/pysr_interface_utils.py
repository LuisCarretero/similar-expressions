import pysr

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

def print_summary_stats(stats):
    print("Summary statistics:")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v}")
        else:  # numpy array
            if len(v) > 0:
                mean = v.mean()
                std = v.std()
                print(f"{k}_mean: {mean:.1f}")
                print(f"{k}_std: {std:.1f}")
            else:
                print(f"{k}: {v}")
