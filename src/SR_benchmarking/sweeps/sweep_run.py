# Single sweep run consisting of multiple SR runs on different datasets but with same set of hyperparameters.
from pathlib import Path
import sys

# Add SR_benchmarking to path if not running in module mode
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from run.run_multiple import run_equations_pooled, get_run_prefix


if __name__ == "__main__":
    log_dir = '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/univar_sweep_neuralParams'
    run_prefix = get_run_prefix(log_dir)
    run_equations_pooled(log_dir, run_prefix)
