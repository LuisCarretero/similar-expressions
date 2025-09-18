import sys
sys.path.append('/cephfs/home/lc865/workspace/similar-expressions')

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

from src.SR_benchmarking.dataset import utils
from src.SR_benchmarking.analysis.utils import load_tensorboard_data

# Define experiment directories for both vanilla and neural
experiment_dirs = {
    'neural': '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/sept-baseline-neural',
    'vanilla': '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/sept-baseline-vanilla',
    'neural-2': '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/sept-baseline-2-neural',
    'vanilla-2': '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/sept-baseline-2-vanilla',
    'neural-3': '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/sept-baseline-3-neural',
    'vanilla-3': '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/sept-baseline-3-vanilla'
}

# Load data from both experiments
data_by_experiment = {}
for experiment_type, experiment_dir in experiment_dirs.items():
        
    # Get all run directories with required prefix
    run_dirs = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith("run")]
    
    # Load data from all runs
    data = []
    for run_dir in tqdm(run_dirs, desc=f"Loading {experiment_type} runs"):
        try:
            df_scalars, _ = load_tensorboard_data(os.path.join(experiment_dir, run_dir))
            df_scalars['run'] = run_dir
            data.append(df_scalars)  # [['step', 'timestamp', 'min_loss', 'pareto_volume', 'run']]
        except Exception as e:
            print(f"Error loading data for {run_dir}: {e}")
    
    if data:
        data_by_experiment[experiment_type] = pd.concat(data, ignore_index=True)
    else:
        print(f"No data found for {experiment_type}")
        data_by_experiment[experiment_type] = pd.DataFrame()

# Store each dataframe in its respective directory
for experiment_type, df in data_by_experiment.items():
    output_path = os.path.join(experiment_dirs[experiment_type], f'tensorboard_scalars.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {experiment_type} data to {output_path}")
