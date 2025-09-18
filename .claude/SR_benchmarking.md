# Symbolic Regression (SR) Benchmarking System

This repository contains a neural-enhanced symbolic regression benchmarking system built on top of PySR. The system is designed for academic research to evaluate and optimize symbolic regression algorithms.

## Project Structure

```
src/SR_benchmarking/
├── run/                    # Core benchmarking execution
│   ├── benchmarking_utils.py    # Core utilities and data classes
│   ├── run_multiple.py         # Run multiple SR experiments
│   ├── pysr_interface_utils.py # PySR and neural stats interface
│   ├── config.yaml            # Configuration for single runs
│   └── logs/                  # Run outputs (ignore for analysis)
├── sweeps/                 # Hyperparameter optimization
│   ├── single_run.py          # Single sweep run execution
│   ├── config.yaml           # WandB sweep configuration
│   └── slurm_sweep_agent.sh  # SLURM job submission script
├── analysis/               # Results analysis
│   └── utils.py              # Data loading and analysis utilities
├── dataset/               # Dataset handling
│   └── utils.py             # Dataset creation and manipulation
├── wandb/                 # WandB logs (IGNORE - contains logs only)
├── outputs/               # Run outputs (IGNORE - contains logs only)
└── *.out files           # SLURM output files (IGNORE - logs only)
```

## Key Components

### 1. Configuration System

**Single Run Configuration** (`run/config.yaml`):
- `run_settings`: Basic run parameters (iterations, early stopping, log directory)
- `dataset`: Dataset selection (feynman, synthetic, pysr-difficult, custom)
- `symbolic_regression`: PySR settings including neural options

**Sweep Configuration** (`sweeps/config.yaml`):
- WandB Bayesian optimization setup
- Hyperparameter search spaces for mutation weights
- Optimization target: `mean-pareto_volume` (maximize)

### 2. Neural Enhancement

The system integrates ONNX models for neural-guided tree mutations:
- **Model Path**: `/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx`
- **Neural Options**: Sampling parameters, similarity thresholds, batch processing
- **Key Parameters**:
  - `sampling_eps`: Exploration vs exploitation (0.02-0.05)
  - `subtree_min/max_nodes`: Tree complexity bounds
  - `similarity_threshold`: Expression similarity requirements
  - `require_novel_skeleton`: Enforce structural diversity

### 3. Data Classes (benchmarking_utils.py)

- **`NeuralOptions`**: Neural mutation configuration
- **`MutationWeights`**: Relative weights for different mutation operators
- **`ModelSettings`**: PySR model configuration (iterations, loss, precision)
- **`DatasetSettings`**: Dataset specification and preprocessing options

### 4. Datasets

Supported dataset types:
- **`feynman`**: Physics equations from Feynman lectures
- **`synthetic`**: Generated synthetic expressions
- **`pysr-difficult`**: Challenging benchmark equations
- **`custom`**: User-defined expressions

Dataset features:
- Configurable noise levels
- Univariate conversion (replace all variables with 'x')
- Sample size control (typically 10,000 samples)
- Forbidden operators (e.g., exclude 'log' for stability)

### 5. Execution Modes

**Single Benchmarking Run**:
```bash
cd src/SR_benchmarking
python run/run_multiple.py --equations 1,2,3 --dataset feynman
```

**WandB Hyperparameter Sweep**:
```bash
# Local execution
cd src/SR_benchmarking
python -m wandb sweep sweeps/config.yaml --project simexp-SR
python -m wandb agent <sweep-id> --count 100

# SLURM execution
sbatch sweeps/slurm_sweep_agent.sh
```

**Interactive Setup**:
```bash
bash run/interactive_setup.sh
```

### 6. Logging and Analysis

**TensorBoard Integration**:
- Scalar metrics (loss, complexity, diversity)
- Expression evolution tracking
- Neural mutation statistics

**WandB Integration**:
- Distributed hyperparameter optimization
- Real-time metric monitoring
- Experiment comparison and visualization

**Analysis Utilities** (`analysis/utils.py`):
- `load_mutations_data()`: Load mutation logs
- `load_neural_stats()`: Load neural mutation summaries
- `load_tensorboard_data()`: Extract TensorBoard scalars and expressions
- `collect_sweep_results()`: Aggregate multi-run results

### 7. SLURM Configuration

The system is optimized for HPC clusters:
- **Partition**: lovelace
- **Resources**: 32 CPUs, 64GB RAM, 1 GPU
- **Time Limit**: 12 hours
- **Requeue Support**: Graceful handling of preemption
- **Environment**: Conda ML environment

## Important File Paths

- **ONNX Model**: `/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx`
- **Log Directory**: `/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/`
- **Conda Environment**: `/cephfs/store/gr-mc2473/lc865/misc/condaforge/`

## Workflow Examples

### Running a Quick Test
```bash
cd src/SR_benchmarking
python run/run_multiple.py \
    --equations 1,2,3 \
    --dataset feynman \
    --niterations 10 \
    --log_dir ./test_run
```

### Hyperparameter Optimization
```bash
cd src/SR_benchmarking
# Create sweep
python -m wandb sweep sweeps/config.yaml --project simexp-SR
# Run agents (replace with actual sweep ID)
python -m wandb agent --count 50 <sweep-id>
```

### Analyzing Results
```python
from analysis.utils import load_tensorboard_data, load_neural_stats
# Load run data
df_scalars, df_exprs = load_tensorboard_data("path/to/logs")
neural_stats = load_neural_stats("path/to/logs")
```

## Key Research Features

1. **Neural-Guided Evolution**: Uses transformer models to suggest meaningful tree mutations
2. **Diversity Enforcement**: Ensures expression novelty through similarity checking  
3. **Multivariate/Univariate Support**: Flexible dataset preprocessing
4. **Distributed Optimization**: Bayesian hyperparameter tuning via WandB
5. **Comprehensive Logging**: Detailed mutation tracking and performance metrics

## Notes for Claude

- **Ignore directories**: `wandb/`, `outputs/`, `*.out` files (these contain logs/outputs only)
- **Main execution entry points**: `run/run_multiple.py`, `sweeps/single_run.py`
- **Configuration hierarchy**: `run/config.yaml` for single runs, `sweeps/config.yaml` for optimization
- **Neural integration**: Requires ONNX model and CUDA support
- **Academic focus**: Designed for symbolic regression research and benchmarking

This system enables systematic evaluation of symbolic regression performance across different datasets, hyperparameters, and neural enhancement configurations.