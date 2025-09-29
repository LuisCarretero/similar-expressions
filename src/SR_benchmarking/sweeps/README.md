# Optuna Hyperparameter Optimization for Symbolic Regression

This directory contains an Optuna-based hyperparameter optimization setup for PySR symbolic regression, designed to maximize pareto volume across multiple equations.

## Files

- **`optuna_config.yaml`** - Configuration file with hyperparameter ranges and study settings
- **`optuna_objective.py`** - Objective function wrapping the existing SR pipeline
- **`optuna_hyperopt.py`** - Main Optuna optimization script
- **`slurm_optuna.sh`** - SLURM submission script with requeuing support
- **`test_optuna_setup.py`** - Validation script to test the setup

## Quick Start

### 1. Test the Setup
```bash
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking
python sweeps/test_optuna_setup.py
```

### 2. Sequential Optimization (Recommended)

**Phase 1: Neural Backend Optimization**
```bash
# Submit neural parameter optimization (8D space, ~1 day)
sbatch sweeps/slurm_optuna.sh sweeps/optuna_neural_config.yaml

# Monitor progress
tail -f sweeps/logs/optuna-<JOB_ID>.out

# View results
optuna-dashboard sqlite:///sweeps/optuna_neural_study.db
```

**Phase 2: Mutation Weights Optimization**
```bash
# After Phase 1 completes, update base config with best neural params
# Then submit weights optimization (11D simplex, ~2-3 days)
sbatch sweeps/slurm_optuna.sh sweeps/optuna_weights_config.yaml

# Monitor progress
tail -f sweeps/logs/optuna-<JOB_ID>.out

# View results
optuna-dashboard sqlite:///sweeps/optuna_weights_study.db
```

### 3. Legacy Joint Optimization
```bash
# Submit the original joint optimization job (19D space, not recommended)
sbatch sweeps/slurm_optuna.sh sweeps/optuna_config.yaml

# Monitor progress
tail -f sweeps/logs/optuna-<JOB_ID>.out

# View results
optuna-dashboard sqlite:///sweeps/optuna_study.db
```

## Configuration

### Configuration Options

**Three optimization strategies available:**

1. **Sequential Optimization (Recommended)**:
   - `optuna_neural_config.yaml`: Neural backend parameters (8D space)
   - `optuna_weights_config.yaml`: Mutation weights simplex (11D space, proper Dirichlet sampling)

2. **Legacy Joint Optimization**:
   - `optuna_config.yaml`: All parameters together (19D space, biased simplex sampling)

### Hyperparameter Space

**Neural Backend Parameters** (`optuna_neural_config.yaml`):
- `sampling_eps`: Neural sampling epsilon (0.005-0.15)
- `subtree_min_nodes`, `subtree_max_nodes_diff`: Tree size constraints
- `max_tree_size_diff`: Maximum tree size difference
- `similarity_threshold`: Expression similarity threshold (0.1-1.2)
- `max_resamples`: Maximum resampling attempts (50-300)
- Boolean flags: `require_tree_size_similarity`, `require_novel_skeleton`, `require_expr_similarity`

**Mutation Weights Simplex** (`optuna_weights_config.yaml`):
- All 11 mutation weights with proper Dirichlet distribution sampling
- Ensures unbiased exploration of the simplex (weights sum to 1)
- Parameters: `weight_add_node`, `weight_insert_node`, `weight_delete_node`, `weight_do_nothing`, `weight_mutate_constant`, `weight_mutate_operator`, `weight_swap_operands`, `weight_rotate_tree`, `weight_randomize`, `weight_simplify`, `weight_optimize`, `weight_neural_mutate_tree`

### Dataset Configuration
- **Equations**: 100 equations (1-100) from `pysr-univariate` dataset
- **Runs per equation**: 5 (for variance reduction)
- **Samples per equation**: 2000
- **Noise level**: 0.0001

## How It Works

### Objective Function
1. **Equation Batching**: Processes equations in batches of 10 for intermediate reporting
2. **Variance Reduction**: Runs each equation 5 times, uses median for robustness
3. **Pareto Volume**: Calculates pareto volume using existing analysis pipeline
4. **Final Metric**: Returns mean pareto volume across all 100 equations

### SLURM Integration
- **Requeuing**: Automatically requeues jobs that hit 12h time limit
- **Resumption**: Optuna study persists in SQLite database across requeues
- **Graceful Shutdown**: Handles USR1 signals for clean termination
- **Resource Allocation**: 40 cores + 1 GPU on lovelace-mc partition

### Noise Handling
- **Conservative Pruning**: 20 startup trials, 30 warmup steps before pruning
- **Robust Aggregation**: Median per equation, mean across equations
- **Large Sample Size**: 500 total runs (100 equations × 5 runs) per trial

## Advanced Usage

### Custom Configuration
Edit `optuna_config.yaml` to modify:
- Hyperparameter ranges and distributions
- Number of trials and batch sizes
- Pruning and sampling strategies
- Dataset and equation selection

### Resume Existing Study
```bash
# Resume a previous study
python sweeps/optuna_hyperopt.py --config optuna_config.yaml --resume
```

### Multi-Node Extension
The current setup uses single-node optimization. For multi-node:
1. Modify `slurm_optuna.sh` to use multiple nodes
2. Use distributed Optuna storage (e.g., MySQL/PostgreSQL)
3. Run multiple workers in parallel

### Parameter Importance Analysis
After sufficient trials (>10), Optuna automatically computes parameter importance:
```python
import optuna

study = optuna.load_study(study_name="sr_pareto_optimization",
                         storage="sqlite:///sweeps/optuna_study.db")
importance = optuna.importance.get_param_importances(study)
```

## Monitoring

### Dashboard Views
- **Optimization History**: Progress over time
- **Parameter Importance**: Which hyperparameters matter most
- **Parallel Coordinates**: Parameter relationships
- **Trial Details**: Individual trial results and pruning decisions

### Log Files
- **SLURM Output**: `sweeps/logs/optuna-<JOB_ID>.out`
- **Results File**: `sweeps/optuna_results.txt` (generated after completion)
- **Study Database**: `sweeps/optuna_study.db` (persistent across requeues)

## Expected Runtime

### Sequential Optimization (Recommended)
- **Phase 1 (Neural)**: ~24-30 hours (40 trials × 150 SR runs each)
- **Phase 2 (Weights)**: ~70-85 hours (70 trials × 500 SR runs each)
- **Total**: ~100-115 hours across 2 phases

### Legacy Joint Optimization
- **Per Trial**: ~3 hours (500 SR runs)
- **Total Optimization**: ~300 hours for 100 trials
- **With Pruning**: ~150-200 hours (assuming 30-50% pruning rate)

**All timings distributed across multiple 12h SLURM jobs with requeuing**

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure conda environment is activated and dependencies installed
2. **Config Not Found**: Check paths in `optuna_config.yaml` are correct
3. **Database Locked**: Only one optimization process per database
4. **SLURM Timeout**: Check requeue logic and signal handling

### Validation
Always run the test script before submitting jobs:
```bash
python sweeps/test_optuna_setup.py
```

### Recovery
If optimization is interrupted:
1. Study state is preserved in SQLite database
2. Restart with `--resume` flag
3. Use `optuna-dashboard sqlite:///sweeps/optuna_study.db` to inspect current state