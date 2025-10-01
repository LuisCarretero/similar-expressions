# Optuna Sweep Refactoring - 3-Layer Architecture

## Overview

This directory contains a refactored version of the Optuna distributed hyperparameter optimization system with a clean 3-layer architecture that separates concerns and improves maintainability.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: OptunaHyperoptRunner (optuna_hyperopt.py)         │
│ • Study management + trial execution (864 lines)            │
│ • Hyperparameter sampling                                   │
│ • Master/worker routing                                     │
│ • Results aggregation                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ uses
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: DistributedTrialExecutor (distributed_executor.py) │
│ • Generic distributed coordination (595 lines)              │
│ • File-based master/worker communication                    │
│ • Asynchronous batch reporting                              │
│ • No SR-specific code                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ uses
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: SRBatchRunner (sr_runner.py)                       │
│ • Pure SR execution (261 lines)                             │
│ • PySR model created once per trial                         │
│ • Multi-level resumption (trial, equation, run)             │
│ • No Optuna or coordination knowledge                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Improvements

### 1. Clear Separation of Concerns
- **Layer 1 (SR execution)**: No knowledge of Optuna or distributed coordination
- **Layer 2 (Coordination)**: Generic, reusable for any experiment type
- **Layer 3 (Optuna)**: Orchestrates everything, combining study + objective

### 2. Simplified Design
- **3 layers instead of 4**: Objective function is naturally a method within study manager
- **Single source of truth**: File-based `processed_trials.json` written only by master
- **No shutdown signal file**: Uses `interruption_manager` everywhere

### 3. Asynchronous Reporting
- **No batch-wise synchronization**: Workers send incremental results, master scans asynchronously
- **Report each batch individually**: Master reports after each of its batches + worker batches as discovered
- **Better resource utilization**: Workers don't wait for master between batches

### 4. Enhanced Resumption
- **Trial-level**: Incomplete trials detected via `.done` marker files
- **Equation-level**: `SRBatchRunner._scan_existing_results()` checks for completed equations
- **Run-level**: Individual runs within equations can be reused

## File Structure

```
sweeps_refactor/
├── sr_runner.py                    # Layer 1: Pure SR execution
├── distributed_executor.py         # Layer 2: Generic coordination
├── optuna_hyperopt.py             # Layer 3: Optuna orchestration
├── slurm_optuna_distributed.sh    # Entry point (updated)
└── README.md                       # This file
```

## Usage

### Basic Usage

```bash
# Single command to run distributed optimization
sbatch sweeps_refactor/slurm_optuna_distributed.sh sweeps/optuna_neural_config.yaml
```

### Monitoring

```bash
# Watch master node log
tail -f sweeps_refactor/logs/optuna_distributed-JOBID_0.out

# Watch worker node log
tail -f sweeps_refactor/logs/optuna_distributed-JOBID_1.out

# Check job status
squeue -u $USER

# View Optuna dashboard
optuna-dashboard sqlite:///sweeps/optuna_neural_study.db
```

### Manual Testing

```bash
# Run test suite (validates graceful shutdown, resumption, communication)
cd tests/
./run_refactor_test.sh
```

## What Gets Tested

The test suite (`tests/slurm_optuna_test.sh`) validates:

1. **Distributed Communication**
   - Master broadcasts trial params
   - Workers acknowledge receipt
   - Workers send incremental results
   - Master collects and reports all results

2. **Graceful Shutdown**
   - SLURM signal (USR1) received 2min before timeout
   - Python `signal_manager` catches signal
   - Workers finish current equation batch
   - Master waits for worker completion
   - Job requeues automatically

3. **Trial Resumption**
   - Incomplete trials detected via missing `.done` marker
   - Trial parameters reconstructed from saved params
   - Equations with all runs complete are skipped
   - Individual runs within incomplete equations are reused

4. **Batch Reporting**
   - Master reports after each batch (2 equations)
   - Workers report incrementally as batches complete
   - Master scans for new worker batches asynchronously
   - Optuna receives updates for early pruning

## Key Design Decisions

### Layer 2: Generic Coordination

**No SR-specific code** - Works with any runner type via factory functions:

```python
def create_runner(params):
    ms, no, mw = reconstruct_objects(params)
    return SRBatchRunner(ms, no, mw, dataset_config, trial_id)

executor.execute_trial_with_batching(
    trial_params, equations, batch_size,
    create_runner,  # Factory function
    report_callback,
    interruption_flag
)
```

### Layer 3: Combined Study + Objective

**Objective as method** instead of separate class:

```python
class OptunaHyperoptRunner:
    def optimize(self, resume=False):
        if is_worker:
            self._run_worker_loop()
        else:
            self._run_master_optimization(resume)

    def _run_master_optimization(self, resume):
        study.optimize(self._objective, ...)

    def _objective(self, trial):
        # Direct access to self.study, self.config, etc.
        # No need to pass state between classes
        ...
```

### Asynchronous Batch Reporting

**Master reports each batch individually**:

```python
# Master loop
for master_batch in master_equations:
    batch_results = runner.run_equations(batch)
    report_callback(batch_results, "master_batch_i")

    # Scan and report new worker batches
    for worker_id in workers:
        worker_batches = get_new_worker_batches(worker_id)
        for batch in worker_batches:
            report_callback(batch, f"worker{id}_batch_i")
```

## Migration from Old Code

### What Changed

| Old Code | New Code | Change |
|----------|----------|--------|
| `optuna_coordinator.py` | `distributed_executor.py` | Removed SR-specific code, made generic |
| `optuna_objective.py` | `optuna_hyperopt.py` | Merged into study manager as `_objective()` method |
| `optuna_hyperopt.py` | `optuna_hyperopt.py` | Combined with objective, added factory functions |
| N/A | `sr_runner.py` | Extracted SR execution logic |

### What Stayed the Same

- ✅ SLURM script interface (same command-line args)
- ✅ Config file format (no changes needed)
- ✅ Signal handling mechanism (`signal_manager.py`)
- ✅ File-based communication protocol
- ✅ Trial resumption behavior

## Coordination Files

Files created during execution:

### Per Trial (ephemeral, cleaned up after trial)
```
optuna_coord_{study_name}/
├── trial_{trial_id}_params.json            # Broadcast by master
├── trial_{trial_id}_worker_{id}_ack.json   # Sent by workers
├── trial_{trial_id}_worker_{id}_results.json  # Incremental results
└── trial_{trial_id}.done                   # Created on completion
```

### Global (persistent across trials)
```
optuna_coord_{study_name}/
└── processed_trials.json                   # Written by master only
```

### Trial Data (persistent, for resumption)
```
optuna_experiment/
└── trial_{trial_id}/
    ├── {dataset}_eq{idx}_run{i}/           # Individual runs
    │   ├── tensorboard_scalars.csv
    │   └── ...
    └── ...
```

## Debugging

### Check Coordination State

```bash
COORD_DIR="/cephfs/store/.../optuna_coord_{study_name}"

# Check processed trials
cat $COORD_DIR/processed_trials.json

# Check active trial params
ls -lah $COORD_DIR/trial_*_params.json

# Check worker acknowledgments
ls -lah $COORD_DIR/trial_*_worker_*_ack.json

# Check worker results
ls -lah $COORD_DIR/trial_*_worker_*_results.json
```

### Check Trial Progress

```bash
TRIAL_DIR="/cephfs/store/.../optuna_experiment"

# List all trials
ls -d $TRIAL_DIR/trial_*/

# Check specific trial
ls -lah $TRIAL_DIR/trial_{trial_id}/

# Count completed runs for equation
ls $TRIAL_DIR/trial_{trial_id}/*_eq{idx}_run*/tensorboard_scalars.csv | wc -l
```

### Check Logs

```bash
# Master log
tail -100 sweeps_refactor/logs/optuna_distributed-JOBID_0.out

# Worker log
tail -100 sweeps_refactor/logs/optuna_distributed-JOBID_1.out

# Search for errors
grep -i error sweeps_refactor/logs/optuna_distributed-JOBID_*.out

# Search for interruption
grep -i interrupt sweeps_refactor/logs/optuna_distributed-JOBID_*.out
```

## Performance Characteristics

### Parallelization
- **Master + N workers**: Each processes 1/N of equations
- **No synchronization**: Asynchronous batch reporting
- **Linear scaling**: 2 nodes = ~2x throughput

### Resumption Overhead
- **Trial-level**: Minimal (just file existence check)
- **Equation-level**: ~O(equations × n_runs) file checks
- **Run-level**: One CSV read per completed run

### Memory Usage
- **PySR model**: Created once per trial per node (~MB)
- **Coordination files**: Tiny (~KB per trial)
- **Trial data**: Depends on equation complexity

## Known Limitations

1. **Fixed node count**: `total_nodes` must match SLURM array size
2. **No dynamic scaling**: Can't add/remove nodes mid-trial
3. **No fault tolerance**: Worker failure blocks master (timeout 2h)
4. **Assumes shared filesystem**: All nodes need access to coordination directory

## Future Improvements

Potential enhancements (not implemented):

1. **Dynamic worker pool**: Support variable number of workers
2. **Fault tolerance**: Continue if some workers fail
3. **Better load balancing**: Distribute equations based on complexity
4. **Incremental checkpointing**: Save progress mid-equation batch
5. **Metrics dashboard**: Real-time progress visualization

## References

- Design documents: `.claude/refactoring_design/`
- Original code: `sweeps/` (preserved for comparison)
- Tests: `tests/slurm_optuna_test.sh`
- Signal handling: `run/signal_manager.py`
