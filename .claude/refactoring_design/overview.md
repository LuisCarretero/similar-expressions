# Optuna Distributed Benchmarking Refactoring Design

## Motivation

The current codebase has grown organically and responsibilities are mixed:
- `OptunaCoordinator` does both coordination AND SR execution
- SR-specific logic mixed with generic distributed coordination
- Hard to test components in isolation
- Unclear layer boundaries

## Goals

1. **Separate concerns** into clear layers
2. **Make coordinator generic** - reusable for other experiments
3. **Improve testability** - each layer testable independently
4. **Preserve all functionality** - file communication, signal handling, resumption

## Proposed 4-Layer Architecture

```
Layer 4: OptunaStudyManager (optuna_hyperopt.py)
         ↓ creates study, runs optimization
Layer 3: OptunaObjective (optuna_objective.py)
         ↓ samples hyperparameters, handles Optuna reporting
Layer 2: DistributedTrialExecutor (distributed_executor.py) [NEW NAME]
         ↓ coordinates master/worker, file communication
Layer 1: SRBatchRunner (sr_runner.py) [NEW FILE]
         ↓ runs SR experiments, PySR model
```

## Key Design Decisions

### Layer 1: SR Execution
- **New file**: `sr_runner.py`
- **Responsibility**: Pure SR execution, no Optuna/coordination knowledge
- **Key feature**: PySR model created ONCE per trial, reused for all equations
- **Resumption**: Scans for existing results at equation and run level

### Layer 2: Distributed Execution
- **Refactored from**: `optuna_coordinator.py` → `distributed_executor.py`
- **Responsibility**: Generic distributed coordination
- **Key changes**:
  - ✅ No batch-wise synchronization (asynchronous)
  - ✅ Master reports after each sub-batch (includes all worker results available)
  - ✅ Shutdown checks in both master and worker loops
  - ✅ File-based `processed_trials` (single source of truth)
  - ✅ Only master marks trials as processed (after workers done)
  - ✅ Removed all SR-specific imports and logic

### Layer 3: Optuna Integration
- **Simplified**: `optuna_objective.py`
- **Responsibility**: Hyperparameter sampling, trial resumption, Optuna reporting
- **Key change**: Composes executor + runner via factory functions

### Layer 4: Study Management
- **Minor changes**: `optuna_hyperopt.py`
- **Responsibility**: Study lifecycle, optimization loop
- **Key change**: Renames coordinator → executor

## File Changes

### New Files
- `src/SR_benchmarking/sweeps/sr_runner.py` - Layer 1

### Refactored Files
- `src/SR_benchmarking/sweeps/distributed_executor.py` - Refactored from `optuna_coordinator.py`
- `src/SR_benchmarking/sweeps/optuna_objective.py` - Simplified
- `src/SR_benchmarking/sweeps/optuna_hyperopt.py` - Minor updates

### Deleted Files
- `src/SR_benchmarking/sweeps/optuna_coordinator.py` - Replaced by `distributed_executor.py`

### Unchanged Files
- `src/SR_benchmarking/sweeps/slurm_optuna_distributed.sh` - No changes needed

## Communication Flow (Revised)

### Master Loop
```python
# Broadcast trial params → wait for acks → create runner ONCE

for each batch in master_equations:
    # Run master's batch
    batch_results = runner.run_equations(batch_equations)

    # Scan all available worker results (non-blocking)
    worker_results = collect_all_worker_results()

    # Report combined results to Optuna
    trial.report(master_results + worker_results)

    # Check for pruning
    if trial.should_prune():
        raise TrialPruned()

# Wait for all workers to finish
final_worker_results = wait_for_all_workers()

# Final report
trial.report(all_results)

# Mark trial as processed (FILE-BASED)
mark_trial_processed(trial_id)
```

### Worker Loop
```python
while True:
    # Check shutdown signals
    if interrupted or shutdown_signal:
        break

    # Wait for trial params
    trial_params = wait_for_trial_params()

    # Check if already processed (FILE-BASED)
    if trial_id in load_processed_trials():
        continue

    # Create runner ONCE for this trial
    runner = create_runner(trial_params)

    # Get worker's equation subset

    for each batch in worker_equations:
        # Check interruption
        if interrupted:
            break

        # Run worker's batch
        batch_results = runner.run_equations(batch_equations)

        # Send incremental results to master
        send_worker_results(all_results_so_far)

    # Send final results
    send_worker_results(all_results, final=True)
```

## Benefits

✅ **Testable**: Each layer independently testable
✅ **Reusable**: `DistributedTrialExecutor` works for any experiment
✅ **Clear**: Single responsibility per class
✅ **Maintainable**: SR changes don't affect coordination
✅ **Preserves**: All file communication and signal handling unchanged
✅ **Simpler**: No batch-wise synchronization, single source of truth for processed trials

## Implementation Plan

1. Create `sr_runner.py` with `SRBatchRunner` class
2. Create `distributed_executor.py` with `DistributedTrialExecutor` class
3. Simplify `optuna_objective.py` to use executor + runner
4. Update `optuna_hyperopt.py` to use new naming
5. Test in single-node mode
6. Test in distributed mode
7. Delete old `optuna_coordinator.py`

## Open Questions for User Feedback

**Layer 3 (OptunaObjective):**
- How to structure with new executor API?
- Keep hyperparameter sampling logic as-is or refactor further?
- Keep trial resumption logic in objective or move to executor?

**Layer 4 (OptunaStudyManager):**
- Any other simplifications desired?
- Keep baseline trial enqueueing logic?

**General:**
- File organization preferences?
- Naming preferences?
- Any other concerns?

## Status

**Current**: Awaiting feedback on Layer 3/4 design before finalizing implementation plan.
