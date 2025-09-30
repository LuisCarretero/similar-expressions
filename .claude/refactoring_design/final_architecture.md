# Final Refactoring Architecture - 3 Layer Design

## Overview

Clean separation into 3 layers with clear responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: OptunaHyperoptRunner (optuna_hyperopt.py)         │
│ • Study management + trial execution (combined layers 3+4) │
│ • Hyperparameter sampling                                   │
│ • Master/worker routing                                     │
│ • Results aggregation                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ uses
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: DistributedTrialExecutor (distributed_executor.py) │
│ • Generic distributed coordination                          │
│ • File-based master/worker communication                    │
│ • Asynchronous batch reporting                              │
│ • No SR-specific code                                       │
└────────────────────┬────────────────────────────────────────┘
                     │ uses
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: SRBatchRunner (sr_runner.py)                       │
│ • Pure SR execution                                         │
│ • PySR model created once per trial                         │
│ • Multi-level resumption (trial, equation, run)             │
│ • No Optuna or coordination knowledge                       │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
src/SR_benchmarking/sweeps/
├── sr_runner.py                  # Layer 1: SR execution
├── distributed_executor.py       # Layer 2: Coordination
├── optuna_hyperopt.py           # Layer 3: Optuna (combined)
└── slurm_optuna_distributed.sh  # Entry point (unchanged)
```

**Total: 3 Python files, ~1500 lines** (down from current scattered structure)

## Why 3 Layers Instead of 4?

### Original Plan (4 layers):
```
Layer 4: OptunaStudyManager    - Study lifecycle
Layer 3: OptunaObjective       - Trial execution
Layer 2: Coordinator           - Distribution
Layer 1: SR execution          - PySR
```

### Problem with 4 layers:
- ❌ Tight coupling between study and objective
- ❌ Objective never used independently
- ❌ Shared state passed between classes
- ❌ Artificial boundary - objective IS part of study in Optuna
- ❌ More boilerplate without added value

### Solution (3 layers):
```
Layer 3: OptunaHyperoptRunner  - Study + trial execution
Layer 2: DistributedExecutor   - Distribution
Layer 1: SRBatchRunner        - PySR
```

### Benefits:
- ✅ Natural structure - objective is a method
- ✅ Direct state access (no passing config/executor)
- ✅ Simpler mental model
- ✅ Less code overall
- ✅ Still modular at layer boundaries

## Layer Responsibilities

### Layer 1: SRBatchRunner
**What it does:**
- Runs SR experiments on equations
- Creates PySR model once, reuses for all equations
- Handles trial/equation/run level resumption
- Extracts pareto volumes from results

**What it doesn't know:**
- Optuna trials or hyperparameter optimization
- Distributed coordination
- Master/worker roles

**Interface:**
```python
runner = SRBatchRunner(model_settings, neural_options, mutation_weights,
                       dataset_config, trial_id)
results = runner.run_equations([1, 2, 3], interruption_flag)
```

**Size:** ~200 lines

---

### Layer 2: DistributedTrialExecutor
**What it does:**
- Coordinates master and workers via files
- Asynchronous batch-wise reporting
- Worker loop for continuous trial processing
- File-based processed trials tracking

**What it doesn't know:**
- SR experiments or PySR
- Optuna trials or sampling
- What a "runner" actually does (generic interface)

**Interface:**
```python
# Master
executor = DistributedTrialExecutor(node_id, total_nodes, coord_dir)
results = executor.execute_trial_with_batching(
    trial_params, equations, batch_size,
    create_runner_fn, report_callback, interruption_flag
)

# Worker
executor.run_worker_loop(equations, batch_size,
                        create_runner_fn, interruption_flag)
```

**Size:** ~500 lines

**Key design decisions:**
- ✅ No batch-wise synchronization (async)
- ✅ Report each batch individually (master and worker)
- ✅ Use interruption_manager everywhere (no shutdown signal file)
- ✅ File-based processed_trials (single source of truth)
- ✅ Master marks trials processed (after workers done)

---

### Layer 3: OptunaHyperoptRunner (Combined)
**What it does:**
- Study creation and configuration
- Master/worker routing
- Hyperparameter sampling (neural, mutation weights)
- Trial resumption detection
- Trial execution via executor
- Results aggregation and printing

**Public API:**
```python
runner = OptunaHyperoptRunner(config_path, node_id, total_nodes)
runner.optimize(resume=False)
```

**Internal structure:**
```python
class OptunaHyperoptRunner:
    # Public API
    def optimize(resume):
        if is_worker: _run_worker_loop()
        else: _run_master_optimization(resume)

    # Master optimization (private)
    def _run_master_optimization(resume):
        study = _create_or_load_study(resume)
        study.optimize(self._objective, ...)
        _print_final_results()

    def _objective(trial):
        # Check resumption
        # Sample/reconstruct hyperparameters
        # Create runner factory and report callback
        # Execute via executor
        # Return metric

    # Worker loop (private)
    def _run_worker_loop():
        executor.run_worker_loop(...)

    # Hyperparameter sampling (private)
    def _sample_hyperparameters(trial)
    def _reconstruct_hyperparameters(trial, saved_params)

    # Trial management (private)
    def _find_incomplete_trial()
    def _mark_trial_complete(trial_id)

    # Study setup (private)
    def _create_or_load_study(resume)
    def _create_sampler()
    def _create_pruner()
    def _enqueue_baseline_trial()

    # Results (private)
    def _print_final_results()
```

**Size:** ~640 lines (well-organized with clear sections)

---

## Communication Flow

### Master Loop
```python
# OptunaHyperoptRunner (Layer 3)
for trial in study:
    # Sample hyperparameters
    model_settings, neural_opts, mutation_weights = _sample_hyperparameters(trial)

    # Create runner factory
    def create_runner(params):
        return SRBatchRunner(ms, no, mw, dataset, trial_id)  # Layer 1

    # Create report callback
    def report_callback(batch_results, batch_id):
        trial.report(np.mean(batch_results))  # Report to Optuna
        if trial.should_prune():
            raise TrialPruned()

    # Execute via distributed executor (Layer 2)
    results = executor.execute_trial_with_batching(
        trial_params, equations, batch_size,
        create_runner, report_callback, interruption_flag
    )
```

### DistributedTrialExecutor (Layer 2)
```python
# Master
for master_batch in master_equations:
    batch_results = runner.run_equations(batch)  # Layer 1
    report_callback(batch_results, "master_batch_i")  # To Layer 3

    # Scan and report new worker batches
    for worker_id in workers:
        worker_batches = get_new_worker_batches(worker_id)
        for batch in worker_batches:
            report_callback(batch, f"worker{id}_batch_i")  # To Layer 3

# Wait for all workers, continue reporting their batches
wait_and_report_remaining_worker_batches()
mark_trial_processed()

# Worker
while True:
    trial_params = wait_for_trial_params()
    runner = create_runner(trial_params)  # Factory from Layer 3

    for worker_batch in worker_equations:
        batch_results = runner.run_equations(batch)  # Layer 1
        send_worker_results(all_results_so_far)

    send_worker_results(all_results, final=True)
```

### SRBatchRunner (Layer 1)
```python
def run_equations(equations, interruption_flag):
    existing = _scan_existing_results(equations)

    for eq in remaining_equations:
        if interruption_flag():
            break

        pv = _run_single_equation(eq)
        # Runs n_runs, extracts PVs, returns mean

    return [results_map.get(eq, 0.0) for eq in equations]
```

## Data Flow

```
Config YAML
    ↓
OptunaHyperoptRunner
    ↓ (trial_params)
DistributedTrialExecutor
    ↓ (create_runner factory)
SRBatchRunner
    ↓ (run equations)
Results (Pareto Volumes)
    ↑ (batch results)
DistributedTrialExecutor
    ↑ (report_callback)
OptunaHyperoptRunner
    ↑ (trial.report)
Optuna Study
```

## Key Design Principles

### 1. Clear Layer Boundaries
- Each layer has well-defined interface
- No cross-layer knowledge
- Dependency only flows downward

### 2. Single Responsibility (at layer level)
- Layer 1: SR execution
- Layer 2: Distribution
- Layer 3: Optuna orchestration

### 3. Generic Lower Layers
- Layer 2 works with any runner (not SR-specific)
- Layer 1 works without Optuna

### 4. Testability
- Each layer independently testable
- Factory functions enable mocking
- Clear inputs/outputs

### 5. Simplicity
- 3 layers (not 4) - no artificial splits
- Public API is minimal
- Private methods for internal structure

## Migration Path

### Current State:
```
optuna_coordinator.py (~674 lines)
  - Coordination + SR execution mixed
optuna_objective.py (~285 lines)
  - Trial execution
optuna_hyperopt.py (~355 lines)
  - Study management
```

### New State:
```
sr_runner.py (~200 lines)
  - Pure SR execution (extracted from coordinator)
distributed_executor.py (~500 lines)
  - Generic coordination (refactored from coordinator)
optuna_hyperopt.py (~640 lines)
  - Study + trial (combined objective + hyperopt)
```

### Steps:
1. Create `sr_runner.py` - Extract SR logic from coordinator
2. Create `distributed_executor.py` - Refactor coordinator, remove SR code
3. Merge objective into `optuna_hyperopt.py` - Combine study + trial
4. Test in single-node mode
5. Test in distributed mode
6. Delete old `optuna_coordinator.py` and `optuna_objective.py`

## Success Criteria

### Correctness:
- ✅ All existing functionality preserved
- ✅ Trial resumption works at all levels
- ✅ Distributed execution functions correctly
- ✅ Intermediate reporting and pruning work

### Code Quality:
- ✅ Clear separation of concerns
- ✅ Each layer independently testable
- ✅ Reduced coupling
- ✅ Improved maintainability

### Simplicity:
- ✅ 3 layers instead of scattered logic
- ✅ Clearer mental model
- ✅ Less boilerplate
- ✅ Easier to understand workflow

## Files to Create/Modify

### New Files:
- `src/SR_benchmarking/sweeps/sr_runner.py`
- `src/SR_benchmarking/sweeps/distributed_executor.py`

### Modified Files:
- `src/SR_benchmarking/sweeps/optuna_hyperopt.py` (major refactor)

### Deleted Files:
- `src/SR_benchmarking/sweeps/optuna_coordinator.py` (replaced)
- `src/SR_benchmarking/sweeps/optuna_objective.py` (merged)

### Unchanged Files:
- `src/SR_benchmarking/sweeps/slurm_optuna_distributed.sh`
- All config files
- All other run infrastructure
