# Signal Handling & SLURM Queuing Comparison

**Date:** 2025-10-01
**Comparison:** `run/slurm_requeuing_run.sh` vs `sweeps/slurm_optuna_distributed.sh`

## Key Differences Found

### 1. **Signal Handler Behavior**

#### Sweeps script (`sweeps/slurm_optuna_distributed.sh`) - MORE RECENT
- Creates flag file for Python ✓
- Waits for Python to exit gracefully ✓
- Cleans up flag file ✓
- **NEW:** Checks for `study.done` marker before requeuing (lines 38-46)
- **NEW:** Directly requeues entire array job: `scontrol requeue ${SLURM_ARRAY_JOB_ID}` (line 50)
- Traps: `USR1 TERM` only (line 55)

#### Run script (`run/slurm_requeuing_run.sh`) - OLDER
- Creates flag file for Python ✓
- Waits for Python to exit gracefully ✓
- Cleans up flag file ✓
- **MISSING:** No completion marker check in signal handler
- Uses separate function `check_and_requeue_if_needed()` with Python script
- Traps: `USR1 TERM INT` (line 54)

### 2. **Resume Logic**

#### Sweeps script has:
```bash
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG=""
fi
```
(lines 72-78)

#### Run script:
**No resume detection logic**

### 3. **Requeuing Strategy**

#### Sweeps approach:
- Signal handler directly requeues if study not complete
- Checks global completion marker: `/path/to/coord_dir/study.done`
- All nodes use same requeue logic

#### Run approach:
- Delegates to Python script (`check_node_completion.py`) that:
  - Checks individual equation `.done` files in `log_dir/completed/`
  - Only requeues if this specific node has remaining work
  - Per-node completion checking

### 4. **Signal Trap Differences**

| Script | Signals Trapped | Comment |
|--------|----------------|---------|
| Sweeps | `USR1 TERM` | Line 55: `# Send INT to kill` |
| Run    | `USR1 TERM INT` | Line 54 |

**Implication:** Sweeps allows `INT` (Ctrl+C) to kill immediately, while Run script captures it for graceful shutdown.

## Commit History Context

Recent signal-related commits:
- `ae99ea9` (Oct 1): Fix: Misc refactor fixes - Added `--open-mode=append` to sweeps
- `ca04a4c` (Sep 30): Fix: Misc requing + signal updates for optuna - **Major signal handler refactor**
- `5086323` (Sep 29): Updated signal handling for controlled requeing - **Initial signal_manager.py**

The sweeps script received the requeuing logic additions in commit `ae99ea9`, which added:
- Study completion marker check
- Direct array job requeue (`scontrol requeue ${SLURM_ARRAY_JOB_ID}`)

## Recommendations

### Option A: Minimal Updates (Conservative)
1. Keep current run script approach (node-specific completion checking)
2. Add comment explaining why it differs from sweeps
3. Consider removing `INT` from trap to match sweeps behavior

**Pros:** Maintains existing logic
**Cons:** Scripts remain inconsistent

### Option B: Add Completion Marker (Recommended)
1. Add overall completion check in signal handler (similar to sweeps)
2. Keep the per-node Python checker as secondary validation
3. Add resume detection using `SLURM_RESTART_COUNT`
4. Update trap to `USR1 TERM` only

**Pros:** More consistent, prevents unnecessary requeues when fully complete
**Cons:** Requires coordination between nodes for completion marker

### Option C: Full Alignment
1. Create an overall "run.done" marker when all equations complete
2. Check this marker in signal handler before calling Python script
3. Add resume logic
4. Simplify to direct requeue instead of Python script

**Pros:** Full consistency across scripts
**Cons:** May lose granular per-node completion tracking

## Execution Model Differences

The sweeps and run scripts have fundamentally different execution models:

- **Optuna sweeps:**
  - All nodes work on shared Optuna study (via SQLite)
  - Trials distributed dynamically across nodes
  - Global completion marker makes sense (study has fixed trial count)

- **Run script:**
  - Each node has pre-assigned equation subset
  - Equations split statically via `get_node_equations()`
  - Per-node checking may be more appropriate

## Current Run Script Flow

1. Python process runs with interruption checking
2. On SLURM signal → bash creates flag file
3. Python detects flag, completes current work, exits
4. Bash cleanup, then calls `check_node_completion.py`
5. Python script checks if this node has remaining equations
6. If yes → `scontrol requeue $SLURM_JOB_ID`

## Current Sweeps Script Flow

1. Python process runs with interruption checking
2. On SLURM signal → bash creates flag file
3. Python detects flag, completes current work, exits
4. Bash checks if `study.done` marker exists
5. If no marker → `scontrol requeue ${SLURM_ARRAY_JOB_ID}`
6. If marker exists → exit without requeue

## Action Items

- [ ] Decide on alignment strategy (A/B/C)
- [ ] Consider if `INT` should trigger graceful shutdown or immediate kill
- [ ] Test resume logic if implemented
- [ ] Document execution model differences if keeping divergent approaches
