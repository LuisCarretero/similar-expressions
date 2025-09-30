#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/test_optuna_dist-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=test-optuna-distributed-fileflag

# TEST: Distributed Optuna with File-Based Flag Shutdown
# This tests the file-based flag approach to avoid Julia signal conflicts
# Runtime: 5 minutes, Signal: 2 minutes before end (at 3-minute mark)

echo "[$(date)] ==================================="
echo "[$(date)] DISTRIBUTED OPTUNA FILE FLAG TEST"
echo "[$(date)] ==================================="
echo "[$(date)] Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME"
echo "[$(date)] Testing: B:USR1@120 + file-based flag + distributed Optuna"

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Store Python PID for monitoring
PYTHON_PID=""

# Signal handler that creates flag file for Python to check
# This avoids conflicts with Julia's signal handling
signal_handler() {
    echo "[$(date)] BASH Node $SLURM_ARRAY_TASK_ID: Received shutdown signal, creating flag file"
    FLAG_FILE="/tmp/slurm_shutdown_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "1" > "$FLAG_FILE"
    echo "[$(date)] BASH Node $SLURM_ARRAY_TASK_ID: Flag file created: $FLAG_FILE"

    # Wait for Python to finish gracefully
    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        wait $PYTHON_PID
        echo "[$(date)] BASH Node $SLURM_ARRAY_TASK_ID: Python exited gracefully"
    fi

    # Clean up flag file
    rm -f "$FLAG_FILE"
    exit 0
}

trap signal_handler USR1 TERM INT

# Setup environment
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# Create logs directory
mkdir -p sweeps/logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if this is a requeued job
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    RESUME_FLAG="--resume"
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Resuming Optuna study (restart #${SLURM_RESTART_COUNT})"
else
    RESUME_FLAG=""
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting new Optuna study"
fi

# Use test config for quick validation
CONFIG_FILE="sweeps/optuna_test_config.yaml"

echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting distributed Optuna test with config: $CONFIG_FILE"
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Expected signal at ~3 minute mark (120s warning)"

# Run distributed Optuna optimization in background to capture PID
python -u sweeps/optuna_hyperopt.py \
    --config $CONFIG_FILE \
    --node_id=$SLURM_ARRAY_TASK_ID \
    --total_nodes=$TOTAL_NODES \
    $RESUME_FLAG &

PYTHON_PID=$!
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Python Optuna process started with PID: $PYTHON_PID"

# Wait for Python script to complete
wait $PYTHON_PID
EXIT_CODE=$?

echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Python script exited with code: $EXIT_CODE"

# Completion message
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: ✅ SUCCESS - Distributed Optuna test completed successfully"
    if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
        echo "[$(date)] View test results: optuna-dashboard sqlite:///sweeps/optuna_test_study.db"
    fi
else
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: ❌ FAILURE - Test failed with exit code $EXIT_CODE"
fi

echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Test completion at $(date)"

exit $EXIT_CODE

# Expected behavior:
# 1. Both nodes start and coordinate via OptunaCoordinator
# 2. Master node (0) creates/loads study and manages trials
# 3. Worker node (1) processes assigned equations
# 4. At ~3 min mark: SLURM sends USR1 to bash
# 5. Bash trap creates flag file /tmp/slurm_shutdown_JOBID_TASKID
# 6. Python checks flag file and detects shutdown request
# 7. Optuna completes current trial and exits gracefully
# 8. Job requeues automatically for continuation