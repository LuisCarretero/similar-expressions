#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=00:14:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/tests/logs/optuna_test-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --job-name=optuna-sr-test

# TEST SCRIPT - Short runtime for signal handling verification
# Runtime: 5 minutes, Signal: 2 minutes before end

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Store Python PID for monitoring
PYTHON_PID=""

# Signal handler that creates flag file for Python to check
# This avoids conflicts with Julia's signal handling
signal_handler() {
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Received shutdown signal, creating flag file"
    FLAG_FILE="/tmp/slurm_shutdown_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "1" > "$FLAG_FILE"
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Flag file created: $FLAG_FILE"

    # Wait for Python to finish gracefully
    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        wait $PYTHON_PID
        echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Python exited gracefully"
    fi

    # Clean up flag file
    rm -f "$FLAG_FILE"

    # All nodes check if study is complete before requeueing
    COORD_DIR="/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment/study_coord-sr_test_distributed_3"
    if [ -f "$COORD_DIR/study.done" ]; then
        echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Study complete (found study.done marker), not requeueing"
        exit 0
    fi

    # All tasks requeue the array job
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Requeuing job array ${SLURM_ARRAY_JOB_ID}"
    scontrol requeue ${SLURM_ARRAY_JOB_ID}
    exit 0
}

trap signal_handler USR1 TERM  # Send INT to kill

# Setup environment
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# Create logs directory
mkdir -p tests/logs

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if this is a requeued job
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    RESUME_FLAG="--resume"
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Resuming test study (restart #${SLURM_RESTART_COUNT})"
else
    RESUME_FLAG=""
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting new test study"
fi

# Basic job info
echo "Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"
echo "TEST RUN: 5min runtime, 2min signal warning"

# Use test config
CONFIG_FILE="tests/optuna_test_config.yaml"

# Run distributed Optuna optimization with REFACTORED code
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting test optimization with REFACTORED code"
echo "[$(date)] Config: $CONFIG_FILE"
python -u sweeps/optuna_hyperopt.py \
    --config $CONFIG_FILE \
    --node_id=$SLURM_ARRAY_TASK_ID \
    --total_nodes=$TOTAL_NODES \
    $RESUME_FLAG &

PYTHON_PID=$!
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Python process started with PID: $PYTHON_PID"

# Wait for Python script to complete
wait $PYTHON_PID
EXIT_CODE=$?

exit $EXIT_CODE