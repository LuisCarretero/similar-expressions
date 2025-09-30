#!/bin/bash
#SBATCH --job-name=optuna_test_dist
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=00:31:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/tests/logs/optuna_test_dist-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800

# Multi-node test script for Optuna distributed optimization
# Uses existing optuna_hyperopt.py infrastructure with test config

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

    # Requeue the job so it resumes after graceful shutdown
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Requeuing job ${SLURM_JOB_ID}"
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}

trap signal_handler USR1 TERM INT

echo "=========================================="
echo "Optuna Multi-Node Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Total nodes: $TOTAL_NODES"
echo "=========================================="

# Setup
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking
mkdir -p tests/logs

# Activate conda
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Set threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if this is a requeued job
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    RESUME_FLAG="--resume"
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Resuming Optuna test (restart #${SLURM_RESTART_COUNT})"
else
    RESUME_FLAG=""
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting new Optuna test"
fi

echo ""
echo "Running distributed test with config: tests/optuna_test_config.yaml"
echo "This is node $SLURM_ARRAY_TASK_ID of $TOTAL_NODES"
echo ""

# Run optimization using existing infrastructure in background to capture PID
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting optimization"
python -u sweeps/optuna_hyperopt.py \
    --config tests/optuna_test_config.yaml \
    --node_id=$SLURM_ARRAY_TASK_ID \
    --total_nodes=$TOTAL_NODES \
    $RESUME_FLAG &

PYTHON_PID=$!
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Python process started with PID: $PYTHON_PID"

# Wait for Python script to complete
wait $PYTHON_PID
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Test completed successfully"
    if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
        echo "View results: optuna-dashboard sqlite:///sweeps/optuna_test_study.db"
    fi
else
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Test failed with exit code $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
