#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/optuna_distributed-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800
#SBATCH --job-name=optuna-sr-distributed

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

# Basic job info
echo "Job ID: $SLURM_JOB_ID | Array Task ID: $SLURM_ARRAY_TASK_ID | Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"

# Get config file from first argument, default to neural config
CONFIG_FILE=${1:-sweeps/optuna_neural_config.yaml}

# Run distributed Optuna optimization in background to capture PID
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting optimization with config: $CONFIG_FILE"
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

# Completion message
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Optimization completed successfully"
    if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
        echo "View results: optuna-dashboard sqlite:///sweeps/optuna_neural_study.db"
    fi
else
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Optimization failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE

# Usage examples:
# sbatch sweeps/slurm_optuna_distributed.sh                                    # Use default neural config
# sbatch sweeps/slurm_optuna_distributed.sh sweeps/optuna_weights_config.yaml # Use weights config
#
# To resume after interruption:
# The script automatically detects requeued jobs and passes --resume flag
#
# To monitor progress:
# tail -f sweeps/logs/optuna_distributed-JOBID_0.out  # Master node log
# tail -f sweeps/logs/optuna_distributed-JOBID_1.out  # Worker node log
# optuna-dashboard sqlite:///src/SR_benchmarking/sweeps/optuna_neural_study.db