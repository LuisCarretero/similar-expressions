#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/optuna_test-%A_%a.out
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --job-name=optuna-sr-test

# TEST SCRIPT - Short runtime for signal handling verification
# Runtime: 5 minutes, Signal: 2 minutes before end

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Setup environment
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# Create logs directory
mkdir -p sweeps/logs

# Signal handling is managed by Python signal_manager.py
# No bash trap to avoid conflicts with Python signal handlers

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
CONFIG_FILE="sweeps/optuna_test_config.yaml"

# Run distributed Optuna optimization
echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Starting test optimization with config: $CONFIG_FILE"
python -u sweeps/optuna_hyperopt.py \
    --config $CONFIG_FILE \
    --node_id=$SLURM_ARRAY_TASK_ID \
    --total_nodes=$TOTAL_NODES \
    $RESUME_FLAG

EXIT_CODE=$?

# Completion message
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Test optimization completed successfully"
    if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
        echo "View test results: optuna-dashboard sqlite:///sweeps/optuna_test_study.db"
    fi
else
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Test optimization failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE