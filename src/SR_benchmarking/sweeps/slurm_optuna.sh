#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=2
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/optuna-%j.out
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800
#SBATCH --job-name=optuna-sr-hyperopt

# Setup environment
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# Create logs directory
mkdir -p sweeps/logs

# Signal handling for graceful shutdown
trap 'echo "[$(date)] Interrupted, exiting for requeue..."; exit 0' USR1 TERM INT

# Set environment variables
export PYTHON_JULIAPKG_PROJECT="/cephfs/home/lc865/workspace/similar-expressions"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Check if this is a requeued job
if [ "${SLURM_RESTART_COUNT:-0}" -gt 0 ]; then
    RESUME_FLAG="--resume"
    echo "[$(date)] Resuming Optuna study (restart #${SLURM_RESTART_COUNT})"
else
    RESUME_FLAG=""
    echo "[$(date)] Starting new Optuna study"
fi

# Basic job info
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | CPUs: $SLURM_CPUS_PER_TASK"

# Get config file from first argument, default to original config
CONFIG_FILE=${1:-sweeps/optuna_neural_config.yaml}

# Run Optuna optimization
echo "[$(date)] Starting optimization with config: $CONFIG_FILE"
srun python -u sweeps/optuna_hyperopt.py --config $CONFIG_FILE $RESUME_FLAG
EXIT_CODE=$?

# Completion message
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] Optimization completed successfully"
    echo "View results: optuna-dashboard sqlite:///sweeps/optuna_study.db"
else
    echo "[$(date)] Optimization failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE