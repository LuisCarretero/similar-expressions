#!/bin/bash
#SBATCH -p lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/%x-%j.out
#SBATCH --signal=B:USR1@90

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Store Python PID for monitoring
PYTHON_PID=""

# Signal handler that creates flag file for Python to check
signal_handler() {
    echo "[$(date)] Received shutdown signal, creating flag file"
    FLAG_FILE="/tmp/slurm_shutdown_${SLURM_JOB_ID}_0"
    echo "1" > "$FLAG_FILE"
    echo "[$(date)] Flag file created: $FLAG_FILE"

    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        wait $PYTHON_PID
        echo "[$(date)] Python exited gracefully"
    fi

    rm -f "$FLAG_FILE"
    exit 0
}

trap signal_handler USR1 TERM INT

# Run neural mode first (with neural mutations enabled)
echo "Starting neural pooled run..."
python -u -m run.run_multiple --config=run/config_neural.yaml --pooled &
PYTHON_PID=$!
wait $PYTHON_PID

echo "Neural run completed. Starting vanilla pooled run..."

# Run vanilla mode (with neural mutations disabled)
python -u -m run.run_multiple --config=run/config_vanilla.yaml --pooled &
PYTHON_PID=$!
wait $PYTHON_PID

echo "Both pooled runs completed."