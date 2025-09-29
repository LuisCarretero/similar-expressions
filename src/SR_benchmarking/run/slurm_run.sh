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

# Signal handling for graceful shutdown
trap 'echo "[$(date)] Interrupted, exiting gracefully..."; exit 0' USR1 TERM INT

# Run neural mode first (with neural mutations enabled)
echo "Starting neural pooled run..."
python -u -m run.run_multiple --config=run/config_neural.yaml --pooled

echo "Neural run completed. Starting vanilla pooled run..."

# Run vanilla mode (with neural mutations disabled)
python -u -m run.run_multiple --config=run/config_vanilla.yaml --pooled

echo "Both pooled runs completed."