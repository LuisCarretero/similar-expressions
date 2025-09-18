#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/analysis/%x-%j.out

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/analysis"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Run the tensorboard data parsing script
echo "Starting tensorboard data parsing..."
python parse_tb_data.py

echo "Tensorboard data parsing completed."