#!/bin/bash

#SBATCH -J simexp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=24:00:00
#SBATCH --partition=lovelace
#SBATCH --output=/cephfs/store/gr-mc2473/lc865/workspace/slurm-logs/single-run/%x-%j.out

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/home/lc865/workspace/similar-expressions"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Run the Python script using the options
CMD="srun python -m src.train.train"

echo "Executing command: $CMD"
eval $CMD