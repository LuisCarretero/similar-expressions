#!/bin/bash

#SBATCH -J simexp-modelcheck
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=lovelace
#SBATCH --output=%x.out

#! Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate /cephfs/store/gr-mc2473/lc865/misc/condaforge/envs/ml  # Temporarily using whole path

#! Working directory
workdir="/home/lc865/workspace/similar-expressions"

#! Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

#! Run the Python script using the options
CMD="srun python -m src.dev.modelcheck"

echo "Executing command: $CMD"
eval $CMD