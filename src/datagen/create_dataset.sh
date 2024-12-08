#!/bin/bash

#SBATCH -J simexp-datagen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=21
#SBATCH --time=04:00:00
#SBATCH --partition=lovelace
#SBATCH --output=/store/DAMTP/lc865/workspace/slurm-logs/datagen/%x-%j.out

#! Load environment (activate conda environment)
source /store/DAMTP/lc865/misc/miniforge3/etc/profile.d/conda.sh
conda activate julia-env

#! Working directory
workdir="/home/lc865/workspace/similar-expressions/src/datagen"

#! Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

#! Run the Python script using the options
CMD="srun julia create_dataset.jl"

echo "Executing command: $CMD"
eval $CMD