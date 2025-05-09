#!/bin/bash

#SBATCH -J simexp_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=lovelace
#SBATCH --output=slurm_envs.out

#! Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

#! Working directory
workdir="$SLURM_SUBMIT_DIR"

#! Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

#! Run the Python script using the options
CMD="srun python test_slurm_envs.py"

echo "Executing command: $CMD"
eval $CMD