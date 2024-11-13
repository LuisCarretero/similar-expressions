#!/bin/bash

#SBATCH -J simexp-agent
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=lovelace
#SBATCH --output=/store/DAMTP/lc865/workspace/slurm-logs/sweeps/%x-%j.out

if [ $# -eq 1 ]; then
    echo "Error: No wandb sweep path or count provided. Usage: $0 <wandb_sweep_path> <count>"
    exit 1
fi
SWEEP_PATH=$1
COUNT=$2

# Load environment (activate conda environment)
source /store/DAMTP/lc865/misc/miniforge3/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/home/lc865/workspace/similar-expressions"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

CMD="srun wandb agent --count $COUNT $SWEEP_PATH"

echo "Executing command: $CMD"
eval $CMD
