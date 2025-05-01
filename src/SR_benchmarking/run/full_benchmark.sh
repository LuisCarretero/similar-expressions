#!/bin/bash

#SBATCH -J simexpSR
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --time=24:00:00
#SBATCH --partition=lovelace
#SBATCH --output=/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/round1/%x-%j.out

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/home/lc865/workspace/similar-expressions"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Run the Python script using the options
CMD="srun python src/sr_inference_benchmarking/full_benchmark.py"

echo "Executing command: $CMD"
eval $CMD