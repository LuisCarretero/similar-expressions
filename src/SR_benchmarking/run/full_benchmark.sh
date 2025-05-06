#!/bin/bash

#SBATCH -J simexpSR
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=2
#SBATCH --time=8:00:00
#SBATCH --partition=lovelace
#SBATCH --output=/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/variance_tests_10iter/slurm_logs/%x-%j.out

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh  # Needed if not interactive.
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Run the Python script using the options
python -m run.run_multiple \
    --dataset feynman \
    --equations 8:20 \
    --log_dir /cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/variance_tests_10iter \
    --niterations 10

# python -m run.run_multiple \
#     --dataset synthetic \
#     --equations 20:30 \
#     --log_dir /cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/variance_tests_10iter \
#     --niterations 10

python -m run.run_multiple \
    --dataset pysr-difficult \
    --equations 11:20 \
    --log_dir /cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/variance_tests_10iter \
    --niterations 10

python -m run.run_multiple \
    --dataset pysr-difficult \
    --equations -20:0 \
    --log_dir /cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/variance_tests_10iter \
    --niterations 10