#! /bin/bash

echo "Activating conda environment"
conda activate ml

echo "Changing directory to similar-expressions/src/SR_benchmarking"
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# srun -p lovelace --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem-per-cpu=16G --gpus=1 --pty --time=2:00:00 bash
# julia --threads auto
# python -m run.benchmarking