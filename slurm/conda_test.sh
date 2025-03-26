#!/bin/bash
#SBATCH --job-name=conda_check
#SBATCH --output=conda_check.out
#SBATCH --partition=lovelace
#SBATCH --ntasks=1

source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

python --version
