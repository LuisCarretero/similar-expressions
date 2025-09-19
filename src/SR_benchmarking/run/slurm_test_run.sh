#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/logs/%x-%A_%a.out

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"
echo "Node ID: $SLURM_ARRAY_TASK_ID"

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Run test configuration distributed across nodes
echo "Starting distributed test run on node $SLURM_ARRAY_TASK_ID..."
python -m run.run_multiple --config=run/config_test.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

echo "Distributed test run completed on node $SLURM_ARRAY_TASK_ID."