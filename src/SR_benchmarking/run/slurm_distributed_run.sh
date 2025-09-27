#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/logs/%x-%A_%a.out

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"
echo "Node ID: $SLURM_ARRAY_TASK_ID"

# Run neural mode first (with neural mutations enabled)
echo "Starting neural distributed run on node $SLURM_ARRAY_TASK_ID..."
python -u -m run.run_multiple --config=run/config_neural.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

echo "Neural run completed. Starting vanilla distributed run on node $SLURM_ARRAY_TASK_ID..."

# Run vanilla mode (with neural mutations disabled)
python -u -m run.run_multiple --config=run/config_vanilla.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

echo "Vanilla distributed run completed on node $SLURM_ARRAY_TASK_ID."