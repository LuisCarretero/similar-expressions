#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=15:00
#SBATCH --job-name=slurm_requeuing_run
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/logs/%x-%A_%a.out
#SBATCH --requeue

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Configuration files to process
CONFIG_FILES=("run/config_vanilla.yaml")  #  "run/config_neural.yaml"

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"
echo "Node ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Start Time: $(date)"

# Set SLURM environment variables for Python time monitoring
export SLURM_JOB_START_TIME=$(date +%s)
export SLURM_TIME_LIMIT=15  # 45 minutes total time limit

# Function to check if this node has remaining work and requeue if needed
check_and_requeue_if_needed() {
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Checking for remaining work..."
    python run/check_node_completion.py \
        --node_id=$SLURM_ARRAY_TASK_ID \
        --total_nodes=$TOTAL_NODES \
        --job_id=$SLURM_JOB_ID \
        --config_files "${CONFIG_FILES[@]}"
}

echo "Starting vanilla distributed run on node $SLURM_ARRAY_TASK_ID..."
python -u -m run.run_multiple --config=run/config_vanilla.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

# echo "Starting neural distributed run on node $SLURM_ARRAY_TASK_ID..."
# python -u -m run.run_multiple --config=run/config_neural.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

echo "Distributed runs completed on node $SLURM_ARRAY_TASK_ID at $(date)"

# Check if this node needs to requeue itself
check_and_requeue_if_needed

echo "Node $SLURM_ARRAY_TASK_ID completed at $(date)"