#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --job-name=slurm_requeuing_run
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/logs/%x-%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Configuration files to process
CONFIG_FILES=("run/config_neural.yaml")  #  "run/config_neural.yaml"

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

# Store Python PID for monitoring
PYTHON_PID=""

# Signal handler that creates flag file for Python to check
signal_handler() {
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Received shutdown signal, creating flag file"
    FLAG_FILE="/tmp/slurm_shutdown_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "1" > "$FLAG_FILE"
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Flag file created: $FLAG_FILE"

    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        wait $PYTHON_PID
        echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Python exited gracefully"
    fi

    rm -f "$FLAG_FILE"
    exit 0
}

trap signal_handler USR1 TERM INT

# Function to check if this node has remaining work and requeue if needed
check_and_requeue_if_needed() {
    echo "[$(date)] Node $SLURM_ARRAY_TASK_ID: Checking for remaining work..."
    python run/check_node_completion.py \
        --node_id=$SLURM_ARRAY_TASK_ID \
        --total_nodes=$TOTAL_NODES \
        --job_id=$SLURM_JOB_ID \
        --config_files "${CONFIG_FILES[@]}"
}

# echo "Starting vanilla distributed run on node $SLURM_ARRAY_TASK_ID..."
# python -u -m run.run_multiple --config=run/config_vanilla.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES &
# PYTHON_PID=$!
# wait $PYTHON_PID

echo "Starting neural distributed run on node $SLURM_ARRAY_TASK_ID..."
python -u -m run.run_multiple --config=run/config_neural.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES &
PYTHON_PID=$!
wait $PYTHON_PID

echo "Distributed runs completed on node $SLURM_ARRAY_TASK_ID at $(date)"

# Check if this node needs to requeue itself
check_and_requeue_if_needed

echo "Node $SLURM_ARRAY_TASK_ID completed at $(date)"