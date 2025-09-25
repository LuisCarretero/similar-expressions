#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --array=0-1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=11:50:00
#SBATCH --job-name=slurm_requeuing_run
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/logs/%x-%A_%a.out

# Total number of nodes (should match array size)
TOTAL_NODES=2

# Configuration file to use for requeing
CONFIG_FILE="run/config_vanilla.yaml"

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
export SLURM_TIME_LIMIT=710 # 11:50 in minutes

echo "Starting neural distributed run on node $SLURM_ARRAY_TASK_ID..."
python -m run.run_multiple --config=run/config_neural.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

echo "Neural run completed. Starting vanilla distributed run on node $SLURM_ARRAY_TASK_ID..."

# Run vanilla mode (with neural mutations disabled)
python -m run.run_multiple --config=run/config_vanilla.yaml --pooled --node_id=$SLURM_ARRAY_TASK_ID --total_nodes=$TOTAL_NODES

echo "Vanilla distributed run completed on node $SLURM_ARRAY_TASK_ID at $(date)"

# Resubmission logic (only run on first array task to avoid duplicates)
if [ "$SLURM_ARRAY_TASK_ID" -eq "0" ]; then
    echo "Checking for remaining work and handling resubmission..."

    # Extract log directory and dataset info from config using helper script
    JOB_INFO=$(python run/get_job_info.py "$CONFIG_FILE" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$JOB_INFO" ]; then
        # Parse space-separated output: log_dir dataset_name total_equations
        LOG_DIR=$(echo $JOB_INFO | cut -d' ' -f1)
        DATASET_NAME=$(echo $JOB_INFO | cut -d' ' -f2)
        TOTAL_EQUATIONS=$(echo $JOB_INFO | cut -d' ' -f3)
    else
        echo "ERROR: Failed to extract job info from config file $CONFIG_FILE"
        LOG_DIR="/tmp/fallback_logs"
        DATASET_NAME="unknown"
        TOTAL_EQUATIONS=0
    fi

    echo "Log directory: $LOG_DIR"
    echo "Dataset: $DATASET_NAME"
    echo "Total equations: $TOTAL_EQUATIONS"

    # Count completed equations using .done files
    COMPLETED_DIR="$LOG_DIR/completed"
    if [ -d "$COMPLETED_DIR" ]; then
        COMPLETED_COUNT=$(find "$COMPLETED_DIR" -name "${DATASET_NAME}_eq*.done" 2>/dev/null | wc -l)
    else
        COMPLETED_COUNT=0
    fi

    echo "Completed equations: $COMPLETED_COUNT"
    echo "Progress: $COMPLETED_COUNT/$TOTAL_EQUATIONS"

    # Check if we need to resubmit
    if [ "$COMPLETED_COUNT" -lt "$TOTAL_EQUATIONS" ]; then
        REMAINING=$((TOTAL_EQUATIONS - COMPLETED_COUNT))
        echo "Found $REMAINING incomplete equations. Resubmitting job..."

        # Resubmit this same script with consistent job name
        NEW_JOB_ID=$(sbatch --parsable --job-name="slurm_requeuing_run" "$0")
        if [ $? -eq 0 ]; then
            echo "Successfully resubmitted as job ID: $NEW_JOB_ID"
        else
            echo "ERROR: Failed to resubmit job"
        fi
    else
        echo "All equations completed! No resubmission needed."
    fi
else
    echo "Node $SLURM_ARRAY_TASK_ID: Skipping resubmission check (only node 0 handles this)"
fi

echo "Node $SLURM_ARRAY_TASK_ID completed at $(date)"