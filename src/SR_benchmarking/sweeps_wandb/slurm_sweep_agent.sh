#!/bin/bash
#SBATCH -p lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/%x-%j_%R.out
#SBATCH --open-mode=append
#SBATCH --requeue
# Send USR1 90s before preemption/timeout so we can requeue gracefully
#SBATCH --signal=B:USR1@90

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Requeue handler: on USR1/TERM, requeue the job and exit cleanly
trap 'echo "[Requeue] Requeuing job $SLURM_JOB_ID"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1 TERM

# Persist sweep ID so requeues skip sweep creation and reuse the same ID
SWEEP_ID_FILE="$workdir/sweep_id.txt"

if [ -f "$SWEEP_ID_FILE" ] && [ -s "$SWEEP_ID_FILE" ]; then
    SWEEP_ID=$(cat "$SWEEP_ID_FILE")
    echo "Using existing sweep ID: $SWEEP_ID"
else
    echo "Creating new sweep from sweeps/config.yaml"
    python -m wandb sweep sweeps/config.yaml --project simexp-SR > sweep_output.txt 2>&1
    cat sweep_output.txt
    echo ""
    # Extract the sweep ID from the output file
    SWEEP_ID=$(grep "wandb agent" sweep_output.txt | awk '{print $NF}')
    if [ -z "$SWEEP_ID" ]; then
        echo "Failed to extract sweep ID. Check the output above."
        exit 1
    fi
    echo "$SWEEP_ID" > "$SWEEP_ID_FILE"
    echo "Extracted sweep ID: $SWEEP_ID"
fi

COUNT=100  # Number of runs to execute

# Run the wandb agent
python -m wandb agent --count $COUNT --project simexp-SR $SWEEP_ID
