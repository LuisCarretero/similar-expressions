#!/bin/bash
#SBATCH -p lovelace
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=2
#SBATCH --time=4:00:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/%x-%j.out

# Load environment (activate conda environment)
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
workdir="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking"

# Move to the working directory
cd $workdir
echo "Running in directory: `pwd`"

# Create the sweep
python -m wandb sweep sweeps/config.yaml --project simexp-SR > sweep_output.txt 2>&1
cat sweep_output.txt
echo ""

# Extract the sweep ID from the output file
SWEEP_ID=$(grep "wandb agent" sweep_output.txt | awk '{print $NF}')
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to extract sweep ID. Check the output above."
    exit 1
fi
echo "Extracted sweep ID: $SWEEP_ID"
COUNT=10  # Number of runs to execute

# Run the wandb agent with the config file
python -m wandb agent --count $COUNT --project simexp-SR $SWEEP_ID
