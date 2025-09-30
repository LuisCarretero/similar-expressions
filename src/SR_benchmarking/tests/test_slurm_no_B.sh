#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/test_no_B-%j.out
#SBATCH --signal=USR1@120
#SBATCH --job-name=test-signal-no-B

# TEST: SLURM without B: prefix (signal to all processes)
# Expected: Signal should be sent directly to all processes including Python
# Runtime: 5 minutes, Signal: 2 minutes before end (at 3-minute mark)

echo "[$(date)] TEST: USR1@120 - Signal to all processes"
echo "[$(date)] Job ID: $SLURM_JOB_ID"
echo "[$(date)] Node: $SLURMD_NODENAME"
echo "[$(date)] No bash trap - testing direct signal to Python process"

# Setup environment
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# Create logs directory
mkdir -p sweeps/logs

echo "[$(date)] Starting Python signal test script..."

# Run the Python signal test script directly
python3 sweeps/test_signal_basic.py

EXIT_CODE=$?

echo "[$(date)] Python script exited with code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] SUCCESS: Signal was received and handled gracefully"
else
    echo "[$(date)] FAILURE: No signal received or script failed"
fi

exit $EXIT_CODE