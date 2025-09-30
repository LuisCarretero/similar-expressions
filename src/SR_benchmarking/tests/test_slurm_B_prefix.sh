#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/test_B_prefix-%j.out
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=test-signal-B-prefix

# TEST: SLURM with B: prefix (signal to batch script only)
# Expected: Signal should be sent to bash script, which then runs Python
# Runtime: 5 minutes, Signal: 2 minutes before end (at 3-minute mark)

echo "[$(date)] TEST: B:USR1@120 - Signal to batch script only"
echo "[$(date)] Job ID: $SLURM_JOB_ID"
echo "[$(date)] Node: $SLURMD_NODENAME"
echo "[$(date)] No bash trap - testing if signal reaches Python directly"

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