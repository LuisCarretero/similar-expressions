#!/bin/bash
#SBATCH -p lovelace-mc
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/sweeps/logs/test_both-%j.out
#SBATCH --signal=B:USR1@120
#SBATCH --job-name=test-signal-both

# TEST: SLURM with B: prefix + bash trap that forwards signal
# Expected: Signal goes to bash, bash trap forwards it to Python child
# Runtime: 5 minutes, Signal: 2 minutes before end (at 3-minute mark)

echo "[$(date)] TEST: B:USR1@120 + bash trap forwarding"
echo "[$(date)] Job ID: $SLURM_JOB_ID"
echo "[$(date)] Node: $SLURMD_NODENAME"

# Store the Python process PID for signal forwarding
PYTHON_PID=""

# Signal handler that forwards to Python child process
signal_handler() {
    echo "[$(date)] BASH: Received signal, forwarding to Python PID: $PYTHON_PID"
    if [ ! -z "$PYTHON_PID" ] && kill -0 $PYTHON_PID 2>/dev/null; then
        echo "[$(date)] BASH: Sending USR1 to Python process..."
        kill -USR1 $PYTHON_PID
        # Give Python time to handle signal gracefully
        sleep 5
        echo "[$(date)] BASH: Forwarding complete, exiting"
    else
        echo "[$(date)] BASH: Python process not found or already dead"
    fi
    exit 0
}

# Set up signal trap
trap signal_handler USR1 TERM INT

# Setup environment
source /cephfs/store/gr-mc2473/lc865/misc/condaforge/etc/profile.d/conda.sh
conda activate ml

# Working directory
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking

# Create logs directory
mkdir -p sweeps/logs

echo "[$(date)] Starting Python signal test script with bash trap forwarding..."

# Run the Python signal test script in background to get PID
python3 sweeps/test_signal_basic.py &
PYTHON_PID=$!

echo "[$(date)] Python script started with PID: $PYTHON_PID"

# Wait for Python script to complete
wait $PYTHON_PID
EXIT_CODE=$?

echo "[$(date)] Python script exited with code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date)] SUCCESS: Signal was forwarded and handled gracefully"
else
    echo "[$(date)] FAILURE: Signal forwarding failed or script failed"
fi

exit $EXIT_CODE