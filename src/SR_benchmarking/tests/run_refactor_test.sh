#!/bin/bash
# Convenience script to run refactored code tests with cleanup

set -e

echo "=========================================="
echo "Testing Refactored Optuna Code"
echo "=========================================="
echo ""

# Define directories
COORD_DIR="/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment/optuna_coord_sr_test_distributed_2"
TRIAL_DIR="/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment"
LOG_DIR="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/tests/logs"
DB_FILE="/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/tests/optuna_test_study.db"

# Check if cleanup is needed
NEEDS_CLEANUP=false
if [ -d "$COORD_DIR" ]; then
    echo "⚠️  Found existing coordination directory: $COORD_DIR"
    NEEDS_CLEANUP=true
fi
if [ -f "$DB_FILE" ]; then
    echo "⚠️  Found existing study database: $DB_FILE"
    NEEDS_CLEANUP=true
fi
if [ -d "$LOG_DIR" ]; then
    echo "⚠️  Found existing log directory: $LOG_DIR"
    NEEDS_CLEANUP=true
fi

# Find existing trial directories
TRIAL_DIRS=$(find "$TRIAL_DIR" -maxdepth 1 -type d -name "trial_*" 2>/dev/null | wc -l)
if [ "$TRIAL_DIRS" -gt 0 ]; then
    echo "⚠️  Found $TRIAL_DIRS existing trial directories in $TRIAL_DIR"
    NEEDS_CLEANUP=true
fi

if [ "$NEEDS_CLEANUP" = true ]; then
    echo ""
    read -p "🗑️  Clean up previous test runs? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning up..."
        rm -rf "$COORD_DIR"
        rm -f "$DB_FILE"
        rm -rf "$LOG_DIR"
        find "$TRIAL_DIR" -maxdepth 1 -type d -name "trial_*" -exec rm -rf {} +
        echo "✅ Cleanup complete"
    else
        echo "⚠️  Keeping existing files (may affect test results)"
    fi
    echo ""
fi

# Submit the test job
echo "🚀 Submitting test job..."
cd /cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking
JOB_ID=$(sbatch tests/slurm_optuna_test.sh | grep -oP '\d+$')

echo ""
echo "✅ Test job submitted: $JOB_ID"
echo ""
echo "📊 Monitor progress:"
echo "  Master: tail -f tests/logs/optuna_test-${JOB_ID}_0.out"
echo "  Worker: tail -f tests/logs/optuna_test-${JOB_ID}_1.out"
echo ""
echo "📋 Check job status:"
echo "  squeue -j $JOB_ID"
echo ""
echo "🔍 What this test validates:"
echo "  ✓ Distributed master-worker communication (2 nodes)"
echo "  ✓ File-based coordination (params, acks, results)"
echo "  ✓ Batch-wise incremental reporting"
echo "  ✓ Trial resumption (via --resume flag on requeue)"
echo "  ✓ Graceful shutdown on SLURM signal (USR1 @ 2min before timeout)"
echo "  ✓ Multi-level resumption (trial/equation/run)"
echo ""
echo "⏰ Test configuration:"
echo "  • Runtime: 5 minutes"
echo "  • Signal: USR1 at 2 minutes before timeout"
echo "  • Trials: 4"
echo "  • Equations: 8 (split across 2 nodes)"
echo "  • Batch size: 2 equations per batch"
echo ""
echo "📁 After completion, check:"
echo "  Coordination: ls -lah $COORD_DIR"
echo "  Trial dirs: ls -d ${TRIAL_DIR}/trial_*"
echo "  Database: sqlite3 $DB_FILE 'SELECT * FROM trials;'"
echo ""
echo "🎯 Expected behavior:"
echo "  1. Master broadcasts trial params → workers ack"
echo "  2. Master processes 4 equations, workers process 4 equations"
echo "  3. Batch reports every 2 equations (8 reports total)"
echo "  4. If signal received: graceful shutdown + requeue"
echo "  5. On requeue: resume from last checkpoint"
echo ""
