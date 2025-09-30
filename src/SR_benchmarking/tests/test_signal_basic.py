#!/usr/bin/env python3
"""
Minimal signal handling test script.

This script tests if Python signal handlers work correctly in SLURM environment.
It runs for ~5 minutes and logs when signals are received.
"""

import signal
import time
import sys
import os
from datetime import datetime

# Global flag to track interruption
interrupted = False
signal_received = None

def signal_handler(signum, frame):
    """Handle signals and log reception."""
    global interrupted, signal_received

    signal_names = {
        signal.SIGUSR1: "USR1",
        signal.SIGTERM: "TERM",
        signal.SIGINT: "INT"
    }

    signal_name = signal_names.get(signum, str(signum))
    signal_received = signal_name

    print(f"[{datetime.now()}] *** SIGNAL RECEIVED: {signal_name} ({signum}) ***", flush=True)
    print(f"[{datetime.now()}] Setting interrupted flag and preparing for graceful shutdown...", flush=True)

    interrupted = True

def main():
    # Set up signal handlers
    signal.signal(signal.SIGUSR1, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Log startup info
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
    print(f"[{datetime.now()}] Starting signal test script", flush=True)
    print(f"[{datetime.now()}] SLURM Job ID: {slurm_job_id}", flush=True)
    print(f"[{datetime.now()}] Process ID: {os.getpid()}", flush=True)
    print(f"[{datetime.now()}] Signal handlers initialized for USR1, TERM, INT", flush=True)
    print(f"[{datetime.now()}] Starting 5-minute loop...", flush=True)

    # Run for ~5 minutes (300 seconds) with 10-second intervals
    start_time = time.time()
    iteration = 0

    while not interrupted and (time.time() - start_time) < 300:
        iteration += 1
        elapsed = time.time() - start_time

        print(f"[{datetime.now()}] Iteration {iteration}, elapsed: {elapsed:.1f}s", flush=True)

        # Sleep in small chunks to be responsive to signals
        for _ in range(10):  # 10 x 1-second sleeps = 10 seconds total
            if interrupted:
                break
            time.sleep(1)

    # Final status
    total_time = time.time() - start_time

    if interrupted:
        print(f"[{datetime.now()}] *** GRACEFUL SHUTDOWN SUCCESSFUL ***", flush=True)
        print(f"[{datetime.now()}] Signal received: {signal_received}", flush=True)
        print(f"[{datetime.now()}] Total runtime: {total_time:.1f}s", flush=True)
        print(f"[{datetime.now()}] Exiting with code 0", flush=True)
        sys.exit(0)
    else:
        print(f"[{datetime.now()}] *** NO SIGNAL RECEIVED - COMPLETED FULL RUNTIME ***", flush=True)
        print(f"[{datetime.now()}] Total runtime: {total_time:.1f}s", flush=True)
        print(f"[{datetime.now()}] Exiting with code 1 (no signal test failed)", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()