#!/usr/bin/env python3
"""
Helper script to extract job information from config files for SLURM resubmission logic.
Returns log_dir, dataset_name, and total_equations count in a clean format.
"""

import sys
import os
from pathlib import Path

def get_job_info(config_file):
    """Extract job information from config file with error handling."""
    try:
        # Add current directory to path to import modules
        sys.path.insert(0, '.')

        # Suppress output during imports
        import contextlib
        import io

        # Capture and discard any output during imports
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from run.utils import load_config
            from run.run_multiple import _parse_eq_idx

        # Load config
        cfg = load_config(config_file)

        # Extract log directory
        log_dir = str(cfg.run_settings.log_dir)

        # Extract dataset name
        dataset_name = str(cfg.dataset.name)

        # Extract and parse equation indices
        equations = cfg.dataset.equation_indices
        if isinstance(equations, str):
            equations = _parse_eq_idx(equations)
        total_equations = len(equations)

        return log_dir, dataset_name, total_equations

    except Exception as e:
        # Fallback values if anything goes wrong
        print(f"ERROR: Failed to parse config file '{config_file}': {e}", file=sys.stderr)
        return "/tmp/fallback_logs", "unknown", 0

def main():
    """Main function to handle command line arguments and output results."""
    if len(sys.argv) != 2:
        print("Usage: python get_job_info.py <config_file>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]

    # Check if config file exists
    if not os.path.exists(config_file):
        print(f"ERROR: Config file '{config_file}' not found", file=sys.stderr)
        print("/tmp/fallback_logs unknown 0")
        sys.exit(1)

    try:
        # Suppress Julia/Python initialization messages that go to stderr
        import warnings
        warnings.filterwarnings("ignore")

        log_dir, dataset_name, total_equations = get_job_info(config_file)

        # Output in space-separated format for easy parsing in bash
        print(f"{log_dir} {dataset_name} {total_equations}")

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        print("/tmp/fallback_logs unknown 0")
        sys.exit(1)

if __name__ == "__main__":
    main()