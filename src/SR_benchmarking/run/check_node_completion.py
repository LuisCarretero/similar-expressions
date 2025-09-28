#!/usr/bin/env python3
"""
Node completion checker for SLURM requeuing.
Checks if a specific node has remaining work and handles requeue decisions.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

def get_node_completion_status(config_file: str, node_id: int, total_nodes: int) -> Tuple[bool, Dict]:
    """Check completion status for a specific node.

    Args:
        config_file: Path to config file
        node_id: Node ID (0-indexed)
        total_nodes: Total number of nodes

    Returns:
        Tuple of (has_remaining_work, status_info)
    """
    try:
        # Add current directory to path to import modules
        sys.path.insert(0, '.')

        # Suppress output during imports
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from run.utils import load_config
            from run.run_multiple import _parse_eq_idx, get_node_equations

        # Load config and get node-specific equations
        cfg = load_config(config_file)
        log_dir = str(cfg.run_settings.log_dir)
        dataset_name = str(cfg.dataset.name)

        # Parse equations and get this node's subset
        equations = cfg.dataset.equation_indices
        if isinstance(equations, str):
            equations = _parse_eq_idx(equations)

        node_equations = get_node_equations(equations, node_id, total_nodes)
        total_assigned = len(node_equations)

        if total_assigned == 0:
            return False, {
                'config': config_file,
                'total_assigned': 0,
                'completed': 0,
                'remaining': 0,
                'log_dir': log_dir,
                'dataset': dataset_name
            }

        # Count completed equations for this node
        completed_dir = Path(log_dir) / "completed"
        completed_count = 0

        if completed_dir.exists():
            for eq_id in node_equations:
                done_file = completed_dir / f"{dataset_name}_eq{eq_id}.done"
                if done_file.exists():
                    completed_count += 1

        remaining = total_assigned - completed_count
        has_remaining = remaining > 0

        status_info = {
            'config': config_file,
            'total_assigned': total_assigned,
            'completed': completed_count,
            'remaining': remaining,
            'log_dir': log_dir,
            'dataset': dataset_name,
            'node_equations': node_equations[:10] if len(node_equations) <= 10 else f"{node_equations[:10]}... (+{len(node_equations)-10} more)"
        }

        return has_remaining, status_info

    except Exception as e:
        print(f"ERROR: Failed to check completion for {config_file}: {e}", file=sys.stderr)
        return True, {  # Assume work remaining on error
            'config': config_file,
            'error': str(e),
            'total_assigned': -1,
            'completed': -1,
            'remaining': -1
        }


def check_node_and_requeue(node_id: int, total_nodes: int, config_files: List[str], job_id: str) -> bool:
    """Check if node has remaining work and requeue if needed.

    Args:
        node_id: Node ID (0-indexed)
        total_nodes: Total number of nodes
        config_files: List of config files to check
        job_id: SLURM job ID for requeuing

    Returns:
        True if requeue was requested, False otherwise
    """
    print(f"[{node_id}] Checking completion status...")

    overall_has_remaining = False
    status_summary = []

    for config_file in config_files:
        has_remaining, status = get_node_completion_status(config_file, node_id, total_nodes)
        status_summary.append(status)

        if 'error' in status:
            print(f"[{node_id}] ERROR checking {config_file}: {status['error']}")
            overall_has_remaining = True  # Conservative: assume work remaining on error
        else:
            print(f"[{node_id}] {config_file}: {status['completed']}/{status['total_assigned']} equations completed")
            if status['total_assigned'] > 0:
                print(f"[{node_id}] {config_file}: {status['remaining']} equations remaining")
                if status['remaining'] > 0:
                    print(f"[{node_id}] {config_file}: Sample equations: {status['node_equations']}")

        if has_remaining:
            overall_has_remaining = True

    # Print summary
    print(f"[{node_id}] Summary:")
    for status in status_summary:
        if 'error' not in status and status['total_assigned'] > 0:
            config_name = os.path.basename(status['config']).replace('.yaml', '')
            print(f"[{node_id}]   {config_name}: {status['completed']}/{status['total_assigned']} done")

    # Decide whether to requeue
    if overall_has_remaining:
        print(f"[{node_id}] Node has remaining work. Requesting requeue...")
        return request_requeue(job_id, node_id)
    else:
        print(f"[{node_id}] All assigned work completed. No requeue needed.")
        return False


def request_requeue(job_id: str, node_id: int) -> bool:
    """Request SLURM to requeue the job.

    Args:
        job_id: SLURM job ID
        node_id: Node ID for logging

    Returns:
        True if requeue was successful, False otherwise
    """
    try:
        subprocess.run(
            ["scontrol", "requeue", job_id],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[{node_id}] Successfully requested requeue for job {job_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{node_id}] ERROR: Failed to requeue job {job_id}: {e.stderr.strip()}")
        return False
    except Exception as e:
        print(f"[{node_id}] ERROR: Unexpected error during requeue: {e}")
        return False


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Check node completion and handle requeuing')
    parser.add_argument('--node_id', type=int, required=True, help='Node ID (0-indexed)')
    parser.add_argument('--total_nodes', type=int, required=True, help='Total number of nodes')
    parser.add_argument('--job_id', required=True, help='SLURM job ID for requeuing')
    parser.add_argument('--config_files', nargs='+', required=True, help='Config files to check')

    args = parser.parse_args()

    try:
        # Suppress warnings during import
        import warnings
        warnings.filterwarnings("ignore")

        requeued = check_node_and_requeue(
            args.node_id,
            args.total_nodes,
            args.config_files,
            args.job_id
        )

        # Exit code indicates whether requeue was requested
        sys.exit(0 if not requeued else 1)

    except Exception as e:
        print(f"[{args.node_id}] FATAL ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()