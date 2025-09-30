"""
Signal manager for clean SLURM interruption handling via file-based flag.

This module uses file-based flags instead of signal handlers to avoid
conflicts with Julia's signal handling. When SLURM sends a signal to the bash
script, the bash trap creates a flag file, which Python checks regularly at
safe interruption points.
"""

import logging
import os
from pathlib import Path


class InterruptionManager:
    """
    Manages SLURM interruption via file-based flag checking.

    Avoids signal handler conflicts with Julia by checking for a flag file that
    the bash trap creates. This allows Julia to complete its work naturally
    without signal interference.
    """

    def __init__(self, logger_name: str = None):
        """
        Initialize the interruption manager.

        Args:
            logger_name: Name for the logger. If None, uses the module name.
        """
        self.interrupted = False
        self.logger = logging.getLogger(logger_name or __name__)

        job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
        array_task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        self.flag_file = Path(f"/tmp/slurm_shutdown_{job_id}_{array_task_id}")
        self.logger.info(f"Interruption manager initialized, watching flag: {self.flag_file}")

    def check_interrupted(self) -> bool:
        """
        Check if execution should be interrupted by checking flag file existence.

        The bash trap creates a flag file when it receives a signal, and this
        method checks for that file at safe interruption points.

        Returns:
            True if interrupted, False otherwise
        """
        # Check flag file (created by bash trap)
        if self.flag_file.exists():
            if not self.interrupted:
                self.logger.info(f"Detected SLURM shutdown request via flag file: {self.flag_file}")
                self.logger.info("Will complete current work and shutdown gracefully")
                self.interrupted = True
            return True

        return self.interrupted

    def create_checker(self):
        """
        Create a callable checker function for use in other modules.

        This allows passing the interruption check as a lambda function,
        compatible with existing code that expects a callable.

        Returns:
            Callable that returns True if interrupted
        """
        return lambda: self.check_interrupted()


def create_interruption_manager(logger_name: str = None) -> InterruptionManager:
    return InterruptionManager(logger_name)
