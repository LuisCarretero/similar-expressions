"""
Signal manager for clean SLURM interruption handling.

This module provides a simple, consistent approach to handling SLURM signals
across all run scripts. It mirrors the clean signal handling approach from
the Optuna implementation in sweeps/.
"""

import signal
import logging
import os


class InterruptionManager:
    """
    Manages SLURM signal interruption with a simple interrupted flag.

    This mirrors the signal handling pattern from sweeps/optuna_hyperopt.py
    but is designed for the general run/ scripts.
    """

    def __init__(self, logger_name: str = None):
        """
        Initialize the interruption manager.

        Args:
            logger_name: Name for the logger. If None, uses the module name.
        """
        self.interrupted = False

        # Set up logging
        self.logger = logging.getLogger(logger_name or __name__)

        # Set up signal handlers for SLURM signals
        signal.signal(signal.SIGUSR1, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Log signal handler setup
        slurm_job_id = os.environ.get('SLURM_JOB_ID', 'N/A')
        self.logger.info(f"Signal handlers initialized for job {slurm_job_id}")

    def _signal_handler(self, signum, frame):
        """
        Handle interruption signals gracefully.

        This mirrors the signal handler from optuna_hyperopt.py exactly.
        """
        signal_names = {
            signal.SIGUSR1: "USR1",
            signal.SIGTERM: "TERM",
            signal.SIGINT: "INT"
        }
        signal_name = signal_names.get(signum, str(signum))

        self.logger.info(f"Received signal {signal_name} ({signum}), preparing for graceful shutdown...")
        self.interrupted = True

    def check_interrupted(self) -> bool:
        """
        Check if execution should be interrupted.

        Returns:
            True if interrupted, False otherwise
        """
        return self.interrupted

    def create_checker(self):
        """
        Create a callable checker function for use in other modules.

        This allows passing the interruption check as a lambda function,
        similar to how it's done in the Optuna implementation.

        Returns:
            Callable that returns True if interrupted
        """
        return lambda: self.interrupted


def create_interruption_manager(logger_name: str = None) -> InterruptionManager:
    """
    Factory function to create an interruption manager.

    Args:
        logger_name: Name for the logger

    Returns:
        InterruptionManager instance
    """
    return InterruptionManager(logger_name)