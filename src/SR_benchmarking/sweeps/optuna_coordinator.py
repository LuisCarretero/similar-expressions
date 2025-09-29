"""
Optuna distributed coordination system.

Handles distributed execution of Optuna trials across multiple SLURM nodes,
with file-based communication and seamless single-node fallback.
"""

import json
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Add parent directories to path for imports
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from run.utils import (
    ModelSettings, NeuralOptions, MutationWeights, DatasetSettings,
    init_pysr_model, run_single
)
from run.run_multiple import get_node_equations


class OptunaCoordinator:
    """
    Coordinates distributed execution of Optuna trials across multiple nodes.

    Handles both single-node and distributed modes seamlessly through file-based
    communication. In single-node mode, acts as a pass-through with no overhead.
    """

    def __init__(self, node_id: Optional[int], total_nodes: Optional[int], config: Dict[str, Any]):
        """
        Initialize the coordinator.

        Args:
            node_id: Node ID (0-indexed), None for single-node mode
            total_nodes: Total number of nodes, None for single-node mode
            config: Full Optuna configuration dictionary
        """
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.config = config

        # Mode detection
        self.single_node = (node_id is None or total_nodes is None or total_nodes == 1)
        self.is_master = (node_id == 0) if not self.single_node else True
        self.is_worker = not self.is_master and not self.single_node

        # Setup coordination directory
        if not self.single_node:
            self.coord_dir = Path("/tmp") / f"optuna_coord_{config['study']['name']}"
            self.coord_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"OptunaCoordinator initialized: "
                        f"node_id={node_id}, total_nodes={total_nodes}, "
                        f"single_node={self.single_node}, is_master={self.is_master}")

    def execute_trial(self, trial_params: Dict[str, Any], interruption_flag=None) -> List[float]:
        """
        Execute a trial, handling both single-node and distributed modes.

        Args:
            trial_params: Dictionary containing model settings, neural options, mutation weights
            interruption_flag: Function that returns True if execution should be interrupted

        Returns:
            List of pareto volumes from all equations
        """
        if self.single_node:
            return self._run_local_trial(trial_params, interruption_flag)
        elif self.is_master:
            return self._run_distributed_trial(trial_params, interruption_flag)
        else:
            raise RuntimeError("Workers should not call execute_trial - use run_worker_loop instead")

    def _run_local_trial(self, trial_params: Dict[str, Any], interruption_flag=None) -> List[float]:
        """Run trial on single node (current behavior)."""
        model_settings = trial_params['model_settings']
        neural_options = trial_params['neural_options']
        mutation_weights = trial_params['mutation_weights']

        # Parse equations
        equations = self._parse_equation_indices(self.config['dataset']['equation_indices'])

        # Run all equations locally
        return self._run_equation_batch(
            equations, model_settings, neural_options, mutation_weights, interruption_flag
        )

    def _run_distributed_trial(self, trial_params: Dict[str, Any], interruption_flag=None) -> List[float]:
        """Run trial distributed across nodes."""
        trial_id = int(time.time() * 1000000)  # Unique trial ID

        try:
            # Broadcast trial parameters to workers
            self._broadcast_trial_params(trial_id, trial_params)

            # Get master's equation subset
            equations = self._parse_equation_indices(self.config['dataset']['equation_indices'])
            master_equations = get_node_equations(equations, 0, self.total_nodes)

            # Run master's equations
            master_results = self._run_equation_batch(
                master_equations,
                trial_params['model_settings'],
                trial_params['neural_options'],
                trial_params['mutation_weights'],
                interruption_flag
            )

            # Wait for worker results
            all_results = [master_results]
            for worker_id in range(1, self.total_nodes):
                worker_results = self._wait_for_worker_results(trial_id, worker_id)
                if worker_results is not None:
                    all_results.append(worker_results)
                else:
                    self.logger.warning(f"No results received from worker {worker_id}")

            # Flatten all results
            final_results = []
            for results in all_results:
                final_results.extend(results)

            return final_results

        finally:
            # Clean up coordination files
            self._cleanup_trial_files(trial_id)

    def run_worker_loop(self, interruption_flag=None):
        """
        Main loop for worker nodes - continuously process trials from master.
        """
        if not self.is_worker:
            raise RuntimeError("Only worker nodes should run worker loop")

        self.logger.info(f"Worker {self.node_id} starting worker loop")

        while True:
            if interruption_flag and interruption_flag():
                self.logger.info(f"Worker {self.node_id} interrupted, shutting down")
                break

            trial_params = self._wait_for_trial_params()
            if trial_params is None:
                # Check for shutdown signal
                if self._check_shutdown_signal():
                    break
                time.sleep(1.0)  # Wait before checking again
                continue

            trial_id = trial_params['trial_id']
            self.logger.info(f"Worker {self.node_id} processing trial {trial_id}")

            try:
                # Get worker's equation subset
                equations = self._parse_equation_indices(self.config['dataset']['equation_indices'])
                worker_equations = get_node_equations(equations, self.node_id, self.total_nodes)

                # Process worker's equations
                results = self._run_equation_batch(
                    worker_equations,
                    trial_params['model_settings'],
                    trial_params['neural_options'],
                    trial_params['mutation_weights'],
                    interruption_flag
                )

                # Send results back to master
                self._send_worker_results(trial_id, results)

            except Exception as e:
                self.logger.error(f"Worker {self.node_id} error processing trial {trial_id}: {e}")
                # Send empty results to unblock master
                self._send_worker_results(trial_id, [])

    def _broadcast_trial_params(self, trial_id: int, trial_params: Dict[str, Any]):
        """Broadcast trial parameters to all workers."""
        broadcast_data = {
            'trial_id': trial_id,
            **trial_params
        }

        params_file = self.coord_dir / f"trial_{trial_id}_params.json"
        with open(params_file, 'w') as f:
            json.dump(broadcast_data, f, indent=2)

        self.logger.info(f"Broadcast trial {trial_id} parameters to workers")

    def _wait_for_trial_params(self, timeout_sec: float = 5.0) -> Optional[Dict[str, Any]]:
        """Worker waits for trial parameters from master."""
        # Look for any trial params file
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            params_files = list(self.coord_dir.glob("trial_*_params.json"))
            if params_files:
                params_file = params_files[0]  # Take first available
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    # Remove params file so other workers don't process it
                    params_file.unlink(missing_ok=True)
                    return params
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
            time.sleep(0.1)
        return None

    def _send_worker_results(self, trial_id: int, results: List[float]):
        """Worker sends results back to master."""
        results_file = self.coord_dir / f"trial_{trial_id}_node_{self.node_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump({'results': results}, f)

        self.logger.info(f"Worker {self.node_id} sent {len(results)} results for trial {trial_id}")

    def _wait_for_worker_results(self, trial_id: int, worker_id: int, timeout_sec: float = 7200.0) -> Optional[List[float]]:
        """Master waits for results from a specific worker."""
        results_file = self.coord_dir / f"trial_{trial_id}_node_{worker_id}_results.json"

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    return data['results']
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            time.sleep(1.0)

        self.logger.error(f"Timeout waiting for results from worker {worker_id} for trial {trial_id}")
        return None

    def _check_shutdown_signal(self) -> bool:
        """Check if master has signaled shutdown."""
        shutdown_file = self.coord_dir / "shutdown.signal"
        return shutdown_file.exists()

    def signal_shutdown(self):
        """Master signals all workers to shutdown."""
        if self.single_node or not self.is_master:
            return

        shutdown_file = self.coord_dir / "shutdown.signal"
        shutdown_file.touch()
        self.logger.info("Signaled shutdown to all workers")

    def _cleanup_trial_files(self, trial_id: int):
        """Clean up coordination files for a trial."""
        if self.single_node:
            return

        patterns = [
            f"trial_{trial_id}_params.json",
            f"trial_{trial_id}_node_*_results.json"
        ]

        for pattern in patterns:
            for file_path in self.coord_dir.glob(pattern):
                file_path.unlink(missing_ok=True)

    def _parse_equation_indices(self, eq_str: str) -> List[int]:
        """Parse equation indices string (reuse logic from run_multiple.py)."""
        from run.run_multiple import _parse_eq_idx
        return _parse_eq_idx(eq_str)

    def _run_equation_batch(
        self,
        equations: List[int],
        model_settings: ModelSettings,
        neural_options: NeuralOptions,
        mutation_weights: MutationWeights,
        interruption_flag=None
    ) -> List[float]:
        """
        Run a batch of equations and return pareto volumes.

        This mirrors the logic from OptunaObjective._run_equation_batch
        """
        if interruption_flag is None:
            interruption_flag = lambda: False

        pareto_volumes = []
        packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)

        # Create temp directory for this batch
        import tempfile
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        node_suffix = f"_node{self.node_id}" if not self.single_node else ""
        temp_dir = Path(tempfile.mkdtemp(suffix=f"_optuna_batch_{timestamp}{node_suffix}"))

        try:
            for eq_idx in equations:
                if interruption_flag():
                    self.logger.info(f"Interrupted during equation batch processing at equation {eq_idx}")
                    break

                try:
                    eq_pareto_volumes = []
                    n_runs = self.config['dataset']['n_runs']

                    # Run multiple times per equation for variance reduction
                    for run_i in range(n_runs):
                        # Create unique run directory
                        run_dir = temp_dir / f"{self.config['dataset']['name']}_eq{eq_idx}_run{run_i}"
                        if run_dir.exists():
                            shutil.rmtree(run_dir)

                        # Create dataset settings
                        dataset_settings = DatasetSettings(
                            dataset_name=self.config['dataset']['name'],
                            eq_idx=eq_idx,
                            num_samples=self.config['dataset']['num_samples'],
                            rel_noise_magn=self.config['dataset']['noise'],
                            remove_op_equations=self.config['dataset'].get('remove_op_equations', {})
                        )

                        # Run single SR experiment
                        run_single(
                            packaged_model=packaged_model,
                            dataset_settings=dataset_settings,
                            log_dir=str(run_dir),
                            wandb_logging=False,  # No WandB logging for individual runs
                            enable_mutation_logging=False  # Disable for speed
                        )

                        # Extract pareto volume from results
                        final_pareto_volume = self._extract_run_pv(run_dir)
                        eq_pareto_volumes.append(final_pareto_volume)

                    # Use mean pareto volume across runs
                    eq_mean_pv = np.mean(eq_pareto_volumes)
                    pareto_volumes.append(eq_mean_pv)

                    node_info = f" (node {self.node_id})" if not self.single_node else ""
                    self.logger.info(f"Equation {eq_idx}{node_info}: PV = {eq_mean_pv:.4f} "
                                   f"(runs: {eq_pareto_volumes})")

                except Exception as e:
                    self.logger.error(f"Failed to run equation {eq_idx}: {e}")
                    pareto_volumes.append(0.0)  # Assign poor score for failed runs

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

        return pareto_volumes

    def _extract_run_pv(self, run_dir: Path) -> float:
        """Extract pareto volume from a single run directory."""
        import pandas as pd

        csv_path = run_dir / 'tensorboard_scalars.csv'
        if not csv_path.exists():
            self.logger.warning(f"No tensorboard CSV found at {csv_path}")
            return 0.0

        try:
            df = pd.read_csv(csv_path)
            if 'pareto_volume' not in df.columns:
                self.logger.warning(f"No pareto_volume column in {csv_path}")
                return 0.0
            final_pareto_volume = df['pareto_volume'].iloc[-1]
            return float(final_pareto_volume)
        except Exception as e:
            self.logger.error(f"Error extracting pareto volume from {csv_path}: {e}")
            return 0.0

    def cleanup(self):
        """Clean up coordinator resources."""
        if not self.single_node and self.is_master:
            self.signal_shutdown()
            # Give workers time to see shutdown signal
            time.sleep(2.0)

            # Clean up coordination directory
            if hasattr(self, 'coord_dir') and self.coord_dir.exists():
                try:
                    shutil.rmtree(self.coord_dir)
                    self.logger.info("Cleaned up coordination directory")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up coordination directory: {e}")