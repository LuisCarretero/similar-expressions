"""
Optuna distributed coordination system with trial resumption support.

Handles:
- Distributed execution of Optuna trials across multiple SLURM nodes
- File-based communication for worker coordination
- Trial resumption after SLURM requeues by scanning existing results
"""

import json
import time
import shutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd

# Add parent directories to path for imports
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from run.utils import (
    ModelSettings, NeuralOptions, MutationWeights, DatasetSettings,
    init_pysr_model, run_single
)
from run.run_multiple import get_node_equations, _parse_eq_idx


class OptunaCoordinator:
    """
    Coordinates distributed execution of Optuna trials across multiple nodes.

    Handles distributed modes through file-based communication.
    """

    def __init__(self, node_id: int, total_nodes: int, config: Dict[str, Any]):
        """
        Initialize the coordinator.

        Args:
            node_id: Node ID (0-indexed)
            total_nodes: Total number of nodes
            config: Full Optuna configuration dictionary
        """
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.config = config

        # Mode detection
        self.is_master = (node_id == 0)
        self.is_worker = not self.is_master

        # Setup logging first
        self.logger = logging.getLogger(__name__)

        # Setup coordination directory - one per study for persistence across job submissions
        base_dir = Path("/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment")
        self.coord_dir = base_dir / f"optuna_coord_{config['study']['name']}"
        self.coord_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Using coordination directory: {self.coord_dir}")

        # Track processed trials to prevent reprocessing
        if self.is_worker:
            self.processed_trials_file = self.coord_dir / f"worker_{self.node_id}_processed.json"
            if self.processed_trials_file.exists():
                with open(self.processed_trials_file, 'r') as f:
                    self.processed_trial_ids = set(json.load(f))
                self.logger.info(f"Worker {self.node_id} loaded {len(self.processed_trial_ids)} processed trials")
            else:
                self.processed_trial_ids = set()
                self._save_processed_trials()
                self.logger.info(f"Worker {self.node_id} initialized with empty processed trials")
        else:
            self.processed_trial_ids = set()

        self.logger.info(f"OptunaCoordinator initialized: "
                        f"node_id={node_id}, total_nodes={total_nodes}, "
                        f"is_master={self.is_master}")

    def _save_processed_trials(self):
        """Save processed trial IDs to file."""
        if not self.is_worker:
            return
        with open(self.processed_trials_file, 'w') as f:
            json.dump(list(self.processed_trial_ids), f)

    def _load_trial_progress(self, trial_id: int) -> Optional[Dict[str, Any]]:
        """Load progress for a trial if it exists."""
        if not self.is_worker:
            return None
        progress_file = self.coord_dir / f"trial_{trial_id}_node_{self.node_id}_progress.json"
        if progress_file.exists():
            try:
                with open(progress_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return None
        return None

    def _save_trial_progress(self, trial_id: int, completed_equations: List[int], results: List[float]):
        """Save progress for a trial."""
        if not self.is_worker:
            return
        progress_file = self.coord_dir / f"trial_{trial_id}_node_{self.node_id}_progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                'completed_equations': completed_equations,
                'results': results
            }, f)

    def _delete_trial_progress(self, trial_id: int):
        """Delete progress file after trial completion."""
        if not self.is_worker:
            return
        progress_file = self.coord_dir / f"trial_{trial_id}_node_{self.node_id}_progress.json"
        progress_file.unlink(missing_ok=True)

    def _reconstruct_objects(self, trial_params: Dict[str, Any]) -> Tuple[ModelSettings, NeuralOptions, MutationWeights]:
        """
        Reconstruct ModelSettings, NeuralOptions, MutationWeights objects from serialized dictionaries.
        """
        # Reconstruct ModelSettings
        ms_dict = trial_params['model_settings']
        model_settings = ModelSettings(
            niterations=ms_dict['niterations'],
            loss_function=ms_dict['loss_function'],
            early_stopping_condition=ms_dict['early_stopping_condition'],
            verbosity=ms_dict['verbosity'],
            precision=ms_dict['precision'],
            batching=ms_dict['batching'],
            batch_size=ms_dict['batch_size']
        )

        # Reconstruct NeuralOptions
        no_dict = trial_params['neural_options']
        neural_options = NeuralOptions(
            active=no_dict['active'],
            model_path=no_dict['model_path'],
            sampling_eps=no_dict['sampling_eps'],
            subtree_min_nodes=no_dict['subtree_min_nodes'],
            subtree_max_nodes=no_dict['subtree_max_nodes'],
            device=no_dict['device'],
            verbose=no_dict['verbose'],
            max_tree_size_diff=no_dict['max_tree_size_diff'],
            require_tree_size_similarity=no_dict['require_tree_size_similarity'],
            require_novel_skeleton=no_dict['require_novel_skeleton'],
            require_expr_similarity=no_dict['require_expr_similarity'],
            similarity_threshold=no_dict['similarity_threshold'],
            log_subtree_strings=no_dict['log_subtree_strings'],
            sample_logits=no_dict['sample_logits'],
            max_resamples=no_dict['max_resamples'],
            sample_batchsize=no_dict['sample_batchsize'],
            subtree_max_features=no_dict['subtree_max_features']
        )

        # Reconstruct MutationWeights
        mw_dict = trial_params['mutation_weights']
        mutation_weights = MutationWeights(
            weight_add_node=mw_dict['weight_add_node'],
            weight_insert_node=mw_dict['weight_insert_node'],
            weight_delete_node=mw_dict['weight_delete_node'],
            weight_do_nothing=mw_dict['weight_do_nothing'],
            weight_mutate_constant=mw_dict['weight_mutate_constant'],
            weight_mutate_operator=mw_dict['weight_mutate_operator'],
            weight_swap_operands=mw_dict['weight_swap_operands'],
            weight_rotate_tree=mw_dict['weight_rotate_tree'],
            weight_randomize=mw_dict['weight_randomize'],
            weight_simplify=mw_dict['weight_simplify'],
            weight_optimize=mw_dict['weight_optimize'],
            weight_neural_mutate_tree=mw_dict['weight_neural_mutate_tree']
        )

        return model_settings, neural_options, mutation_weights

    def execute_trial(self, trial_params: Dict[str, Any], interruption_flag=None) -> List[float]:
        """
        Execute a distributed trial.

        Args:
            trial_params: Dictionary containing model settings, neural options, mutation weights
            interruption_flag: Function that returns True if execution should be interrupted

        Returns:
            List of pareto volumes from all equations
        """
        if self.is_master:
            return self._run_distributed_trial(trial_params, interruption_flag)
        else:
            raise RuntimeError("Workers should not call execute_trial - use run_worker_loop instead")

    def _run_distributed_trial(self, trial_params: Dict[str, Any], interruption_flag=None) -> List[float]:
        """Run trial distributed across nodes."""
        # Get trial_id (must be provided by objective function)
        trial_id = trial_params['trial_id']

        try:
            # Broadcast trial parameters to workers
            self._broadcast_trial_params(trial_id, trial_params)

            # Wait for all workers to acknowledge receipt
            self._wait_for_worker_acknowledgments(trial_id, interruption_flag)

            # Get master's equation subset
            equations = _parse_eq_idx(self.config['dataset']['equation_indices'])
            master_equations = get_node_equations(equations, 0, self.total_nodes)

            # Reconstruct objects from trial params
            model_settings, neural_options, mutation_weights = self._reconstruct_objects(trial_params)

            # Run master's equations
            master_results = self._run_equation_batch(
                master_equations,
                model_settings,
                neural_options,
                mutation_weights,
                trial_id,
                interruption_flag
            )

            # Wait for worker results
            all_results = [master_results]
            for worker_id in range(1, self.total_nodes):
                worker_results = self._wait_for_worker_results(trial_id, worker_id, interruption_flag=interruption_flag)
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

            self.logger.info(f"Worker {self.node_id} waiting for trial parameters...")
            trial_params = self._wait_for_trial_params(timeout_sec=10.0)  # Longer timeout
            if trial_params is None:
                # Check for shutdown signal
                if self._check_shutdown_signal():
                    self.logger.info(f"Worker {self.node_id} received shutdown signal")
                    break
                self.logger.info(f"Worker {self.node_id} no trial parameters found, continuing to wait...")
                time.sleep(1.0)  # Wait before checking again
                continue

            trial_id = trial_params['trial_id']

            # Check for existing progress (resumption after requeue)
            progress = self._load_trial_progress(trial_id)
            if progress:
                self.logger.info(f"Worker {self.node_id} resuming trial {trial_id} with {len(progress['completed_equations'])} equations already completed")
            else:
                self.logger.info(f"Worker {self.node_id} starting fresh trial {trial_id}")

            try:
                # Get worker's equation subset
                equations = _parse_eq_idx(self.config['dataset']['equation_indices'])
                worker_equations = get_node_equations(equations, self.node_id, self.total_nodes)

                self.logger.info(f"Worker {self.node_id} will process equations {worker_equations} for trial {trial_id}")

                # Reconstruct objects from trial params
                model_settings, neural_options, mutation_weights = self._reconstruct_objects(trial_params)

                # Process worker's equations (with resumption support)
                results = self._run_equation_batch(
                    worker_equations,
                    model_settings,
                    neural_options,
                    mutation_weights,
                    trial_id,
                    interruption_flag,
                    progress
                )

                # Send results back to master
                self._send_worker_results(trial_id, results)

                # Mark as processed and cleanup
                self.processed_trial_ids.add(trial_id)
                self._save_processed_trials()
                self._delete_trial_progress(trial_id)
                self.logger.info(f"Worker {self.node_id} marked trial {trial_id} as processed")

            except Exception as e:
                self.logger.error(f"Worker {self.node_id} error processing trial {trial_id}: {e}")
                # Send empty results to unblock master
                self._send_worker_results(trial_id, [])

                # Mark as processed even on error to prevent retry loop
                self.processed_trial_ids.add(trial_id)
                self._save_processed_trials()
                self._delete_trial_progress(trial_id)
                self.logger.info(f"Worker {self.node_id} marked failed trial {trial_id} as processed")

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

    def _wait_for_worker_acknowledgments(self, trial_id: int, interruption_flag=None, timeout_sec: float = 30.0):
        """Master waits for all workers to acknowledge they received parameters."""
        if interruption_flag is None:
            interruption_flag = lambda: False

        expected_workers = list(range(1, self.total_nodes))  # Workers are nodes 1, 2, 3, etc.
        received_acks = set()

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            # Check for interruption
            if interruption_flag():
                self.logger.info("Interrupted while waiting for worker acknowledgments")
                return

            # Check for acknowledgment files
            for worker_id in expected_workers:
                if worker_id not in received_acks:
                    ack_file = self.coord_dir / f"trial_{trial_id}_worker_{worker_id}_ack.json"
                    if ack_file.exists():
                        received_acks.add(worker_id)
                        self.logger.info(f"Received acknowledgment from worker {worker_id} for trial {trial_id}")

            # Check if all workers have acknowledged
            if len(received_acks) == len(expected_workers):
                self.logger.info(f"All workers acknowledged trial {trial_id} parameters")
                return

            time.sleep(0.1)

        # Log which workers didn't respond
        missing_workers = set(expected_workers) - received_acks
        self.logger.warning(f"Timeout waiting for acknowledgments from workers {missing_workers} for trial {trial_id}")

    def _wait_for_trial_params(self, timeout_sec: float = 5.0) -> Optional[Dict[str, Any]]:
        """Worker waits for trial parameters from master."""
        # Look for any trial params file
        self.logger.info(f"Worker {self.node_id} looking for params files in: {self.coord_dir}")
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            params_files = list(self.coord_dir.glob("trial_*_params.json"))
            self.logger.debug(f"Worker {self.node_id} found {len(params_files)} params files")
            if params_files:
                params_file = params_files[0]  # Take first available
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)

                    trial_id = params['trial_id']

                    # Skip if already processed
                    if trial_id in self.processed_trial_ids:
                        self.logger.info(f"Worker {self.node_id} skipping already processed trial {trial_id}")
                        time.sleep(0.1)
                        continue

                    # Create acknowledgment file to signal we got the parameters
                    ack_file = self.coord_dir / f"trial_{trial_id}_worker_{self.node_id}_ack.json"
                    with open(ack_file, 'w') as f:
                        json.dump({'worker_id': self.node_id, 'status': 'received'}, f)

                    self.logger.info(f"Worker {self.node_id} received trial {trial_id} parameters")
                    return params
                except (json.JSONDecodeError, FileNotFoundError):
                    continue
            time.sleep(0.1)
        return None

    def _send_worker_results(self, trial_id: int, results: List[float]):
        """Worker sends results back to master."""
        results_file = self.coord_dir / f"trial_{trial_id}_node_{self.node_id}_results.json"
        # Ensure parent directory exists (in case of requeue or cleanup)
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump({'results': results}, f)

        self.logger.info(f"Worker {self.node_id} sent {len(results)} results for trial {trial_id}")

    def _wait_for_worker_results(self, trial_id: int, worker_id: int, interruption_flag=None, timeout_sec: float = 7200.0) -> Optional[List[float]]:
        """Master waits for results from a specific worker."""
        if interruption_flag is None:
            interruption_flag = lambda: False

        results_file = self.coord_dir / f"trial_{trial_id}_node_{worker_id}_results.json"

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            # Check for interruption
            if interruption_flag():
                self.logger.info(f"Interrupted while waiting for results from worker {worker_id}")
                return None

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
        if not self.is_master:
            return

        shutdown_file = self.coord_dir / "shutdown.signal"
        shutdown_file.touch()
        self.logger.info("Signaled shutdown to all workers")

    def _cleanup_trial_files(self, trial_id: int):
        """Clean up ephemeral coordination files for a trial.

        Note: params.json files are preserved for resumption logic.
        They are only cleaned up when a trial completes (has .done marker).
        """
        patterns = [
            f"trial_{trial_id}_worker_*_ack.json",
            f"trial_{trial_id}_node_*_results.json"
        ]

        for pattern in patterns:
            for file_path in self.coord_dir.glob(pattern):
                file_path.unlink(missing_ok=True)

    def _run_equation_batch(
        self,
        equations: List[int],
        model_settings: ModelSettings,
        neural_options: NeuralOptions,
        mutation_weights: MutationWeights,
        trial_id: int,
        interruption_flag=None,
        progress: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """
        Run a batch of equations and return pareto volumes.

        Scans for existing results from previous runs and only executes remaining equations.

        Args:
            equations: List of equation indices to process
            model_settings: Model configuration
            neural_options: Neural network options
            mutation_weights: Mutation weight configuration
            trial_id: Unique trial identifier for directory organization
            interruption_flag: Function to check for interruption
            progress: Optional dict with 'completed_equations' and 'results' for resumption (worker-level)
        """
        if interruption_flag is None:
            interruption_flag = lambda: False

        # First, scan for existing results from previous runs (trial-level resumption)
        existing_results = self._scan_existing_results(equations, trial_id)

        # Load worker-level progress if resuming (for distributed mode)
        worker_completed = []
        worker_results = []
        if progress:
            worker_completed = progress.get('completed_equations', [])
            worker_results = progress.get('results', [])
            self.logger.info(f"Worker resuming with {len(worker_completed)} completed equations: {worker_completed}")

        # Build final results list in the original equation order
        # Combine: existing results (from disk) + worker progress (from memory) + to-be-computed
        all_results_map = {}  # eq_idx -> pv

        # Add existing results from disk
        all_results_map.update(existing_results)

        # Add worker progress (takes precedence if both exist)
        for eq_idx, result in zip(worker_completed, worker_results):
            all_results_map[eq_idx] = result

        # Determine which equations still need to be run
        remaining_equations = [eq for eq in equations if eq not in all_results_map]

        if not remaining_equations:
            self.logger.info(f"All equations already completed, returning cached results")
            # Return results in original equation order
            return [all_results_map[eq] for eq in equations]

        self.logger.info(f"Processing {len(remaining_equations)} remaining equations: {remaining_equations}")

        packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)

        # Create experiment directory for this trial
        base_dir = Path("/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment")
        temp_dir = base_dir / f"trial_{trial_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for eq_idx in remaining_equations:
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

                    # Skip if run already completed (check for results file)
                    results_csv = run_dir / 'tensorboard_scalars.csv'
                    if results_csv.exists():
                        try:
                            pv = self._extract_run_pv(run_dir)
                            eq_pareto_volumes.append(pv)
                            self.logger.info(f"Equation {eq_idx} run {run_i}: reusing existing result PV = {pv:.4f}")
                            continue
                        except Exception:
                            # Results file corrupt, re-run
                            shutil.rmtree(run_dir)
                    elif run_dir.exists():
                        # Incomplete run, clean up
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

                # Add to results map and save worker progress incrementally
                all_results_map[eq_idx] = eq_mean_pv
                if self.is_worker:
                    # Save worker-level progress (for distributed mode)
                    worker_completed.append(eq_idx)
                    worker_results.append(eq_mean_pv)
                    self._save_trial_progress(trial_id, worker_completed, worker_results)

                node_info = f" (node {self.node_id})"
                self.logger.info(f"Equation {eq_idx}{node_info}: PV = {eq_mean_pv:.4f} "
                               f"(runs: {eq_pareto_volumes})")

            except Exception as e:
                self.logger.error(f"Failed to run equation {eq_idx}: {e}")
                eq_mean_pv = 0.0
                all_results_map[eq_idx] = eq_mean_pv
                if self.is_worker:
                    worker_completed.append(eq_idx)
                    worker_results.append(eq_mean_pv)
                    self._save_trial_progress(trial_id, worker_completed, worker_results)

        # Return results in original equation order
        # Handle early shutdown by only returning results we have
        return [all_results_map.get(eq, 0.0) for eq in equations]

    def _extract_run_pv(self, run_dir: Path) -> float:
        """Extract pareto volume from a single run directory."""
        csv_path = run_dir / 'tensorboard_scalars.csv'
        df = pd.read_csv(csv_path)
        final_pareto_volume = df['pareto_volume'].iloc[-1]
        return float(final_pareto_volume)

    def _scan_existing_results(self, equations: List[int], trial_id: int) -> Dict[int, float]:
        """
        Scan experiment directory for existing completed runs and extract their pareto volumes.

        Args:
            equations: List of equation indices to check
            trial_id: Trial ID to scan for

        Returns:
            Dictionary mapping equation index to mean pareto volume for completed equations
        """
        base_dir = Path("/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment")
        trial_dir = base_dir / f"trial_{trial_id}"

        if not trial_dir.exists():
            self.logger.info(f"No existing trial directory found for trial {trial_id}")
            return {}

        existing_results = {}
        n_runs = self.config['dataset']['n_runs']
        dataset_name = self.config['dataset']['name']

        for eq_idx in equations:
            eq_pareto_volumes = []
            all_runs_complete = True

            # Check all runs for this equation
            for run_i in range(n_runs):
                run_dir = trial_dir / f"{dataset_name}_eq{eq_idx}_run{run_i}"
                results_csv = run_dir / 'tensorboard_scalars.csv'

                if results_csv.exists():
                    try:
                        pv = self._extract_run_pv(run_dir)
                        eq_pareto_volumes.append(pv)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract PV from {run_dir}: {e}")
                        all_runs_complete = False
                        break
                else:
                    all_runs_complete = False
                    break

            # Only include equation if ALL runs are complete
            if all_runs_complete and len(eq_pareto_volumes) == n_runs:
                eq_mean_pv = np.mean(eq_pareto_volumes)
                existing_results[eq_idx] = eq_mean_pv
                self.logger.info(f"Found existing results for equation {eq_idx}: PV = {eq_mean_pv:.4f}")

        if existing_results:
            self.logger.info(f"Loaded {len(existing_results)} equation results from trial {trial_id}")

        return existing_results

    def cleanup(self):
        """Clean up coordinator resources."""
        if self.is_master:
            self.signal_shutdown()
