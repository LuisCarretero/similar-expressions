"""
Distributed Trial Executor - Generic distributed coordination.

This module coordinates distributed execution across master and worker nodes
via file-based communication. It's designed to be:
- Generic: Works with any runner type (not SR-specific)
- Asynchronous: No batch-wise synchronization between nodes
- Resumable: Tracks processed trials to prevent reprocessing
- Fault-tolerant: Handles worker failures gracefully
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple

# Add parent directories to path for imports
import sys
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from run.run_multiple import get_node_equations


class TrialIncompleteError(Exception):
    """Raised when a trial is interrupted before all workers complete."""
    pass


class DistributedTrialExecutor:
    """
    Coordinates distributed batch execution with asynchronous reporting.

    Handles file-based communication between master and workers, with support
    for incremental batch reporting and trial resumption.
    """

    def __init__(self, node_id: int, total_nodes: int, coord_dir: Path):
        """
        Initialize the distributed executor.

        Args:
            node_id: Node ID (0-indexed, 0 is master)
            total_nodes: Total number of nodes
            coord_dir: Coordination directory for file communication
        """
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.is_master = (node_id == 0)
        self.is_worker = not self.is_master
        self.coord_dir = coord_dir
        self.logger = logging.getLogger('[EXECUTOR]')

        self.coord_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"DistributedTrialExecutor initialized: "
                        f"node_id={node_id}, total_nodes={total_nodes}, "
                        f"is_master={self.is_master}")

    # ========================================================================
    # TRIAL COMPLETION CHECKING
    # ========================================================================

    def _is_trial_complete(self, trial_id: int) -> bool:
        """
        Check if trial is complete or pruned by looking for marker files.

        This uses the same mechanism as optuna_hyperopt.py's
        _mark_trial_done() method, ensuring consistency.

        Args:
            trial_id: Trial ID to check

        Returns:
            True if .done or .pruned file exists, False otherwise
        """
        done_file = self.coord_dir / f"trial_{trial_id}.done"
        pruned_file = self.coord_dir / f"trial_{trial_id}.pruned"
        return done_file.exists() or pruned_file.exists()

    # ========================================================================
    # MASTER API
    # ========================================================================

    def execute_trial_with_batching(
        self,
        trial_params: Dict[str, Any],
        all_equations: List[int],
        batch_size: int,
        create_runner_fn: Callable,
        report_callback: Callable,
        interruption_flag: Callable
    ) -> List[float]:
        """
        Master: Run trial in batches with asynchronous worker coordination.

        Args:
            trial_params: Trial configuration dictionary
            all_equations: Complete list of equations (will be split across nodes)
            batch_size: Equations per batch for reporting
            create_runner_fn: Callable that creates runner from trial_params
            report_callback: Called with (batch_results, batch_id, step) for EACH batch
            interruption_flag: Callable to check for shutdown signal (from interruption_manager)

        Returns:
            Combined results from master + all workers
        """
        if not self.is_master:
            raise RuntimeError("Only master can execute trials")

        trial_id = trial_params['trial_id']
        self.logger.info(f"Master executing trial {trial_id}. Waiting for worker acks.")

        try:
            # 2. Wait for worker acks
            self._wait_for_worker_acks(trial_id, interruption_flag, timeout_sec=60*120.0)  # 2h

            # 3. Create runner ONCE
            self.logger.info(f"Creating runner for trial {trial_id}")
            runner = create_runner_fn(trial_params)

            # 4. Get master's equation subset
            master_equations = get_node_equations(all_equations, 0, self.total_nodes)
            self.logger.info(f"Master will process {len(master_equations)} equations: {master_equations}")

            # 5. Track which worker batches we've reported
            worker_batches_reported = {worker_id: 0 for worker_id in range(1, self.total_nodes)}

            # 6. Track cumulative equations processed for step reporting
            equations_processed = 0

            # 7. Collect all results (for return value)
            all_master_results = []

            # 8. BATCH LOOP - process master's equations and report each batch
            master_batch_id, master_completed = 0, True
            for batch_start in range(0, len(master_equations), batch_size):
                # Check for interruption
                if interruption_flag():
                    self.logger.info("Master interrupted during batch processing")
                    master_completed = False
                    break

                batch_end = min(batch_start + batch_size, len(master_equations))
                batch_equations = master_equations[batch_start:batch_end]

                # Run master's batch
                self.logger.info(f"Master processing batch {master_batch_id}: equations {batch_equations}")
                batch_results = runner.run_equations(batch_equations, interruption_flag)
                all_master_results.extend(batch_results)

                # Update cumulative counter and REPORT master's batch
                equations_processed += len(batch_results)
                report_callback(batch_results, f"master_batch_{master_batch_id}", equations_processed)
                master_batch_id += 1

                # Check for and report new worker batches
                equations_processed = self._report_new_worker_batches(
                    trial_id, worker_batches_reported, batch_size,
                    report_callback, equations_processed
                )

            # 9. Continue reporting worker batches until all workers finish
            if master_completed:
                self.logger.info("Master finished, waiting for workers and reporting their batches...")
            else:
                self.logger.info("Master interrupted, waiting for workers to finish current work...")

            workers_completed, equations_processed = self._wait_and_report_remaining_worker_batches(
                trial_id, worker_batches_reported, batch_size,
                report_callback, interruption_flag, equations_processed,
                timeout_sec=60*30  # 30 min
            )

            # 10. Collect all final worker results (for return value)
            final_worker_results = self._collect_all_worker_results(trial_id)

            combined_results = all_master_results + final_worker_results

            # 11. Check if trial completed successfully
            if not (master_completed and workers_completed):
                raise TrialIncompleteError(
                    f"Trial {trial_id} was interrupted before completion "
                    f"(master: {master_completed}, workers: {workers_completed})"
                )

            self.logger.info(f"Trial {trial_id} complete: {len(combined_results)} total results "
                           f"({equations_processed} equations processed)")

            return combined_results

        finally:
            # Clean up coordination files
            self._cleanup_trial_files(trial_id)

    def _report_new_worker_batches(
        self,
        trial_id: int,
        worker_batches_reported: Dict[int, int],
        batch_size: int,
        report_callback: Callable,
        equations_processed: int
    ) -> int:
        """
        Scan for new worker batches and report them.

        Updates worker_batches_reported in place.

        Args:
            trial_id: Trial ID
            worker_batches_reported: Dict tracking batches reported per worker
            batch_size: Equations per batch
            report_callback: Callback to report batch results
            equations_processed: Current cumulative equation count

        Returns:
            Updated cumulative equation count
        """
        for worker_id in range(1, self.total_nodes):
            worker_results = self._get_worker_results(trial_id, worker_id)
            if worker_results is None:
                continue

            num_results = len(worker_results)
            num_reported = worker_batches_reported[worker_id]

            # Calculate how many complete new batches we have
            num_batches_available = num_results // batch_size

            # Report any new complete batches
            while num_reported < num_batches_available:
                batch_start = num_reported * batch_size
                batch_end = batch_start + batch_size
                batch_results = worker_results[batch_start:batch_end]

                equations_processed += len(batch_results)
                report_callback(batch_results, f"worker{worker_id}_batch_{num_reported}", equations_processed)

                num_reported += 1
                worker_batches_reported[worker_id] = num_reported

        return equations_processed

    def _wait_and_report_remaining_worker_batches(
        self,
        trial_id: int,
        worker_batches_reported: Dict[int, int],
        batch_size: int,
        report_callback: Callable,
        interruption_flag: Callable,
        equations_processed: int,
        timeout_sec: float
    ) -> Tuple[bool, int]:
        """
        Wait for all workers to finish, reporting their batches as they arrive.

        Args:
            trial_id: Trial ID
            worker_batches_reported: Dict tracking batches reported per worker
            batch_size: Equations per batch
            report_callback: Callback to report batch results
            interruption_flag: Function to check for interruption
            equations_processed: Current cumulative equation count
            timeout_sec: Timeout in seconds

        Returns:
            Tuple of (all_workers_finished, updated_equations_processed)
        """
        expected_workers = set(range(1, self.total_nodes))
        workers_done = set()

        start_time = time.time()

        while len(workers_done) < len(expected_workers):
            # Check for interruption
            if interruption_flag():
                self.logger.warning("Interrupted while waiting for workers")
                break

            # Check timeout
            if time.time() - start_time > timeout_sec:
                missing_workers = expected_workers - workers_done
                self.logger.error(f"Timeout waiting for workers {missing_workers}")
                break

            # Check which workers are done
            for worker_id in expected_workers:
                if worker_id not in workers_done:
                    if self._is_worker_done(trial_id, worker_id):
                        workers_done.add(worker_id)
                        self.logger.info(f"Worker {worker_id} finished")

            # Report any new worker batches
            equations_processed = self._report_new_worker_batches(
                trial_id, worker_batches_reported, batch_size,
                report_callback, equations_processed
            )

            time.sleep(1.0)

        # Final check for any remaining partial batches
        for worker_id in range(1, self.total_nodes):
            worker_results = self._get_worker_results(trial_id, worker_id)
            if worker_results is None:
                continue

            num_results = len(worker_results)
            num_reported = worker_batches_reported[worker_id]

            # Report any remaining partial batch
            if num_results > num_reported * batch_size:
                batch_start = num_reported * batch_size
                remaining_results = worker_results[batch_start:]

                equations_processed += len(remaining_results)
                report_callback(remaining_results, f"worker{worker_id}_batch_{num_reported}_final", equations_processed)
                self.logger.info(f"Reported worker {worker_id} final partial batch ({len(remaining_results)} equations)")

        # Return whether all workers completed and final equation count
        all_workers_finished = len(workers_done) == len(expected_workers)
        return all_workers_finished, equations_processed

    # ========================================================================
    # WORKER API
    # ========================================================================

    def run_worker_loop(
        self,
        all_equations: List[int],
        batch_size: int,
        create_runner_fn: Callable,
        interruption_flag: Callable
    ):
        """
        Worker: Continuously process trials from master.

        Args:
            all_equations: Complete equation list (worker will extract its subset)
            batch_size: Equations per batch for incremental result reporting
            create_runner_fn: Callable that creates runner from trial_params
            interruption_flag: Callable to check for shutdown signal (from interruption_manager)
        """
        if not self.is_worker:
            raise RuntimeError("Only workers can run worker loop")

        self.logger.info(f"Worker {self.node_id} starting worker loop")

        runner_stop_flag = lambda: interruption_flag() or self._is_trial_complete(trial_id)

        while True:
            # Check for interruption (from interruption_manager)
            if interruption_flag():
                self.logger.info(f"Worker {self.node_id} interrupted, shutting down")
                break

            # Wait for trial params
            trial_params = self._wait_for_trial_params(timeout_sec=30.0)
            if trial_params is None:
                time.sleep(1.0)
                continue

            trial_id = trial_params['trial_id']

            # Edge case: _wait_for_trial_params should check for this, but just to make sure
            if self._is_trial_complete(trial_id) or self._is_worker_done(trial_id, self.node_id):
                self.logger.warning(f"Worker {self.node_id} skipping already-processed trial {trial_id}")
                continue

            self.logger.info(f"Worker {self.node_id} starting trial {trial_id}")

            try:
                # Create runner ONCE for this trial
                runner = create_runner_fn(trial_params)

                # Get worker's equation subset
                worker_equations = get_node_equations(all_equations, self.node_id, self.total_nodes)

                self.logger.info(f"Worker {self.node_id} processing {len(worker_equations)} equations: {worker_equations}")

                # BATCH LOOP - process worker's equations with incremental result sending
                all_worker_results = []

                for batch_start in range(0, len(worker_equations), batch_size):
                    if runner_stop_flag():
                        reason = "interrupted" if interruption_flag() else f"trial {trial_id} was pruned"
                        self.logger.info(f"Worker {self.node_id} {reason}, stopping work")
                        break

                    batch_end = min(batch_start + batch_size, len(worker_equations))
                    batch_equations = worker_equations[batch_start:batch_end]

                    # Run worker's batch
                    self.logger.info(f"Worker {self.node_id} processing batch: equations {batch_equations}")
                    batch_results = runner.run_equations(batch_equations, runner_stop_flag)
                    all_worker_results.extend(batch_results)

                    # Send incremental results to master (no new data, just update)
                    self._send_worker_results(trial_id, all_worker_results, final=False)

                    self.logger.info(f"Worker {self.node_id}: Sent {len(all_worker_results)} results so far")

                # Send final marker (no new data, just final=True)
                self._send_worker_results(trial_id, all_worker_results, final=True)
                self.logger.info(f"Worker {self.node_id} completed trial {trial_id}")

            except Exception as e:
                self.logger.error(f"Worker {self.node_id} error processing trial {trial_id}: {e}")
                # Send empty results with final=True to unblock master
                self._send_worker_results(trial_id, [], final=True)

    # ========================================================================
    # COMMUNICATION PRIMITIVES
    # ========================================================================

    def _wait_for_worker_acks(
        self,
        trial_id: int,
        interruption_flag: Callable,
        timeout_sec: float
    ):
        """
        Master waits for worker acknowledgments.

        Args:
            trial_id: Trial ID
            interruption_flag: Function to check for interruption
            timeout_sec: Timeout in seconds
        """
        expected_workers = list(range(1, self.total_nodes))
        received_acks = set()

        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            if interruption_flag():
                self.logger.info("Interrupted while waiting for worker acknowledgments")
                return

            for worker_id in expected_workers:
                if worker_id not in received_acks:
                    ack_file = self.coord_dir / f"trial_{trial_id}_worker_{worker_id}_ack.json"
                    if ack_file.exists():
                        received_acks.add(worker_id)
                        self.logger.info(f"Received ack from worker {worker_id}")

            if len(received_acks) == len(expected_workers):
                self.logger.info(f"All workers acknowledged trial {trial_id} parameters")
                return

            time.sleep(0.2)

        # Log which workers didn't respond
        missing_workers = set(expected_workers) - received_acks
        self.logger.warning(f"Timeout waiting for acknowledgments from workers {missing_workers} for trial {trial_id}")

    def _wait_for_trial_params(self, timeout_sec: float) -> Optional[Dict[str, Any]]:
        """
        Worker waits for trial parameters from master.

        Args:
            timeout_sec: Timeout in seconds

        Returns:
            Trial parameters dict if found, None otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            params_files = list(self.coord_dir.glob("trial_*_params.json"))

            for params_file in params_files:
                try:
                    with open(params_file, 'r') as f:
                        params = json.load(f)
                    trial_id = params['trial_id']

                    # Skip if already completed (via .done marker file)
                    if self._is_trial_complete(trial_id) or self._is_worker_done(trial_id, self.node_id):
                        continue

                    # Send acknowledgment
                    ack_file = self.coord_dir / f"trial_{trial_id}_worker_{self.node_id}_ack.json"
                    with open(ack_file, 'w') as f:
                        json.dump({'worker_id': self.node_id, 'status': 'received'}, f)

                    self.logger.info(f"Worker {self.node_id} received trial {trial_id} parameters")
                    return params

                except (json.JSONDecodeError, FileNotFoundError):
                    continue

            time.sleep(0.2)

        return None

    def _send_worker_results(self, trial_id: int, results: List[float], final: bool = False):
        """
        Worker sends (incremental or final) results to master.

        Args:
            trial_id: Trial ID
            results: List of results so far
            final: Whether this is the final update
        """
        results_file = self.coord_dir / f"trial_{trial_id}_worker_{self.node_id}_results.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump({
                'results': results,
                'final': final,
                'worker_id': self.node_id
            }, f)

    def _get_worker_results(self, trial_id: int, worker_id: int) -> Optional[List[float]]:
        """
        Master gets current worker results (non-blocking).

        Args:
            trial_id: Trial ID
            worker_id: Worker node ID

        Returns:
            List of results if available, None otherwise
        """
        results_file = self.coord_dir / f"trial_{trial_id}_worker_{worker_id}_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                return data['results']
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return None

    def _is_worker_done(self, trial_id: int, worker_id: int) -> bool:
        """
        Check if worker has set final=True.

        Args:
            trial_id: Trial ID
            worker_id: Worker node ID

        Returns:
            True if worker is done, False otherwise
        """
        results_file = self.coord_dir / f"trial_{trial_id}_worker_{worker_id}_results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                return data.get('final', False)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return False

    def _collect_all_worker_results(self, trial_id: int) -> List[float]:
        """
        Master collects ALL worker results (for final return value).

        Args:
            trial_id: Trial ID

        Returns:
            Concatenated list of all worker results
        """
        all_results = []

        for worker_id in range(1, self.total_nodes):
            worker_results = self._get_worker_results(trial_id, worker_id)
            if worker_results is not None:
                all_results.extend(worker_results)

        return all_results

    def _cleanup_trial_files(self, trial_id: int):
        """
        Clean up coordination files for completed trial.

        Note: Only removing ack file for now.

        Args:
            trial_id: Trial ID
        """
        patterns = [
            # f"trial_{trial_id}_params.json",
            f"trial_{trial_id}_worker_*_ack.json",
            # f"trial_{trial_id}_worker_*_results.json"
        ]
        for pattern in patterns:
            for f in self.coord_dir.glob(pattern):
                f.unlink(missing_ok=True)

        self.logger.info(f"Cleaned up coordination files for trial {trial_id}")
