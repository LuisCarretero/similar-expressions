"""
SR Batch Runner - Pure SR execution logic.

This module handles symbolic regression experiment execution with no knowledge
of Optuna trials or distributed coordination. It's designed to be:
- Stateful: Created once per trial with fixed hyperparameters
- Resumable: Scans for existing results at equation and run level
- Pure SR: No Optuna or coordination concerns
"""

import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
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


class SRBatchRunner:
    """
    Runs SR experiments on batches of equations.

    Initialized ONCE per trial with fixed hyperparameters, then reused for all equations.
    Supports multi-level resumption (trial, equation, run).
    """

    def __init__(
        self,
        model_settings: ModelSettings,
        neural_options: NeuralOptions,
        mutation_weights: MutationWeights,
        dataset_config: Dict[str, Any],
        trial_id: int,
        coord_base_dir: Path
    ):
        """
        Initialize runner with hyperparameters.

        Key: PySR model is created ONCE here and reused for all equations.

        Args:
            model_settings: Model configuration
            neural_options: Neural network options
            mutation_weights: Mutation weight configuration
            dataset_config: Dataset configuration (from optuna config)
            trial_id: Unique trial identifier for directory organization
        """
        self.logger = logging.getLogger('[RUNNER]')

        # Initialize PySR model ONCE for this trial
        self.packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)
        self.dataset_config = dataset_config
        self.trial_id = trial_id

        # Create experiment directory for this trial
        self.trial_dir = coord_base_dir / f"trial_{trial_id}"
        self.trial_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"SRBatchRunner initialized for trial {trial_id} (trial directory: {self.trial_dir})")

    def run_equations(self, equations: List[int], interruption_flag=None) -> List[float]:
        """
        Run a list of equations and return their pareto volumes (in order).

        Supports resumption: scans for existing results and only runs remaining equations.

        Args:
            equations: List of equation indices to process
            interruption_flag: Function to check for interruption

        Returns:
            List of mean pareto volumes for each equation (in original order)
        """
        if interruption_flag is None:
            interruption_flag = lambda: False

        # Scan for existing results (trial-level resumption)
        existing_results = self._scan_existing_results(equations)

        if existing_results:
            self.logger.info(f"Found {len(existing_results)} equations with existing results")

        # Determine remaining equations to run
        remaining_equations = [eq for eq in equations if eq not in existing_results]

        if not remaining_equations:
            self.logger.info(f"All {len(equations)} equations already completed")
            return [existing_results[eq] for eq in equations]

        self.logger.info(f"Processing {len(remaining_equations)} remaining equations: {remaining_equations}")

        # Initialize results map with existing results
        all_results_map = existing_results.copy()

        # Run remaining equations
        for eq_idx in remaining_equations:
            if interruption_flag():
                self.logger.info(f"Interrupted during equation processing at equation {eq_idx}")
                break

            try:
                eq_mean_pv = self._run_single_equation(eq_idx, interruption_flag)
                all_results_map[eq_idx] = eq_mean_pv

                self.logger.info(f"Equation {eq_idx}: PV = {eq_mean_pv:.4f}")

            except Exception as e:
                self.logger.error(f"Failed to run equation {eq_idx}: {e}")
                # Use 0.0 for failed equations
                all_results_map[eq_idx] = 0.0

        # Return results in original order (0.0 for any interrupted/failed)
        return [all_results_map.get(eq, 0.0) for eq in equations]

    def _run_single_equation(self, eq_idx: int, interruption_flag=None) -> float:
        """
        Run n_runs for one equation, return mean pareto volume.

        Handles run-level resumption: reuses existing completed runs.

        Args:
            eq_idx: Equation index
            interruption_flag: Function to check for interruption

        Returns:
            Mean pareto volume across all runs
        """
        if interruption_flag is None:
            interruption_flag = lambda: False

        eq_pareto_volumes = []
        n_runs = self.dataset_config['n_runs']
        dataset_name = self.dataset_config['name']

        for run_i in range(n_runs):
            if interruption_flag():
                self.logger.info(f"Interrupted during runs for equation {eq_idx} at run {run_i}")
                break

            run_dir = self.trial_dir / f"{dataset_name}_eq{eq_idx}_run{run_i}"

            # Skip if run already completed
            results_csv = run_dir / 'tensorboard_scalars.csv'
            if results_csv.exists():
                try:
                    pv = self._extract_run_pv(run_dir)
                    eq_pareto_volumes.append(pv)
                    self.logger.info(f"Equation {eq_idx} run {run_i}: reusing existing result PV = {pv:.4f}")
                    continue
                except Exception as e:
                    self.logger.warning(f"Failed to extract PV from {run_dir}: {e}")
                    # Results file corrupt, re-run
                    shutil.rmtree(run_dir)

            # Clean incomplete runs
            if run_dir.exists():
                shutil.rmtree(run_dir)

            # Create dataset settings for this equation
            dataset_settings = DatasetSettings(
                dataset_name=dataset_name,
                eq_idx=eq_idx,
                num_samples=self.dataset_config['num_samples'],
                rel_noise_magn=self.dataset_config['noise'],
                remove_op_equations=self.dataset_config.get('remove_op_equations', {})
            )

            # Run single SR experiment
            run_single(
                packaged_model=self.packaged_model,
                dataset_settings=dataset_settings,
                log_dir=str(run_dir),
                wandb_logging=False,  # No WandB logging for individual runs
                enable_mutation_logging=False  # Disable for speed
            )

            # Extract pareto volume from results
            pv = self._extract_run_pv(run_dir)
            eq_pareto_volumes.append(pv)
            self.logger.info(f"Equation {eq_idx} run {run_i}: PV = {pv:.4f}")

        # Return mean pareto volume across completed runs
        if not eq_pareto_volumes:
            self.logger.warning(f"No completed runs for equation {eq_idx}, returning 0.0")
            return 0.0

        return float(np.mean(eq_pareto_volumes))

    def _scan_existing_results(self, equations: List[int]) -> Dict[int, float]:
        """
        Scan trial directory for completed runs.

        Only includes equations where ALL n_runs are complete.

        Args:
            equations: List of equation indices to check

        Returns:
            Dictionary mapping equation index to mean pareto volume for completed equations
        """
        if not self.trial_dir.exists():
            return {}

        existing_results = {}
        n_runs = self.dataset_config['n_runs']
        dataset_name = self.dataset_config['name']

        for eq_idx in equations:
            eq_pareto_volumes = []
            all_runs_complete = True

            # Check all runs for this equation
            for run_i in range(n_runs):
                run_dir = self.trial_dir / f"{dataset_name}_eq{eq_idx}_run{run_i}"
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
                eq_mean_pv = float(np.mean(eq_pareto_volumes))
                existing_results[eq_idx] = eq_mean_pv
                self.logger.info(f"Found existing results for equation {eq_idx}: PV = {eq_mean_pv:.4f}")

        return existing_results

    def _extract_run_pv(self, run_dir: Path) -> float:
        """
        Extract pareto volume from a single run directory.

        Args:
            run_dir: Path to run directory

        Returns:
            Final pareto volume from tensorboard CSV
        """
        csv_path = run_dir / 'tensorboard_scalars.csv'
        df = pd.read_csv(csv_path)
        
        # Use pareto_volume_calculated if it exists, otherwise fall back to pareto_volume
        # if 'pareto_volume_calculated' in df.columns:
        #     final_pareto_volume = df['pareto_volume_calculated'].iloc[-1]
        # else:
        final_pareto_volume = df['pareto_volume'].iloc[-1]
        
        return float(final_pareto_volume)
