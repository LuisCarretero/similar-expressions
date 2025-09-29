#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for symbolic regression.

This script sets up and runs an Optuna study for optimizing PySR hyperparameters
to maximize pareto volume across multiple equations. Supports SLURM requeuing
and graceful interruption handling.

Usage:
    python optuna_hyperopt.py --config optuna_config.yaml [--resume]
"""

import argparse
import os
import sys
import time
from pathlib import Path
import logging
import numpy as np

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from omegaconf import OmegaConf

from sweeps.optuna_objective import create_objective
from sweeps.optuna_coordinator import OptunaCoordinator
from run.signal_manager import create_interruption_manager


class OptunaStudyManager:
    """
    Manages Optuna study creation, execution, and graceful interruption handling.
    """

    def __init__(self, config_path: str, node_id: int = None, total_nodes: int = None):
        """
        Initialize the study manager with configuration.

        Args:
            config_path: Path to optuna_config.yaml
            node_id: Node ID for distributed execution (None for single-node)
            total_nodes: Total number of nodes for distributed execution (None for single-node)
        """
        self.config_path = config_path
        self.config = OmegaConf.load(config_path)
        self.config = OmegaConf.to_container(self.config, resolve=True)

        self.study = None
        self.objective = None

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Set up signal manager for graceful interruption
        self.signal_manager = create_interruption_manager(__name__)

        # Set up coordinator for distributed execution
        self.coordinator = OptunaCoordinator(node_id, total_nodes, self.config)


    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create the Optuna sampler based on configuration."""
        sampler_config = self.config['sampler']

        if sampler_config['type'] == 'TPESampler':
            return TPESampler(
                n_startup_trials=sampler_config.get('n_startup_trials', 50),
                n_ei_candidates=sampler_config.get('n_ei_candidates', 24),
                multivariate=sampler_config.get('multivariate', True)
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_config['type']}")

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create the Optuna pruner based on configuration."""
        pruner_config = self.config['pruner']

        if pruner_config['type'] == 'MedianPruner':
            return MedianPruner(
                n_startup_trials=pruner_config.get('n_startup_trials', 20),
                n_warmup_steps=pruner_config.get('n_warmup_steps', 30),
                interval_steps=pruner_config.get('interval_steps', 10)
            )
        else:
            raise ValueError(f"Unknown pruner type: {pruner_config['type']}")

    def create_or_load_study(self, resume: bool = False) -> optuna.Study:
        """
        Create a new study or load an existing one.

        Args:
            resume: If True, attempt to resume existing study

        Returns:
            Optuna Study object
        """
        study_config = self.config['study']
        storage_url = study_config['storage']
        study_name = study_config['name']

        # Create sampler and pruner
        sampler = self._create_sampler()
        pruner = self._create_pruner()

        if resume:
            try:
                # Try to load existing study
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                    sampler=sampler,
                    pruner=pruner
                )
                self.logger.info(f"Resumed existing study '{study_name}' with {len(study.trials)} trials")
                return study
            except KeyError:
                self.logger.info(f"No existing study '{study_name}' found, creating new one")

        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction=study_config['direction'],
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True  # Load if exists, create if not
        )

        self.logger.info(f"Created/loaded study '{study_name}' with direction '{study_config['direction']}'")
        return study

    def _enqueue_baseline_trial(self):
        """
        Enqueue a trial with baseline values from base config for parameters being optimized.
        Works for any hyperparameter configuration (neural-only, weights-only, or combined).
        """
        baseline_params = {}

        # Process neural options if being optimized
        if 'neural_options' in self.config['hyperparameters']:
            base_neural = self.objective.base_config.symbolic_regression.neural_options
            neural_params = self.config['hyperparameters']['neural_options']

            for param_name in neural_params.keys():
                if hasattr(base_neural, param_name):
                    baseline_value = getattr(base_neural, param_name)

                    # Handle special case: subtree_max_nodes_diff
                    if param_name == 'subtree_max_nodes_diff':
                        baseline_value = base_neural.subtree_max_nodes - base_neural.subtree_min_nodes

                    baseline_params[param_name] = baseline_value

        # Process mutation weights if being optimized
        if 'mutation_weights' in self.config['hyperparameters']:
            base_weights = self.objective.base_config.symbolic_regression.mutation_weights
            weight_params = self.config['hyperparameters']['mutation_weights']

            # Get all weight names being optimized
            weight_names = [name for name in weight_params.keys() if name != 'use_dirichlet_sampling']

            # Extract base values and convert to Dirichlet raw values
            weight_values = [getattr(base_weights, name) for name in weight_names]
            total = sum(weight_values)

            # Convert to raw format for Dirichlet sampling
            for name, value in zip(weight_names, weight_values):
                normalized_weight = value / total
                raw_value = -np.log(normalized_weight)
                baseline_params[f"raw_{name}"] = raw_value

        # Enqueue the baseline trial
        if baseline_params:
            self.study.enqueue_trial(baseline_params)
            self.logger.info(f"Enqueued baseline trial with {len(baseline_params)} parameters from base config")
            self.logger.info(f"Baseline parameters: {baseline_params}")
        else:
            self.logger.info("No hyperparameters found to optimize - skipping baseline trial")

    def optimize(self, resume: bool = False):
        """
        Run the optimization process.

        Args:
            resume: If True, attempt to resume existing study
        """
        try:
            if self.coordinator.is_worker:
                # Worker nodes run the worker loop
                self.logger.info(f"Worker {self.coordinator.node_id} starting worker loop")
                self.coordinator.run_worker_loop(self.signal_manager.create_checker())
                self.logger.info(f"Worker {self.coordinator.node_id} finished")
                return

            # Master or single-node: run normal Optuna optimization
            self._run_master_optimization(resume)

        except Exception as e:
            self.logger.error(f"Critical error in optimization: {e}")
            raise

        finally:
            # Clean up
            if self.objective:
                self.objective.cleanup()
            if self.coordinator:
                self.coordinator.cleanup()

    def _run_master_optimization(self, resume: bool = False):
        """
        Run master/single-node optimization process.
        """
        # Create study and objective
        self.study = self.create_or_load_study(resume)
        self.objective = create_objective(self.config_path, self.signal_manager.create_checker(), self.coordinator)

        # Enqueue baseline trial with current config defaults (only on fresh start)
        if not resume:
            self._enqueue_baseline_trial()

        n_trials = self.config['execution']['n_trials']
        self.logger.info(f"Starting optimization with {n_trials} trials")

        # Print study statistics
        existing_trials = len(self.study.trials)
        if existing_trials > 0:
            self.logger.info(f"Study has {existing_trials} existing trials")
            if self.study.best_trial:
                self.logger.info(f"Current best value: {self.study.best_value:.6f}")
                self.logger.info(f"Current best params: {self.study.best_params}")

        # Custom optimization loop with interruption handling
        trials_completed = 0
        max_trials_per_iteration = 1  # Process one trial at a time for better control

        while trials_completed < n_trials and not self.signal_manager.interrupted:
            try:
                # Run a small batch of trials
                remaining_trials = min(max_trials_per_iteration, n_trials - trials_completed)

                self.logger.info(f"Running trial batch: {trials_completed + 1}-{trials_completed + remaining_trials} of {n_trials}")

                self.study.optimize(
                    self.objective,
                    n_trials=remaining_trials,
                    timeout=None,
                    show_progress_bar=False,  # We'll handle progress ourselves
                    callbacks=[self._trial_callback]
                )

                trials_completed += remaining_trials

                # Print progress
                self.logger.info(f"Completed {trials_completed}/{n_trials} trials")
                if self.study.best_trial:
                    self.logger.info(f"Best value so far: {self.study.best_value:.6f}")

            except optuna.TrialPruned:
                # This is normal - pruned trials are handled by Optuna
                trials_completed += 1
                continue

            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                self.signal_manager.interrupted = True
                break

            except Exception as e:
                self.logger.error(f"Error during optimization: {e}")
                # Continue with next trial unless it's a critical error
                trials_completed += 1
                continue

        # Print final results
        self._print_final_results()

    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial):
        """Callback function called after each trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.logger.info(f"Trial {trial.number} completed with value {trial.value:.6f}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.logger.info(f"Trial {trial.number} was pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            self.logger.warning(f"Trial {trial.number} failed")

    def _print_final_results(self):
        """Print final optimization results."""
        if not self.study:
            return

        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]

        self.logger.info("="*60)
        self.logger.info("OPTIMIZATION RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Total trials: {len(self.study.trials)}")
        self.logger.info(f"  Completed: {len(completed_trials)}")
        self.logger.info(f"  Pruned: {len(pruned_trials)}")
        self.logger.info(f"  Failed: {len(failed_trials)}")

        if self.study.best_trial:
            self.logger.info(f"\nBest trial:")
            self.logger.info(f"  Value: {self.study.best_value:.6f}")
            self.logger.info(f"  Params:")
            for key, value in self.study.best_params.items():
                if isinstance(value, float):
                    self.logger.info(f"    {key}: {value:.6f}")
                else:
                    self.logger.info(f"    {key}: {value}")

            # Print parameter importance if we have enough trials
            if len(completed_trials) >= 10:
                try:
                    importance = optuna.importance.get_param_importances(self.study)
                    self.logger.info(f"\nParameter importance:")
                    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
                        self.logger.info(f"  {param}: {imp:.4f}")
                except Exception as e:
                    self.logger.warning(f"Could not compute parameter importance: {e}")

        else:
            self.logger.warning("No completed trials found")

        # Save results to file
        results_file = Path("sweeps/optuna_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Optuna Optimization Results\n")
            f.write(f"==========================\n\n")
            f.write(f"Total trials: {len(self.study.trials)}\n")
            f.write(f"Completed: {len(completed_trials)}\n")
            f.write(f"Pruned: {len(pruned_trials)}\n")
            f.write(f"Failed: {len(failed_trials)}\n\n")

            if self.study.best_trial:
                f.write(f"Best trial value: {self.study.best_value:.6f}\n")
                f.write(f"Best parameters:\n")
                for key, value in self.study.best_params.items():
                    f.write(f"  {key}: {value}\n")

        self.logger.info(f"\nResults saved to: {results_file}")

        # Print instructions for dashboard
        storage_path = self.config['study']['storage']
        if storage_path.startswith('sqlite:///'):
            db_path = storage_path.replace('sqlite:///', '')
            self.logger.info(f"\nTo view results in Optuna dashboard:")
            self.logger.info(f"  optuna-dashboard {storage_path}")
            self.logger.info(f"  # or if installed globally:")
            self.logger.info(f"  optuna-dashboard {db_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Optuna hyperparameter optimization for SR')
    parser.add_argument('--config', type=str, default='optuna_config.yaml',
                       help='Path to configuration file (default: optuna_config.yaml)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing study if it exists')
    parser.add_argument('--node_id', type=int, help='Node ID for distributed execution (0-indexed)')
    parser.add_argument('--total_nodes', type=int, help='Total number of nodes for distributed execution')

    args = parser.parse_args()

    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    # Validate distributed arguments
    if (args.node_id is None) != (args.total_nodes is None):
        print("Error: Both --node_id and --total_nodes must be specified together")
        sys.exit(1)

    if args.node_id is not None and (args.node_id < 0 or args.node_id >= args.total_nodes):
        print(f"Error: node_id must be between 0 and {args.total_nodes-1}")
        sys.exit(1)

    # Create and run study manager
    manager = OptunaStudyManager(str(config_path), args.node_id, args.total_nodes)
    manager.optimize(resume=args.resume)


if __name__ == "__main__":
    main()