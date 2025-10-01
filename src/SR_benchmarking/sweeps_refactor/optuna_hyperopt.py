#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Runner - Combines study management + trial execution.

This module orchestrates the complete Optuna study lifecycle including:
- Study creation, configuration, and management
- Master/worker routing for distributed execution
- Hyperparameter sampling (neural options, mutation weights)
- Trial resumption after interruptions
- Intermediate reporting and pruning
- Results aggregation and visualization
"""

import argparse
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from omegaconf import OmegaConf

from run.utils import (
    ModelSettings, NeuralOptions, MutationWeights, load_config
)
from run.run_multiple import _parse_eq_idx
from run.signal_manager import create_interruption_manager

# Import refactored layers
from sweeps_refactor.sr_runner import SRBatchRunner
from sweeps_refactor.distributed_executor import DistributedTrialExecutor, TrialIncompleteError


class OptunaHyperoptRunner:
    """
    Manages complete Optuna hyperparameter optimization for symbolic regression.

    Combines study management and objective function in a single class for simplicity.
    The objective function is implemented as a private method that can directly access
    study state and configuration.
    """

    def __init__(self, config_path: str, node_id: int = None, total_nodes: int = None):
        """
        Initialize the hyperopt runner.

        Args:
            config_path: Path to optuna_config.yaml
            node_id: Node ID for distributed execution (None for single-node)
            total_nodes: Total number of nodes (None for single-node)
        """
        self.config_path = config_path
        self.config = OmegaConf.load(config_path)
        self.config = OmegaConf.to_container(self.config, resolve=True)

        self.study = None
        self.base_config = load_config(self.config['base_config'])
        self.equations = _parse_eq_idx(self.config['dataset']['equation_indices'])

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(name)s %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('[HYPEROPT]')

        # Set up signal manager for graceful interruption
        self.signal_manager = create_interruption_manager(__name__)

        # Set up distributed executor
        coord_dir = self._get_coord_dir()
        self.executor = DistributedTrialExecutor(
            node_id if node_id is not None else 0,
            total_nodes if total_nodes is not None else 1,
            coord_dir
        )

        self.logger.info(f"OptunaHyperoptRunner initialized: "
                        f"node_id={self.executor.node_id}, total_nodes={self.executor.total_nodes}, "
                        f"is_master={self.executor.is_master}")

    def _get_coord_dir(self) -> Path:
        """
        Get coordination directory for distributed execution.

        Returns:
            Path to coordination directory
        """
        base_dir = Path("/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optuna_experiment")
        coord_dir = base_dir / f"optuna_coord_{self.config['study']['name']}"
        coord_dir.mkdir(parents=True, exist_ok=True)
        return coord_dir

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def optimize(self, resume: bool = False):
        """
        Run the optimization process.

        Routes to worker loop or master optimization based on node role.

        Args:
            resume: If True, attempt to resume existing study
        """
        try:
            if self.executor.is_worker:
                self._run_worker_loop()
                self.logger.info(f"Worker {self.executor.node_id} finished")
            else:
                self._run_master_optimization(resume)

        except Exception as e:
            self.logger.error(f"Critical error in optimization: {e}")
            raise

    # ========================================================================
    # MASTER OPTIMIZATION (PRIVATE)
    # ========================================================================

    def _run_master_optimization(self, resume: bool = False):
        """
        Master: Run the complete Optuna study.

        Creates/loads study, enqueues baseline, runs optimization loop, prints results.

        Args:
            resume: If True, attempt to resume existing study
        """
        # Create study
        self.study = self._create_or_load_study(resume)

        # Enqueue baseline trial with current config defaults (only on fresh start)
        if not resume:
            self._enqueue_baseline_trial()

        n_trials = self.config['execution']['n_trials']

        # Print study statistics and calculate remaining trials
        existing_trials = len(self.study.trials)
        trials_already_complete = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL])
        remaining_trials = n_trials - trials_already_complete

        self.logger.info(f"Target: {n_trials} trials | Completed: {trials_already_complete} | Remaining: {remaining_trials}")
        
        if existing_trials > 0:
            self.logger.info(f"Existing trials: {existing_trials} (Completed: {trials_already_complete}, Pruned: {pruned_trials}, Failed: {failed_trials})")

        # Check if target already reached
        if remaining_trials <= 0:
            self.logger.info(f"Target of {n_trials} trials already reached!")
            self._print_final_results()
            return

        # Run remaining trials (one study.optimize call per trial for better control)
        trials_completed = 0
        while trials_completed < remaining_trials and not self.signal_manager.check_interrupted():
            try:
                self.study.optimize(
                    self._objective,
                    n_trials=1,
                    timeout=None,
                    show_progress_bar=False
                )
                trials_completed += 1

            except optuna.TrialPruned:
                # Pruned trials are expected - count and continue
                trials_completed += 1
                continue

            except KeyboardInterrupt:
                self.logger.info("Received keyboard interrupt")
                self.signal_manager.interrupted = True
                break

            except Exception as e:
                self.logger.error(f"Error during optimization: {e}")
                trials_completed += 1
                continue

        # Check if we've reached the target
        final_completed = len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if final_completed >= n_trials:
            self._print_final_results()
        else:
            self.logger.info(f"Progress: {final_completed}/{n_trials} trials completed")

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Objective function called by Optuna for each trial.

        Handles:
        - Trial resumption checking
        - Hyperparameter sampling or reconstruction
        - Distributed execution via executor
        - Intermediate reporting for pruning

        Args:
            trial: Optuna trial object

        Returns:
            Mean pareto volume across all equations (to be maximized)
        """
        self.logger.info(f"\n[OPTUNA] Starting trial {trial.number}")

        # 1. Check for incomplete trial to resume
        incomplete = self._find_incomplete_trial()

        if incomplete is not None:
            # Resume incomplete trial
            trial_id, saved_params = incomplete
            self.logger.info(f"[OPTUNA] Resuming incomplete trial {trial_id}")

            # Reconstruct hyperparameters from saved state
            model_settings, neural_options, mutation_weights = \
                self._reconstruct_hyperparameters(trial, saved_params)

            self.logger.info(f"[OPTUNA] Reconstructed hyperparameters (trial {trial_id})")
        else:
            # Fresh trial - generate new trial_id and sample hyperparameters
            trial_id = int(time.time() * 1000000)
            self.logger.info(f"[OPTUNA] Fresh trial {trial.number}, assigned trial_id {trial_id}")

            # Sample hyperparameters
            model_settings, neural_options, mutation_weights = \
                self._sample_hyperparameters(trial)

            self.logger.info(f"[OPTUNA] Trial {trial.number} hyperparameters:")
            self.logger.info(f"  Neural: {neural_options}")
            self.logger.info(f"  Mutations: {mutation_weights}")

        # 2. Prepare trial parameters for executor
        trial_params = {
            'trial_id': trial_id,
            'model_settings': self._to_dict(model_settings),
            'neural_options': self._to_dict(neural_options),
            'mutation_weights': self._to_dict(mutation_weights),
            'raw_params': trial.params  # For resumption
        }

        # Save params for potential resumption (only if fresh trial)
        if incomplete is None:
            self._save_trial_params(trial_id, trial_params)

        # 3. Create runner factory for executor
        def create_runner(params):
            """Factory function to create SRBatchRunner from trial params."""
            ms, no, mw = self._reconstruct_objects(params)
            return SRBatchRunner(ms, no, mw, self.config['dataset'], params['trial_id'])

        # 4. Create reporting callback for intermediate pruning
        def report_callback(batch_results, batch_id):
            """
            Called by executor after each batch (master or worker).

            Reports to Optuna and checks for pruning.
            """
            current_mean = np.mean(batch_results)
            trial.report(current_mean, step=len(batch_results))

            self.logger.info(f"[OPTUNA] {batch_id}: PV = {current_mean:.4f}")

            if trial.should_prune():
                self.logger.info(f"[OPTUNA] Trial {trial.number} pruned at {batch_id}")
                raise optuna.TrialPruned()

        # 5. Execute trial via distributed executor
        batch_size = self.config['execution']['equations_per_batch']

        try:
            all_results = self.executor.execute_trial_with_batching(
                trial_params,
                self.equations,
                batch_size,
                create_runner,
                report_callback,
                self.signal_manager.create_checker()
            )
        except optuna.TrialPruned:
            # Mark trial as pruned and re-raise
            self._mark_trial_done(trial_id, pruned=True)
            raise
        except TrialIncompleteError:
            # Don't call _mark_trial_done() - leave trial incomplete for resumption
            raise  # Let Optuna mark trial as failed, but we keep params for resumption

        # 6. Calculate final objective value
        final_mean_pv = np.mean(all_results)
        self.logger.info(f"[OPTUNA] Trial {trial.number} COMPLETE: Final PV = {final_mean_pv:.4f}")

        # 7. Mark trial as complete
        self._mark_trial_done(trial_id, pruned=False)

        return final_mean_pv

    # ========================================================================
    # WORKER LOOP (PRIVATE)
    # ========================================================================

    def _run_worker_loop(self):
        """
        Worker: Process trials from master in continuous loop.

        Creates runner factory and delegates to executor's worker loop.
        """
        def create_runner(params):
            """Factory function to create SRBatchRunner from trial params."""
            ms, no, mw = self._reconstruct_objects(params)
            return SRBatchRunner(ms, no, mw, self.config['dataset'], params['trial_id'])

        batch_size = self.config['execution']['equations_per_batch']

        self.executor.run_worker_loop(
            self.equations,
            batch_size,
            create_runner,
            self.signal_manager.create_checker()
        )

    # ========================================================================
    # HYPERPARAMETER SAMPLING (PRIVATE)
    # ========================================================================

    def _sample_hyperparameters(
        self,
        trial: optuna.Trial
    ) -> Tuple[ModelSettings, NeuralOptions, MutationWeights]:
        """
        Sample hyperparameters from the trial using configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Tuple of (ModelSettings, NeuralOptions, MutationWeights)
        """
        # Start with base configuration
        base_model_cfg = self.base_config.model_settings
        base_neural_cfg = self.base_config.symbolic_regression.neural_options
        base_mutation_cfg = self.base_config.symbolic_regression.mutation_weights

        # Model settings (not optimized, use base)
        model_settings = ModelSettings(
            niterations=base_model_cfg.niterations,
            loss_function=base_model_cfg.loss_function,
            early_stopping_condition=base_model_cfg.early_stopping_condition,
            verbosity=base_model_cfg.verbosity,
            precision=base_model_cfg.precision,
            batching=base_model_cfg.batching,
            batch_size=base_model_cfg.batch_size
        )

        # Neural options (start with base, then sample)
        neural_options = NeuralOptions(
            active=base_neural_cfg.active,
            model_path=base_neural_cfg.model_path,
            sampling_eps=base_neural_cfg.sampling_eps,
            subtree_min_nodes=base_neural_cfg.subtree_min_nodes,
            subtree_max_nodes=base_neural_cfg.subtree_max_nodes,
            device=base_neural_cfg.device,
            verbose=base_neural_cfg.verbose,
            max_tree_size_diff=base_neural_cfg.max_tree_size_diff,
            require_tree_size_similarity=base_neural_cfg.require_tree_size_similarity,
            require_novel_skeleton=base_neural_cfg.require_novel_skeleton,
            require_expr_similarity=base_neural_cfg.require_expr_similarity,
            similarity_threshold=base_neural_cfg.similarity_threshold,
            log_subtree_strings=base_neural_cfg.log_subtree_strings,
            sample_logits=base_neural_cfg.sample_logits,
            max_resamples=base_neural_cfg.max_resamples,
            sample_batchsize=base_neural_cfg.sample_batchsize,
            subtree_max_features=base_neural_cfg.subtree_max_features
        )

        # Sample neural hyperparameters if specified in config
        if 'neural_options' in self.config['hyperparameters']:
            neural_params = self.config['hyperparameters']['neural_options']

            if 'sampling_eps' in neural_params:
                neural_options.sampling_eps = trial.suggest_float('sampling_eps',
                    neural_params['sampling_eps']['low'],
                    neural_params['sampling_eps']['high'])

            if 'subtree_min_nodes' in neural_params:
                neural_options.subtree_min_nodes = trial.suggest_int('subtree_min_nodes',
                    neural_params['subtree_min_nodes']['low'],
                    neural_params['subtree_min_nodes']['high'])

            # Handle special case: subtree_max_nodes_diff
            if 'subtree_max_nodes_diff' in neural_params:
                subtree_max_nodes_diff = trial.suggest_int('subtree_max_nodes_diff',
                    neural_params['subtree_max_nodes_diff']['low'],
                    neural_params['subtree_max_nodes_diff']['high'])
                neural_options.subtree_max_nodes = neural_options.subtree_min_nodes + subtree_max_nodes_diff

            if 'max_tree_size_diff' in neural_params:
                neural_options.max_tree_size_diff = trial.suggest_int('max_tree_size_diff',
                    neural_params['max_tree_size_diff']['low'],
                    neural_params['max_tree_size_diff']['high'])

            if 'require_tree_size_similarity' in neural_params:
                neural_options.require_tree_size_similarity = trial.suggest_categorical(
                    'require_tree_size_similarity',
                    neural_params['require_tree_size_similarity']['choices'])

            if 'require_novel_skeleton' in neural_params:
                neural_options.require_novel_skeleton = trial.suggest_categorical(
                    'require_novel_skeleton',
                    neural_params['require_novel_skeleton']['choices'])

            if 'require_expr_similarity' in neural_params:
                neural_options.require_expr_similarity = trial.suggest_categorical(
                    'require_expr_similarity',
                    neural_params['require_expr_similarity']['choices'])

            if 'similarity_threshold' in neural_params:
                neural_options.similarity_threshold = trial.suggest_float('similarity_threshold',
                    neural_params['similarity_threshold']['low'],
                    neural_params['similarity_threshold']['high'])

            if 'max_resamples' in neural_params:
                if neural_params['max_resamples']['type'] == 'categorical':
                    neural_options.max_resamples = trial.suggest_categorical('max_resamples',
                        neural_params['max_resamples']['choices'])
                else:
                    neural_options.max_resamples = trial.suggest_int('max_resamples',
                        neural_params['max_resamples']['low'],
                        neural_params['max_resamples']['high'])

        # Sample mutation weights using Dirichlet distribution if specified
        if 'mutation_weights' in self.config['hyperparameters']:
            mutation_params = self.config['hyperparameters']['mutation_weights']
            mutation_weights = MutationWeights()

            # Get all weight parameter names
            weight_names = [name for name in mutation_params.keys() if name != 'use_dirichlet_sampling']

            # Sample from uniform and transform to exponential for Dirichlet
            raw_weights = []
            for weight_name in weight_names:
                u = trial.suggest_float(f"raw_{weight_name}", 1e-10, 1.0 - 1e-10)
                raw_weights.append(-np.log(u))

            # Normalize to create proper Dirichlet sample
            total = sum(raw_weights)
            normalized_weights = [w / total for w in raw_weights]

            # Assign to mutation_weights object
            for weight_name, weight_value in zip(weight_names, normalized_weights):
                setattr(mutation_weights, weight_name, weight_value)

            self.logger.info(f"[OPTUNA] Dirichlet sampled weights: {dict(zip(weight_names, normalized_weights))}")
        else:
            # Use base configuration
            mutation_weights = MutationWeights(
                weight_add_node=base_mutation_cfg.weight_add_node,
                weight_insert_node=base_mutation_cfg.weight_insert_node,
                weight_delete_node=base_mutation_cfg.weight_delete_node,
                weight_do_nothing=base_mutation_cfg.weight_do_nothing,
                weight_mutate_constant=base_mutation_cfg.weight_mutate_constant,
                weight_mutate_operator=base_mutation_cfg.weight_mutate_operator,
                weight_swap_operands=base_mutation_cfg.weight_swap_operands,
                weight_rotate_tree=base_mutation_cfg.weight_rotate_tree,
                weight_randomize=base_mutation_cfg.weight_randomize,
                weight_simplify=base_mutation_cfg.weight_simplify,
                weight_optimize=base_mutation_cfg.weight_optimize,
                weight_neural_mutate_tree=base_mutation_cfg.weight_neural_mutate_tree
            )

        return model_settings, neural_options, mutation_weights

    def _reconstruct_hyperparameters(
        self,
        trial: optuna.Trial,
        saved_params: Dict
    ) -> Tuple[ModelSettings, NeuralOptions, MutationWeights]:
        """
        Reconstruct hyperparameters from saved trial by suggesting fixed values.

        Forces Optuna to use exact saved values for resumption.

        Args:
            trial: Optuna trial object
            saved_params: Saved trial parameters

        Returns:
            Tuple of (ModelSettings, NeuralOptions, MutationWeights)
        """
        # Reconstruct Optuna trial params
        for param_name, param_value in saved_params.get('raw_params', {}).items():
            if isinstance(param_value, float):
                trial.suggest_float(param_name, param_value, param_value)
            elif isinstance(param_value, int):
                trial.suggest_int(param_name, param_value, param_value)
            elif isinstance(param_value, bool):
                trial.suggest_categorical(param_name, [param_value])

        # Reconstruct objects from dicts
        return self._reconstruct_objects(saved_params)

    def _reconstruct_objects(
        self,
        params: Dict
    ) -> Tuple[ModelSettings, NeuralOptions, MutationWeights]:
        """
        Reconstruct dataclass objects from serialized dictionaries.

        Args:
            params: Trial parameters dictionary

        Returns:
            Tuple of (ModelSettings, NeuralOptions, MutationWeights)
        """
        model_settings = ModelSettings(**params['model_settings'])
        neural_options = NeuralOptions(**params['neural_options'])
        mutation_weights = MutationWeights(**params['mutation_weights'])
        return model_settings, neural_options, mutation_weights

    # ========================================================================
    # TRIAL MANAGEMENT (PRIVATE)
    # ========================================================================

    def _find_incomplete_trial(self) -> Optional[Tuple[int, Dict[str, Any]]]:
        """
        Find the most recent incomplete trial.

        Returns:
            Tuple of (trial_id, params_dict) if found, None otherwise
        """
        coord_dir = self.executor.coord_dir
        params_files = list(coord_dir.glob("trial_*_params.json"))

        incomplete_trials = []
        for params_file in params_files:
            trial_id_str = params_file.stem.replace("_params", "").replace("trial_", "")
            try:
                trial_id = int(trial_id_str)
            except ValueError:
                continue

            # Skip trials that are either completed or pruned
            done_file = coord_dir / f"trial_{trial_id}.done"
            pruned_file = coord_dir / f"trial_{trial_id}.pruned"
            if not done_file.exists() and not pruned_file.exists():
                with open(params_file, 'r') as f:
                    params = json.load(f)
                incomplete_trials.append((trial_id, params))

        if not incomplete_trials:
            return None

        # Return most recent (highest trial_id)
        incomplete_trials.sort(key=lambda x: x[0], reverse=True)
        return incomplete_trials[0]

    def _save_trial_params(self, trial_id: int, trial_params: Dict):
        """
        Save trial parameters for potential resumption.

        Args:
            trial_id: Trial ID
            trial_params: Trial parameters dictionary
        """
        coord_dir = self.executor.coord_dir
        params_file = coord_dir / f"trial_{trial_id}_params.json"
        with open(params_file, 'w') as f:
            json.dump(trial_params, f, indent=2)

    def _mark_trial_done(self, trial_id: int, pruned: bool = False):
        """
        Create marker file for completed or pruned trial.

        Args:
            trial_id: Trial ID
            pruned: If True, mark as pruned; otherwise mark as completed
        """
        coord_dir = self.executor.coord_dir
        suffix = "pruned" if pruned else "done"
        marker_file = coord_dir / f"trial_{trial_id}.{suffix}"
        current_time = datetime.now().isoformat()
        marker_file.write_text(current_time)
        
        self.logger.info(f"Marked trial {trial_id} as {suffix}")

    # ========================================================================
    # STUDY SETUP (PRIVATE)
    # ========================================================================

    def _create_or_load_study(self, resume: bool = False) -> optuna.Study:
        """
        Create new study or load existing one.

        Args:
            resume: If True, attempt to resume existing study

        Returns:
            Optuna Study object
        """
        study_config = self.config['study']
        storage_url = study_config['storage']
        study_name = study_config['name']

        sampler = self._create_sampler()
        pruner = self._create_pruner()

        if resume:
            try:
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
            load_if_exists=True
        )

        self.logger.info(f"Created/loaded study '{study_name}' with direction '{study_config['direction']}'")
        return study

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """
        Create Optuna sampler from configuration.

        Returns:
            Optuna sampler instance
        """
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
        """
        Create Optuna pruner from configuration.

        Returns:
            Optuna pruner instance
        """
        pruner_config = self.config['pruner']

        if pruner_config['type'] == 'MedianPruner':
            return MedianPruner(
                n_startup_trials=pruner_config.get('n_startup_trials', 20),
                n_warmup_steps=pruner_config.get('n_warmup_steps', 30),
                interval_steps=pruner_config.get('interval_steps', 10)
            )
        else:
            raise ValueError(f"Unknown pruner type: {pruner_config['type']}")

    def _enqueue_baseline_trial(self):
        """
        Enqueue trial with baseline values from base config.

        Works for any hyperparameter configuration (neural-only, weights-only, or combined).
        """
        baseline_params = {}

        # Process neural options if being optimized
        if 'neural_options' in self.config['hyperparameters']:
            base_neural = self.base_config.symbolic_regression.neural_options
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
            base_weights = self.base_config.symbolic_regression.mutation_weights
            weight_params = self.config['hyperparameters']['mutation_weights']

            weight_names = [name for name in weight_params.keys() if name != 'use_dirichlet_sampling']

            # Extract base values and convert to Dirichlet raw values
            weight_values = [getattr(base_weights, name) for name in weight_names]
            total = sum(weight_values)

            # Convert to raw format
            for name, value in zip(weight_names, weight_values):
                normalized_weight = value / total
                raw_value = -np.log(normalized_weight)
                baseline_params[f"raw_{name}"] = raw_value

        # Enqueue the baseline trial
        if baseline_params:
            self.study.enqueue_trial(baseline_params)
            self.logger.info(f"Enqueued baseline trial with {len(baseline_params)} parameters")
            self.logger.info(f"Baseline parameters: {baseline_params}")
        else:
            self.logger.info("No hyperparameters found to optimize - skipping baseline trial")

    # ========================================================================
    # RESULTS (PRIVATE)
    # ========================================================================

    def _print_final_results(self):
        """Print and save final optimization results."""
        if not self.study:
            return

        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]

        self.logger.info("="*60)
        self.logger.info("FINISHED STUDY - OPTIMIZATION RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Total trials: {len(self.study.trials)}")
        self.logger.info(f"  Completed: {len(completed_trials)}")
        self.logger.info(f"  Pruned: {len(pruned_trials)}")
        self.logger.info(f"  Failed: {len(failed_trials)}")

        if completed_trials:
            self.logger.info(f"\nBest trial:")
            self.logger.info(f"  Value: {self.study.best_value:.6f}")
            self.logger.info(f"  Params:")
            for key, value in self.study.best_params.items():
                if isinstance(value, float):
                    self.logger.info(f"    {key}: {value:.6f}")
                else:
                    self.logger.info(f"    {key}: {value}")

            # Print parameter importance if enough trials
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
        results_file = Path("sweeps_refactor/optuna_results.txt")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, "w") as f:
            f.write(f"Optuna Optimization Results\n")
            f.write(f"==========================\n\n")
            f.write(f"Total trials: {len(self.study.trials)}\n")
            f.write(f"Completed: {len(completed_trials)}\n")
            f.write(f"Pruned: {len(pruned_trials)}\n")
            f.write(f"Failed: {len(failed_trials)}\n\n")

            if completed_trials:
                f.write(f"Best trial value: {self.study.best_value:.6f}\n")
                f.write(f"Best parameters:\n")
                for key, value in self.study.best_params.items():
                    f.write(f"  {key}: {value}\n")

        self.logger.info(f"\nResults saved to: {results_file}")

        # Print dashboard instructions
        storage_path = self.config['study']['storage']
        if storage_path.startswith('sqlite:///'):
            db_path = storage_path.replace('sqlite:///', '')
            self.logger.info(f"\nTo view results in Optuna dashboard:")
            self.logger.info(f"  optuna-dashboard {storage_path}")

    # ========================================================================
    # UTILITIES (PRIVATE)
    # ========================================================================

    def _to_dict(self, obj):
        """
        Convert dataclass to dict (handles both OmegaConf and regular dataclasses).

        Args:
            obj: Object to convert

        Returns:
            Dictionary representation
        """
        if hasattr(obj, '_content'):  # OmegaConf
            return OmegaConf.to_container(obj)
        return obj.__dict__


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Run Optuna hyperparameter optimization for SR')
    parser.add_argument('--config', type=str, default='optuna_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume existing study if it exists')
    parser.add_argument('--node_id', type=int, help='Node ID for distributed execution (0-indexed)')
    parser.add_argument('--total_nodes', type=int, help='Total number of nodes')

    args = parser.parse_args()

    # Validate config exists
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

    # Create and run optimizer
    runner = OptunaHyperoptRunner(str(config_path), args.node_id, args.total_nodes)
    runner.optimize(resume=args.resume)


if __name__ == "__main__":
    main()
