"""
Optuna objective function for symbolic regression hyperparameter optimization.

This module wraps the existing SR benchmarking pipeline to work with Optuna's
optimization framework, handling noisy objectives and graceful SLURM requeuing.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import tempfile
import shutil

# Add parent directories to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import optuna
from omegaconf import OmegaConf

# Import existing SR benchmarking infrastructure
from run.utils import (
    ModelSettings, NeuralOptions, MutationWeights, DatasetSettings,
    init_pysr_model, run_single, load_config
)
from run.run_multiple import _parse_eq_idx
from analysis.utils import collect_sweep_results


class OptunaObjective:
    """
    Optuna objective function that wraps the existing SR benchmarking pipeline.

    Handles equation batching, intermediate reporting, and graceful SLURM requeuing.
    """

    def __init__(self, config: Dict[str, Any], interruption_flag=None):
        """
        Initialize the objective function with configuration.

        Args:
            config: Configuration dictionary loaded from optuna_config.yaml
            interruption_flag: Callable that returns True if execution should be interrupted
        """
        self.config = config
        self.base_config = load_config(config['base_config'])
        self.equations = _parse_eq_idx(config['dataset']['equation_indices'])
        self.n_runs = config['dataset']['n_runs']
        self.equations_per_batch = config['execution']['equations_per_batch']
        self.interruption_flag = interruption_flag or (lambda: False)

        # Create temporary directory for this optimization session
        self.temp_dir = Path(tempfile.mkdtemp(prefix="optuna_sr_"))

        print(f"[OPTUNA] Initialized objective with {len(self.equations)} equations, "
              f"{self.n_runs} runs each, batch size {self.equations_per_batch}")
        print(f"[OPTUNA] Temporary directory: {self.temp_dir}")

    def _sample_hyperparameters(self, trial: optuna.Trial) -> Tuple[ModelSettings, NeuralOptions, MutationWeights]:
        """
        Sample hyperparameters from the trial using the configuration.

        Args:
            trial: Optuna trial object

        Returns:
            Tuple of (ModelSettings, NeuralOptions, MutationWeights) dataclass instances
        """
        # Start with base configuration
        base_model_cfg = self.base_config.model_settings
        base_neural_cfg = self.base_config.symbolic_regression.neural_options
        base_mutation_cfg = self.base_config.symbolic_regression.mutation_weights

        # Sample model settings
        model_settings = ModelSettings(
            niterations=base_model_cfg.niterations,
            loss_function=base_model_cfg.loss_function,
            early_stopping_condition=base_model_cfg.early_stopping_condition,
            verbosity=base_model_cfg.verbosity,
            precision=base_model_cfg.precision,
            batching=base_model_cfg.batching,
            batch_size=base_model_cfg.batch_size
        )

        # Sample neural options
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

        # Sample neural hyperparameters if specified
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

            if 'subtree_max_nodes_diff' in neural_params:
                neural_options.subtree_max_nodes_diff = trial.suggest_int('subtree_max_nodes_diff',
                    neural_params['subtree_max_nodes_diff']['low'],
                    neural_params['subtree_max_nodes_diff']['high'])

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

        # Sample mutation weights using proper Dirichlet distribution if present
        if 'mutation_weights' in self.config['hyperparameters']:
            mutation_params = self.config['hyperparameters']['mutation_weights']
            mutation_weights = MutationWeights()

            # Get all weight parameter names
            weight_names = [name for name in mutation_params.keys() if name != 'use_dirichlet_sampling']

            # Sample from uniform distribution and transform to exponential for Dirichlet distribution
            raw_weights = []
            for weight_name in weight_names:
                u = trial.suggest_float(f"raw_{weight_name}", 1e-10, 1.0 - 1e-10)
                raw_weights.append(-np.log(u))

            # Normalize to create proper Dirichlet sample (uniform over simplex)
            total = sum(raw_weights)
            normalized_weights = [w / total for w in raw_weights]

            # Assign normalized weights to mutation_weights object
            for weight_name, weight_value in zip(weight_names, normalized_weights):
                setattr(mutation_weights, weight_name, weight_value)

            print(f"[OPTUNA] Dirichlet sampled weights: {dict(zip(weight_names, normalized_weights))}")
        else:
            # Use base configuration mutation weights if not optimizing
            base_mutation_cfg = self.base_config.symbolic_regression.mutation_weights
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

    def _run_equation_batch(self, equations: List[int], model_settings: ModelSettings,
                           neural_options: NeuralOptions, mutation_weights: MutationWeights) -> List[float]:
        """
        Run a batch of equations and return their pareto volumes.

        Args:
            equations: List of equation indices to run
            model_settings: Model configuration
            neural_options: Neural mutation configuration
            mutation_weights: Mutation weight configuration

        Returns:
            List of pareto volumes (one per equation)
        """
        pareto_volumes = []
        packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)

        for eq_idx in equations:
            # Check for interruption before starting each equation
            if self.interruption_flag():
                print(f"[OPTUNA] Interrupted during batch processing at equation {eq_idx}")
                break

            try:
                eq_pareto_volumes = []

                # Run multiple times per equation for variance reduction
                for run_i in range(self.n_runs):
                    # Create unique run directory
                    run_dir = self.temp_dir / f"{self.config['dataset']['name']}_eq{eq_idx}_run{run_i}"

                    # Create dataset settings
                    dataset_settings = DatasetSettings(
                        dataset_name=self.config['dataset']['name'],
                        eq_idx=eq_idx,
                        num_samples=self.config['dataset']['num_samples'],
                        rel_noise_magn=self.config['dataset']['noise']
                    )

                    # Run single SR experiment
                    run_single(
                        packaged_model=packaged_model,
                        dataset_settings=dataset_settings,
                        log_dir=str(run_dir),
                        wandb_logging=False,  # No WandB logging for individual runs
                        enable_mutation_logging=False  # Disable for speed
                    )

                    # Extract pareto volume from tensorboard data
                    final_pareto_volume = self._extract_run_pv(run_dir)
                    eq_pareto_volumes.append(final_pareto_volume)

                # Use mean (previous results actually show lower variance compared to median)
                eq_mean_pv = np.mean(eq_pareto_volumes)
                pareto_volumes.append(eq_mean_pv)

                print(f"[OPTUNA] Equation {eq_idx}: PV = {eq_mean_pv:.4f} "
                      f"(runs: {eq_pareto_volumes})")

            except Exception as e:
                print(f"[ERROR] Failed to run equation {eq_idx}: {e}")
                pareto_volumes.append(0.0)  # Assign poor score for failed runs

        return pareto_volumes

    def _extract_run_pv(self, run_dir: Path) -> float:
        """Extract results from a single run directory."""
        step_stats, _ = collect_sweep_results(
            str(run_dir.parent),
            [run_dir.name],
            keep_single_runs=False,
            combined_prefix=''
        )
        final_pareto_volume = step_stats[-1][1]['mean/pareto_volume']
        return final_pareto_volume


    def __call__(self, trial: optuna.Trial) -> float:
        """
        Main objective function called by Optuna.

        Args:
            trial: Optuna trial object

        Returns:
            Mean pareto volume across all equations (to be maximized)
        """
        print(f"\n[OPTUNA] Starting trial {trial.number}")

        # Sample hyperparameters
        model_settings, neural_options, mutation_weights = self._sample_hyperparameters(trial)

        print(f"[OPTUNA] Trial {trial.number} hyperparameters:")
        print(f"  Model: niterations={model_settings.niterations}")
        print(f"  Neural: eps={getattr(neural_options, 'sampling_eps', 'default')}")
        print(f"  Mutations: neural={mutation_weights.weight_neural_mutate_tree:.3f}, "
              f"add={mutation_weights.weight_add_node:.3f}")

        all_pareto_volumes = []

        # Process equations in batches for intermediate reporting
        for batch_start in range(0, len(self.equations), self.equations_per_batch):

            batch_end = min(batch_start + self.equations_per_batch, len(self.equations))
            batch_equations = self.equations[batch_start:batch_end]

            print(f"[OPTUNA] Trial {trial.number}: Running equations "
                  f"{batch_equations[0]}-{batch_equations[-1]} "
                  f"(batch {len(all_pareto_volumes)//self.equations_per_batch + 1})")

            # Run this batch of equations
            batch_pareto_volumes = self._run_equation_batch(
                batch_equations, model_settings, neural_options, mutation_weights
            )

            all_pareto_volumes.extend(batch_pareto_volumes)

            # Report intermediate value for pruning
            current_mean = np.mean(all_pareto_volumes)
            trial.report(current_mean, step=len(all_pareto_volumes))

            print(f"[OPTUNA] Trial {trial.number}: Intermediate PV = {current_mean:.4f} "
                  f"(after {len(all_pareto_volumes)} equations)")

            # Check if trial should be pruned
            if trial.should_prune():
                print(f"[OPTUNA] Trial {trial.number} pruned after {len(all_pareto_volumes)} equations")
                raise optuna.TrialPruned()

        # Calculate final objective value
        final_pareto_volume = float(np.mean(all_pareto_volumes))

        print(f"[OPTUNA] Trial {trial.number} completed: Final PV = {final_pareto_volume:.4f}")
        print(f"[OPTUNA] PV statistics: min={np.min(all_pareto_volumes):.4f}, "
              f"max={np.max(all_pareto_volumes):.4f}, std={np.std(all_pareto_volumes):.4f}")

        return final_pareto_volume

    def cleanup(self):
        """Clean up temporary directories."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"[OPTUNA] Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"[WARNING] Failed to clean up {self.temp_dir}: {e}")


def create_objective(config_path: str, interruption_flag=None) -> OptunaObjective:
    """
    Create an Optuna objective function from configuration file.

    Args:
        config_path: Path to optuna_config.yaml
        interruption_flag: Callable that returns True if execution should be interrupted

    Returns:
        OptunaObjective instance
    """
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    return OptunaObjective(config, interruption_flag)