from pathlib import Path
import argparse
import sys
from typing import List, Tuple, Dict, Any, Union
import wandb
import os
import json

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from utils.config import load_config
from omegaconf import OmegaConf
from run.benchmarking_utils import (
    NeuralOptions,
    MutationWeights,
    ModelSettings,
    DatasetSettings,
    init_pysr_model,
    run_single
)
from analysis.utils import collect_sweep_results


def load_config_with_overrides(config_path: str, args) -> Tuple[Any, ModelSettings, NeuralOptions, MutationWeights, str, str, List[int]]:
    """
    Load config file, convert to dataclass instances, and apply CLI overrides.
    """
    cfg = load_config(config_path)

    # Create dataclass instances from config sections
    model_settings = ModelSettings(
        niterations=cfg.model_settings.niterations,
        loss_function=cfg.model_settings.loss_function,
        early_stopping_condition=cfg.model_settings.early_stopping_condition,
        verbosity=cfg.model_settings.verbosity,
        precision=cfg.model_settings.precision,
        batching=cfg.model_settings.batching,
        batch_size=cfg.model_settings.batch_size
    )

    # Create neural options from config
    neural_cfg = cfg.symbolic_regression.neural_options
    neural_options = NeuralOptions(
        active=neural_cfg.active,
        model_path=neural_cfg.model_path,
        sampling_eps=neural_cfg.sampling_eps,
        subtree_min_nodes=neural_cfg.subtree_min_nodes,
        subtree_max_nodes=neural_cfg.subtree_max_nodes,
        device=neural_cfg.device,
        verbose=neural_cfg.verbose,
        max_tree_size_diff=neural_cfg.max_tree_size_diff,
        require_tree_size_similarity=neural_cfg.require_tree_size_similarity,
        require_novel_skeleton=neural_cfg.require_novel_skeleton,
        require_expr_similarity=neural_cfg.require_expr_similarity,
        similarity_threshold=neural_cfg.similarity_threshold,
        log_subtree_strings=neural_cfg.log_subtree_strings,
        sample_logits=neural_cfg.sample_logits,
        max_resamples=neural_cfg.max_resamples,
        sample_batchsize=neural_cfg.sample_batchsize,
        subtree_max_features=neural_cfg.subtree_max_features
    )

    # Create mutation weights from config
    mutation_cfg = cfg.symbolic_regression.mutation_weights
    mutation_weights = MutationWeights(
        weight_add_node=mutation_cfg.weight_add_node,
        weight_insert_node=mutation_cfg.weight_insert_node,
        weight_delete_node=mutation_cfg.weight_delete_node,
        weight_do_nothing=mutation_cfg.weight_do_nothing,
        weight_mutate_constant=mutation_cfg.weight_mutate_constant,
        weight_mutate_operator=mutation_cfg.weight_mutate_operator,
        weight_swap_operands=mutation_cfg.weight_swap_operands,
        weight_rotate_tree=mutation_cfg.weight_rotate_tree,
        weight_randomize=mutation_cfg.weight_randomize,
        weight_simplify=mutation_cfg.weight_simplify,
        weight_optimize=mutation_cfg.weight_optimize,
        weight_neural_mutate_tree=mutation_cfg.weight_neural_mutate_tree
    )

    # Apply CLI overrides
    # Model settings overrides
    if hasattr(args, 'niterations') and args.niterations is not None:
        model_settings.niterations = args.niterations
    if hasattr(args, 'pysr_verbosity') and args.pysr_verbosity is not None:
        model_settings.verbosity = args.pysr_verbosity

    # Config value overrides
    log_dir = args.log_dir if hasattr(args, 'log_dir') and args.log_dir is not None else cfg.run_settings.log_dir
    dataset_name = args.dataset if hasattr(args, 'dataset') and args.dataset is not None else cfg.dataset.name
    equations = args.equations if hasattr(args, 'equations') and args.equations is not None else cfg.dataset.equation_indices

    return cfg, model_settings, neural_options, mutation_weights, log_dir, dataset_name, equations


def merge_configs(
        model_settings: ModelSettings, 
        mutation_weights: MutationWeights, 
        neural_options: NeuralOptions, 
        wandb_config: Dict[str, Any]
) -> Tuple[ModelSettings, MutationWeights, NeuralOptions]:
    
    def _set_attr_safely(obj, key, value):
        if hasattr(obj, key):
            setattr(obj, key, value)
        else:
            raise AttributeError(f"Settings object {type(obj).__name__} has no attribute '{key}'")
        
    def _lit2num(x: str) -> Union[float, int, str]:
        try:
            tmp = float(x)
            if tmp.is_integer():
                tmp = int(tmp)
        except:
            tmp = x
        return tmp

    for k, v in wandb_config.items():
        category, key = k.split('.')
        if category == 'model_settings':
            _set_attr_safely(model_settings, key, _lit2num(v))
        elif category == 'mutation_weights':
            _set_attr_safely(mutation_weights, key, _lit2num(v))
        elif category == 'neural_options':
            _set_attr_safely(neural_options, key, _lit2num(v))
        else:
            raise ValueError(f'Unknown category: {category}')

    return model_settings, mutation_weights, neural_options


def apply_cli_overrides(cfg, model_settings, args):
    """Apply CLI argument overrides to config objects"""
    # Model settings overrides
    if hasattr(args, 'niterations') and args.niterations is not None:
        model_settings.niterations = args.niterations
    if hasattr(args, 'pysr_verbosity') and args.pysr_verbosity is not None:
        model_settings.verbosity = args.pysr_verbosity

    # Config value overrides
    log_dir = args.log_dir if hasattr(args, 'log_dir') and args.log_dir is not None else cfg.run_settings.log_dir
    dataset_name = args.dataset if hasattr(args, 'dataset') and args.dataset is not None else cfg.dataset.name
    equations = args.equations if hasattr(args, 'equations') and args.equations is not None else cfg.dataset.equation_indices

    return log_dir, dataset_name, equations


def run_equations(
    cfg,
    model_settings,
    neural_options,
    mutation_weights,
    equations: List[int],
    dataset_name: str,
    log_dir: str,
    pooled: bool = False,
) -> None:
    """
    Run multiple SR experiments with unified logic.

    Args:
        pooled: If True, aggregate results into single WandB run. If False, separate WandB runs.
    """
    wandb_logging = cfg.run_settings.wandb_logging

    # Filter out completed equations
    completed_equations = [eq for eq in equations if is_equation_completed(eq, dataset_name, log_dir, pooled)]
    remaining_equations = sorted(list(set(equations) - set(completed_equations)))

    # Print informative messages and early return if nothing to do
    if completed_equations:
        print(f'[INFO] Skipping {len(completed_equations)} completed equations: {completed_equations}')
    
    if not remaining_equations:
        print('[INFO] All equations already completed, nothing to run')
        return
    
    print(f'[INFO] Running {len(remaining_equations)} remaining equations: {remaining_equations}')
    equations = remaining_equations

    if pooled:
        # Pooled mode: pool multiple runs per equation
        for eq_idx in equations:
            print(f'[INFO] Starting pooled runs for equation {eq_idx}')

            # Init wandb for this equation
            os.makedirs(log_dir, exist_ok=True)

            # Generate descriptive run name
            experiment_name = log_dir.split('/')[-1]
            run_name = f"{experiment_name}_{dataset_name}_eq{eq_idx}"

            wandb_run = wandb.init(
                dir=log_dir,
                project='simexp-SR',
                name=run_name,
                config=OmegaConf.to_container(cfg, resolve=True)
            )

            # Create dataset settings
            dataset_settings = DatasetSettings(
                dataset_name=dataset_name,
                eq_idx=eq_idx,
                num_samples=cfg.dataset.num_samples,
                noise=cfg.dataset.noise,
                forbid_ops=OmegaConf.to_container(cfg.dataset.forbid_ops, resolve=True),
                univariate=cfg.dataset.univariate
            )

            # Create fresh model settings for each run
            model_settings = ModelSettings(
                niterations=model_settings.niterations,
                early_stopping_condition=model_settings.early_stopping_condition,
                verbosity=model_settings.verbosity
            )
            # If we want to run sweeps, update this code. I think in that case we don't pass cfg to wandb.init
            # print('WandB config:', wandb.config)
            # # Merge with wandb.config <- This is needed for WandB sweeps
            # model_settings, mutation_weights, neural_options = merge_configs(
            #     model_settings,
            #     mutation_weights,
            #     neural_options,
            #     wandb.config
            # )
            mutation_weights.normalize()

            # Create model for this run
            model = init_pysr_model(model_settings, mutation_weights, neural_options)

            run_dirs = []
            for run_i in range(cfg.run_settings.n_runs):
                print(f'[INFO] Running equation {eq_idx}, run {run_i+1}/{cfg.run_settings.n_runs}')

                # Run SR for this equation/run combination
                run_dir = os.path.join(log_dir, f'{dataset_name}_eq{eq_idx}_run{run_i}')
                run_dirs.append(f'{dataset_name}_eq{eq_idx}_run{run_i}')

                try:
                    run_single(
                        model,
                        dataset_settings,
                        log_dir=run_dir,
                        wandb_logging=False  # <- Important, to not interfere with batched runs
                    )
                except Exception as e:
                    print(f'[ERROR] Error running equation {eq_idx}, run {run_i}: {e}')
                    raise e
                    # continue

            # Aggregate results across all runs for this equation
            try:
                all_step_stats, all_summary_stats_combined = collect_sweep_results(
                    log_dir,
                    run_dirs,
                    keep_single_runs=False,
                    combined_prefix='mean-'
                )

                # Log to WandB
                for step, values in all_step_stats:
                    wandb_run.log(values, step=step, commit=False)
                wandb_run.log({}, commit=True)
                wandb_run.summary.update(all_summary_stats_combined)
            except Exception as e:
                print(f'[ERROR] Error collecting results for equation {eq_idx}: {e}')
                continue
            finally:
                wandb.finish()

    else:
        # Separate mode: individual WandB runs
        dataset_cfg = cfg.dataset

        # Create model
        packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)

        # Run benchmark for each equation
        for eq_idx in equations:
            print(f'[INFO] Running equation {eq_idx} from dataset {dataset_name}')
            dataset_settings = DatasetSettings(
                dataset_name=dataset_name,
                eq_idx=eq_idx,
                num_samples=dataset_cfg.num_samples,
                noise=dataset_cfg.noise,
                forbid_ops=OmegaConf.to_container(dataset_cfg.forbid_ops, resolve=True),
                univariate=dataset_cfg.univariate
            )
            try:
                run_single(
                    packaged_model,
                    dataset_settings,
                    log_dir=str(Path(log_dir) / f'{dataset_name}_eq{eq_idx}'),
                    wandb_logging=wandb_logging,
                )
            except Exception as e:
                print(f'[ERROR] Error running equation {eq_idx} from dataset {dataset_name}: {e}')
                continue

def parse_equations(s: str) -> List[int]:
    """Parse equation string into list of integers.

    Supports:
    - Single: "102" -> [102]
    - List: "102,202,302" -> [102, 202, 302]
    - Range: "102:105" -> [102, 103, 104]
    - Step: "102:110:2" -> [102, 104, 106, 108]
    - Mixed: "102,200:203" -> [102, 200, 201, 202]
    """
    if not s:
        return []

    result = []
    for segment in s.split(','):
        segment = segment.strip()
        if ':' in segment:
            # Range: reuse existing logic
            result.extend(list(range(*map(lambda x: int(x.replace('m', '-')) if x else None, segment.split(':')))))
        else:
            # Single number
            result.append(int(segment.replace('m', '-')))

    return result


def is_equation_completed(eq_idx: int, dataset_name: str, log_dir: str, pooled: bool) -> bool:
    """Check if an equation has already been completed by checking directory existence.

    Args:
        eq_idx: Equation index to check
        dataset_name: Dataset name
        log_dir: Base log directory
        pooled: Whether running in pooled mode

    Returns:
        True if any directory exists for this equation (indicating it was started)
    """
    if pooled:
        # For pooled mode, check if any run directory exists for this equation
        # Format: {log_dir}/{dataset_name}_eq{eq_idx}_run{run_i}
        run_i = 0
        run_dir = os.path.join(log_dir, f'{dataset_name}_eq{eq_idx}_run{run_i}')
        return os.path.exists(run_dir)
    else:
        # For separate mode, check if equation directory exists
        # Format: {log_dir}/{dataset_name}_eq{eq_idx}
        eq_dir = os.path.join(log_dir, f'{dataset_name}_eq{eq_idx}')
        return os.path.exists(eq_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SR experiments using config file')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--node_id', type=int, help='Node ID for distributed runs (0-indexed)')
    parser.add_argument('--total_nodes', type=int, help='Total number of nodes for distributed runs')
    parser.add_argument('--pooled', action='store_true', help='Run equations with pooled/aggregated results instead of separate runs')

    parser.add_argument('--log_dir', type=str, help='Override log directory from config')
    parser.add_argument('--niterations', type=int, help='Override number of iterations from config')
    args = parser.parse_args()

    # Load config and apply CLI overrides in one step
    cfg, model_settings, neural_options, mutation_weights, log_dir, dataset_name, equations = load_config_with_overrides(args.config, args)

    # Distribute equations across nodes if distributed mode is enabled
    if args.node_id is not None and args.total_nodes is not None:
        equations = equations[args.node_id::args.total_nodes]
        print(f'[INFO] Node {args.node_id}/{args.total_nodes}: Running {len(equations)} equations: {equations}')

    run_equations(
        cfg=cfg,
        model_settings=model_settings,
        neural_options=neural_options,
        mutation_weights=mutation_weights,
        equations=equations,
        dataset_name=dataset_name,
        log_dir=log_dir,
        pooled=args.pooled
    )

    # Examples:
    # Individual runs (separate logging): python -m run.run_multiple --config=run/config.yaml --equations="102,202:205" --dataset=feynman --niterations=10 --pysr_verbosity=1
    # Individual runs with config defaults: python -m run.run_multiple --config=run/config.yaml
    # Pooled runs (aggregated logging): python -m run.run_multiple --config=run/config.yaml --pooled
    
    # To run equations programmatically:
    # from run.run_multiple import run_equations
    # run_equations(config_path='run/config.yaml', log_dir='path/to/logs', pooled=True)  # For pooled mode
    # run_equations(config_path='run/config.yaml', log_dir='path/to/logs', pooled=False) # For separate mode