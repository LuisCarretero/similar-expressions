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


def load_config_to_dataclasses(config_path: str) -> Tuple[ModelSettings, NeuralOptions, MutationWeights, dict]:
    """
    Load config file and convert to dataclass instances.
    """
    cfg = load_config(config_path)
    
    # Extract run_settings for use by main()
    run_settings = {
        'n_runs': cfg.run_settings.n_runs,
        'log_dir': cfg.run_settings.log_dir,
        'run_prefix': cfg.run_settings.run_prefix,
        'do_vanilla': cfg.run_settings.do_vanilla,
        'do_neural': cfg.run_settings.do_neural
    }
    
    # Create dataclass instances from config sections
    model_settings = ModelSettings(
        niterations=cfg.run_settings.max_iter,
        early_stopping_condition=cfg.run_settings.early_stopping_condition,
        verbosity=0  # Keep low verbosity for multiple runs
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
    
    return model_settings, neural_options, mutation_weights, run_settings


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


def run_equations_pooled(
    log_dir: str,
    run_prefix: str,
    config_path: str = "run/config.yaml",
    equations: List[int] = None,
    dataset_name: str = None
) -> None:
    """
    Run multiple SR experiments with pooled results using config file.
    All equations are run and results are aggregated into a single WandB run.
    """
    # Load default settings from config
    model_settings, neural_options, mutation_weights, _ = load_config_to_dataclasses(config_path)
    cfg = load_config(config_path)
    
    # Override specific settings for multiple equation runs
    model_settings.verbosity = 0  # Keep low verbosity for multiple runs
    neural_options.subtree_max_features = 1  # Univariate!

    # Define datasets - use parameters, then config
    dataset_name = cfg.dataset.name if dataset_name is None else dataset_name
    eq_indices = cfg.dataset.equation_indices if equations is None else equations
        
    datasets = [
        DatasetSettings(dataset_name=dataset_name, eq_idx=eq_idx, univariate=cfg.dataset.univariate)
        for eq_idx in eq_indices
    ]

    # Init wandb
    os.makedirs(log_dir, exist_ok=True)
    wandb_run = wandb.init(
        dir=log_dir,
        project='simexp-SR',
        config=cfg
    )

    # Merge above with wandb.config
    model_settings, mutation_weights, neural_options = merge_configs(
        model_settings, 
        mutation_weights, 
        neural_options, 
        wandb.config
    )
    mutation_weights.normalize()
    
    # Create model
    model = init_pysr_model(model_settings, mutation_weights, neural_options)

    # Run SR for each dataset (disable WandB!)
    for dataset in datasets:
        run_single(
            model, 
            dataset, 
            log_dir=os.path.join(log_dir, f'{run_prefix}_{dataset.dataset_name}_eq{dataset.eq_idx}'), 
            wandb_logging=False  # <- Important, to not interfere with batched runs
        )

    all_step_stats, all_summary_stats_combined = collect_sweep_results(
        log_dir,
        [f'{run_prefix}_{dataset.dataset_name}_eq{dataset.eq_idx}' for dataset in datasets],
        keep_single_runs=False,
        combined_prefix='mean-'
    )
    
    for step, values in all_step_stats:
        wandb_run.log(values, step=step, commit=False)
    wandb_run.log({}, commit=True)
    wandb_run.summary.update(all_summary_stats_combined)
    wandb.finish()


def get_run_prefix(log_dir: str) -> str:
    """
    Get a unique file name for the current run by incrementing a counter stored in a JSON file.
        
    Returns:
        A string with the current iteration number
    """
    counter_file = os.path.join(log_dir, 'sweep_metadata.json')

    os.makedirs(os.path.dirname(counter_file), exist_ok=True)
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as f:
            counter = json.load(f).get('counter', 0)
    else:
        counter = 0
    
    # Save the updated counter
    with open(counter_file, 'w') as f:
        json.dump({'counter': int(counter+1)}, f)

    return f'run{int(counter)}'


def run_equations_separate(
    equations: List[int], 
    dataset_name: str, 
    log_dir: str,
    config_path: str,
    niterations: int = None,
    pysr_verbosity: int = None,
    wandb_logging: bool = True,
) -> None:
    """
    Run multiple SR experiments separately using config file.
    Each equation is logged as a separate WandB run.
    """
    # Load configuration
    model_settings, neural_options, mutation_weights, run_settings = load_config_to_dataclasses(config_path)
    
    # Override with command line arguments if provided
    if niterations is not None:
        model_settings.niterations = niterations
    if pysr_verbosity is not None:
        model_settings.verbosity = pysr_verbosity
    
    # Load dataset settings from config
    cfg = load_config(config_path)
    dataset_cfg = cfg.dataset

    # Create model
    packaged_model = init_pysr_model(model_settings, mutation_weights, neural_options)

    # Run benchmark
    for eq_idx in equations:
        print(f'[INFO] Running equation {eq_idx} from dataset {dataset_name}')
        dataset_settings = DatasetSettings(
            dataset_name=dataset_name,
            eq_idx=eq_idx,
            num_samples=dataset_cfg.num_samples,
            noise=dataset_cfg.noise,
            forbid_ops=OmegaConf.to_container(getattr(dataset_cfg, 'forbid_ops', None))
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

def str_to_list(s: str) -> List[int]:
    """
    Convert a string to a slice.
    """
    if ':' in s:
        return list(range(*map(lambda x: int(x.replace('m', '-')) if x else None, s.split(':'))))
    else:
        return [int(s)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run SR experiments using config file')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--equations', type=str_to_list, help='Equation indices to run (e.g., "1,2,3" or "1:10"). If not provided, uses config defaults.')
    parser.add_argument('--dataset', type=str, help='Dataset name to use. If not provided, uses config defaults.')
    parser.add_argument('--pooled', action='store_true', help='Run equations with pooled/aggregated results instead of separate runs')
    parser.add_argument('--log_dir', type=str, help='Override log directory from config')
    parser.add_argument('--pysr_verbosity', type=int, help='Override Pysr verbosity from config')
    parser.add_argument('--niterations', type=int, help='Override number of iterations from config')
    parser.add_argument('--wandb_logging', type=bool, default=True, help='Enable WandB logging')
    parser.add_argument('--node_id', type=int, help='Node ID for distributed runs (0-indexed)')
    parser.add_argument('--total_nodes', type=int, help='Total number of nodes for distributed runs')
    args = parser.parse_args()

    # Load config for defaults
    cfg = load_config(args.config)
    
    # Use log_dir from config if not provided via CLI
    if args.log_dir is None:
        log_dir = cfg.run_settings.log_dir
    else:
        log_dir = args.log_dir
    
    if args.pooled:
        # Pooled mode: run_equations_pooled
        for i in range(cfg.run_settings.n_runs):
            run_prefix = get_run_prefix(log_dir)
            run_equations_pooled(log_dir, run_prefix, args.config)
    else:
        # Separate mode: run_equations_separate
        # Load equations and dataset from config if not provided
        equations = cfg.dataset.equation_indices if args.equations is None else args.equations
        dataset_name = cfg.dataset.name if args.dataset is None else args.dataset

        # Distribute equations across nodes if distributed mode is enabled
        if args.node_id is not None and args.total_nodes is not None:
            equations = equations[args.node_id::args.total_nodes]
            print(f'[INFO] Node {args.node_id}/{args.total_nodes}: Running {len(equations)} equations: {equations}')

        run_equations_separate(
            equations=equations, 
            dataset_name=dataset_name, 
            log_dir=log_dir,
            config_path=args.config,
            pysr_verbosity=args.pysr_verbosity,
            niterations=args.niterations,
            wandb_logging=args.wandb_logging
        )

    # Examples:
    # Individual runs (separate logging): python -m run.run_multiple --config=run/config.yaml --equations=1:5 --dataset=feynman --niterations=10 --pysr_verbosity=1
    # Individual runs with config defaults: python -m run.run_multiple --config=run/config.yaml
    # Pooled runs (aggregated logging): python -m run.run_multiple --config=run/config.yaml --pooled
    
    # To run pooled equations programmatically:
    # from run.run_multiple import run_equations_pooled, get_run_prefix
    # log_dir = 'path/to/logs'
    # run_prefix = get_run_prefix(log_dir)  
    # run_equations_pooled(log_dir, run_prefix, config_path='run/config.yaml')