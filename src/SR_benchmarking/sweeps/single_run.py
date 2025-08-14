# Single sweep run consisting of multiple SR runs on different datasets but with same set of hyperparameters.
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import wandb
import os
import sys
import json
from pathlib import Path

# Add SR_benchmarking to path if not running in module mode
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from run.benchmarking_utils import (
    NeuralOptions,
    MutationWeights,
    ModelSettings,
    DatasetSettings,
    init_pysr_model,
    run_single
)
from analysis.utils import collect_sweep_results


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


def single_sweep_run(log_dir: str, run_prefix: str) -> None:
    # Model settings
    model_settings = ModelSettings(
        niterations=20,
        verbosity=0,
    )
    neural_options = NeuralOptions(
        active=True,
        model_path='/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx',
        device='cuda',
        verbose=False,
        log_subtree_strings=False,
        sample_logits=True,

        sample_batchsize=32,

        sampling_eps=0.05,
        subtree_min_nodes=3,
        subtree_max_nodes_diff=8,
        max_resamples=95,
        max_tree_size_diff=5,
        require_tree_size_similarity=True,
        require_novel_skeleton=True,
        require_expr_similarity=True,
        similarity_threshold=0.8,
        subtree_max_features=2
    )
    mutation_weights = MutationWeights(
        weight_add_node=0.1610783001322248,
        weight_delete_node=0.558056984593573,
        weight_do_nothing=0.5322796201659097,
        weight_insert_node=0.9124190624970774,
        weight_mutate_constant=0.2643340049588264,
        weight_mutate_operator=0.4471111685390209,
        weight_neural_mutate_tree=1,
        weight_optimize=0.5091563535691276,
        weight_randomize=0.1081032643312303,
        weight_rotate_tree=0.3020124627925661,
        weight_simplify=0.08332806744756971,
        weight_swap_operands=0.3239389265186603,
    )

    # Define datasets
    dataset_name = 'pysr-difficult'
    eq_indices = list(range(3))
    datasets = [
        DatasetSettings(dataset_name=dataset_name, eq_idx=eq_idx)
        for eq_idx in eq_indices
    ]

    # Init wandb
    os.makedirs(log_dir, exist_ok=True)
    wandb_run = wandb.init(dir=log_dir)

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


if __name__ == "__main__":
    log_dir = '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optim_sweep_1'
    run_prefix = get_run_prefix(log_dir)
    single_sweep_run(log_dir, run_prefix)
