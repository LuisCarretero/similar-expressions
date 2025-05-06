# Single sweep run consisting of multiple SR runs on different datasets but with same set of hyperparameters.
from pathlib import Path
from typing import List, Dict, Any, Tuple
import wandb
import os
import sys
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
    if (wandb_model_settings := wandb_config.get('model_settings', None)) is not None:
        model_settings.update(wandb_model_settings)

    if (wandb_mutation_weights := wandb_config.get('mutation_weights', None)) is not None:
        mutation_weights.update(wandb_mutation_weights)

    if (wandb_neural_options := wandb_config.get('neural_options', None)) is not None:
        neural_options.update(wandb_neural_options)

    return model_settings, mutation_weights, neural_options


def single_sweep_run():
    LOG_DIR = '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/optim_sweep_1'
    
    # Model settings
    model_settings = ModelSettings(
        niterations=4,
        verbosity=0,
    )
    neural_options = NeuralOptions(
        active=True,
        model_path='/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx',
        device='cuda',
    )
    mutation_weights = MutationWeights(
        weight_neural_mutate_tree=1.0,
        weight_mutate_constant = 0.0353,
        weight_mutate_operator = 3.63,
        weight_swap_operands = 0.00608,
        weight_rotate_tree = 1.42,
        weight_add_node = 0.0771,
        weight_insert_node = 2.44,
        weight_delete_node = 0.369,
        weight_simplify = 0.00148,
        weight_randomize = 0.00695,
        weight_do_nothing = 0.431,
        weight_optimize = 0.0,
    )

    # Define datasets
    dataset_name = 'pysr-difficult'
    eq_indices = list(range(2))
    datasets = [
        DatasetSettings(dataset_name=dataset_name, eq_idx=eq_idx)
        for eq_idx in eq_indices
    ]

    # Init wandb
    os.makedirs(LOG_DIR, exist_ok=True)
    wandb_run = wandb.init(dir=LOG_DIR)
    # Merge above with wandb.config
    model_settings, mutation_weights, neural_options = merge_configs(
        model_settings, 
        mutation_weights, 
        neural_options, 
        wandb.config
    )
    
    # Create model
    model = init_pysr_model(model_settings, mutation_weights, neural_options)

    # Run SR for each dataset (disable WandB!)
    for dataset in datasets:
        run_single(
            model, 
            dataset, 
            log_dir=os.path.join(LOG_DIR, f'{dataset.dataset_name}_eq{dataset.eq_idx}'), 
            wandb_logging=False  # <- Important, to not interfere with batched runs
        )

    all_step_stats, all_summary_stats_combined = collect_sweep_results(
        LOG_DIR,
        [f'{dataset.dataset_name}_eq{dataset.eq_idx}' for dataset in datasets],
        keep_single_runs=False,
        combined_prefix='mean-'
    )
    
    for step, values in all_step_stats:
        wandb_run.log(values, step=step, commit=False)
    wandb_run.log({}, commit=True)
    wandb_run.summary.update(all_summary_stats_combined)
    wandb.finish()


# def main():
#     sweep_configuration = {
#         "method": "random",
#         "metric": {"goal": "maximize", "name": "mean-pareto_volume"},
#         "parameters": {
#             "mutation_weights.weight_neural_mutate_tree": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_mutate_constant": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_mutate_operator": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_swap_operands": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_rotate_tree": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_add_node": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_insert_node": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_delete_node": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_simplify": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_randomize": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_do_nothing": {"max": 1.0, "min": 0.0},
#             "mutation_weights.weight_optimize": {"max": 1.0, "min": 0.0},
#         },
#     }

#     sweep_id = wandb.sweep(sweep=sweep_configuration, project="simexp-SR")
#     wandb.agent(sweep_id, function=single_sweep_run, count=4)


if __name__ == "__main__":
    single_sweep_run()
