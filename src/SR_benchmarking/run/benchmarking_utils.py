import os
from dataclasses import dataclass, asdict, field
from typing import Iterable, Literal
import json

from pysr import PySRRegressor, TensorBoardLoggerSpec
from dataset import utils as dataset_utils
from run.pysr_interface_utils import (
    init_mutation_logger, 
    close_mutation_logger, 
    get_neural_mutation_stats, 
    reset_neural_mutation_stats,
    summarize_stats_dict
)


# Model-specific settings

@dataclass
class NeuralOptions:
    active: bool = False
    model_path: str = ""
    sampling_eps: float = 0.05
    subtree_min_nodes: int = 3
    subtree_max_nodes: int = 10
    device: str = "cpu"
    verbose: bool = False
    max_resamples: int = 100
    max_tree_size_diff: int = 5
    require_tree_size_similarity: bool = True
    require_novel_skeleton: bool = True
    require_expr_similarity: bool = True
    similarity_threshold: float = 0.8
    sample_batchsize: int = 32
    sample_logits: bool = True
    log_subtree_strings: bool = False
    subtree_max_features: int = 10

@dataclass
class MutationWeights:  # TODO: Rm weight_ prefix
    weight_add_node: float = 2.47
    weight_insert_node: float = 0.0112
    weight_delete_node: float = 0.870
    weight_do_nothing: float = 0.273
    weight_mutate_constant: float = 0.0346
    weight_mutate_operator: float = 0.293
    weight_swap_operands: float = 0.198
    weight_rotate_tree: float = 4.26
    weight_randomize: float = 0.000502
    weight_simplify: float = 0.00209
    weight_optimize: float = 0.0
    weight_neural_mutate_tree: float = 0.0

@dataclass
class ModelSettings:
    niterations: int = 40
    loss_function: str | None = None  # None for default loss
    early_stopping_condition: float = 0.0  # Loss threshold to stop training. Use =0.0 to deactivate
    verbosity: int = 1
    precision: int = 64
    batching: bool = False
    batch_size: int = 50

# Run-specific settings

@dataclass
class DatasetSettings:
    dataset_name: Literal['synthetic', 'feynman', 'pysr-difficult', 'custom'] = 'feynman'
    num_samples: int = 2000
    noise: float = 0.0001
    eq_idx: int = 10
    forbid_ops: Iterable[str] | None = None
    custom_expr: str | None = None  # If dataset_name == 'custom', this is the expression to use.


def create_LaSR_custom_loss():
    custom_loss = """
    function eval_loss(tree, dataset::Dataset{T,L}, options, idx)::L where {T,L}
        X = isnothing(idx) ? dataset.X : dataset.X[:, idx]
        y = isnothing(idx) ? dataset.y : dataset.y[idx]
        n = isnothing(idx) ? dataset.n : length(idx)
        
        prediction, flag = eval_tree_array(tree, X, options)
        if !flag
            return L(Inf)
        end
        return sum( (1000 .* (prediction .- y) ) .^ 2) / n
    end
    """
    return custom_loss

# Dataclass to store package and it's hyperparams
@dataclass
class PackagedModel:
    model: PySRRegressor
    neural_options: NeuralOptions
    mutation_weights: MutationWeights
    model_settings: ModelSettings
    model_args: dict = field(default_factory=dict)


def init_pysr_model(
    model_settings: ModelSettings = ModelSettings(),
    mutation_weights: MutationWeights = MutationWeights(),
    neural_options: NeuralOptions = NeuralOptions(),
    model_args: dict = {}  # Additional arguments to pass to the model constructor. Only for quick debug/dev.
) -> PackagedModel:
    """
    Initialize the PySRRegressor model with the given configuration. Model can then be used to fit datasets 
    multiple times without having to re-initialize.
    """
    # Keys that are not args to PySRRegressor but are in ModelSettings
    model_settings_special_keys = ['early_stopping_condition']

    neural_options_dict = asdict(neural_options)
    mutation_weights_dict = asdict(mutation_weights)
    model_settings_dict = asdict(model_settings)
    model_settings_dict = {k: v for k, v in model_settings_dict.items() if k not in model_settings_special_keys}

    logger_spec = TensorBoardLoggerSpec(
        log_dir='logs/run',  # Will be replaced during runs
        log_interval=1,  # Log every 10 iterations
        overwrite='append'
    )

    model = PySRRegressor(
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos", "exp", "sin", "zero_sqrt(x) = x >= 0 ? sqrt(x) : zero(x)"],
        extra_sympy_mappings={"zero_sqrt": lambda x: x},  # TODO: Not using Sympy rn. Fix this.
        # Above settings should stay fixed.
        early_stop_condition=f"f(loss, complexity) = (loss < {model_settings.early_stopping_condition:e})",
        neural_options=neural_options_dict,
        logger_spec=logger_spec,
        **mutation_weights_dict,
        **model_settings_dict,
        **model_args
    )

    return PackagedModel(
        model=model,
        neural_options=neural_options,
        mutation_weights=mutation_weights,
        model_settings=model_settings,
        model_args=model_args
    )

def save_run_metadata(
    packaged_model: PackagedModel,
    dataset_settings: DatasetSettings,
    log_dir: str,
) -> None:
    """
    Save the run metadata to the log directory.
    """

    metadata = {
        'model_settings': asdict(packaged_model.model_settings),
        'neural_options': asdict(packaged_model.neural_options),
        'mutation_weights': asdict(packaged_model.mutation_weights),
        'model_args': packaged_model.model_args,
        'dataset_settings': asdict(dataset_settings),
    }

    # Save the model settings
    with open(os.path.join(log_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
def run_single(
    packaged_model: PackagedModel,
    dataset_settings: DatasetSettings,
    log_dir: str,
) -> None:
    # TODO: Update log dir AND output directory!
    # output_directory=f'/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/round1/ckpts/{cfg.run_settings.run_prefix}',

    if dataset_settings.dataset_name == 'custom':
        assert dataset_settings.custom_expr is not None, "Custom expression is required for custom dataset."
        dataset = dataset_utils.create_dataset_from_expression(
            dataset_settings.custom_expr,
            dataset_settings.num_samples,
            dataset_settings.noise,
        )
    else:
        dataset = dataset_utils.load_datasets(
            which=dataset_settings.dataset_name,
            num_samples=dataset_settings.num_samples,
            noise=dataset_settings.noise,
            equation_indices=[dataset_settings.eq_idx],
            forbid_ops=dataset_settings.forbid_ops,
        )[0]

    model = packaged_model.model

    # Setup logging directories
    os.makedirs(log_dir, exist_ok=False)
    save_run_metadata(packaged_model, dataset_settings, log_dir)
    model.logger_spec.log_dir = log_dir
    model.output_directory = log_dir
    # log_dir = '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/round2'
    init_mutation_logger(log_dir, prefix='mutations')
    reset_neural_mutation_stats()

    # Run the model
    model.fit(dataset.X, dataset.y)

    # Close the mutation logger (flushing remaining data to disk)
    close_mutation_logger()

    # Get neural mutation stats and save to file
    neural_stats = summarize_stats_dict(get_neural_mutation_stats())
    with open(os.path.join(log_dir, 'neural_stats.json'), 'w') as f:
        json.dump(neural_stats, f, indent=4)
    

if __name__ == '__main__':
    from pathlib import Path

    model_settings = ModelSettings(
        niterations=4,
        early_stopping_condition=0.0
    )

    neural_options = NeuralOptions(
        active=True,
        model_path='/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-e51hcsb9.onnx',
        sampling_eps=0.05,
        subtree_min_nodes=3,
        subtree_max_nodes=10,
        device='cuda',
        verbose=False,
        max_resamples=100,
        sample_batchsize=32,
        max_tree_size_diff=5,
        require_tree_size_similarity=True
    )
    mutation_weights = MutationWeights(weight_neural_mutate_tree=1.0)

    model = init_pysr_model(
        model_settings=model_settings,
        neural_options=neural_options,
        mutation_weights=mutation_weights
    )

    dataset_settings = DatasetSettings(
        dataset_name='feynman',
        num_samples=2000,
        noise=0.0001,
        eq_idx=10
    )
    log_dir = Path(__file__).parent / 'logs'
    os.makedirs(log_dir, exist_ok=False)

    run_single(
        model=model, 
        dataset_settings=dataset_settings, 
        log_dir=log_dir
    )
