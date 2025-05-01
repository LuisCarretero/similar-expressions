from pysr import PySRRegressor, TensorBoardLoggerSpec
import os
from tqdm import trange
import time
from omegaconf import OmegaConf

from dataset import utils as dataset_utils
from pysr_interface_utils import print_summary_stats, reset_mutation_stats, get_neural_mutation_stats

def setup_model(cfg, use_neural=False):
    neural_options = dict(cfg.symbolic_regression.neural_options)
    n_runs = cfg.run_settings.n_runs
    max_iter = cfg.run_settings.max_iter
    early_stopping_condition = cfg.run_settings.early_stopping_condition
    log_dir = cfg.run_settings.log_dir

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

    logger_spec = TensorBoardLoggerSpec(
        log_dir='logs/run',  # Will be replaced during runs
        log_interval=10,  # Log every 10 iterations
    )

    if use_neural:
        neural_options["active"] = True
        weight_neural_mutate_tree = cfg.symbolic_regression.weight_neural_mutate_tree
    else:
        neural_options["active"] = False
        weight_neural_mutate_tree = 0.0

    model = PySRRegressor(
        niterations=max_iter,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos","exp","sin","zero_sqrt(x) = x >= 0 ? sqrt(x) : zero(x)"],
        extra_sympy_mappings={"zero_sqrt": lambda x: x},  # TODO: Not using Sympy rn. Fix this.
        precision=64,
        neural_options=neural_options,
        weight_neural_mutate_tree=weight_neural_mutate_tree,
        loss_function=custom_loss,
        early_stop_condition=f"f(loss, complexity) = (loss < {early_stopping_condition:e})",
        logger_spec=logger_spec,
        verbosity=0,
        batching=True,
        batch_size=50,
        output_directory=f'/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/round1/ckpts/{cfg.run_settings.run_prefix}'
    )
    return model

def run_benchmark(cfg, model, dataset, start_time, log_postfix):
    eq_idx = dataset.idx
    print(f"\nBenchmarking equation {eq_idx}: {dataset.equation}")
    X, y = dataset.X, dataset.y
    
    log_name = f'{start_time:.0f}_{cfg.run_settings.run_prefix}_eq{eq_idx}_{log_postfix}'
    for i in trange(cfg.run_settings.n_runs, desc=f'Running neural benchmark for equation {eq_idx}'):
        run_log_dir = os.path.join(cfg.run_settings.log_dir, cfg.run_settings.run_prefix, f'{log_name}_{i}')
        model.logger_spec.log_dir = run_log_dir
        reset_mutation_stats()
        model.fit(X, y)
        print_summary_stats(get_neural_mutation_stats())
        
def main(cfg):
    start_time = time.time()

    # Load all datasets
    datasets = dataset_utils.load_datasets(
        cfg.dataset.name, 
        num_samples=cfg.dataset.num_samples, 
        noise=cfg.dataset.noise, 
        equation_indices=cfg.dataset.equation_indices,
        forbid_ops=cfg.dataset.forbid_ops
    )    
    
    if cfg.run_settings.do_neural:
        neural_model = setup_model(cfg, use_neural=True)
        for dataset in datasets:
            run_benchmark(cfg, neural_model, dataset, start_time, 'neural')

    if cfg.run_settings.do_vanilla:
        vanilla_model = setup_model(cfg, use_neural=False)
        for dataset in datasets:
            run_benchmark(cfg, vanilla_model, dataset, start_time, 'vanilla')



if __name__ == "__main__":
    cfg = OmegaConf.load('./src/sr_inference_benchmarking/config.yaml')
    main(cfg)
