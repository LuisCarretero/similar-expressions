from pysr import PySRRegressor, TensorBoardLoggerSpec
import sys
sys.path.append("/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/sr_inference_benchmarking")
import dataset_utils
from tqdm import trange
import os

def eval_equation(X, y, neural_options, log_dir, max_iter=10, early_stopping_condition=1e-8, weight_neural_mutate_tree=0.0):
    custom_loss = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
    end
    """    
    logger_spec = TensorBoardLoggerSpec(
        log_dir=log_dir,
        log_interval=10,  # Log every 10 iterations
    )

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
    )
    model.fit(X, y)
    return model

def run_benchmark(n_runs=5):
    model_id = 'e51hcsb9'

    log_basedir = '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/sr_inference_benchmarking/logs'
    # log_basedir = '/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/logs'
    neural_options=dict(
        active=True,
        model_path=f"/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/dev/ONNX/onnx-models/model-{model_id}.onnx",
        sampling_eps=0.08,
        subtree_min_nodes=3,
        subtree_max_nodes=10,
    )

    # expr = 'cos(x0)/(x0+2)+x0 + (x1 - sqrt(x1))/exp(x1+3)'
    # dataset = dataset_utils.create_dataset_from_expression(expr, 500, 0)
    dataset = dataset_utils.load_datasets('feynman', num_samples=2000, noise=0.0001, equation_indices=[3])[0]
    print(dataset.equation)
    X, y = dataset.X, dataset.y
    
    for i in trange(n_runs, desc='Running benchmark'):
        log_dir = os.path.join(log_basedir, f'250121_eq3_neural_8e-2_{i}')  # neural_4e-2
        eval_equation(X, y, neural_options, log_dir, max_iter=20, early_stopping_condition=1e-8, weight_neural_mutate_tree=1.0)

run_benchmark(n_runs=5)