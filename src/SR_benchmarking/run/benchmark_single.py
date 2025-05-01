from pysr import PySRRegressor, TensorBoardLoggerSpec
import sys
sys.path.append("/home/lc865/workspace/similar-expressions/src/sr_inference_benchmarking")
import dataset_utils
from tqdm import trange
import os

# def eval_equation(X, y, neural_options, log_dir, max_iter=10, early_stopping_condition=1e-8, weight_neural_mutate_tree=0.0):
#     custom_loss = """
#     function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
#         prediction, flag = eval_tree_array(tree, dataset.X, options)
#         if !flag
#             return L(Inf)
#         end
#         return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
#     end
#     """    
#     logger_spec = TensorBoardLoggerSpec(
#         log_dir=log_dir,
#         log_interval=10,  # Log every 10 iterations
#     )

#     model = PySRRegressor(
#         niterations=max_iter,
#         binary_operators=["+", "*", "-", "/"],
#         unary_operators=["cos","exp","sin","zero_sqrt(x) = x >= 0 ? sqrt(x) : zero(x)"],
#         extra_sympy_mappings={"zero_sqrt": lambda x: x},  # TODO: Not using Sympy rn. Fix this.
#         precision=64,
#         neural_options=neural_options,
#         weight_neural_mutate_tree=weight_neural_mutate_tree,
#         loss_function=custom_loss,
#         early_stop_condition=f"f(loss, complexity) = (loss < {early_stopping_condition:e})",
#         logger_spec=logger_spec,
#     )
#     model.fit(X, y)
#     return model

def run_benchmark(n_runs=5):
    model_id = 'e51hcsb9'
    model_path = f"/home/lc865/workspace/similar-expressions/src/dev/ONNX/onnx-models/model-{model_id}.onnx"
    log_basedir = '/cephfs/store/gr-mc2473/lc865/workspace/benchmark_data/round1/logs'

    sampling_eps = 0.05
    neural_mutate_tree_weight = 0.0

    neural_options=dict(
        active=True,
        model_path=model_path,
        sampling_eps=sampling_eps,
        subtree_min_nodes=3,
        subtree_max_nodes=10,
        device="cuda",
        verbose=True,
    )

    # expr = 'cos(x0)/(x0+2)+x0 + (x1 - sqrt(x1))/exp(x1+3)'
    # dataset = dataset_utils.create_dataset_from_expression(expr, 500, 0)
    eq_idx = 10
    dataset = dataset_utils.load_datasets('feynman', num_samples=2000, noise=0.0001, equation_indices=[eq_idx])[0]
    print(dataset.equation)
    X, y = dataset.X, dataset.y

    is_neural = neural_options["active"] and neural_mutate_tree_weight > 0.0
    log_name = f'250122_eq{eq_idx}_{"neural" if is_neural else "vanilla"}_{sampling_eps:.0e}_test'
    
    logger_spec = TensorBoardLoggerSpec(
        log_dir='logs/run',
        log_interval=10,  # Log every 10 iterations
    )

    custom_loss = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
    end
    """
    early_stopping_condition = 1e-8
    model = PySRRegressor(
        niterations=40,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["cos","exp","sin","zero_sqrt(x) = x >= 0 ? sqrt(x) : zero(x)"],
        extra_sympy_mappings={"zero_sqrt": lambda x: x},  # TODO: Not using Sympy rn. Fix this.
        precision=64,
        neural_options=neural_options,
        weight_neural_mutate_tree=neural_mutate_tree_weight,
        loss_function=custom_loss,
        early_stop_condition=f"f(loss, complexity) = (loss < {early_stopping_condition:e})",
        logger_spec=logger_spec,
    )

    for i in trange(n_runs, desc='Running benchmark'):
        log_dir = os.path.join(log_basedir, f'{log_name}_{i}')
        logger_spec.log_dir = log_dir
        model.fit(X, y)
        # eval_equation(X, y, neural_options, log_dir, max_iter=40, early_stopping_condition=1e-8, weight_neural_mutate_tree=neural_mutate_tree_weight)

run_benchmark(n_runs=5)