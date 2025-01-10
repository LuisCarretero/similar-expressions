from pysr import PySRRegressor
from pysr_interface_utils import get_mutation_stats
import numpy as np
import sys
sys.path.append("/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/sr_inference_benchmarking")
from importlib import reload
import dataset_utils
reload(dataset_utils)

def eval_equation(X, y, n_iterations=10, early_stopping_condition=1e-8):
    custom_loss = """
    function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        return sum( (1000 .* (prediction .- dataset.y) ) .^ 2) / dataset.n
    end
    """

    model = PySRRegressor(
        niterations=n_iterations,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "cos",
            "exp",
            "sin",
            "tanh",
            "cosh",
            "sinh"
        ],
        precision=64,
        neural_options=dict(
            active=True,
            model_path="/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/dev/ONNX/onnx-models/model-zwrgtnj0.onnx",
            sampling_eps=0.01,
            subtree_min_nodes=1,
            subtree_max_nodes=10,
        ),
        weight_neural_mutate_tree=1.0,
        # elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        loss_function=custom_loss,
        early_stop_condition=f"f(loss, complexity) = (loss < {early_stopping_condition:e})"
    )
    model.fit(X, y)
    return model


dataset = dataset_utils.load_datasets('synthetic', num_samples=2000, noise=0, equation_indices=[9])[0]
dataset.equation

model = eval_equation(dataset.X, dataset.Y)




stats = get_mutation_stats()
in_sizes, out_sizes = stats['subtree_in_sizes'], stats['subtree_out_sizes']



# X = 2 * np.random.randn(100, 5)
# y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

########################################################
sys.path.append("/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/sr_inference_benchmarking")
import dataset_utils
from importlib import reload
reload(dataset_utils)
from dataset_utils import get_synthetic_equations, load_feynman_equations, load_datasets

equations = get_synthetic_equations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

for idx, (equation, bounds) in equations:
    print(equation)
    print(bounds)
    print()

a = load_feynman_equations('/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/sr_inference_benchmarking/data/FeynmanEquations.csv')
a[0]

a = load_datasets('feynman', 100, 1e-4, equation_indices=set(range(0, 100)))
a[0]