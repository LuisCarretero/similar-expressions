from pysr import PySRRegressor
import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

model = PySRRegressor(
    # maxsize=20,
    niterations=10,  # < Increase me for better results
    # population_size=10,
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
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    neural_options=dict(
        active=True,
        model_path="/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/dev/ONNX/onnx-models/model-zwrgtnj0.onnx",
        sampling_eps=0.01,
        subtree_min_nodes=1,
        subtree_max_nodes=10,
    ),
    weight_neural_mutate_tree=1.0,
)

model.fit(X, y)


import pysr
stats = pysr.julia_import.SymbolicRegression.NeuralMutationsModule.get_mutation_stats()

def get_mutation_stats():
    raw = pysr.julia_import.SymbolicRegression.NeuralMutationsModule.get_mutation_stats()
    stats = {}
    for k in filter(lambda x: not x.startswith("_"), dir(raw)):
        val = getattr(raw, k)
        if isinstance(val, pysr.julia_import.VectorValue):
            stats[k] = val.to_numpy()
        else:
            stats[k] = val
    return stats
        

get_mutation_stats()