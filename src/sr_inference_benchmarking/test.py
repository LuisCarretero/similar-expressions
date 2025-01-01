from pysr import PySRRegressor
import numpy as np

X = 2 * np.random.randn(100, 5)
y = 2.5382 * np.cos(X[:, 3]) + X[:, 0] ** 2 - 0.5

model = PySRRegressor(
    maxsize=20,
    niterations=40,  # < Increase me for better results
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
        subtree_min_nodes=5,
        subtree_max_nodes=10,
    ),
    weight_neural_mutate_tree=1.0,
)

model.fit(X, y)