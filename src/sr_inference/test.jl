using SymbolicRegression
using DynamicExpressions: parse_expression, Expression
using Revise

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, cosh, sinh, tanh],
    populations=40,
    neural_options=NeuralOptions(
        active=true,  # If not active, will still be called according to MutationWeights.neural_mutate_tree rate but will return the original tree
        sampling_eps=0.01,
        subtree_min_nodes=5,
        subtree_max_nodes=10,
        model_path="/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/dev/ONNX/onnx-models/model-zwrgtnj0.onnx",
    ),
    mutation_weights=MutationWeights(
        mutate_constant = 0.0353,
        mutate_operator = 3.63,
        swap_operands = 0.00608,
        rotate_tree = 1.42,
        add_node = 0.0771,
        insert_node = 2.44,
        delete_node = 0.369,
        simplify = 0.00148,
        randomize = 0.00695,
        do_nothing = 0.431,
        optimize = 0.0,
        form_connection = 0.5,
        break_connection = 0.1,
        neural_mutate_tree = 0.0
    ),
)

# ex = parse_expression(:((x1*x1 * 3) + cos(x2)*2 +5), operators=options.operators, variable_names=["x1", "x2"])
ex = parse_expression(:(y1 * (((y1 * y1) / (2.189911201366985 / cos((1.2114819663272414 - y4) + -0.20111570724898717))) / exp(-0.08661496242802426 * y5))), operators=options.operators, variable_names=["y1", "y2", "y3", "y4", "y5"])

ex_out = nothing
function mutate_multiple(ex, options, n)
    for i in 1:n
        ex_out = SymbolicRegression.NeuralMutationsModule.neural_mutate_tree(copy(ex), options)
    end
end

mutate_multiple(ex, options, 1000)

stats = SymbolicRegression.NeuralMutationsModule.get_mutation_stats()

# SymbolicRegression.NeuralMutationsModule.reset_mutation_stats!()
