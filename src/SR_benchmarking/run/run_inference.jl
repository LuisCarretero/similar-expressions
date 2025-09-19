"""
Dry run script to test the SR inference pipeline.
"""

using Revise
using SymbolicRegression
using DynamicExpressions: parse_expression, Expression
using SymbolicRegression.NeuralMutationsModule: zero_sqrt
using SymbolicRegression.MutationLoggingModule: init_logger, close_global_logger!

# ----- Create options

model_id = "e51hcsb9"
model_path = joinpath(@__DIR__, "../../../onnx-models/model-$model_id.onnx")

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, zero_sqrt],
    populations=40,
    neural_options=NeuralOptions(
        active=true,
        sampling_eps=0.02,
        subtree_min_nodes=8,
        subtree_max_nodes=14,
        model_path=model_path,
        device="cuda",
        verbose=true,
        max_tree_size_diff=7,
        require_tree_size_similarity=true,
        require_novel_skeleton=true,
        require_expr_similarity=true,
        similarity_threshold=0.2,
        max_resamples=127,
        sample_batchsize=32,
        sample_logits=false,
        log_subtree_strings=false,
        subtree_max_features=1
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
        neural_mutate_tree = 1.0
    ),
)
# ----- Create data and run SR

X = (rand(2, 1000) .- 0.5) .* 20
y = X[1, :] .^ 3 .- 2 + 2 * cos.(X[2, :]) + sin.(X[1, :] .* X[2, :]) ./ 3

log_path = joinpath(@__DIR__, "julia_test_logs")
init_logger(log_path)

hall_of_fame = equation_search(
    X, y, niterations=2, options=options,
    parallelism=:multithreading
);

close_global_logger!()