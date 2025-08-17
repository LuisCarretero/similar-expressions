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
options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, zero_sqrt],
    populations=40,
    neural_options=NeuralOptions(
        active=true,  # If not active, will still be called according to MutationWeights.neural_mutate_tree rate but will return the original tree
        sampling_eps=0.02,
        subtree_min_nodes=8,
        subtree_max_nodes=14,
        model_path="/cephfs/home/lc865/workspace/similar-expressions/onnx-models/model-$model_id.onnx",
        # model_path="/home/lc865/workspace/similar-expressions/src/dev/ONNX/onnx-models/model-$model_id.onnx",
        device="cuda",  # "cuda"
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
# y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2
y = X[1, :] .^ 3 .- 2 + 2 * cos.(X[2, :]) + sin.(X[1, :] .* X[2, :]) ./ 3

# I.6.2b,3,f,exp(-((theta-theta1)/sigma)**2/2)/(sqrt(2*pi)*sigma),3,sigma,1,3,theta,1,3,theta1,1,3
# X = rand(3, 1000) .* 10
# X[1:2, :] .= (X[1:2, :] .- 5)
# y = 1 ./ (sqrt(2 * pi) .* X[3, :]) .* exp.(-((X[1, :] .- X[2, :]) ./ X[3, :]) .^ 2 / 2)

init_logger("/cephfs/home/lc865/workspace/similar-expressions/src/SR_benchmarking/run/julia_test_logs")

hall_of_fame = equation_search(
    X, y, niterations=2, options=options,
    parallelism=:multithreading
);

close_global_logger!()