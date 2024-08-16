include("ExpressionGenerator.jl")
include("utils.jl")
include("Dataset.jl")

using .ExpressionGenerator
using .Utils: eval_trees, encode_trees
using .DatasetModule: Dataset
using DynamicExpressions: OperatorEnum, string_tree
using Serialization

# Settings
total_ops = 3
nfeatures = 1
ops = OperatorEnum((+, -, *), (sin, exp))
op_probs = ExpressionGenerator.OperatorProbEnum(ops, [1.0, 1.0, 1.0], [1.0, 1.0])
seq_len = 15  # Max number of nodes in the tree
N = 100_000  # You can adjust this number as needed

# Generate trees
println("Generating trees...")
generator_config = ExpressionGenerator.ExpressionGeneratorConfig(total_ops, Float64, ops, op_probs, nfeatures, seq_len, 0)
trees = [ExpressionGenerator.generate_expr_tree(generator_config) for _ in 1:N]
dataset = Dataset(generator_config, trees)

# Evaluate trees
println("Evaluating trees...")
eval_x = reshape(collect(range(-10, 10, length=100)), (1, 100))
eval_y, success = eval_trees(trees, generator_config.ops, eval_x)
trees = trees[success]
eval_y = eval_y[success, :]
dataset = Dataset(generator_config, trees, eval_x, eval_y)

# Encode trees
println("Encoding trees...")
onehot, consts, success = encode_trees(trees, generator_config)
onehot = onehot[success, :, :]
consts = consts[success, :]
dataset = Dataset(generator_config, trees[success], eval_x, eval_y[success, :], onehot, consts)

# Save dataset
println("Saving dataset...")
output_file = "./data/dataset_240816.jls"
open(output_file, "w") do io
    serialize(io, dataset)
end
