include("ExpressionGenerator.jl")
include("utils.jl")
include("Dataset.jl")

using .ExpressionGenerator
using .Utils: eval_trees, encode_trees, get_onehot_legend
using .DatasetModule: Dataset
using DynamicExpressions: OperatorEnum, string_tree
using Serialization
using HDF5

# Settings
op_cnt_min = 1
op_cnt_max = 5
nfeatures = 1
ops = OperatorEnum((+, -, *, /), (sin, exp))
op_probs = ExpressionGenerator.OperatorProbEnum(ops, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0])
seq_len = 15  # Max number of nodes in the tree
N = 100_000  # You can adjust this number as needed
name = "dataset_240817_2"

# Generate trees
println("Generating trees...")
generator_config = ExpressionGenerator.ExpressionGeneratorConfig(op_cnt_min, op_cnt_max, Float64, ops, op_probs, nfeatures, seq_len, 0)
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
open("./data/$name.jls", "w") do io
    serialize(io, dataset)
end
# Save as HDF5 file to be used in python
h5open("./data/$name.h5", "w") do file
    file["eval_y"] = dataset.eval_y
    file["onehot"] = Array(dataset.onehot)
    file["consts"] = dataset.consts
    file["onehot_legend"] = get_onehot_legend(dataset);
end
