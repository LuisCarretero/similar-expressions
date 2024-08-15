include("ExpressionGenerator.jl")
include("Utils.jl")
include("misc.jl")

using .ExpressionGenerator
using .Utils: Dataset
using .Misc: eval_trees
using DynamicExpressions: OperatorEnum, string_tree
using Serialization

total_ops = 3
nfeatures = 1
ops = OperatorEnum((+, -, *), (sin, exp))
op_probs = ExpressionGenerator.OperatorProbEnum(ops, [1.0, 1.0, 1.0], [1.0, 1.0])
N = 100_000  # You can adjust this number as needed


generator_config = ExpressionGenerator.ExpressionGeneratorConfig(total_ops, Float64, ops, op_probs, nfeatures, 0)
trees = [ExpressionGenerator.generate_expr_tree(generator_config) for _ in 1:N]
dataset = Dataset(trees, ops, generator_config)

# output_file = "./data/dataset_240815.jls"
# open(output_file, "w") do io
#     serialize(io, dataset)
# end

# input_file = "dataset.jls"
# loaded_trees = open(input_file, "r") do io
#     deserialize(io)
# end

x = reshape(range(-10, 10, length=100), (1, 100))
res_mat, is_complete = eval_trees(trees, generator_config.ops, x)
trees_new = trees[is_complete]
res_mat_new = res_mat[is_complete, :]
dataset = Dataset(trees_new, res_mat_new, generator_config.ops, generator_config)