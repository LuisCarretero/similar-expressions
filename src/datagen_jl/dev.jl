include("ExpressionGenerator.jl")
using .ExpressionGenerator
using DynamicExpressions: OperatorEnum, string_tree

total_ops = 3
nfeatures = 1
ops = OperatorEnum((+, -, *), (sin, exp))
op_probs = ExpressionGenerator.OperatorProbEnum(ops, [1.0, 1.0, 1.0], [1.0, 1.0])

generator_config = ExpressionGenerator.ExpressionGeneratorConfig(total_ops, Float64, ops, op_probs, nfeatures, 0)

tree = ExpressionGenerator.generate_expr_tree(generator_config)

string_tree(tree, ops)

# Create N trees using list comprehension
N = 10_000  # You can adjust this number as needed
trees = [ExpressionGenerator.generate_expr_tree(generator_config) for _ in 1:N]
