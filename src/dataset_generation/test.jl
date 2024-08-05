# using DynamicExpressions

# max_node_cnt = 5
# seq_len = 2 * max_node_cnt + 1  # Worst case: linear binary tree with each node having two children (const + binary) and last binary has two const nodes
# operators = OperatorEnum(binary_operators=(+, -, *), unary_operators=(sin, exp))
# variable_names = [:x1]


# function test(str::String, operators::OperatorEnum, variable_names::Vector{Symbol})
#     expr = Meta.parse(str)
#     ex = parse_expression(expr, operators=operators, variable_names=variable_names, node_type=GraphNode)

#     # Parse tree into normal Polish (prefix) notation of fixed length (incl padding)
#     function tree_to_seq(node::GraphNode, operators::OperatorEnum)
#         if node.constant
#             # @info "constant"
#             return [string(node.val)]
#         elseif node.degree == 0  # Feature
#             # @info "feature"
#             return ["x" * string(node.feature)]
#         elseif node.degree == 1  # Unary op
#             # @info "unary"
#             return vcat([string(operators.unaops[node.op])], tree_to_seq(node.l, operators))
#         elseif node.degree == 2  # Binary op
#             # @info "binary"
#             return vcat([string(operators.binops[node.op])], tree_to_seq(node.l, operators), tree_to_seq(node.r, operators))
#         end
#     end
#     seq = tree_to_seq(ex.tree, operators)
#     println(seq)
# end

# expression_str = "(x1 - 0.125192826477114) + 0.43174347075339936"

# # for _ in 1:10
# #     test(expression_str, operators, variable_names)
# # end

# function tree_to_seq(node::GraphNode, operators::OperatorEnum)
#     if node.constant
#         # @info "constant"
#         return [string(node.val)]
#     elseif node.degree == 0  # Feature
#         # @info "feature"
#         return ["x" * string(node.feature)]
#     elseif node.degree == 1  # Unary op
#         # @info "unary"
#         return vcat([string(operators.unaops[node.op])], tree_to_seq(node.l, operators))
#     elseif node.degree == 2  # Binary op
#         # @info "binary"
#         return vcat([string(operators.binops[node.op])], tree_to_seq(node.l, operators), tree_to_seq(node.r, operators))
#     end
# end

# expr = Meta.parse(expression_str)
# ex = parse_expression(expr, operators=operators, variable_names=variable_names, node_type=GraphNode)

# for _ in 1:10
#     println(ex.tree)
#     println(ex.tree.constant)
#     seq = tree_to_seq(ex.tree, operators)
#     println(seq)
# end

a = rand(Float64, 100_000)
println(maximum(a), "   ", minimum(a))