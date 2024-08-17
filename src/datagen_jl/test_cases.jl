import ..ExpressionGenerator: ExpressionGeneratorConfig
import ..Utils: node_to_token_idx
using DynamicExpressions: Node, OperatorEnum
using Distributions: Normal


function _make_childless_op(degree::Int, op_index::Int, ::Type{T})::Node{T} where {T<:Number}
    op_node = Node{T}()
    op_node.degree = degree
    op_node.op = op_index
    return op_node
end

function testnode_to_token_idx(generator_config::ExpressionGeneratorConfig)

    # Test binary operator nodes
    for (i, op) in enumerate(generator_config.ops.binops)
        binary_node = _make_childless_op(2, i, Float64)
        binary_token = node_to_token_idx(binary_node, generator_config)
        println("Binary node ($op): ", binary_token)
    end

    # Test unary operator nodes
    for (i, op) in enumerate(generator_config.ops.unaops)
        unary_node = _make_childless_op(1, i, Float64)
        unary_token = node_to_token_idx(unary_node, generator_config)
        println("Unary node ($op): ", unary_token)
    end

    # Test constant leaf node
    const_node = Node{Float64}(; val=1.5)
    const_token = node_to_token_idx(const_node, generator_config)
    println("Constant node (1.5): ", const_token)

    # Test variable leaf nodes
    for i in 1:generator_config.nfeatures
        var_node = Node{Float64}(; feature=i)
        var_token = node_to_token_idx(var_node, generator_config)
        println("Variable node (x$i): ", var_token)
    end
end

# Run the test
testnode_to_token_idx(generator_config)
