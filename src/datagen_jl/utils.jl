module Utils

import ..ExpressionGenerator: ExpressionGeneratorConfig
using DynamicExpressions: eval_tree_array, OperatorEnum, Node

export eval_trees, encode_trees

function eval_trees(trees::Vector{Node{T}}, ops::OperatorEnum, x::AbstractMatrix{T}) where T <: Number
    # Initialize a matrix to store results for all trees
    res_mat = Matrix{Float64}(undef, length(trees), size(x, 2))
    success = Vector{Bool}(undef, length(trees))

    # Evaluate each tree and store the results
    for (i, tree) in enumerate(trees)
        (res, complete) = eval_tree_array(tree, x, ops)
        good = complete && all(abs.(res) .< 1e5)  # FIXME: Make this customizable.
        success[i] = good
        if good
            res_mat[i, :] = res
        end
    end

    @assert !any(isnan, res_mat)
    @assert !any(isinf, res_mat)
    @assert all(isfinite, res_mat)

    return res_mat, success
end

function _tree_to_prefix(tree::Node{T})::Vector{Node{T}} where T <: Number
    result = [tree]
    if tree.degree >= 1
        append!(result, _tree_to_prefix(tree.l))
    end
    if tree.degree == 2
        append!(result, _tree_to_prefix(tree.r))
    end
    return result
end

function _node_to_token_idx(node::Node{T}, generator_config::ExpressionGeneratorConfig)::Tuple{Int, Float64} where T <: Number
    offset_unaop = generator_config.nbin
    offset_const = offset_unaop + generator_config.nuna
    offset_var = offset_const + 1
    if node.degree == 2
        return (node.op, 0)
    elseif node.degree == 1
        return (offset_unaop + node.op, 0)
    elseif node.degree == 0
        if node.constant
            return (offset_const, node.val)
        else
            return (offset_var + node.feature, 0)
        end
    end
end

function _onehot_encode(idx::Vector{Vector{Tuple{Int, Float64}}}, generator_config::ExpressionGeneratorConfig)
    onehot = falses(length(idx), generator_config.seq_len, generator_config.nb_onehot_cats)
    consts = zeros(Float64, length(idx), generator_config.seq_len)
    success = trues(length(idx))

    for (expr_i, expr_token_list) in enumerate(idx)
        if length(expr_token_list) < generator_config.seq_len
            for (node_i, (token_idx, token_val)) in enumerate(expr_token_list)  # Each node has corresponding index and value
                onehot[expr_i, node_i, token_idx] = true
                consts[expr_i, node_i] = token_val
            end
            # Pad with empty token (last category)
            onehot[expr_i, length(expr_token_list)+1:end, end] .= true 
        else
            success[expr_i] = false
        end
    end
    return onehot, consts, success
end

function encode_trees(trees::Vector{Node{T}}, generator_config::ExpressionGeneratorConfig) where T <: Number

    prefix = [_tree_to_prefix(tree) for tree in trees] # Vector{Vector{Node}}

    idx = [[_node_to_token_idx(node, generator_config) for node in expr_prefix] for expr_prefix in prefix]

    onehot, consts, success = _onehot_encode(idx, generator_config)

    # TODO: Add some validation

    return onehot, consts, success
end

end