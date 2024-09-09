module Utils

import ..ExpressionGenerator: ExpressionGeneratorConfig
# import ..DatasetModule: Dataset
using DynamicExpressions: eval_tree_array, OperatorEnum, Node

export eval_trees, encode_trees, node_to_token_idx



struct FilterSettings
    max_abs_value::Float64  # -1 if not used
    max_1st_deriv::Float64  # -1 if not used
    filter_unique_skeletons::Bool
end

function filter_evaluated_trees(trees::Vector{Node{T}}, eval_y::AbstractMatrix{T}, success::Vector{Bool}, eval_x::AbstractMatrix{T}, settings::FilterSettings) where T <: Number
    # TODO: Use dataset?
    valid = success
    
    # Checks
    if settings.max_abs_value != -1
        valid = valid .& all((abs.(eval_y) .< settings.max_abs_value), dims=2)
    end
    if settings.max_1st_deriv != -1
        valid = valid .& all((abs.(first_deriv(eval_x, eval_y)) .< settings.max_1st_deriv), dims=2)
    end

    # Filter
    valid = vec(valid)
    trees = trees[valid]
    eval_y = eval_y[valid, :]

    @assert !any(isnan, eval_y)
    @assert !any(isinf, eval_y)
    @assert all(isfinite, eval_y)

    return trees, eval_y
end

function first_deriv(x::AbstractMatrix{T}, y::AbstractMatrix{T}) where T <: Number
    return diff(y, dims=2) ./ diff(x, dims=2)
end

function eval_trees(trees::Vector{Node{T}}, ops::OperatorEnum, x::AbstractMatrix{T}) where T <: Number
    # Initialize a matrix to store results for all trees
    res_mat = Matrix{Float64}(undef, length(trees), size(x, 2))
    success = Vector{Bool}(undef, length(trees))

    # Evaluate each tree and store the results
    for (i, tree) in enumerate(trees)
        (res, complete) = eval_tree_array(tree, x, ops)
        good = complete && all((res .< prevfloat(typemax(Float64))) .& (res .> nextfloat(typemin(Float64)))) && !any(isnan, res) && !any(isinf, res)
        success[i] = good
        if good
            res_mat[i, :] = res
        end
    end
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

function node_to_token_idx(node::Node{T}, generator_config::ExpressionGeneratorConfig)::Tuple{Int, Float64} where T <: Number
    offset_unaop = generator_config.nbin
    offset_const = offset_unaop + generator_config.nuna
    offset_var = offset_const + 1
    if node.degree == 2
        return (node.op, 0)
    elseif node.degree == 1
        return (offset_unaop + node.op, 0)
    elseif node.degree == 0
        if node.constant
            return (offset_const + 1, node.val)  # Only 1 const token
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

    idx = [[node_to_token_idx(node, generator_config) for node in expr_prefix] for expr_prefix in prefix]

    onehot, consts, success = _onehot_encode(idx, generator_config)

    # TODO: Add some validation

    return onehot, consts, success
end

function filter_encoded_trees(onehot::BitArray{3}, consts::AbstractMatrix{Float64}, success::AbstractVector{Bool}, settings::FilterSettings)
    # TODO: Use dataset?
    valid = success
    # Checks
    if settings.filter_unique_skeletons
        # Check if there are duplicates in the onehot matrix
        already_seen = Set{Vector{Bool}}()
        for i in axes(onehot, 1)
            row = vec(onehot[i, :, :])
            if row ∈ already_seen
                valid[i] = false
            else
                push!(already_seen, row)
            end
        end
    end

    # Filter
    onehot = onehot[valid, :, :]
    consts = consts[valid, :]

    return onehot, consts, valid
end

function get_onehot_legend(dataset)::Vector{String}  # ::DatasetModule.Dataset
    ops = (dataset.ops.binops..., dataset.ops.unaops...)

    # Map ops to their string versions using the provided mapping
    op_map = Dict(
        typeof(+) => "ADD",
        typeof(-) => "SUB",
        typeof(*) => "MUL",
        typeof(/) => "DIV",
        typeof(sin) => "SIN",
        typeof(exp) => "EXP"
    )

    # Create a new array with the string versions in the same order as ops
    mapped_ops = [op_map[typeof(op)] for op in ops]
    push!(mapped_ops, "CON", "x1", "END")  # FIXME: Add multiple var

    return mapped_ops
end

end