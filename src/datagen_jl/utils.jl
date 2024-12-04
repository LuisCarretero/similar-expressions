module Utils

using ..ExpressionGenerator: ExpressionGeneratorConfig
using DynamicExpressions: eval_tree_array, OperatorEnum, Node
using Statistics: mean, std 

export eval_trees, encode_trees, node_to_token_idx, create_value_transform, FilterSettings, ValueTransformSettings

struct FilterSettings
    max_abs_value::Float64
    max_1st_deriv::Float64
    max_2nd_deriv::Float64
    max_3rd_deriv::Float64
    max_4th_deriv::Float64
    min_range::Float64
    filter_unique_skeletons::Bool
    filter_unique_expressions::Bool
    unique_expression_const_tol::Int  # digits of precision for considering two expressions as the same

    # Constructor with keyword arguments
    FilterSettings(;
        max_abs_value::Float64 = 1e2,
        max_1st_deriv::Float64 = 1e2,
        max_2nd_deriv::Float64 = 1e2,
        max_3rd_deriv::Float64 = 1e2,
        max_4th_deriv::Float64 = 1e2,
        min_range::Float64 = 1e-3,  # Minimum range: If the range is smaller we will reach
        filter_unique_skeletons::Bool = true,
        filter_unique_expressions::Bool = true,
        unique_expression_const_tol::Int = 3
    ) = new(max_abs_value, max_1st_deriv, max_2nd_deriv, max_3rd_deriv, max_4th_deriv, min_range, filter_unique_skeletons, filter_unique_expressions, unique_expression_const_tol)
end

struct ValueTransformSettings
    mapping::Union{String, Nothing}
    bias::Union{String, Nothing} 
    scale::Union{String, Nothing}

    # Constructor with keyword arguments
    ValueTransformSettings(;
        mapping::Union{String, Nothing} = nothing,
        bias::Union{String, Nothing} = nothing,
        scale::Union{String, Nothing} = nothing
    ) = new(mapping, bias, scale)
end

function create_value_transform(settings::ValueTransformSettings, values::AbstractMatrix{T}) where T <: Number
    # Define mapping functions
    mapping = if settings.mapping == "arcsinh"
        x -> asinh.(x)
    else
        x -> x
    end

    # Pre-compute mapped values if needed for dataset stats
    values_mapped = if settings.bias == "dataset" || settings.scale in ["dataset-std", "dataset-range"]
        mapping(values)
    else
        nothing
    end

    # Define bias function
    bias = if settings.bias == "dataset"
        _ -> mean(values_mapped)
    elseif settings.bias == "sample"
        x -> reshape(mean(x, dims=2), :, 1)
    else
        _ -> zero(T)
    end

    # Helper to replace values
    function replace_with(values_mapped, value, replacement)
        values_mapped[values_mapped .== value] .= replacement
        values_mapped
    end

    # Define scale function 
    scale = if settings.scale == "dataset-std"
        _ -> std(values_mapped)
    elseif settings.scale == "sample-std"
        x -> reshape(replace_with(std(x, dims=2), 0, 1), :, 1)
    elseif settings.scale == "dataset-range"
        _ -> maximum(values_mapped) - minimum(values_mapped)
    elseif settings.scale == "sample-range"
        x -> reshape(replace_with(maximum(x, dims=2) - minimum(x, dims=2), 0, 1), :, 1)
    else
        _ -> one(T)
    end

    # Return combined transform function
    function combined_transform(x)
        x_mapped = mapping(x)
        x_scaled = x_mapped ./ scale(x_mapped)
        return x_scaled .- bias(x_scaled)
    end

    return combined_transform
end




function filter_evaluated_trees(trees::Vector{Node{T}}, eval_y::AbstractMatrix{T}, success::Vector{Bool}, eval_x::AbstractMatrix{T}, settings::FilterSettings, transform_settings::ValueTransformSettings) where T <: Number
    # TODO: Use dataset?
    println("Checking ", length(success), " expressions.")
    println("Number of valid expressions after evaluation: ", sum(success) / length(success))
    valid = success

    # Absolute value check (original values)
    valid = valid .& all((abs.(eval_y) .< settings.max_abs_value), dims=2)  # FIXME: Use range for consistency?
    println("Number of valid expressions after abs value check: ", sum(valid) / length(valid))

    # Checks (using transformed values)  # FIXME: Allow to provide transformation to be applied?
    value_transform = create_value_transform(transform_settings, eval_y)
    eval_y_transformed = value_transform(eval_y)

    deriv_1 = first_deriv(eval_x, eval_y_transformed)
    deriv_2 = first_deriv(eval_x[:, 2:end], deriv_1)
    deriv_3 = first_deriv(eval_x[:, 3:end], deriv_2)
    deriv_4 = first_deriv(eval_x[:, 4:end], deriv_3)

    range = maximum(eval_y_transformed, dims=2) - minimum(eval_y_transformed, dims=2)
    valid = valid .& all((range .> settings.min_range) .| (range .== 0), dims=2)
    println("Number of valid expressions after range check: ", sum(valid) / length(valid))

    valid = valid .& all((abs.(deriv_1) .< settings.max_1st_deriv), dims=2)
    println("Number of valid expressions after 1st deriv check: ", sum(valid) / length(valid))
    valid = valid .& all((abs.(deriv_2) .< settings.max_2nd_deriv), dims=2)
    println("Number of valid expressions after 2nd deriv check: ", sum(valid) / length(valid))
    valid = valid .& all((abs.(deriv_3) .< settings.max_3rd_deriv), dims=2)
    println("Number of valid expressions after 3rd deriv check: ", sum(valid) / length(valid))
    valid = valid .& all((abs.(deriv_4) .< settings.max_4th_deriv), dims=2)
    println("Number of valid expressions after 4th deriv check: ", sum(valid) / length(valid))

    # Filter
    valid = vec(valid)
    trees = trees[valid]
    eval_y = eval_y[valid, :]
    eval_y_transformed = eval_y_transformed[valid, :]

    @assert !any(isnan, eval_y)
    @assert !any(isinf, eval_y)
    @assert all(isfinite, eval_y)

    return trees, eval_y, eval_y_transformed
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
    println("Checking ", length(valid), " expressions.")
    # Checks
    if settings.filter_unique_skeletons
        check_unique_skeletons!(onehot, valid)
        println("Number of valid expressions after skeleton check: ", sum(valid) / length(valid))
    end
    if settings.filter_unique_expressions
        check_unique_expressions!(onehot, consts, valid, settings)
        println("Number of valid expressions after expression check: ", sum(valid) / length(valid))
    end

    # Filter
    onehot = onehot[valid, :, :]
    consts = consts[valid, :]

    return onehot, consts, valid
end

function check_unique_skeletons!(onehot::BitArray{3}, valid::AbstractVector{Bool})
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

function check_unique_expressions!(onehot::BitArray{3}, consts::AbstractMatrix{Float64}, valid::AbstractVector{Bool}, settings::FilterSettings)
    
    consts = round.(consts, digits=settings.unique_expression_const_tol)
    already_seen = Set{Tuple{Vector{Bool}, Vector{Float64}}}()
    for i in axes(onehot, 1)
        row = (vec(onehot[i, :, :]), consts[i, :])
        if row ∈ already_seen
            valid[i] = false
        else
            push!(already_seen, row)
        end
    end
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
        typeof(cos) => "COS",
        typeof(log) => "LOG",
        typeof(log2) => "LOG2",
        typeof(log10) => "LOG10",
        typeof(sqrt) => "SQRT",
        typeof(exp) => "EXP",
        typeof(tanh) => "TANH",
        typeof(cosh) => "COSH",
        typeof(sinh) => "SINH",
    )

    # Create a new array with the string versions in the same order as ops
    mapped_ops = [op_map[typeof(op)] for op in ops]
    push!(mapped_ops, "CON", "x1", "END")  # FIXME: Add multiple var

    return mapped_ops
end

end