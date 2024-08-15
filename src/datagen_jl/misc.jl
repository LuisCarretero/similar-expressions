module Misc

import ..ExpressionGenerator: ExpressionGeneratorConfig
using DynamicExpressions: eval_tree_array, AbstractExpressionNode, OperatorEnum


export eval_trees

function eval_trees(trees::Vector{<:AbstractExpressionNode}, ops::OperatorEnum, x::AbstractMatrix{<:Number})
    # Initialize a matrix to store results for all trees
    res_mat = Matrix{Float64}(undef, length(trees), size(x, 2))
    is_complete = Vector{Bool}(undef, length(trees))

    # Evaluate each tree and store the results
    for (i, tree) in enumerate(trees)
        (res, complete) = eval_tree_array(tree, x, ops)
        is_complete[i] = complete
        if complete
            res_mat[i, :] = res
        end
    end

    @assert !any(isnan, res_mat)
    @assert !any(isinf, res_mat)
    @assert all(isfinite, res_mat)

    return res_mat, is_complete
end

end