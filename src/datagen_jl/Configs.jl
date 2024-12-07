module Configs

using DynamicExpressions: OperatorEnum
using Random: AbstractRNG
using Distributions

export FilterSettings, ValueTransformSettings, ExpressionGeneratorConfig, OperatorProbEnum

struct OperatorProbEnum
    binops_probs::Vector{Float64}
    unaops_probs::Vector{Float64}
end

function OperatorProbEnum(ops::OperatorEnum, binops_probs::Vector{Float64}, unaops_probs::Vector{Float64})
    @assert length(binops_probs) == length(ops.binops)
    @assert length(unaops_probs) == length(ops.unaops)

    binops_probs = [binops_probs[i] / sum(binops_probs) for i in 1:length(ops.binops)]
    unaops_probs = [unaops_probs[i] / sum(unaops_probs) for i in 1:length(ops.unaops)]

    return OperatorProbEnum(binops_probs, unaops_probs)
end

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

struct ExpressionGeneratorConfig
    op_cnt_min::Int
    op_cnt_max::Int
    data_type::Type

    ubi_dist::Vector{Vector{Int}}
    rng::AbstractRNG

    ops::OperatorEnum
    op_probs::OperatorProbEnum
    nuna::Int
    nbin::Int

    nfeatures::Int

    const_distr::Distribution

    nl::Int
    p1::Int
    p2::Int

    seq_len::Int
    nb_onehot_cats::Int
    value_transform_settings::ValueTransformSettings
    filter_settings::FilterSettings
    eval_x::Matrix{Float64}

    save_transformed::Bool
end

end  # End of module ConfigModule
