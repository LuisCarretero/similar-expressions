module ConfigModule

export FilterSettings, ValueTransformSettings, ExpressionGeneratorConfig

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

mutable struct ExpressionGeneratorConfig
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
    eval_x::Matrix{Float64}

    save_transformed::Bool
end

function ExpressionGeneratorConfig(op_cnt_min::Int, op_cnt_max::Int, data_type::Type, ops::OperatorEnum, op_probs::OperatorProbEnum, nfeatures::Int, seq_len::Int, seed::Int=0, value_transform_settings::ValueTransformSettings=ValueTransformSettings(), eval_x::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), save_transformed::Bool=true)
    @assert op_cnt_min <= op_cnt_max
    @assert op_cnt_min > 0

    rng = Random.MersenneTwister(seed)

    nuna = length(ops.unaops)
    nbin = length(ops.binops)

    nl = 1 # FIXME: Adjust these values?
    p1 = 1
    p2 = 1

    ubi_dist = _generate_ubi_dist(op_cnt_max, nl, p1, p2)
    nb_onehot_cats = nbin + nuna + nfeatures + 2  # +1 for constants, +1 for end token
    const_distr = truncated(Normal(), -5, 5)  # mean=0, std=1, min=-5, max=5 TODO: Make this customizable.
    ExpressionGeneratorConfig(op_cnt_min, op_cnt_max, data_type, ubi_dist, rng, ops, op_probs, nuna, nbin, nfeatures, const_distr,nl, p1, p2, seq_len, nb_onehot_cats, value_transform_settings, eval_x, save_transformed)
end

end  # End of module ConfigModule
