module DatasetModule

using DynamicExpressions: Node, OperatorEnum

using ..Configs: ExpressionGeneratorConfig

export Dataset

struct Dataset{T<:Number}
    generatorConfig::ExpressionGeneratorConfig
    len::Int

    trees::Vector{Node{T}}
    ops::OperatorEnum

    eval_x::Matrix{Float64}
    eval_y::Matrix{Float32}

    onehot::BitArray
    consts::Matrix{Float64}
end

function Dataset(generatorConfig::ExpressionGeneratorConfig, trees::Vector{Node{T}}) where T <: Number
    # Create a Dataset with only trees (and their ops, and generatorConfig)
    return Dataset{T}(generatorConfig, length(trees), trees, generatorConfig.ops, Matrix{Float64}(undef, 0, 0), Matrix{Float32}(undef, 0, 0), Matrix{Bool}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
end

function Dataset(generatorConfig::ExpressionGeneratorConfig, trees::Vector{Node{T}}, eval_x::Matrix{Float64}, eval_y::Matrix{Float32}) where T <: Number
    @assert length(trees) == size(eval_y, 1)

    # Create a Dataset with trees and evaluated values
    return Dataset{T}(generatorConfig, length(trees), trees, generatorConfig.ops, eval_x, eval_y, Matrix{Bool}(undef, 0, 0), Matrix{Float64}(undef, 0, 0))
end

function Dataset(generatorConfig::ExpressionGeneratorConfig, trees::Vector{Node{T}}, eval_x::Matrix{Float64}, eval_y::Matrix{Float32}, onehot::BitArray, consts::Matrix{Float64}) where T <: Number
    @assert length(trees) == size(onehot, 1) == size(eval_y, 1)
    @assert size(onehot) == (length(trees), generatorConfig.seq_len, generatorConfig.nb_onehot_cats)
    @assert size(consts) == (length(trees), generatorConfig.seq_len)

    # Create a Dataset with trees, evaluated values, and onehot encoding
    return Dataset{T}(generatorConfig, length(trees), trees, generatorConfig.ops, eval_x, eval_y, onehot, consts)
end

end