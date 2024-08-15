module Utils

import ..ExpressionGenerator: ExpressionGeneratorConfig
using DynamicExpressions: Node, OperatorEnum

export Dataset

struct Dataset{T<:Number}
    trees::Vector{Node{T}}
    values::Matrix{T}
    ops::OperatorEnum
    
    generatorConfig::ExpressionGeneratorConfig
end

function Dataset(trees::Vector{Node{T}}, ops::OperatorEnum, generatorConfig::ExpressionGeneratorConfig) where T <: Number
    # Create a Dataset without pre-computed values
    return Dataset{T}(trees, Matrix{T}(undef, 0, 0), ops, generatorConfig)
end

function Dataset(trees::Vector{Node{T}}, values::Matrix{T}, ops::OperatorEnum, generatorConfig::ExpressionGeneratorConfig) where T <: Number
    # Create a Dataset with pre-computed values
    return Dataset{T}(trees, values, ops, generatorConfig)
end


end