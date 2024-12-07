using Statistics: std, mean

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

function create_value_transform(settings::ValueTransformSettings)::Function
    # Define mapping functions
    mapping = if settings.mapping == "arcsinh"
        x -> asinh.(x)
    else
        x -> x
    end

    # Pre-compute mapped values if needed for dataset stats
    # values_mapped = if settings.bias == "dataset" || settings.scale in ["dataset-std", "dataset-range"]
    #     mapping(values)
    # else
    #     nothing
    # end

    # Define bias function
    # bias = if settings.bias == "dataset"
    #     _ -> mean(values_mapped)
    bias = if settings.bias == "sample"
        x -> reshape(mean(x, dims=2), :, 1)
    else
        _ -> zero(Float32)
    end

    # Helper to replace values
    function replace_with(values_mapped, value, replacement)
        values_mapped[values_mapped .== value] .= replacement
        values_mapped
    end

    # Define scale function 
    # scale = if settings.scale == "dataset-std"
    #     _ -> std(values_mapped)
    # elseif settings.scale == "dataset-range"
    #     _ -> maximum(values_mapped) - minimum(values_mapped)
    scale = if settings.scale == "sample-std"
        x -> reshape(replace_with(std(x, dims=2), 0, 1), :, 1)
    elseif settings.scale == "sample-range"
        x -> reshape(replace_with(maximum(x, dims=2) - minimum(x, dims=2), 0, 1), :, 1)
    else
        _ -> one(Float32)
    end

    # Return combined transform function
    function combined_transform(x::Matrix{Float32})::Matrix{Float32}
        x_mapped = mapping(x)
        x_scaled = x_mapped ./ scale(x_mapped)
        return x_scaled .- bias(x_scaled)
    end

    return combined_transform
end

value_transform_settings = ValueTransformSettings(mapping="arcsinh", bias="sample", scale="sample-range")
value_transform = create_value_transform(value_transform_settings)

x = rand(Float32, 1, 10)
value_transform(x)


probs = [0.1, 0.2, 0.7]
rand(MersenneTwister(1), Categorical(probs))