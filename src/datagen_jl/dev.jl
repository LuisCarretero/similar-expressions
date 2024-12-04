include("ExpressionGenerator.jl")
include("utils.jl")

using .Utils: eval_trees, encode_trees, get_onehot_legend, FilterSettings, filter_evaluated_trees, filter_encoded_trees, ValueTransformSettings

value_transform_settings = ValueTransformSettings(
    mapping="arcsinh",
    bias="sample",
    scale="sample-range"
)

values = rand(1000, 100)

create_value_transform(value_transform_settings, values)