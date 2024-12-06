module CreateDataset

include("ExpressionGenerator.jl")
include("Dataset.jl")
include("utils.jl")
include("Configs.jl")

using .ExpressionGenerator
using .Utils: get_onehot_legend, FilterSettings, generate_dataset, merge_datasets, create_value_transform, ValueTransformSettings, generate_datasets_parallel
using .DatasetModule: Dataset

using DynamicExpressions: OperatorEnum, string_tree
using Serialization
using HDF5
using Distributed


# Settings
op_cnt_min = 1
op_cnt_max = 7
nfeatures = 1
ops = OperatorEnum((+, -, *, /), (sin, cos, exp, tanh, cosh, sinh))
op_probs = ExpressionGenerator.OperatorProbEnum(ops, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
seq_len = 15
save_transformed = true
N = 1_000
name = "dataset_241204_2"
max_workers = 2

eval_x = reshape(collect(range(-10, 10, length=100)), (1, 100))
filter_settings = FilterSettings(
    max_abs_value=1e5,  # Used on original values, arcsinh(1e5) ~ 12
    max_1st_deriv=1e2,  # Used on transformed values (everything afterwards)
    max_2nd_deriv=1e2,
    max_3rd_deriv=1e2,
    max_4th_deriv=1e2,
    min_range=1e-10,  # spacing of float64 for O(1) is 1e-16
    filter_unique_skeletons=false,
    filter_unique_expressions=true,
    unique_expression_const_tol=3,  # digits of precision for considering two expressions to be the same
)
value_transform_settings = ValueTransformSettings(
    mapping="arcsinh",
    bias="sample",
    scale="sample-range"
)

# Setup
num_workers = min(Sys.CPU_THREADS, max_workers)
if nprocs() < num_workers
    addprocs(num_workers - nprocs())
end

# Load necessary modules on all workers
@everywhere workers() begin  # Exclude main thread
    include("ExpressionGenerator.jl")
    include("Dataset.jl")
    include("utils.jl")

    using .ExpressionGenerator
    using .Utils: eval_trees, encode_trees, get_onehot_legend, FilterSettings, filter_evaluated_trees, filter_encoded_trees, create_value_transform, generate_dataset
    using .DatasetModule: Dataset
    using DynamicExpressions: OperatorEnum, string_tree
end


generator_config = ExpressionGenerator.ExpressionGeneratorConfig(op_cnt_min, op_cnt_max, Float64, ops, op_probs, nfeatures, seq_len, 0, value_transform_settings, eval_x, save_transformed)

# Generate datasets by workers and merge
datasets = generate_datasets_parallel(generator_config, N)
dataset = merge_datasets(datasets)

# Save dataset
# println("Saving dataset...")
# open("./data/$name.jls", "w") do io
#     serialize(io, dataset)
# end

# # Save as HDF5 file to be used in python
# h5open("./data/$name.h5", "w") do file
#     file["eval_x"] = dataset.eval_x
#     file["eval_y"] = dataset.eval_y
#     file["onehot"] = Array(dataset.onehot)
#     file["consts"] = dataset.consts
#     file["onehot_legend"] = get_onehot_legend(dataset);
# end

# Kill workers
kill_workers()

end  # End of module CreateDataset