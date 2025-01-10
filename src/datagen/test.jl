include("configs.jl")
include("Dataset.jl")
include("utils.jl")
include("ExpressionGenerator.jl")
include("dataset_generation.jl")

using .Utils: get_onehot_legend, create_value_transform, kill_workers, eval_trees
using .DatasetModule: Dataset
using .Configs: OperatorProbEnum, ExpressionGeneratorConfig, ValueTransformSettings, FilterSettings
using .DatasetGeneration: generate_datasets_parallel, merge_datasets
using .ExpressionGenerator: build_expression_generator_config, generate_expr_tree

using DynamicExpressions: OperatorEnum, string_tree
using Serialization
using HDF5
using Distributed

zero_sqrt(x) = x >= 0 ? sqrt(x) : zero(x)

# Settings
op_cnt_min = 1
op_cnt_max = 7
nfeatures = 1
ops = OperatorEnum((+, -, *, /), (sin, cos, exp, zero_sqrt))
op_probs = OperatorProbEnum(ops, [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
seq_len = 15
save_transformed = true
N = 100  # 20M -> 7.3M
datapath = "/mnt/cephfs/store/gr-mc2473/lc865/workspace/data"
name = "dataset_250110_1"
max_procs = 2  # Number of workers + 1

eval_x = reshape(collect(range(-10, 10, length=100)), (1, 100))
filter_settings = FilterSettings(
    max_abs_value=1e5,  # Used on original values, arcsinh(1e5) ~ 12
    max_1st_deriv=2e2,  # Used on transformed values (everything afterwards)
    max_2nd_deriv=2e2,
    max_3rd_deriv=2e2,
    max_4th_deriv=2e2,
    min_range=1e-11,  # spacing of float64 for O(1) is 1e-16
    filter_unique_skeletons=false,
    filter_unique_expressions=true,
    unique_expression_const_tol=3,  # digits of precision for considering two expressions to be the same
)
value_transform_settings = ValueTransformSettings(
    mapping="arcsinh",
    bias=nothing,
    scale=nothing
)

config = build_expression_generator_config(op_cnt_min, op_cnt_max, Float64, ops, op_probs, nfeatures, seq_len, 0, value_transform_settings, filter_settings, eval_x, save_transformed)

# Actual generation

trees = [generate_expr_tree(config) for _ in 1:N]

value_transform = create_value_transform(config.value_transform_settings)
eval_y, eval_y_transformed, success = eval_trees(trees, config.ops, config.eval_x, value_transform)