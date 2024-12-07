module DatasetGeneration

using ..ExpressionGenerator: generate_expr_tree
using ..Utils: create_value_transform, eval_trees, encode_trees, filter_evaluated_trees, filter_encoded_trees
using ..DatasetModule: Dataset
using ..Configs: ExpressionGeneratorConfig

using Distributed: pmap, nworkers, workers

"""
Generate a single dataset.
"""
function generate_dataset(config::ExpressionGeneratorConfig, chunk_size::Int)
    # println(generate_expr_tree)

    # This will be called by each worker: Generate trees, evaluate, filter, encode; return dataset
    @assert (chunk_size > 0) "Chunk size must be greater than 0"

    println("Generating trees...")
    trees = [generate_expr_tree(config) for _ in 1:chunk_size]

    println("Evaluating trees...")
    value_transform = create_value_transform(config.value_transform_settings)
    eval_y, eval_y_transformed, success = eval_trees(trees, config.ops, config.eval_x, value_transform)

    println("Filtering trees...")  # NB: This requires transformed values
    trees, eval_y, eval_y_transformed = filter_evaluated_trees(trees, eval_y, eval_y_transformed, success, config.eval_x, config.filter_settings, value_transform)

    if config.save_transformed
        eval_y_to_save = eval_y_transformed
    else
        eval_y_to_save = eval_y
    end

    println("Encoding trees...")
    onehot, consts, success = encode_trees(trees, config)

    println("Filtering encoded trees...")
    onehot, consts, valid = filter_encoded_trees(onehot, consts, success, config.filter_settings)
    trees = trees[valid]
    eval_y_to_save = eval_y_to_save[valid, :]

    return Dataset(config, trees, config.eval_x, eval_y_to_save, onehot, consts)
end

function merge_datasets(datasets::Vector{Dataset{T}}) where T <: Number
    println("Merging datasets...")
    config = datasets[1].generatorConfig

    # Concatenate all components
    trees = vcat([d.trees for d in datasets]...)
    eval_y = vcat([d.eval_y for d in datasets]...)
    onehot = vcat([d.onehot for d in datasets]...)
    consts = vcat([d.consts for d in datasets]...)

    # Filter duplicates from combined dataset 
    success = trues(size(onehot, 1))
    onehot_filtered, consts_filtered, valid = filter_encoded_trees(onehot, consts, success, config.filter_settings)

    # Apply filtering to other components
    trees_filtered = trees[valid]
    eval_y_filtered = eval_y[valid, :]

    return Dataset(config, trees_filtered, config.eval_x, eval_y_filtered, onehot_filtered, consts_filtered)
end

function generate_datasets_parallel(generator_config::ExpressionGeneratorConfig, N::Int)
    # Calculate chunk size and generate trees in parallel
    chunk_size = ceil(Int, N / nworkers())
    println("Using ", nworkers(), " workers to generate ", N, " trees in chunks of ", chunk_size)
    datasets = pmap(w -> generate_dataset(generator_config, chunk_size), workers())
    return datasets
end

end  # End of module DatasetCreation