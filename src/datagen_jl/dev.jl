include("./ExpressionGenerator.jl")
include("./utils.jl")
include("./Dataset.jl")
    
using Serialization
using .DatasetModule: Dataset

# Load the dataset
dataset = deserialize("/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data/dataset_240816_2.jls")

# # Extract relevant information
using HDF5

# Extract relevant information
eval_y = dataset.eval_y
onehot = Array(dataset.onehot)
consts = dataset.consts

# Save as HDF5 file
h5open("/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data/dataset_240816_2.h5", "w") do file
    file["eval_y"] = eval_y
    file["onehot"] = onehot
    file["consts"] = consts
end

println("Dataset saved as HDF5 file.")