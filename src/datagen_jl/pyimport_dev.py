import numpy as np
from julia.api import Julia
Julia(compiled_modules=False)
from julia import Main


def load_jl_dataset(datapath: str, name: str):

    fpath = f"{datapath}/{name}.jls"

    Main.eval(f"""
        include("ExpressionGenerator.jl")
        include("utils.jl")
        include("Dataset.jl")
            
        using Serialization
        using .DatasetModule: Dataset
        
        # Load the dataset
        dataset = deserialize("{fpath}")
        
        # # Extract relevant information
        eval_y = dataset.eval_y
        onehot = dataset.onehot
        consts = dataset.consts
    """)

    # Convert Julia arrays to NumPy arrays
    eval_y = np.array(Main.eval_y).astype(np.float32)
    onehot = np.array(Main.onehot).astype(np.float32)
    consts = np.array(Main.consts).astype(np.float32)

    syntax_data = np.concatenate([onehot, consts[:, :, np.newaxis]], axis=-1)
    value_data = eval_y

    return syntax_data, value_data

datapath = "/Users/luis/Desktop/Cranmer 2024/Workplace/smallMutations/similar-expressions/data"
name = "dataset_240816"
syntax_data, value_data = load_jl_dataset(datapath, name)