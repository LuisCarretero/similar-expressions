using SymbolicRegression: AbstractMutationWeights

# Define custom mutation weights with default values
Base.@kwdef struct MyMutationWeights <: AbstractMutationWeights
    neural_mutate_tree::Float64 = 0.7
end

# Define the list of mutation names (symbols)
const MY_MUTATIONS = [
    :neural_mutate_tree
]

# Import the `sample_mutation` function to overload it
import SymbolicRegression: sample_mutation
using StatsBase: StatsBase

# Overload the `sample_mutation` function
function sample_mutation(w::MyMutationWeights)
    weights = [
        w.neural_mutate_tree,
    ]
    weights = weights ./ sum(weights)  # Normalize weights to sum to 1.0
    return StatsBase.sample(MY_MUTATIONS, StatsBase.Weights(weights))
end

# Pass it when defining `Options`:
using SymbolicRegression: Options
options = Options()

import SymbolicRegression: Options, equation_search

X = randn(2, 100)
y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[cos, exp],
    populations=20,
    mutation_weights=MyMutationWeights()
)

hall_of_fame = equation_search(
    X, y, niterations=40, options=options,
    parallelism=:multithreading
)
