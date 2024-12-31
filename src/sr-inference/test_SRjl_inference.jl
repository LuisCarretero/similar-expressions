using SymbolicRegression
import SymbolicRegression: Options, equation_search
using Revise
using StatsBase
using Plots


X = randn(2, 100)
# y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2
y = tanh.(X[1, :] .^ 2) .- 2 + 2 * cos.(X[2, :])

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, cosh, sinh, tanh],
    populations=40)

hall_of_fame = equation_search(
    X, y, niterations=20, options=options,
    parallelism=:multithreading
);

stats = SymbolicRegression.NeuralMutationsModule.get_mutation_stats()
success_mask = stats.subtree_out_sizes .> 0
in_sizes = stats.subtree_in_sizes[success_mask]
out_sizes = stats.subtree_out_sizes[success_mask]
size_diff = out_sizes - in_sizes;
0


# p1 = histogram(
#     size_diff,
#     bins=minimum(size_diff):1:maximum(size_diff),
#     xlabel="Size Difference (Output - Input)", 
#     ylabel="Count",
#     title="Distribution of Size Changes in Neural Mutations",
#     legend=false
# )

# p2 = histogram(
#     [in_sizes, out_sizes],
#     bins=minimum([in_sizes; out_sizes]):1:maximum([in_sizes; out_sizes]),
#     label=["Input Size" "Output Size"],
#     xlabel="Tree Size",
#     ylabel="Count", 
#     title="Distribution of Input and Output Sizes",
#     alpha=0.5
# )

# plot(p1, p2, layout=(2,1), size=(800,600))

# using StatsBase
# using Plots

# # Create 2D histogram data
# h = fit(Histogram, (in_sizes, out_sizes), nbins=14)
# # Convert to heatmap data
# heatmap(
#     h.edges[1][1:end-1], h.edges[2][1:end-1], h.weights',
#     xlabel="Input Subtree Size",
#     ylabel="Output Subtree Size",
#     title="Distribution of Input vs Output Subtree Sizes",
#     color=:viridis
# )
# # Add correlation coefficient
# cor_val = cor(in_sizes, out_sizes)
# annotate!(
#     minimum(in_sizes),
#     maximum(out_sizes),
#     text("r = $(round(cor_val, digits=3))", :left, :top)
# )
