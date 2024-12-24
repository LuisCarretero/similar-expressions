import SymbolicRegression: Options, equation_search
using Revise

X = randn(2, 100)
y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, cosh, sinh, tanh],
    populations=20 )

hall_of_fame = equation_search(
    X, y, niterations=20, options=options,
    parallelism=:multithreading
);