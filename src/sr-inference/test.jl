using SymbolicRegression
using DynamicExpressions: parse_expression, Expression
using Revise

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, cosh, sinh, tanh],
    populations=40)


ex = parse_expression(:((x1*x1 * 3) + cos(x2)*2 +5), operators=options.operators, variable_names=["x1", "x2"])


function mutate_multiple(ex, options, n)
    for i in 1:n
        ex_out = SymbolicRegression.NeuralMutationsModule.neural_mutate_tree(copy(ex), options)
    end
end

@profview mutate_multiple(ex, options, 10000)

stats = SymbolicRegression.NeuralMutationsModule.get_mutation_stats()

# SymbolicRegression.NeuralMutationsModule.reset_mutation_stats!()