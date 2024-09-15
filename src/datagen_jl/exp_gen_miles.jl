using Comonicon: @main
using SymbolicRegression: SymbolicRegression as SR
using DynamicExpressions: DynamicExpressions as DE
using Zygote: Zygote
using Random: MersenneTwister, seed!
using Statistics: mean, std
using Serialization: serialize, deserialize
using Plots: Plots
using MLJBase: MLJBase as MLJ
using LoopVectorization: LoopVectorization

"""
This treats the binary tree like an iterator, and checks if
there are any cases where 3 or more unary operators are nested.
"""
function has_overly_nested_unary(eq)
    nestedness = DE.tree_mapreduce(
        leaf -> 0,
        branch -> branch,
        (parent, children...) -> Int(parent.degree == 1) + max(children...),
        eq,
    )
    return nestedness >= 3
end

"""
This checks if an equation has a good range of values,
or whether there are anomalous points.
"""
function has_good_range(sample_y)
    return any(isnan, sample_y) || maximum(abs, sample_y) > 1e4
end

"""
This checks if an equation has a good range of derivative values,
or whether there are anomalous points.
"""
function has_good_derivative_range(eq, sample_X, options)
    # Also check derivative with respect to input:
    sample_dy_dX = eq'(sample_X, options)
    return has_good_range(sample_dy_dX)
end

"""
Check if there are n x as many constants as variables
"""
function has_too_many_constants(eq, n = 4)
    num_constants = count(t -> t.degree == 0 && t.constant, eq)
    num_variables = count(t -> t.degree == 0 && !t.constant, eq)
    return num_constants > n * num_variables
end

"""
This combines the above three tests into one, returning false
for any equation that breaks any of the tests.
"""
function equation_filter(eq, sample_X, sample_y, options)
    !(has_good_range(sample_y) || has_good_derivative_range(eq, sample_X, options))
end
function equation_filter_without_eval(eq)
    return !(has_overly_nested_unary(eq) || has_too_many_constants(eq))
end

"""
Generate a vector of equations that satisfy `equation_filter`
and use the input properties.
"""
function generate_equations(;
    num_equations = 10_000,
    max_attempts = 100_000,
    T = Float64,
    max_num_features = 5,
    num_samples = 1000,
    binary_operators = (+, -, *, /),
    unary_operators = (cos, sqrt),
    rng = MersenneTwister(1),
)
    options = SR.Options(; enable_autodiff = true, binary_operators, unary_operators)
    SR.@extend_operators options
    sample_X = rand(MersenneTwister(0), T, max_num_features, num_samples) .* 10
    num_nodes = rand(rng, 5:30, num_equations)
    num_features = rand(rng, 1:max_num_features, num_equations)

    equations = SR.Node{T}[]
    i = 0
    while length(equations) < num_equations && i < max_attempts
        i += 1
        seed!(i)
        if isinteger(log2(i))
            println(
                "Tried $i equations. Currently have $(length(equations)) equations saved.",
            )
        end

        equation_index = length(equations) + 1
        eq = SR.gen_random_tree_fixed_size(
            num_nodes[equation_index],
            options,
            num_features[equation_index],
            T,
        )

        DE.simplify_tree!(eq, options.operators)

        if num_nodes[equation_index] - length(eq) > 0.1 * num_nodes[equation_index]
            # Simplified too much; it's probably a simple equation
            continue
        end

        if !equation_filter_without_eval(eq)
            continue
        end

        sample_y = eq(sample_X, options)

        if !equation_filter(eq, sample_X, sample_y, options)
            continue
        end

        push!(equations, eq)
    end

    return equations, options, sample_X
end

# Plot a few:
function plot_some_equations(equations, options, n = 10)
    add_to_plot(x, y, f::F = Plots.plot; kws...) where {F} = f(x, y; kws...)
    local p

    first = true
    equations_1d =
        filter(eq -> all(t -> t.degree != 0 || t.constant || t.feature == 1, eq), equations)

    for equation in equations_1d[1:n]
        x = 0.001:0.001:10.0
        raw_y = equation(x', options)
        μ = mean(filter(isfinite, raw_y))
        sigma = std(filter(isfinite, raw_y))

        y = (raw_y .- μ) ./ sigma
        p = add_to_plot(x, y, first ? Plots.plot : Plots.plot!)
        first = false
    end
    p
end


function load_equations(; regenerate = false, plot = false, print_equations = false)
    if regenerate
        rm("equations.jls")
    end
    # First, we get a list of interesting equations:
    equations, options, sample_X = if isfile("equations.jls")
        deserialize("equations.jls")
    else
        generate_equations(
            num_equations = 20_000,
            max_attempts = 1_000_000_000,
            num_samples = 2_000,
            binary_operators = (+, -, *, /),
            unary_operators = (cos, sqrt, exp, log),
        )
    end
    if !isfile("equations.jls")
        serialize("equations.jls", (equations, options, sample_X))
    end

    @show length(equations)

    if print_equations
        println("Here are some equations:")
        for equation in equations[1:10]
            println(SR.string_tree(equation, options))
        end
    end

    if plot
        plot_some_equations(equations, options, 10)
    end

    (equations, options, sample_X)
end

"""
Main function to process a specific equation from a pre-generated list.

This function loads a set of equations, selects one based on the provided index,
and fits a symbolic regression model to the data generated by this equation.
The results, including the model and its performance metrics, are serialized for later use.

# Arguments

- `equation_index`: Index of the equation to process (between 1 and 20000)

# Options

- `--seed <int>`: Random seed for the search.
- `--parallelism <str>`: Type of parallelism. If serial, will use deterministic mode.
- `--niterations <int>`: Number of iterations to use for the search.
- `--maxsize <int>`: Maximum size of the tree.
- `--output_dir <str>`: Directory to save the output.
- `--take_only <int>`: Take only the first `take_only` data points during training.
"""
@main function main(
    equation_index::Int;
    seed::Int=1,
    parallelism::String="serial",
    niterations::Int=400,
    maxsize::Int=30,
    output_dir::String="output",
    take_only::Int=1000,
)
    @info "Loading equations from file"
    (equations, options, sample_X) = load_equations()
    X = deepcopy(sample_X)
    eq = equations[equation_index]
    y = eq(X, options)
    @info "Creating symbolic regression model"
    model = SR.SRRegressor(;
        binary_operators = options.operators.binops,
        unary_operators = options.operators.unaops,
        niterations = niterations,
        maxsize = maxsize,
        parallelism = Symbol(parallelism),
        deterministic = (parallelism == "serial"),
        seed,
        turbo=true,
        progress=false,
        save_to_file=false,
    )
    mach = MLJ.machine(model, copy(X'), y)
    MLJ.fit!(mach; verbosity=0)
    r = MLJ.report(mach)

    if !isdir(output_dir)
        mkpath(output_dir)
    end

    @info "Serializing the results"
    serialize(
        joinpath(output_dir, "equation_$(equation_index).jls"),
        (;
            equation = eq,
            repr = SR.string_tree(eq, options),
            complexity = SR.compute_complexity(eq, options),
            X,
            y,
            options,
            mach,
            report=r,
            best_loss = last(r.losses),
        ),
    )
end