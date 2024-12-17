import SymbolicRegression: Options, equation_search
import SymbolicRegression.SymbolicRegression: Node
using DynamicExpressions: parse_expression
import ONNXRunTime as ORT

model = ORT.load_inference("src/dev/ONNX/onnx-models/model-9j0cbuui.onnx")

include("parsing.jl")
using .ParsingModule: nn_config, node_to_onehot


# X = randn(2, 100)
# y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

# options = Options(
#     binary_operators=[+, *, /, -],
#     unary_operators=[sin, cos, exp, cosh, sinh, tanh],
#     populations=20
# )

# hall_of_fame = equation_search(
#     X, y, niterations=40, options=options,
#     parallelism=:multithreading
# )

# t = hall_of_fame.members[3].tree

cfg = nn_config(
    nbin=6,
    nuna=4,
    nvar=1,
    seq_len=15
)

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, cosh, sinh, tanh],
    populations=20
)

t = parse_expression(:((1 * 3) + x1), operators=options.operators, variable_names=["x1"])
success, x = node_to_onehot(t.tree, cfg)


input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)))
raw = model(input)
x_out = raw["155"][1, :, :]






function _create_grammar_masks()
    # Define grammar rules similar to Python version
    grammar_str = """
    S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
    S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S 
    S -> 'CON'
    S -> 'x1'
    END -> 'END'
    """

    # Collect all LHS symbols and unique set
    all_lhs = String[]
    unique_lhs = String[]
    
    # Parse grammar string to get productions
    for line in split(grammar_str, "\n")
        if !isempty(strip(line))
            lhs = strip(split(line, "->")[1])
            push!(all_lhs, lhs)
            if !(lhs in unique_lhs)
                push!(unique_lhs, lhs)
            end
        end
    end

    # Create masks matrix - each row corresponds to a unique LHS symbol
    # and has 1s for productions with that LHS
    masks = falses(length(unique_lhs), length(all_lhs))
    for (i, symbol) in enumerate(unique_lhs)
        for (j, lhs) in enumerate(all_lhs)
            masks[i,j] = (symbol == lhs)
        end
    end

    allowed_prod_idx = findall(vec(any(masks, dims=1)))

    return masks, allowed_prod_idx
end

masks, allowed_prod_idx = _create_grammar_masks()

# Define grammar rules similar to Python version
grammar_str = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S 
S -> 'CON'
S -> 'x1'
END -> 'END'
"""

# Collect all LHS symbols and unique set
all_lhs = String[]
unique_lhs = String[]

# Parse grammar string to get productions
for line in split(grammar_str, "\n")
    if !isempty(strip(line))
        lhs = strip(split(line, "->")[1])
        push!(all_lhs, lhs)
        if !(lhs in unique_lhs)
            push!(unique_lhs, lhs)
        end
    end
end

# Create masks matrix - each row corresponds to a unique LHS symbol
# and has 1s for productions with that LHS
masks = falses(length(unique_lhs), length(all_lhs))
for (i, symbol) in enumerate(unique_lhs)
    for (j, lhs) in enumerate(all_lhs)
        masks[i,j] = (symbol == lhs)
    end
end

allowed_prod_idx = findall(vec(any(masks, dims=1)))