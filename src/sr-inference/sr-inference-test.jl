import SymbolicRegression: Options, equation_search
import SymbolicRegression.SymbolicRegression: Node
using DynamicExpressions: parse_expression
import ONNXRunTime as ORT

model = ORT.load_inference("src/dev/ONNX/onnx-models/model-9j0cbuui.onnx")

include("parsing.jl")
using .ParsingModule: nn_config, node_to_onehot, _create_grammar_masks, logits_to_prods, prods_to_tree


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
    populations=20 )

ops = [options.operators.binops..., options.operators.unaops...]
const OP_INDEX = Dict{String, Int}(
    "ADD" => findfirst(==(+), ops),
    "SUB" => findfirst(==(-), ops),
    "MUL" => findfirst(==(*), ops),
    "DIV" => findfirst(==(/), ops),
    "SIN" => findfirst(==(sin), ops)-4,
    "COS" => findfirst(==(cos), ops)-4,
    "EXP" => findfirst(==(exp), ops)-4,
    "TANH" => findfirst(==(tanh), ops)-4,
    "COSH" => findfirst(==(cosh), ops)-4,
    "SINH" => findfirst(==(sinh), ops)-4,
)


t = parse_expression(:((cos(x1) * 3) + x1), operators=options.operators, variable_names=["x1"])
success, x = node_to_onehot(t.tree, cfg)

input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)))
raw = model(input)
x_out = raw["155"][1, :, :]
prods = logits_to_prods(x_out, true)
println(prods)

t2 = prods_to_tree(prods, OP_INDEX)


