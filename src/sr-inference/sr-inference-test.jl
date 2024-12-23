import SymbolicRegression: Options, equation_search
import SymbolicRegression.SymbolicRegression: Node
using DynamicExpressions: parse_expression, Expression
using Revise
import ONNXRunTime as ORT

model = ORT.load_inference("src/dev/ONNX/onnx-models/model-zwrgtnj0.onnx")

include("parsing.jl")
using .ParsingModule: nn_config, node_to_onehot, _create_grammar_masks, logits_to_prods, prods_to_tree, select_subtree

cfg = nn_config(
    nbin=4,
    nuna=6,
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


t = parse_expression(:((x1*x1 * 3) + x1), operators=options.operators, variable_names=["x1"])
try
    success, x = node_to_onehot(t.tree, cfg)
    input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [0.05])
    raw = model(input)
    x_out = raw["276"][1, :, :]
    prods = logits_to_prods(x_out, true)
    t2 = prods_to_tree(prods, OP_INDEX)
    t2 = Expression(t2, t.metadata)
catch e
    t2 = t
end
println("$i: $t -> $t2")







X = randn(2, 100)
y = 2 * cos.(X[2, :]) + X[1, :] .^ 2 .- 2

hall_of_fame = equation_search(
    X, y, niterations=40, options=options,
    parallelism=:multithreading
)

t = hall_of_fame.members[16].tree
subtree, parent, feature = select_subtree(t.tree)
t2 = t
try
    encode_success, x = node_to_onehot(subtree, cfg)
    if encode_success
        input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [0.05])
        raw = model(input)
        x_out = raw["276"][1, :, :]
        prods = logits_to_prods(x_out, true)
        t2 = prods_to_tree(prods, OP_INDEX, feature)
        t2 = Expression(t2, t.metadata)
    else
        t2 = t
    end
catch e
    t2 = t
end
println("$t -> $t2")




function mutate_tree(tree::Node)
    # Select a viable subtree to mutate
    subtree, parent, feature = select_subtree(tree)
    try
        # Encode the subtree into a one-hot vector
        encode_success, x = node_to_onehot(subtree, cfg)
        if encode_success
            # Sample new subtree
            input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [0.05])
            raw_out = MODEL_REF[](input)
            x_out = raw_out["276"][1, :, :]
            prods = logits_to_prods(x_out, true)
            new_subtree = prods_to_tree(prods, OP_INDEX, feature)

            # Replace the old subtree with the new one
            if parent === nothing
                # If there's no parent, this means we're replacing the root
                return new_subtree
            else
                # Replace the appropriate child in the parent node
                for i in 1:length(parent.children)
                    if parent.children[i] === subtree
                        parent.children[i] = new_subtree
                        break
                    end
                end
                return tree
            end
        else
            return tree
        end
    catch e
        return tree
    end
end

tree = hall_of_fame.members[16].tree
new_tree = mutate_tree(tree)


