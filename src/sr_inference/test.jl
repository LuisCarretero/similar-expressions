using Revise
using SymbolicRegression
using DynamicExpressions: parse_expression, Expression
using SymbolicRegression.NeuralMutationsModule: zero_sqrt

# model_id = "zwrgtnj0"
model_id = "e51hcsb9"
options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, zero_sqrt],
    populations=40,
    neural_options=NeuralOptions(
        active=true,  # If not active, will still be called according to MutationWeights.neural_mutate_tree rate but will return the original tree
        sampling_eps=1e-10,
        subtree_min_nodes=5,
        subtree_max_nodes=10,
        # model_path="/Users/luis/Desktop/Cranmer2024/Workplace/smallMutations/similar-expressions/src/dev/ONNX/onnx-models/model-$model_id.onnx",
        model_path="/home/lc865/workspace/similar-expressions/src/dev/ONNX/onnx-models/model-$model_id.onnx",
        device="cuda",
        verbose=true,
    ),
    mutation_weights=MutationWeights(
        mutate_constant = 0.0353,
        mutate_operator = 3.63,
        swap_operands = 0.00608,
        rotate_tree = 1.42,
        add_node = 0.0771,
        insert_node = 2.44,
        delete_node = 0.369,
        simplify = 0.00148,
        randomize = 0.00695,
        do_nothing = 0.431,
        optimize = 0.0,
        form_connection = 0.5,
        break_connection = 0.1,
        neural_mutate_tree = 0.0
    ),
)


# ex = parse_expression(:((x1*x1 * 3) + cos(x2)*2 +5), operators=options.operators, variable_names=["x1", "x2"])
ex = parse_expression(:(y1*y1+y1-exp(y1)*cos(y1)), operators=options.operators, variable_names=["y1", "y2", "y3", "y4", "y5"])

# Sample single
ex_out = SymbolicRegression.NeuralMutationsModule.neural_mutate_tree(copy(ex), options)

# Sample multiple
ex_out = nothing
function mutate_multiple(ex, options, n)
    for i in 1:n
        ex_out = SymbolicRegression.NeuralMutationsModule.neural_mutate_tree(copy(ex), options)
    end
end

mutate_multiple(ex, options, 10000)

stats = SymbolicRegression.NeuralMutationsModule.get_mutation_stats()

# SymbolicRegression.NeuralMutationsModule.reset_mutation_stats!()

using Distributions
# %% Check how close to 1 we need probs to work

logits_prods = Float32[-10.712675, 10.319759, -16.263983, 6.586913, -14.957393, 6.610545, 15.951616, 13.097554, 27.844213, 1.0822272, -5.8365364, -5.5205793, -5.6031995]
mask = Float32[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
probs = mask .* exp.(logits_prods)
tot = sum(probs)
a = (tot == 0 || !isfinite(tot))
probs = probs ./ tot
b = (any(!isfinite, probs) || abs(sum(probs) - 1) > 1e-10)
println("tot: $tot, a: $a, b: $b")

Categorical(probs.+1)

a = Float32[-88.0, -88.0, -60.72827, 88.0, 88.0, 88.0, -88.0, -88.0, -88.0, -88.0, -88.0, -88.0, -88.0] * 84/88
b = exp.(a)
b = b ./ sum(b)
Categorical(b)


c = clamp.(Float32[-1e4, 1e2], -88.0f0, 88.0f0)
exp.(c)


log(floatmax(Float32)/50)


#############################


# Import from Parsing.jl
using SymbolicRegression.ParsingModule: OPERATOR_ARITY

# Create dict and print sorted by arity
op_index = Dict(
    "SIN" => 1,
    "COS" => 2, 
    "EXP" => 3,
    "ZERO_SQRT" => 4,
    "ADD" => 1,
    "SUB" => 4,
    "MUL" => 2,
    "DIV" => 3
)

println("Unary operators:")
for (op, idx) in sort(collect(filter(p -> OPERATOR_ARITY[p.first] == 1, op_index)), by=x->x[2])
    println("  $op => $idx")
end

println("\nBinary operators:") 
for (op, idx) in sort(collect(filter(p -> OPERATOR_ARITY[p.first] == 2, op_index)), by=x->x[2])
    println("  $op => $idx")
end

logit_index = ["ADD", "SUB", "MUL", "DIV", "SIN", "COS", "EXP", "ZERO_SQRT", "CON", "x1", "END"]

op_to_logits: Dict(
    (3, 1) => 7, 
    (1, 1) => 5, 
    (2, 1) => 6,
    (4, 1) => 8, 


    (3, 2) => 4, 
    (1, 2) => 1, 
    (4, 2) => 2, 
    (2, 2) => 3, 
    )


#######

grammar_str = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'COS' S | 'EXP' S | 'ZERO_SQRT' S
S -> 'CON'
S -> 'x1'
END -> 'END'
"""

# Split each production rule into separate lines
grammar_lines = String[]
for line in split(grammar_str, "\n")
    line = strip(line)
    if !isempty(line)
        lhs, rhs = split(line, "->")
        lhs = strip(lhs)
        # Split on | and create new lines
        for rule in split(rhs, "|")
            push!(grammar_lines, "$lhs -> $(strip(rule))")
        end
    end
end
grammar_str = join(grammar_lines, "\n")
println("grammar_str: ", grammar_str)


#####
import CUDA
CUDA.set_runtime_version!(v"12.6")

import CUDA, cuDNN
import ONNXRunTime as ORT

model_id = "e51hcsb9"
model_path="/home/lc865/workspace/similar-expressions/src/dev/ONNX/onnx-models/model-$model_id.onnx"
model = ORT.load_inference(model_path, execution_provider=:cuda)


x = rand(Float32, 15, 12)
input = Dict("onnx::Flatten_0" => reshape(x, (1, size(x)...)), "sample_eps" => [0.01])

for i in 1:100000
    raw_out = model(input)
    x_out = raw_out["276"][1, :, :]
end
