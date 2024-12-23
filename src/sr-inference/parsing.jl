module ParsingModule

import SymbolicRegression: Node
using Distributions: Categorical

# Define grammar rules similar to Python version
grammar_str = """
S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S 
S -> 'CON'
S -> 'x1'
END -> 'END'
"""

const OPERATOR_ARITY = Dict{String, Int}(
    # Elementary functions
    "ADD" => 2,
    "SUB" => 2,
    "MUL" => 2,
    "DIV" => 2,

    # Trigonometric Functions
    "SIN" => 1,
    "COS" => 1,
    "EXP" => 1,

    # Hyperbolic Functions
    "SINH" => 1,
    "COSH" => 1,
    "TANH" => 1,

    # FIXME: Handle this differently!
    "x1" => 0
)

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

function _create_grammar_masks(grammar_str::String)
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

    return masks, allowed_prod_idx, unique_lhs
end
masks, allowed_prod_idx, unique_lhs = _create_grammar_masks(grammar_str)

mutable struct nn_config
    nbin::Int
    nuna::Int
    nvar::Int

    seq_len::Int
    nb_onehot_cats::Int

    function nn_config(;nbin::Int, nuna::Int, nvar::Int, seq_len::Int)
        nb_onehot_cats = nbin + nuna + nvar + 2  # +1 for constants, +1 for end token
        return new(nbin, nuna, nvar, seq_len, nb_onehot_cats)
    end
end


function node_to_onehot(node::Node{T}, cfg::nn_config)::Tuple{Bool, Matrix{Float32}} where T <: Number
    prefix = _tree_to_prefix(node)
    idx = [_node_to_token_idx(node, cfg) for node in prefix]
    success, onehot, consts = _onehot_encode(idx, cfg)
    ~success && return false, nothing

    x = zeros(Float32, cfg.seq_len, cfg.nb_onehot_cats+1)
    x[:, 1:cfg.nb_onehot_cats] = onehot
    x[:, cfg.nb_onehot_cats+1] = consts
    return true, x
end

function _onehot_encode(idx::Vector{Tuple{Int, Float64}}, cfg::nn_config)::Tuple{Bool, BitMatrix, Vector{Float64}}
    onehot = falses(cfg.seq_len, cfg.nb_onehot_cats)
    consts = zeros(Float64, cfg.seq_len)

    if length(idx) > cfg.seq_len
        return false, nothing, nothing
    end
    for (node_i, (token_idx, token_val)) in enumerate(idx)  # Each node has corresponding index and value
        onehot[node_i, token_idx] = true
        consts[node_i] = token_val
    end
    # Pad with empty token (last category)
    onehot[length(idx)+1:end, end] .= true 
    return true, onehot, consts
end

function _tree_to_prefix(tree::Node{T})::Vector{Node{T}} where T <: Number
    result = [tree]
    if tree.degree >= 1
        append!(result, _tree_to_prefix(tree.l))
    end
    if tree.degree == 2
        append!(result, _tree_to_prefix(tree.r))
    end
    return result
end

function _node_to_token_idx(node::Node{T}, cfg::nn_config)::Tuple{Int, Float64} where T <: Number
    offset_unaop = cfg.nbin
    offset_const = offset_unaop + cfg.nuna
    offset_var = offset_const + cfg.nvar
    if node.degree == 2
        return (node.op, 0)
    elseif node.degree == 1
        return (offset_unaop + node.op, 0)
    elseif node.degree == 0
        if node.constant
            return (offset_const + 1, node.val)  # Only 1 const token
        else
            return (offset_var + node.feature, 0)
        end
    end
end


function logits_to_prods(logits::Matrix{Float32}, sample::Bool=false, max_length::Int=15)::Vector{Tuple{String, String}}
    # Initialize empty stack with start symbol 'S'
    stack = ["S"]
    
    # Split logits into productions and constants
    logits_prods = logits[:, 1:end-1] 
    constants = logits[:, end]
    
    prods = []
    t = 1
    
    while !isempty(stack)
        alpha = pop!(stack)  # Current LHS toke
        
        # Get mask for current symbol
        symbol_idx = findfirst(==(alpha), unique_lhs)
        mask = masks[symbol_idx, :]
        
        # Calculate probabilities
        probs = mask .* exp.(logits_prods[t, :])
        tot = sum(probs)
        @assert tot > 0 "Sum of probs is 0 at t=$t. Probably due to bad mask or invalid logits?"
        probs = probs ./ tot
        
        # Select production rule
        if sample
            i = rand(Categorical(probs))
        else
            _, i = findmax(probs)
        end
        
        # Get selected rule
        rule = split(grammar_str, "\n")[i]
        lhs, rhs = split(strip(rule), "->")
        lhs = strip(lhs)
        rhs = strip(rhs)
        
        # If rule produces CONST, replace with actual constant
        if rhs == "'CON'"
            rhs = string(constants[t])
        end
        
        # Add production to list
        push!(prods, (lhs, rhs))
        
        # Add RHS nonterminals to stack in reverse order
        rhs_symbols = split(rhs)
        for symbol in reverse(rhs_symbols)
            clean_symbol = replace(symbol, "'" => "")
            if clean_symbol in unique_lhs
                push!(stack, clean_symbol)
            end
        end
        
        t += 1
        if t > max_length
            break
        end
    end
    
    return prods
end

function prods_to_tree(prods::Vector{Tuple{String, String}}, OP_INDEX::Dict{String, Int}, feature::Int)
    # global prods stack
    # Create node for each production
    # Depending on arity, create 0, 1 or 2 children
    # 
    prefix_list = _prods_to_prefix(prods, OP_INDEX, feature)
    tree = _prefix_to_tree!(prefix_list)

    return tree
end

function _prods_to_prefix(prods::Vector{Tuple{String, String}}, OP_INDEX::Dict{String, Int}, feature::Int)::Vector{Node{Float64}}
    prefix_list = []
    for prod in prods
        op_match = match(r"'([^']+)'", prod[2])  # Alternatively, use prod to infer arity?
        if op_match !== nothing
            op = op_match.captures[1]
            arity = OPERATOR_ARITY[op]
            if arity == 0
                push!(prefix_list, Node{Float64}(; feature=feature))  # FIXME: Only univariate for now
            elseif arity == 1
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], Float64))
            elseif arity == 2
                push!(prefix_list, _make_childless_op(arity, OP_INDEX[op], Float64))
            end
        else  # Constant
            push!(prefix_list, Node{Float64}(; val=parse(Float64, prod[2])))
        end
    end
    return prefix_list
end


"""
Copied from datagen/ExpressionGenerator.jl. Consolidate.
"""
function _make_childless_op(degree::Int, op_index::Int, ::Type{T})::Node{T} where {T<:Number}
    op_node = Node{T}()
    op_node.degree = degree
    op_node.op = op_index
    return op_node
end

"""
Copied from datagen/ExpressionGenerator.jl. Consolidate.
"""
function _prefix_to_tree!(prefix_list::Vector{Node{T}})::Node{T} where {T<:Number}
    function build_subtree()
        if isempty(prefix_list)
            error("Unexpected end of prefix list")
        end
        node = popfirst!(prefix_list)
        if node.degree == 0
            # Leaf node, no children to add
        elseif node.degree == 1
            node.l = build_subtree()
        elseif node.degree == 2
            node.l = build_subtree()
            node.r = build_subtree()
        else
            error("Invalid node degree: $(node.degree)")
        end
        return node
    end

    if isempty(prefix_list)
        error("Empty prefix list")
    end
    return build_subtree()
end

"""
Selects a subtree to use for neural sampling. Requirements for the subtree:
    - Univariate (only one feature)
    - 2-14 nodes

Returns the subtree and the feature used in the subtree.

FIXME: Make stochastic.
"""
function select_subtree(t::Node)::Tuple{Node, Node, Int}
    node_list = _tree_to_prefix(t)

    valid_subtrees = []
    for node in node_list
        subtree_nodes = _tree_to_prefix(node)
        
        # Check size constraint
        if length(subtree_nodes) < 2 || length(subtree_nodes) > 14
            continue
        end
        
        # Track which features are used
        features_used = Set{Int}()
        for n in subtree_nodes
            if n.degree == 0 && !n.constant
                push!(features_used, n.feature)
            end
        end
        
        # Check univariate constraint
        if length(features_used) == 1
            # Find parent node
            parent = nothing
            for potential_parent in node_list
                if potential_parent.l === node || potential_parent.r === node
                    parent = potential_parent
                    break
                end
            end
            push!(valid_subtrees, (node, parent, first(features_used)))
        end
    end

    if isempty(valid_subtrees)
        return nothing
    end

    return valid_subtrees[rand(1:length(valid_subtrees))]
    
end

end