module ParsingModule

import SymbolicRegression: Node

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



function _create_grammar_masks()
    # Define grammar rules similar to Python version
    grammar_str = """
    S -> 'ADD' S S | 'SUB' S S | 'MUL' S S | 'DIV' S S
    S -> 'SIN' S | 'COS' S | 'EXP' S | 'TANH' S | 'COSH' S | 'SINH' S 
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



end