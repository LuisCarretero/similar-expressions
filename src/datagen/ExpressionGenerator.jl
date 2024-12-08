module ExpressionGenerator

using ..Configs: ExpressionGeneratorConfig, OperatorProbEnum, ValueTransformSettings, FilterSettings

using DynamicExpressions: Node
using Random: default_rng, AbstractRNG, MersenneTwister, shuffle!
using Distributions: truncated, Normal, Distribution, Categorical
using StatsBase
using DynamicExpressions: OperatorEnum

export generate_expr_tree

function build_expression_generator_config(op_cnt_min::Int, op_cnt_max::Int, data_type::Type, ops::OperatorEnum, op_probs::OperatorProbEnum, nfeatures::Int, seq_len::Int, seed::Int=0, value_transform_settings::ValueTransformSettings=ValueTransformSettings(), filter_settings::FilterSettings=FilterSettings(), eval_x::Matrix{Float64}=Matrix{Float64}(undef, 0, 0), save_transformed::Bool=true)
        @assert op_cnt_min <= op_cnt_max
        @assert op_cnt_min > 0
    
        rng = MersenneTwister(seed)
    
        nuna = length(ops.unaops)
        nbin = length(ops.binops)
    
        nl = 1 # FIXME: Adjust these values?
        p1 = 1
        p2 = 1
    
        ubi_dist = _generate_ubi_dist(op_cnt_max, nl, p1, p2)
        nb_onehot_cats = nbin + nuna + nfeatures + 2  # +1 for constants, +1 for end token
        const_distr = truncated(Normal(), -5, 5)  # mean=0, std=1, min=-5, max=5 TODO: Make this customizable.
        return ExpressionGeneratorConfig(op_cnt_min, op_cnt_max, data_type, ubi_dist, rng, ops, op_probs, nuna, nbin, nfeatures, const_distr,nl, p1, p2, seq_len, nb_onehot_cats, value_transform_settings, filter_settings, eval_x, save_transformed)
    end


function generate_expr_tree(config::ExpressionGeneratorConfig)::Node
    stack = _generate_expr_prefix(config, config.data_type)
    return _prefix_to_tree!(stack)
end

"""
`max_ops`: maximum number of operators
Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
D[e][n] represents the number of different binary trees with n nodes that
can be generated from e empty nodes, using the following recursion:
    D(0, n) = 0
    D(e, 0) = L ** e
    D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
"""
function _generate_ubi_dist(max_ops::Int, nl::Int, p1::Int, p2::Int)::Vector{Vector{Int}}
    # enumerate possible trees
    # first generate the transposed version of D, then transpose it
    D = Vector{Vector{Int}}()
    push!(D, [0; [nl^i for i in 1:(2*max_ops+1)]])
    
    for n in 1:(2*max_ops)  # number of operators
        s = [0]
        for e in 1:(2*max_ops-n+1)  # number of empty nodes
            push!(s, 
                nl * s[e] +
                p1 * D[n][e+1] +
                p2 * D[n][e+2]
            )
        end
        push!(D, s)
    end
    
    @assert all(length(D[i]) >= length(D[i+1]) for i in 1:length(D)-1)
    
    D = [
        [D[j][i] for j in 1:length(D) if i <= length(D[j])]
        for i in 1:maximum(length(x) for x in D)
    ]
    
    return D
end

"""
Sample the position of the next node (unary-binary case).
Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
"""
function _sample_next_pos_ubi(
    config::ExpressionGeneratorConfig,
    nb_empty::Int,
    nb_ops::Int
)::Tuple{Int, Int}
    @assert nb_empty > 0
    @assert nb_ops > 0
    probs = Float64[]
    for i in 0:(nb_empty-1)
        push!(probs, 
            (config.nl ^ i) * config.p1 * config.ubi_dist[nb_empty - i + 1][nb_ops]
        )
    end
    for i in 0:(nb_empty-1)
        push!(probs, 
            (config.nl ^ i) * config.p2 * config.ubi_dist[nb_empty - i + 2][nb_ops]
        )
    end
    probs ./= config.ubi_dist[nb_empty + 1][nb_ops + 1]
    e = rand(config.rng, Categorical(probs)) - 1
    arity = e < nb_empty ? 1 : 2
    e = e % nb_empty
    return e, arity
end

function _make_random_leafs(leaf_cnt::Int, feature_leaf_cnt::Int, nfeatures::Int, const_distr::Distribution, ::Type{T}, rng::AbstractRNG=default_rng())::Vector{Node{T}} where {T<:Number}
    leaves = Vector{Node{T}}()
    for _ in 1:feature_leaf_cnt
        # FIXME: If we want all features to be used at least once, need to change this.
        push!(leaves, Node{T}(; feature=rand(rng, 1:nfeatures)))
    end
    for _ in 1:(leaf_cnt - feature_leaf_cnt)
        push!(leaves, Node{T}(; val=rand(rng, const_distr))) # TODO: Customizable distribution. Currently normal with mu=0, sigma=1.
    end
    shuffle!(rng, leaves)
    return leaves
end

"""
Careful, cannot be printed currently (will throw error)
"""
function _make_childless_op(degree::Int, op_index::Int, ::Type{T})::Node{T} where {T<:Number}
    op_node = Node{T}()
    op_node.degree = degree
    op_node.op = op_index
    return op_node
end

"""
Dont know total number of nodes beforehand (depends on unary-binary choices)
"""
function _generate_expr_prefix(config::ExpressionGeneratorConfig, ::Type{T})::Vector{Node{T}} where {T<:Number}
    nb_empty = 1  # number of empty nodes
    l_leaves = 0  # left leaves - nothing states reserved for leaves
    t_leaves = 1  # total number of leaves (just used for sanity check)

    stack = [nothing]

    op_cnt_total = rand(config.rng, config.op_cnt_min:config.op_cnt_max)  # TODO: Add distribution?

    for nb_ops in op_cnt_total:-1:1
        skipped, arity = _sample_next_pos_ubi(config, nb_empty, nb_ops)

        if arity == 1
            op_index = StatsBase.sample(1:config.nuna, StatsBase.Weights(config.op_probs.unaops_probs))
        else
            op_index = StatsBase.sample(1:config.nbin, StatsBase.Weights(config.op_probs.binops_probs))
        end
        op_node = _make_childless_op(arity, op_index, T)

        nb_empty += (arity - 1 - skipped)
        t_leaves += arity - 1
        l_leaves += skipped

        # update tree
        pos = findall(x -> x === nothing, stack)[l_leaves+1]
        stack = [
            stack[1:pos-1]...,
            op_node,
            [nothing for _ in 1:arity]...,
            stack[pos+1:end]...
        ]
    end

    # sanity check
    @assert count(x -> isa(x, Node), stack) == op_cnt_total
    @assert count(x -> x === nothing, stack) == t_leaves

    # 2. Create leaves
    feature_leaf_cnt = rand(config.rng, 1:t_leaves)  # TODO: Make this customizable. currently always one feature leaf. Could also introduce a feature leaf probability.
    leaves = _make_random_leafs(t_leaves, feature_leaf_cnt, config.nfeatures, config.const_distr, T, config.rng)

    # 3. Insert leaves into tree
    for pos in length(stack):-1:1
        if stack[pos] === nothing
            stack = [stack[1:pos-1]..., pop!(leaves), stack[pos+1:end]...]
        end
    end
    @assert isempty(leaves)
    return stack
end

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

end # module