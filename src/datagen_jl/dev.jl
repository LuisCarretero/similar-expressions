using Random: default_rng, AbstractRNG
using DynamicExpressions: OperatorEnum
using Random
using Distributions
using StatsBase
using DynamicExpressions:
    AbstractExpressionNode,
    AbstractExpression,
    AbstractNode,
    NodeSampler,
    Node,
    get_contents,
    with_contents,
    constructorof,
    copy_node,
    set_node!,
    count_nodes,
    has_constants,
    has_operators,
    string_tree

DATA_TYPE = Number

ops = OperatorEnum((+, -, *), (sin, exp))
nuna = length(ops.unaops)
nbin = length(ops.binops)
ops
all_ops = (ops.unaops..., ops.binops...)

nl = 1
p1 = 1
p2 = 1

struct OperatorProbEnum
    binops_probs::Vector{Float64}
    unaops_probs::Vector{Float64}
end

op_probs = OperatorProbEnum(
    [0.33, 0.33, 0.33],
    [0.5, 0.5]
)

total_ops = 3
nfeatures = 1

function _generate_ubi_dist(max_ops::Int)
    """
    `max_ops`: maximum number of operators
    Enumerate the number of possible unary-binary trees that can be generated from empty nodes.
    D[e][n] represents the number of different binary trees with n nodes that
    can be generated from e empty nodes, using the following recursion:
        D(0, n) = 0
        D(e, 0) = L ** e
        D(e, n) = L * D(e - 1, n) + p_1 * D(e, n - 1) + p_2 * D(e + 1, n - 1)
    """

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

ubi_dist = _generate_ubi_dist(total_ops)

function _sample_next_pos_ubi(nb_empty::Int, nb_ops::Int, rng = Random.MersenneTwister(123))
    """
    Sample the position of the next node (unary-binary case).
    Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
    """
    @assert nb_empty > 0
    @assert nb_ops > 0
    probs = Float64[]
    for i in 0:(nb_empty-1)
        push!(probs, 
            (nl ^ i) * p1 * ubi_dist[nb_empty - i + 1][nb_ops]
        )
    end
    for i in 0:(nb_empty-1)
        push!(probs, 
            (nl ^ i) * p2 * ubi_dist[nb_empty - i + 2][nb_ops]
        )
    end
    probs ./= ubi_dist[nb_empty + 1][nb_ops + 1]
    e = rand(rng, Categorical(probs)) - 1
    arity = e < nb_empty ? 1 : 2
    e = e % nb_empty
    return e, arity
end

function make_random_leaf(
    nfeatures::Int,
    ::Type{T},
    ::Type{N},
    rng::AbstractRNG=default_rng(),
) where {T<:DATA_TYPE,N<:AbstractExpressionNode}
"""From SymbolicRegression.jl"""
    if rand(rng, Bool)  # TODO: Add probs
        return constructorof(N)(; val=randn(rng, T))
    else
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
end

"""
Careful, cannot be printed currently (will throw error)
"""
function make_childless_op(degree::Int, op_index::Int, ::Type{T})::Node{T} where {T<:DATA_TYPE}
    op_node = Node{T}()
    op_node.degree = degree
    op_node.op = op_index
    return op_node
end

"""
Use empty nodes as placeholders for leaves
Dont know total number of nodes beforehand (depends on unary-binary choices)
"""
function _generate_prefix_expr(
    nb_total_ops::Int,
    ::Type{T},
    rng::AbstractRNG=default_rng()
)::Vector{Node{T}} where {T<:DATA_TYPE}
    nb_empty = 1  # number of empty nodes
    l_leaves = 0  # left leaves - nothing states reserved for leaves
    t_leaves = 1  # total number of leaves (just used for sanity check)

    stack = [nothing]

    for nb_ops in nb_total_ops:-1:1
        skipped, arity = _sample_next_pos_ubi(nb_empty, nb_ops)

        if arity == 1
            op_index = StatsBase.sample(1:nuna, StatsBase.Weights(op_probs.unaops_probs))
        else
            op_index = StatsBase.sample(1:nbin, StatsBase.Weights(op_probs.binops_probs))
        end
        op_node = make_childless_op(arity, op_index, T)

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
    @assert count(x -> isa(x, Node), stack) == nb_total_ops
    @assert count(x -> x === nothing, stack) == t_leaves

    # 2. Create leaves
    leaves = [make_random_leaf(nfeatures, T, Node{T}, rng) for _ in 1:t_leaves]

    # 3. Insert leaves into tree
    for pos in length(stack):-1:1
        if stack[pos] === nothing
            stack = [stack[1:pos-1]..., pop!(leaves), stack[pos+1:end]...]
        end
    end
    @assert isempty(leaves)
    return stack
end

function prefix_to_tree!(prefix_list::Vector{Node{T}})::Node{T} where {T<:DATA_TYPE}
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


function _generate_tree(nb_total_ops::Int, ::Type{T}, rng::AbstractRNG=default_rng())::Node{T} where {T<:DATA_TYPE}
    stack = _generate_prefix_expr(nb_total_ops, T, rng)
    return prefix_to_tree!(stack)
end

tree = _generate_tree(3, Float64)

string_tree(tree, ops)