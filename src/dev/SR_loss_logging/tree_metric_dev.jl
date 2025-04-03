using Revise
using SymbolicRegression
using DynamicExpressions: parse_expression, Expression
using SymbolicRegression.NeuralMutationsModule: zero_sqrt
using SymbolicRegression.LoggerModule: init_logger, close_global_logger!
using Hungarian

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, cos, exp, zero_sqrt],
    populations=40,
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

# Helper function to get all subtrees (clades) from a tree
function get_subtrees(node::Node{T}) where T
    subtrees = Set{Node{T}}()
    
    # Add current node to the set
    push!(subtrees, node)
    
    # Recursively add children based on node degree
    if node.degree >= 1
        # Add all subtrees from left child
        for subtree in get_subtrees(node.l)
            push!(subtrees, subtree)
        end
        
        # Add all subtrees from right child if binary operator
        if node.degree == 2
            for subtree in get_subtrees(node.r)
                push!(subtrees, subtree)
            end
        end
    end
    
    return subtrees
end

# Helper function to compare two nodes for structural equality
function is_structurally_equal(n1::Node{T}, n2::Node{T}) where T
    if n1.degree != n2.degree
        return false
    end

    if n1.degree == 0  # Feature or value
        if n1.constant != n2.constant
            return false
        end

        if n1.constant
            return n1.val == n2.val
        end

        return n1.feature == n2.feature
    elseif n1.degree == 1
        # Unary operator
        if n1.op != n2.op
            return false
        end

        return is_structurally_equal(n1.l, n2.l)
    elseif n1.degree == 2
        # Binary operator
        if n1.op != n2.op
            return false
        end

        if is_commutative(n1.op)
            return (is_structurally_equal(n1.l, n2.l) && is_structurally_equal(n1.r, n2.r)) ||
                   (is_structurally_equal(n1.l, n2.r) && is_structurally_equal(n1.r, n2.l))
        else
            return is_structurally_equal(n1.l, n2.l) && is_structurally_equal(n1.r, n2.r)
        end
    end    
    return true
end

# Function to calculate Generalized Robinson-Foulds distance between two expression trees
function generalized_robinson_foulds(tree1::Node{T}, tree2::Node{T}) where T    
    # Get all subtrees from both trees
    subtrees1 = get_subtrees(tree1)
    subtrees2 = get_subtrees(tree2)
    
    # Calculate symmetric difference
    unique_to_tree1 = 0
    unique_to_tree2 = 0
    
    # Count subtrees unique to tree1
    for st1 in subtrees1
        found_match = false
        for st2 in subtrees2
            if is_structurally_equal(st1, st2)
                found_match = true
                break
            end
        end
        if !found_match
            unique_to_tree1 += 1
        end
    end
    
    # Count subtrees unique to tree2
    for st2 in subtrees2
        found_match = false
        for st1 in subtrees1
            if is_structurally_equal(st2, st1)
                found_match = true
                break
            end
        end
        if !found_match
            unique_to_tree2 += 1
        end
    end
    
    # Return the sum of unique subtrees
    return unique_to_tree1 + unique_to_tree2
end



ex1 = parse_expression(:((x1*x1 * x2+2) + cos(x2)*x1*2), operators=options.operators, variable_names=["x1", "x2"])
ex2 = parse_expression(:((x1*x1 * x2 * 3) + cos(x2)*x1*2), operators=options.operators, variable_names=["x1", "x2"])
t1, t2 = ex1.tree, ex2.tree

is_structurally_equal(t1, t2)
# Test the function with our example trees
distance = generalized_robinson_foulds(t1, t2)
println("Generalized Robinson-Foulds distance: ", distance)
# Should be 0 since the trees are identical
@assert distance == 0 "Distance should be 0 for identical trees"




# -----------------------

# function is_commutative(binop_idx::Integer, options::Options)
#     binops = options.operators.binops
#     op = binops[binop_idx]

#     commutative_map = Dict(
#         "+" => true,
#         "*" => true,
#         "-" => false,
#         "/" => false
#     )
#     return get(commutative_map, string(op), false)
# end

"""
    tree_edit_distance(T1, T2, options)

Compute the tree edit distance between two expression trees.
This distance considers the structure and operators of the trees,
with special handling for commutative operators.
"""
function tree_edit_distance(T1::Node, T2::Node, options::Options)
    # Base cases: one of the trees is empty
    T1 === nothing && return cost_insert_tree(T2)
    T2 === nothing && return cost_delete_tree(T1)

    # Retrieve children based on node degree
    children1 = Node{Float64}[]
    children2 = Node{Float64}[]
    T1.degree >= 1 && push!(children1, T1.l)
    T1.degree == 2 && push!(children1, T1.r)
    T2.degree >= 1 && push!(children2, T2.l)
    T2.degree == 2 && push!(children2, T2.r)

    # One or both nodes are leaves
    isempty(children1) && return sum([cost_insert_tree(child) for child in children2], init=0.0)
    isempty(children2) && return sum([cost_delete_tree(child) for child in children1], init=0.0)

    # Compute the substitution cost for the roots
    substitution_cost = cost_substitute(T1, T2)

    # Special case: one or both nodes are unary operators
    n1, n2 = length(children1), length(children2)
    if n1 == 1 || n2 == 1
        if n1 == 1 && n2 == 1
            # Both are unary, direct comparison
            return substitution_cost + tree_edit_distance(children1[1], children2[1], options)
        elseif n1 == 1
            # T1 is unary, T2 is binary
            # Try matching the unary child with each of T2's children and take the minimum
            cost1 = tree_edit_distance(children1[1], children2[1], options) + cost_insert_tree(children2[2])
            cost2 = tree_edit_distance(children1[1], children2[2], options) + cost_insert_tree(children2[1])
            return substitution_cost + min(cost1, cost2)
        else # n2 == 1
            # T2 is unary, T1 is binary
            # Try matching the unary child with each of T1's children and take the minimum
            cost1 = tree_edit_distance(children1[1], children2[1], options) + cost_delete_tree(children1[2])
            cost2 = tree_edit_distance(children1[2], children2[1], options) + cost_delete_tree(children1[1])
            return substitution_cost + min(cost1, cost2)
        end
    end

    # Both nodes are binary
    if is_commutative(T1.op, options)
        # For commutative operators, we need to find the optimal matching between children
        cost_matrix = zeros(Float64, n1, n2)
        for i in 1:n1
            for j in 1:n2
                cost_matrix[i, j] = tree_edit_distance(children1[i], children2[j], options)
            end
        end
        
        # Use the Hungarian algorithm to find the optimal assignment
        _, cost = hungarian(cost_matrix)
        
        # Add penalties for unmatched children
        extra_cost = 0.0
        if n1 > n2
            extra_cost = sum([cost_delete_tree(child) for child in children1[n2+1:end]], init=0.0)
        elseif n2 > n1
            extra_cost = sum([cost_insert_tree(child) for child in children2[n1+1:end]], init=0.0)
        end
        return substitution_cost + cost + extra_cost
    else
        # For non-commutative (ordered) operators, use Zhang-Shasha algorithm
        # We use a dynamic programming approach for ordered trees
        
        # Initialize the forest distance matrix
        # fd[i,j] represents the distance between the forest of the first i nodes of T1
        # and the forest of the first j nodes of T2
        fd = zeros(Float64, n1+1, n2+1)
        
        # Initialize base cases
        fd[1, 1] = 0.0
        for i in 2:n1+1
            fd[i, 1] = fd[i-1, 1] + cost_delete_tree(children1[i-1])
        end
        for j in 2:n2+1
            fd[1, j] = fd[1, j-1] + cost_insert_tree(children2[j-1])
        end
        
        # Fill the forest distance matrix
        for i in 2:n1+1
            for j in 2:n2+1
                # Three operations: delete, insert, or match/replace
                delete_cost = fd[i-1, j] + cost_delete_tree(children1[i-1])
                insert_cost = fd[i, j-1] + cost_insert_tree(children2[j-1])
                match_cost = fd[i-1, j-1] + tree_edit_distance(children1[i-1], children2[j-1], options)
                fd[i, j] = min(delete_cost, insert_cost, match_cost)
            end
        end
        
        return substitution_cost + fd[n1+1, n2+1]
    end
end

function cost_insert_tree(T::Node)
    T === nothing && return 0.0

    cost = 1.0  # Base cost for inserting a node
    T.degree >= 1 && (cost += cost_insert_tree(T.l))
    T.degree == 2 && (cost += cost_insert_tree(T.r))
    return cost
end

function cost_delete_tree(T::Node)
    T === nothing && return 0.0

    cost = 1.0  # Base cost for deleting a node
    T.degree >= 1 && (cost += cost_delete_tree(T.l))
    T.degree == 2 && (cost += cost_delete_tree(T.r))
    return cost
end

function cost_substitute(T1::Node, T2::Node)
    @assert T1.degree >= 1 && T2.degree >= 1 "T1 and T2 must have at least one child"

    if T1.degree == T2.degree
        return T1.op == T2.op ? 0.0 : 1.0
    else
        return 0.0
    end
end

"""
    is_commutative(op, options)

Determine if an operator is commutative.
"""
function is_commutative(op::Int, options::Options)
    # Check if the operator is in the list of commutative operators
    # Typically, addition and multiplication are commutative
    commutative_ops = ["+", "*"]
    op_str = string(options.operators.binops[op])
    return op_str in commutative_ops
end


ex1 = parse_expression(:((x1*3 + x2) - cos(x1)- sin(x2)), operators=options.operators, variable_names=["x1", "x2"])
ex2 = parse_expression(:((3*x1 + x2) + cos(x1)- sin(x2)), operators=options.operators, variable_names=["x1", "x2"])
t1, t2 = ex1.tree, ex2.tree

# Test the tree edit distance function
println("Testing tree edit distance:")
distance_ted = tree_edit_distance(t1, t2, options)
println("Tree Edit Distance: ", distance_ted)
