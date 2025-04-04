using Revise
using SymbolicRegression
using DynamicExpressions: parse_expression, Expression
using SymbolicRegression.NeuralMutationsModule: zero_sqrt
using SymbolicRegression.NeuralLoggingModule: init_logger, close_global_logger!

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


"""
    is_commutative(op, options)

Determine if an operator is commutative.
"""
function is_commutative(op::Integer, options::Options)
    # Check if the operator is in the list of commutative operators
    # Typically, addition and multiplication are commutative
    commutative_ops = ["+", "*"]
    op_str = string(options.operators.binops[op])
    return op_str in commutative_ops
end

"""
    tree_edit_distance(T1, T2, options)

Compute the tree edit distance between two expression trees.
This distance considers the structure and operators of the trees,
with special handling for commutative operators.
"""
function tree_edit_distance(T1::Node, T2::Node, options::Options)::Float32
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

ex1 = parse_expression(:((x1*3 + x2) - cos(x1)- sin(x2)), operators=options.operators, variable_names=["x1", "x2"])
ex2 = parse_expression(:((3*x1 + x2) + cos(x1)- sin(x2)), operators=options.operators, variable_names=["x1", "x2"])
t1, t2 = ex1.tree, ex2.tree

# Test the tree edit distance function
println("Testing tree edit distance:")
start_time = time()
for i in 1:1000
    distance_ted = tree_edit_distance(t1, t2, options)
end
elapsed_time = time() - start_time
println("Time taken for tree_edit_distance: $(elapsed_time) seconds")
println("Tree Edit Distance: ", distance_ted)
