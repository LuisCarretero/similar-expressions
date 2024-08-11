#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.calculus.util import AccumBounds
from sympy_utils import simplify
from collections import Counter
from dclasses import GeneratorDetails
from torch.distributions import Uniform

CLEAR_SYMPY_CACHE_FREQ = 10000


class NotCorrectIndependentVariables(Exception):
    pass

class UnknownSymPyOperator(Exception):
    pass

class ValueErrorExpression(Exception):
    pass

class ImAccomulationBounds(Exception):
    pass

class InvalidPrefixExpression(Exception):
    pass

class Generator(object):
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: "add",
        sp.Mul: "mul",
        sp.Pow: "pow",
        sp.exp: "exp",
        sp.log: "ln",
        sp.Abs: 'abs',

        # Trigonometric Functions
        sp.sin: "sin",
        sp.cos: "cos",
        sp.tan: "tan",

        # Trigonometric Inverses
        sp.asin: "asin",
        sp.acos: "acos",
        sp.atan: "atan",

        # Hyperbolic Functions
        sp.sinh: "sinh",
        sp.cosh: "cosh",
        sp.tanh: "tanh",

    }

    OPERATORS = {
        # Elementary functions
        "add": 2,
        "sub": 2,
        "mul": 2,
        "div": 2,
        "pow": 2,
        "inv": 1,
        "pow2": 1,
        "pow3": 1,
        "pow4": 1,
        "pow5": 1,
        "sqrt": 1,
        "exp": 1,
        "ln": 1,
        "abs": 1,

        # Trigonometric Functions
        "sin": 1,
        "cos": 1,
        "tan": 1,

        # Trigonometric Inverses
        "asin": 1,
        "acos": 1,
        "atan": 1,

        # Hyperbolic Functions
        "sinh": 1,
        "cosh": 1,
        "tanh": 1,
        "coth": 1,
    }
    operators = sorted(list(OPERATORS.keys()))
    constants = ["pi", "E"]
    def __init__(self, params: GeneratorDetails):
        self.min_ops = params.min_ops
        self.max_ops = params.max_ops
        self.max_len = params.max_len
        #self.positive = params.positive


        # parse operators with their weights
        ops = params.operators.split(",")
        ops = sorted([x.split(":") for x in ops])
        assert len(ops) >= 1 and all(o in self.OPERATORS for o, _ in ops)
        self.all_ops = [o for o, _ in ops]
        self.una_ops = [o for o, _ in ops if self.OPERATORS[o] == 1]
        self.bin_ops = [o for o, _ in ops if self.OPERATORS[o] == 2]
        self.all_ops_probs = np.array([float(w) for _, w in ops]).astype(np.float64)
        self.una_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 1]
        ).astype(np.float64)
        self.bin_ops_probs = np.array(
            [float(w) for o, w in ops if self.OPERATORS[o] == 2]
        ).astype(np.float64)
        self.all_ops_probs = self.all_ops_probs / self.all_ops_probs.sum()
        self.una_ops_probs = self.una_ops_probs / self.una_ops_probs.sum()
        self.bin_ops_probs = self.bin_ops_probs / self.bin_ops_probs.sum()

        assert len(self.all_ops) == len(set(self.all_ops)) >= 1
        assert set(self.all_ops).issubset(set(self.operators))
        assert len(self.all_ops) == len(self.una_ops) + len(self.bin_ops)

        # symbols / elements
        self.variables = OrderedDict({})
        for var in params.variables: 
            self.variables[str(var)] =sp.Symbol(str(var), real=True, nonzero=True)
        self.n_variables = len(self.variables)
        self.var_symbols = list(self.variables)
        self.pos_dict = {x:idx for idx, x in enumerate(self.var_symbols)}        
        self.placeholders = {}
        self.placeholders["cm"] = sp.Symbol("cm", real=True, nonzero=True)
        self.placeholders["ca"] = sp.Symbol("ca",real=True, nonzero=True)
        self.const_placeholder = sp.Symbol("c", real=True, nonzero=True)
        assert 1 <= len(self.variables)
        # We do not no a priori how many coefficients an expression has, so to be on the same side we equal to two times the maximum number of expressions
        self.coefficients = [f"{x}_{i}" for x in self.placeholders.keys() for i in range(2*params.max_len)] 
        assert all(v in self.OPERATORS for v in self.SYMPY_OPERATORS.values())

        # SymPy elements
        self.local_dict = {}
        for k, v in list(
            self.variables.items()
        ):  
            assert k not in self.local_dict
            self.local_dict[k] = v

        digits = [str(i) for i in range(-3, abs(6))]
        self.words = (
            list(self.variables.keys())
            + [
                x
                for x in self.operators
                if x not in ("pow2", "pow3", "pow4", "pow5", "sub", "inv")
            ]
            + digits
        )  


        self.id2word = {i: s for i, s in enumerate(self.words, 4)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        # ADD Start and Finish
        self.word2id["P"] = 0
        self.word2id["S"] = 1
        self.word2id["F"] = 2
        self.id2word[1] = "S"
        self.id2word[2] = "F"

        # ADD Constant Placeholder
        self.word2id["c"] = 3
        self.id2word[3] = "c"

        assert len(set(self.word2id.values())) == len(self.word2id.values())
        assert len(set(self.id2word.values())) == len(self.id2word.values())

        #assert len(self.words) == len(set(self.words))

        # leaf probabilities. TODO: Check if params.leaf_probs is defined.
        # FIXME: Only have two options: const or var
        self.leaf_probs = np.array(params.leaf_probs).astype(np.float64)
        assert len(self.leaf_probs) == 2 and all(self.leaf_probs >= 0)
        self.leaf_probs = self.leaf_probs / self.leaf_probs.sum()

        # number of words / indices
        self.n_words = params.n_words = len(self.words)

        # generation parameters. FIXME?
        self.nl = 1  # self.n_leaves
        self.p1 = 1  # len(self.una_ops)
        self.p2 = 1  # len(self.bin_ops)

        # initialize distribution for binary and unary-binary trees
        self.bin_dist = self.generate_bin_dist(params.max_ops)
        self.ubi_dist = self.generate_ubi_dist(params.max_ops)

        # rewrite expressions
        self.rewrite_functions = self.return_rewrite_functions(params)


    @classmethod
    def return_local_dict(cls, variables=None):
        local_dict = {}
        for k, v in list(
            variables.items()
        ):  
            assert k not in local_dict
            local_dict[k] = v
        return local_dict


    @classmethod
    def return_rewrite_functions(cls,params):
        r =  [
            x for x in params.rewrite_functions.split(",") if x != ""
        ]
        assert len(r) == len(set(r))
        assert all(
            x in ["expand", "factor", "expand_log", "logcombine", "powsimp", "simplify"]
            for x in r
        )
        return r

    def generate_bin_dist(self, max_ops):
        """
        `max_ops`: maximum number of operators
        Enumerate the number of possible binary trees that can be generated from empty nodes.
        D[e][n] represents the number of different binary trees with n nodes that
        can be generated from e empty nodes, using the following recursion:
            D(0, n) = 0
            D(1, n) = C_n (n-th Catalan number)
            D(e, n) = D(e - 1, n + 1) - D(e - 2, n + 1)
        """
        # initialize Catalan numbers
        catalans = [1]
        for i in range(1, 2 * max_ops + 1):
            catalans.append((4 * i - 2) * catalans[i - 1] // (i + 1))

        # enumerate possible trees
        D = []
        for e in range(max_ops + 1):  # number of empty nodes
            s = []
            for n in range(2 * max_ops - e + 1):  # number of operators
                if e == 0:
                    s.append(0)
                elif e == 1:
                    s.append(catalans[n])
                else:
                    s.append(D[e - 1][n + 1] - D[e - 2][n + 1])
            D.append(s)
        return D

    def generate_ubi_dist(self, max_ops):
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
        # first generate the tranposed version of D, then transpose it
        D = []
        D.append([0] + ([self.nl ** i for i in range(1, 2 * max_ops + 1)]))
        for n in range(1, 2 * max_ops + 1):  # number of operators
            s = [0]
            for e in range(1, 2 * max_ops - n + 1):  # number of empty nodes
                s.append(
                    self.nl * s[e - 1]
                    + self.p1 * D[n - 1][e]
                    + self.p2 * D[n - 1][e + 1]
                )
            D.append(s)
        assert all(len(D[i]) >= len(D[i + 1]) for i in range(len(D) - 1))
        D = [
            [D[j][i] for j in range(len(D)) if i < len(D[j])]
            for i in range(max(len(x) for x in D))
        ]
        return D

    def sample_next_pos_ubi(self, nb_empty, nb_ops, rng):
        """
        Sample the position of the next node (unary-binary case).
        Sample a position in {0, ..., `nb_empty` - 1}, along with an arity.
        """
        assert nb_empty > 0
        assert nb_ops > 0
        probs = []
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p1 * self.ubi_dist[nb_empty - i][nb_ops - 1]
            )
        for i in range(nb_empty):
            probs.append(
                (self.nl ** i) * self.p2 * self.ubi_dist[nb_empty - i + 1][nb_ops - 1]
            )
        probs = [p / self.ubi_dist[nb_empty][nb_ops] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        e = rng.choice(2 * nb_empty, p=probs)
        arity = 1 if e < nb_empty else 2
        e = e % nb_empty
        return e, arity

    def get_leaf(self, rng) -> List[str]:
        """
        Generate a leaf.

        Use DLSM implementation.

        Options: CONST placeholder (sympy symbol), variable
        """
        leaf_type = rng.choice(2, p=self.leaf_probs)
        if leaf_type == 0:  # variable
            return [list(self.variables.keys())[rng.randint(self.n_variables)]]
        else:  # constant placeholder
            return ['c']  # self.const_placeholder

    def _generate_expr(self, nb_total_ops: int, rng) -> List[str]:
        """
        Create a tree with exactly `nb_total_ops` operators.

        The tree is a list of operators and leaves in prefix notation (each one a str).

        TODO: Add requires_x flag

        Steps:
        1. Create tree stack (prefix notation, with None as leaves)
            Example iterations:
            ['sin', None]
            ['sin', 'add', None, None]
            ['sin', 'add', None, 'add', None, None]

        2. Create leaves list (same size as number of empty leaves)
            Example: 
            leaves = [['x_1'], ['x_1'], ['x_1']]
        3. Insert leaves into tree
            Example:
            ['sin', 'add', 'x_1', 'add', 'x_1', 'x_1']
        """
        stack = [None]
        nb_empty = 1  # number of empty nodes
        l_leaves = 0  # left leaves - None states reserved for leaves
        t_leaves = 1  # total number of leaves (just used for sanity check)

        # 1. Create tree
        for nb_ops in range(nb_total_ops, 0, -1):

            # next operator, arity and position
            skipped, arity = self.sample_next_pos_ubi(nb_empty, nb_ops, rng)
            if arity == 1:
                op = rng.choice(self.una_ops, p=self.una_ops_probs)
            else:
                op = rng.choice(self.bin_ops, p=self.bin_ops_probs)

            nb_empty += (
                self.OPERATORS[op] - 1 - skipped
            )  # created empty nodes - skipped future leaves
            t_leaves += self.OPERATORS[op] - 1  # update number of total leaves
            l_leaves += skipped  # update number of left leaves

            # update tree
            pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
            stack = (
                stack[:pos]
                + [op]
                + [None for _ in range(self.OPERATORS[op])]
                + stack[pos + 1 :]
            )

        # sanity check
        assert len([1 for v in stack if v in self.all_ops]) == nb_total_ops
        assert len([1 for v in stack if v is None]) == t_leaves

        # 2. Create leaves
        leaves = [self.get_leaf(rng) for _ in range(t_leaves)]
        rng.shuffle(leaves)

        # 3. Insert leaves into tree
        for pos in range(len(stack) - 1, -1, -1):
            if stack[pos] is None:
                stack = stack[:pos] + leaves.pop() + stack[pos + 1 :]
        assert len(leaves) == 0
        return stack
    
    @classmethod
    def is_leaf(cls, token: str) -> bool:
        """
        A leaf node does not need paranthesis around it.

        TODO: Check if this are all the cases that need parenthesis.
        """
        return not any(op in token for op in ['*', '+', '-', '/', '**'])

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.

        FIXME: Remove unnecessary parenthesis in case argument is already enclosed by unary operator.
        """

        UNARY_OP_TOKENS  = ["sign", "sqrt", "exp", "ln", "sin", "cos", "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot", "asec", "acsc", "sinh", "cosh", "tanh", "coth", "sech", "csch", "asinh", "acosh", "atanh", "acoth", "asech", "acsch"]
        
        if token in self.una_ops:  # FIXME: Check parenthesis
            return f"{token}({args[0]})"
        else:
            args = ['(' + arg + ')' if not self.is_leaf(arg) else arg for arg in args]

            if token == "add":
                return f"{args[0]}+{args[1]}"
            elif token == "sub":
                return f"{args[0]}- {args[1]}"  # Important to have space between '-' and the next argument (to discern from negative constant)
            elif token == "mul":
                return f"{args[0]}*{args[1]}"
            elif token == "div":
                return f"{args[0]}/{args[1]}"
            elif token == "pow":
                return f"{args[0]}**{args[1]}"
            elif token == "rac":
                return f"{args[0]}**(1/{args[1]})"
            elif token == "abs":
                return f"Abs({args[0]})"
            elif token == "inv":
                return f"1/{args[0]}"
            elif token == "pow2":
                return f"{args[0]}**2"
            elif token == "pow3":
                return f"{args[0]}**3"
            elif token == "pow4":
                return f"{args[0]}**4"
            elif token == "pow5":
                return f"{args[0]}**5"
            else:
                raise InvalidPrefixExpression(
                    f"Unknown token in prefix expression: {token}, with arguments {args}"
                )
        # elif token == "derivative":
        #     return f"Derivative({args[0]},{args[1]})"
        # elif token == "f":
        #     return f"f({args[0]})"
        # elif token == "g":
        #     return f"g({args[0]},{args[1]})"
        # elif token == "h":
        #     return f"h({args[0]},{args[1]},{args[2]})"
        # elif token.startswith("INT"):
        #     return f"{token[-1]}{args[0]}"

    @classmethod
    def add_identifier_constants(cls, expr_list):
        curr = Counter()
        curr["cm"] = 0
        curr["ca"] = 0
        for i in range(len(expr_list)):
            if expr_list[i] == "cm":
                expr_list[i] = "cm_{}".format(curr["cm"])
                curr["cm"] += 1
            if expr_list[i] == "ca":
                expr_list[i] = "ca_{}".format(curr["ca"])
                curr["ca"] += 1
        return expr_list

    def return_constants(self,expr_list):
        #string = "".join(expr_list)
        curr = Counter()
        curr["cm"] = [x for x in expr_list if x[:3] == "cm_"]
        curr["ca"] = [x for x in expr_list if x[:3] == "ca_"]
        return curr
            


    # def sign(self, x):
    #     return ("", "-")[x < 0]
    def _prefix_to_infix(self, expr: List[str], coefficients=None, variables=None):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        # print(f'{expr = }')
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):  # Arity
                i1, l1 = self._prefix_to_infix(l1,  coefficients=coefficients, variables=variables)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in coefficients:
            return "{" + t + "}", expr[1:]
        elif (
            t in variables
            or t in self.constants
            or t == "I"
        ):
            # print(f'{t = }')
            return t, expr[1:]
        else: #INT
            val = expr[0]
            # print(f'{val = }; {str(val) = }')
            return str(val), expr[1:]

    def _prefix_to_edges(self, expr):
        t = expr[0][1]
        edges = []
        li = expr[1:]
        if t in self.operators:
            args = []
            for _ in range(self.OPERATORS[t]):
                new_edge = [expr[0][0], li[0][0]]
                edges.append(new_edge)
                inner_edges, li = self._prefix_to_edges(li)
                edges.extend(inner_edges)
        return edges, li

    def prefix_to_infix(self, expr: List[str], coefficients=None, variables=None):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr, coefficients=coefficients, variables=variables)
        if len(r) > 0:
            raise InvalidPrefixExpression(
                f'Incorrect prefix expression "{expr}". "{r}" was not parsed.'
            )
        # print(f'{p = }')
        return p  # f"({p})"

    @classmethod
    def rewrite_sympy_expr(cfg, expr, rewrite_functions=None):
        """
        Rewrite a SymPy expression.
        """
        expr_rw = expr
        for f in rewrite_functions:
            if f == "expand":
                expr_rw = sp.expand(expr_rw)
            elif f == "factor":
                expr_rw = sp.factor(expr_rw)
            elif f == "expand_log":
                expr_rw = sp.expand_log(expr_rw, force=True)
            elif f == "logcombine":
                expr_rw = sp.logcombine(expr_rw, force=True)
            elif f == "powsimp":
                expr_rw = sp.powsimp(expr_rw, force=True)
            elif f == "simplify":
                expr_rw = simplify(expr_rw, seconds=1)
        return expr_rw

    @classmethod
    def infix_to_sympy(cls, infix, variables, rewrite_functions, no_rewrite=False):
        """
        Convert an infix expression to SymPy.
        """
        try:
            expr = parse_expr(infix, evaluate=True, local_dict=cls.return_local_dict(variables))
        except ValueError:
            raise ImAccomulationBounds
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression
        if not no_rewrite:
            expr = cls.rewrite_sympy_expr(expr, rewrite_functions)
        return expr

    @classmethod
    def _sympy_to_prefix(cls, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)
    
        assert (
            (op == "add" or op == "mul")
            and (n_args >= 2)
            or (op != "add" and op != "mul")
            and (1 <= n_args <= 2)
        )

        # square root
        if (
            op == "pow"
            and isinstance(expr.args[1], sp.Rational)
            and expr.args[1].p == 1
            and expr.args[1].q == 2
        ):
            return ["sqrt"] + Generator.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += Generator.sympy_to_prefix(expr.args[i])

        return parse_list

    @classmethod
    def sympy_to_prefix(cls,expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return [str(expr)]  # self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Float):
            return [str(expr)]  # self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return (
                ["div"] + [str(expr.p)] + [str(expr.q)]
            )  # self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ["E"]
        elif expr == sp.pi:
            return ["pi"]
        elif expr == sp.I:
            return ["I"]
        # SymPy operator
        for op_type, op_name in cls.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return cls._sympy_to_prefix(op_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")

    def replace_const_placeholder(self, expr: List[str], const_min: float, const_max: float) -> List[str]:
        """
        Replace the const placeholder with the actual constant.
        """
        def sample_const(token: str):
            if token == 'c':  # Replace const placeholder with a random constant
                return Uniform(const_min, const_max).sample().item()
            else:  # Leave other tokens untouched
                return token

        return list(map(sample_const, expr))

    def get_minmax_constants(self, f_prefix_list: List[str]) -> Tuple[float, float]:
        """
        Get the minimum and maximum constants in the expression.
        """
        consts = []
        for token in f_prefix_list:
            try:
                consts.append(float(token))
            except ValueError:
                pass
        return (None, None) if not consts else (min(consts), max(consts))

    def generate_equation(self, rng, const_min: float, const_max: float, simplify=False, check_const_range=False) -> List[str]:
        """
        Generates a single equation skeleton as a prefix list.
        """
        nb_ops = rng.randint(self.min_ops, self.max_ops + 1)

        f_prefix_list = self._generate_expr(nb_ops, rng)
        f_prefix_list = self.replace_const_placeholder(f_prefix_list, const_min, const_max)
        f_infix_str = self.prefix_to_infix(f_prefix_list, coefficients=self.coefficients, variables=self.variables)
        if simplify:
            # Conversion to sympy expression automatically simplifies expression
            f_sp_expr = self.infix_to_sympy(f_infix_str, self.variables, self.rewrite_functions)
            f_prefix_list = self.sympy_to_prefix(f_sp_expr)
            if check_const_range:
                min_, max_ = self.get_minmax_constants(f_prefix_list)
                if min_ is not None and (min_ < const_min or max_ > const_max):
                    return ''
            f_infix_str = self.prefix_to_infix(f_prefix_list, coefficients=self.coefficients, variables=self.variables)

        # skip too long sequences
        if len(f_prefix_list) + 2 > self.max_len:
            # raise ValueErrorExpression("Sequence longer than max length")
            return ''
            #return None, "Sequence longer than max length"

        # skip when the number of operators is too far from expected
        # real_nb_ops = sum(1 if op in self.OPERATORS else 0 for op in f_prefix_list)
        # if real_nb_ops < nb_ops / 2:
        #     print(f'{real_nb_ops = }; {f_prefix_list = }')
        #     raise ValueErrorExpression("Too few operators")
            #return None, "Too many operators"

        return f_infix_str



   





