"""
This file defines the EFO language and connect it to graph ML representation DNF.
In the future, it can be also used in CNF.

When consider the EFO language, we need to consider the following assumptions:
1. The language is first-order logic.
2. The formula is already in prenex normal form.

There are three parts of the code
1. Language
2. Partial Interpretation
3. PyG graph

## Part 1. Language
The language is, in general, disentangled from the structure.
In this part, we handle the string presentation and python object
presentation of the language.
- Python objects and classes:
    - Lobject
        - Term (can be partially interpreted)
        - Formula
            - Atomic (the relation / predicate can be partially interpreted)
            - Connective
                - Negation
                - Conjunction
                - Disjunction
- String presentation, to and from:
    - Formula.lstr(): get lstr
    - parse_lstr_to_lformula(): parse lstr

## Part 1.2 Partial Interpretation

The partial interpretation is the interpretation of the constants and relations.
In the partial interpretation, we only interpret the constants and relations.
This is managed in a dict of stuff.

This is implemented in all Lobject classes.
- Lobject.set_interpretation()

## Part 2. Query

The query is the main object that connects the language to the graph.
There are two classes that we need to implement
1. EFO query
2. Conjunctive query

The first one is the main object that we need to implement.
The second one put does not allow the disjunction. Therefore, conjunctive query
is a special case and can be converted into PyG graphs.

## Part 3. Graph

"""

from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, NewType, Tuple

import torch
from torch_geometric.data import Data

LStr = NewType("LStr", str)


############################################
# Language
############################################


class Lobject:
    """
    Lobject is the abstract class for all Language objects.

    Here is the inheritance structure of the language objects:
    Lobject
        - Term
        - Formula
            - Atomic
            - Connective
                - Negation
                - Conjunction
                - Disjunction
                - ...
    """

    op = "ABSTRACTLOBJECT"
    kvcache = {}  # a general kv-cache for storage of all levels.

    @abstractmethod
    def lstr(self) -> LStr:
        pass

    @abstractmethod
    def parse(self, lstr: LStr) -> "Lobject":
        pass

    @abstractmethod
    def append_interpretation(self, interp_dict: Dict[str, int]):
        pass

    def __repr__(self):
        return self.lstr()


class Term(Lobject):
    CONSTANT = 0
    EXISTENTIAL = 1
    FREE = 2

    op = "term"

    def __init__(self, label, name):
        self.name, self.label = name, label
        self.parent_predicate = None

        # the map that maps the term into specific entity.
        # it is a list of entity ids.
        self.interp_list = []

    def lstr(self) -> str:
        return self.name

    def append_interpretation(self, interp_dict: Dict[str, int]):
        if self.label == Term.CONSTANT:
            val = interp_dict[self.name]
            if val is not None:
                self.interp_list.append(val)
                interp_dict[self.name] = None
                # once we append the interpretation, we remove the key from the
                # dictionary. the name is the unique key for the term.
                # also, we allow one term name to be used multiple times in the
                # same query.


class Formula(Lobject):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_atomics(self) -> Dict[str, "Atomic"]:
        pass


class Atomic(Formula):
    op = "pred"

    def __init__(self, relation_name: str, head: Term, tail: Term) -> None:
        self.relation_name = relation_name
        self.head = head
        self.tail = tail

        self.negated = False

        self.head.parent_predicate = self
        self.tail.parent_predicate = self

        self.interp_list = []

    def lstr(self):
        lstr = f"{self.relation_name}({self.head.name},{self.tail.name})"
        return lstr

    def get_atomics(self) -> Dict[str, "Atomic"]:
        ans = {self.lstr(): self}
        return ans

    def append_interpretation(self, interp_dict: Dict[str, int]):
        val = interp_dict[self.relation_name]

        self.head.append_interpretation(interp_dict)
        self.tail.append_interpretation(interp_dict)

        if val is not None:
            self.interp_list.append(val)
            interp_dict[self.relation_name] = None


class Connective(Formula): ...


class Negation(Connective):
    op = "neg"

    def __init__(self, formula: Formula) -> None:
        self.formula = formula

    def lstr(self) -> str:
        lstr = f"!({self.formula.lstr()})"
        return lstr

    def get_atomics(self) -> Dict[str, "Atomic"]:
        ans = {}
        ans.update(self.formula.get_atomics())
        return ans

    def append_interpretation(self, interp_dict):
        self.formula.append_interpretation(interp_dict)


class Conjunction(Connective):
    op = "conj"

    def __init__(self, formulas: List[Formula]) -> None:
        self.formulas = formulas

    def lstr(self):
        lstr = "&".join(f"({f.lstr()})" for f in self.formulas)
        return lstr

    def get_atomics(self) -> Dict[str, "Atomic"]:
        ans = {}
        for f in self.formulas:
            ans.update(f.get_atomics())
        return ans

    def append_interpretation(self, interp_dict):
        for f in self.formulas:
            f.append_interpretation(interp_dict)


class Disjunction(Connective):
    op = "disj"

    def __init__(self, formulas: List[Formula]) -> None:
        self.formulas = formulas

    def lstr(self):
        lstr = "|".join(f"({f.lstr()})" for f in self.formulas)
        return lstr

    def get_atomics(self) -> Dict[str, "Atomic"]:
        ans = {}
        for f in self.formulas:
            ans.update(f.get_atomics())
        return ans

    def append_interpretation(self, interp_dict):
        for f in self.formulas:
            f.append_interpretation(interp_dict)


def remove_outmost_backets(lstr: str):
    if not (lstr[0] == "(" and lstr[-1] == ")"):
        return lstr

    bracket_stack = []
    for i, c in enumerate(lstr):
        if c == "(":
            bracket_stack.append(i)
        elif c == ")":
            left_bracket_index = bracket_stack.pop(-1)

    assert len(bracket_stack) == 0
    if left_bracket_index == 0:
        return lstr[1:-1]
    else:
        return lstr


def remove_brackets(lstr: str):
    new_lstr = remove_outmost_backets(lstr)
    while new_lstr != lstr:
        lstr = new_lstr
        new_lstr = remove_outmost_backets(lstr)
    return lstr


def map_term_name_to_type(name: str):
    c = name[0]

    lookup_dict = {
        "e": Term.EXISTENTIAL,
        "f": Term.FREE,
        "s": Term.CONSTANT,
    }

    if c in lookup_dict:
        return lookup_dict[c]
    else:
        raise ValueError(f"Unknown term type {c}")


def identify_top_binary_operator(lstr: str):
    """
    identify the top-level binary operator
    """
    _lstr = remove_brackets(lstr)
    bracket_stack = []
    for i, c in enumerate(_lstr):
        if c == "(":
            bracket_stack.append(i)
        elif c == ")":
            bracket_stack.pop(-1)
        elif c in "&|" and len(bracket_stack) == 0:
            return c, i
    return None, -1


def parse_lstr_to_lformula(lstr: str) -> Formula:
    term_registry = {}
    atomic_registry = {}

    def _parse_term(term_name):
        if term_name not in term_registry:
            term = Term(label=map_term_name_to_type(term_name), name=term_name)
            term_registry[term_name] = term
        return term_registry[term_name]

    def _parse_atomic(relation_name, term1_name, term2_name):
        term1 = _parse_term(term1_name)
        term2 = _parse_term(term2_name)
        _atomic = Atomic(relation_name=relation_name, head=term1, tail=term2)
        alstr = _atomic.lstr()
        if alstr not in atomic_registry:
            atomic_registry[_atomic.lstr()] = _atomic
        return atomic_registry[_atomic.lstr()]

    def _parse_lstr_to_lformula(lstr: str) -> Formula:
        """
        Parse the string (lstr) to a logical formula object.

        Args:
            lstr (str): The logical string to parse.

        Returns:
            Formula: The parsed logical formula object.

        Raises:
            ValueError: If the input string is invalid or cannot be parsed.
        """
        if not lstr:
            raise ValueError("Input string is empty")

        _lstr = remove_brackets(lstr)

        if not _lstr:
            raise ValueError("Input string is invalid after removing brackets")

        binary_operator, binary_operator_index = identify_top_binary_operator(
            _lstr
        )

        if binary_operator_index >= 0:
            left_lstr = _lstr[:binary_operator_index]
            right_lstr = _lstr[binary_operator_index + 1 :]
            left_formula = _parse_lstr_to_lformula(left_lstr)
            right_formula = _parse_lstr_to_lformula(right_lstr)
            if binary_operator == "&":
                return Conjunction(formulas=[left_formula, right_formula])
            elif binary_operator == "|":
                return Disjunction(formulas=[left_formula, right_formula])
            else:
                raise ValueError(f"Unknown binary operator: {binary_operator}")
        # Identify top-level operator
        elif _lstr[0] == "!":
            sub_lstr = _lstr[1:]
            sub_formula = _parse_lstr_to_lformula(sub_lstr)
            if sub_formula.op == "pred":
                return Negation(formula=sub_formula)
            else:
                raise ValueError("Invalid negation format", _lstr)
        else:  # Parse predicate
            if _lstr[-1] != ")":
                raise ValueError("Predicate string must end with ')'")

        try:
            relation_name, right_lstr = _lstr.split("(")
            right_lstr = right_lstr[:-1]
            term1_name, term2_name = right_lstr.split(",")
        except ValueError:
            raise ValueError("Invalid predicate format {}".format(_lstr))

        predicate = _parse_atomic(relation_name, term1_name, term2_name)
        return predicate

    return _parse_lstr_to_lformula(lstr)


def push_negations(formula: Formula):
    """
    Push the negations down to the atomic formula.
    """
    fop = formula.op
    if fop == "pred":
        return formula
    elif fop in ["conj", "disj"]:
        return type(formula)([push_negations(f) for f in formula.formulas])
    elif fop == "neg":
        sub = formula.formula
        sop = sub.op
        if sop == "pred":
            return formula
        elif sop in ["conj", "disj"]:
            new_f = [push_negations(Negation(f)) for f in sub.formulas]
            return Disjunction(new_f) if sop == "conj" else Conjunction(new_f)
        elif sop == "neg":
            return push_negations(sub.formula)
        else:
            raise ValueError("Not supported sub-formula type {}".format(sop))
    else:
        raise ValueError("Not supported formula type {}".format(fop))


def push_conjunctions(formula: Formula):
    """
    Push conjunctions down to the atomic formulas.

    This function follows the assumption
    1. All disjunction and conjunction are binary.
    """
    if formula.op in ("pred", "neg"):
        return formula

    if formula.op in ("conj", "disj"):
        left, right = (push_conjunctions(f) for f in formula.formulas)

        if formula.op == "conj":
            # Only distribute if either side is a disjunction
            if left.op == "disj":
                la, lb = left.formulas
                return Disjunction(
                    [
                        push_conjunctions(Conjunction([la, right])),
                        push_conjunctions(Conjunction([lb, right])),
                    ]
                )
            if right.op == "disj":
                ra, rb = right.formulas
                return Disjunction(
                    [
                        push_conjunctions(Conjunction([left, ra])),
                        push_conjunctions(Conjunction([left, rb])),
                    ]
                )
            return Conjunction([left, right])

        # formula.op == "disj"
        return Disjunction([left, right])

    raise ValueError("Not supported formula type {}".format(formula.op))


def retrieve_conjunctive_queries_from_dnf(formula: Formula):
    """
    Retrieve the conjunctive queries from the formula.
    """
    if formula.op == "disj":
        conjunctive_query_list = []
        for f in formula.formulas:
            conjunctive_query_list.extend(
                retrieve_conjunctive_queries_from_dnf(f)
            )
        return conjunctive_query_list
    elif formula.op in ["conj", "pred", "neg"]:
        return [formula]
    else:
        raise ValueError("Not supported formula type {}".format(formula.op))


def retrieve_atomic_queries_from_conjunctive_queries(formula: Formula):
    """
    Retrieve the atomic queries from the formula.
    """
    if formula.op == "conj":
        atomic_query_list = []
        for f in formula.formulas:
            atomic_query_list.extend(
                retrieve_atomic_queries_from_conjunctive_queries(f)
            )
        return atomic_query_list
    elif formula.op in ["pred", "neg"]:
        return [formula]
    else:
        raise ValueError("Not supported formula type {}".format(formula.op))


def transform_to_dnf(formula: Formula):
    """
    Transform the disjunction to the top level.

    The goal is that:
    1. The disjunction only appears in the top level.
    2. The negation only appears in the atomic formula.
    3. Additionally, the top-level disjunction contains all formulaes.
    """

    # step 1: push the negations
    formula = push_negations(formula)

    # step 2: push the disjunctions
    formula = push_conjunctions(formula)

    # step 3: make the disjunction and conjunction multi-ary
    if formula.op == "disj":
        conj_list = retrieve_conjunctive_queries_from_dnf(formula)
    else:
        conj_list = [formula]

    flatten_conj_list = []
    for conj in conj_list:
        flatten_conj_list.append(
            Conjunction(retrieve_atomic_queries_from_conjunctive_queries(conj))
        )

    dnf = Disjunction(flatten_conj_list)
    return dnf


class EFOQuery:
    """
    The EFO formula with only one variable.

    self.formula is parsed from the formula and provide the operator tree for
        evaluation
    self.atomic_dict stores each predicates by its name, which are edges
    self.term_dict stores each symbol by its name

    self.easy_answer_list list for easy answers
    self.hard_answer_list list for hard answers

    each 'answer' is a dict whose keys are the variable and values are the
    list of possible answers

    There are three functions

    1. parse the query language and organize it into its internal presentation
        - internal presentation: self.atomic_dict, self.term_dict
    2. add instantiation to the internal presentation (entities and relations)
        - append_qaa_instance
    3. convert the internal presentation to PyG graph
        - get_pyg_graph_list_by_sub_conjunctive_queries
    """

    def __init__(self, formula: Formula) -> None:
        # this is the formula and we don't change it!
        self.formula: Formula = formula
        # we don't maintain the additional version of formula
        # self.dnf_formula: Formula = transform_to_dnf(formula)

        # easy and hard answers
        self.easy_answer_list = []
        self.hard_answer_list = []

        # calculate the term and atomic dict
        self.term_dict: Dict[LStr, Term] = {}
        self.atomic_dict: Dict[LStr, Atomic] = {}
        self.term_name2atomic_name_list = defaultdict(list)

        for alstr, atomic in self.formula.get_atomics().items():
            self.atomic_dict[alstr] = atomic
            self.term_name2atomic_name_list[atomic.head.name].append(alstr)
            self.term_name2atomic_name_list[atomic.tail.name].append(alstr)

            if atomic.head.name not in self.term_dict:
                self.term_dict[atomic.head.name] = atomic.head
            else:
                assert id(self.term_dict[atomic.head.name]) == id(atomic.head)

            if atomic.tail.name not in self.term_dict:
                self.term_dict[atomic.tail.name] = atomic.tail
            else:
                assert id(self.term_dict[atomic.tail.name]) == id(atomic.tail)

    @classmethod
    def from_lstr(cls, lstr: LStr) -> "EFOQuery":
        return cls(formula=parse_lstr_to_lformula(lstr))

    def append_qaa_instance(self, append_dict, easy_answers, hard_answers):
        """
        Append a query-easy_answer-hard_answer instance to the EFOQuery object
        """
        # prepare the easy and hard answers
        self.easy_answer_list.append(easy_answers)
        self.hard_answer_list.append(hard_answers)

        # append the interpretation
        # Note:
        # Because we don't create or distroy new Terms and Atomic in
        # transformation of DNF, we can update all python objects of Term and
        # Atomic directly from the self.formula. All changes also applies to
        # self.dnf_formula.

        _copy_append_dict = append_dict.copy()

        self.formula.append_interpretation(_copy_append_dict)

        assert len(self.easy_answer_list) == self.num_instances
        assert len(self.easy_answer_list) == len(self.hard_answer_list)

    @property
    def free_variable_dict(self):
        return {
            k: v for k, v in self.term_dict.items() if v.label == Term.FREE
        }

    @property
    def existential_variable_dict(self):
        return {
            k: v
            for k, v in self.term_dict.items()
            if v.label == Term.EXISTENTIAL
        }

    @property
    def symbol_dict(self):
        return {
            k: v for k, v in self.term_dict.items() if v.label == Term.CONSTANT
        }

    @property
    def is_sentence(self):
        """
        Determine the state of the formula
        A formula is sentence when all variables are quantified
        """
        return len(self.free_variable_dict) == 0

    def lstr(self):
        return self.formula.lstr()

    def get_pyg_graph_list_by_sub_conjunctive_queries(
        self,
    ) -> Tuple[List[Data]]:
        """
        Get the PyG graph from the EFOQuery
        """
        conj_list = transform_to_dnf(self.formula).formulas
        conj_query_list = []
        for conj in conj_list:
            _pyg_list_per_conj = ConjunctiveQuery(conj).get_pyg_graph_list()
            for g in _pyg_list_per_conj:
                g.lstr = self.lstr()
            conj_query_list.append(_pyg_list_per_conj)
        return conj_query_list

    @property
    def num_instances(self):
        num_instances = None
        for k in self.atomic_dict:
            if num_instances is None:
                num_instances = len(self.atomic_dict[k].interp_list)
            assert num_instances == len(self.atomic_dict[k].interp_list), k

        for k in self.symbol_dict:
            assert num_instances == len(self.term_dict[k].interp_list), k

        return num_instances

    @property
    def num_predicates(self):
        return self.formula.num_atomics

    @property
    def num_variables(self):
        return len(self.existential_variable_dict) + len(
            self.free_variable_dict
        )


class ConjunctiveQuery(EFOQuery):
    """
    The conjunctive query is directly connected to the graph presentation.
    """

    def __init__(self, formula: Formula) -> None:
        super().__init__(formula)

        assert self.formula.op == "conj"
        assert all(f.op in ["pred", "neg"] for f in self.formula.formulas)

        # get negation status of all atomics
        self.negative_alstr = []
        for f in self.formula.formulas:
            if f.op == "neg":
                assert f.formula.op == "pred"
                self.negative_alstr.append(f.formula.lstr())

    def get_pyg_graph_list(self):
        """
        Get the PyG graph from the conjunctive query.
        The output forms a list because there are multiple instances.

        four piece of information are stored in the PyG graph
        say g is a PyG graph

        g.x: the node feature, [#node, 2],
            - the first column is the entity id
            - the second column is the type of the entity,
                - 0 for constant
                - 1 for existential
                - 2 for free
        g.edge_index: the edge index, [#edge, 2]
            - the first column is the head node id
            - the second column is the tail node id
        g.edge_attr: the edge attribute, [#edge, 2]
            - the first column is the relation id
            - the second column is the negative indicator 1 for negated.
        g.num_vars: the number of variables in the query
        """
        pyg_data_list = []
        # iterate samples
        for _instance_index in range(self.num_instances):
            edge_index = []
            edge_attr = []
            x = []

            # node_id
            term_name_to_node_id = {}
            node_id_to_term = {}
            for i, (term_name, term) in enumerate(self.term_dict.items()):
                term_name_to_node_id[term_name] = i
                node_id_to_term[i] = term
                if term.label == Term.CONSTANT:
                    x.append([term.interp_list[_instance_index], 0])
                elif term.label == Term.EXISTENTIAL:
                    x.append([0, 1])
                elif term.label == Term.FREE:
                    x.append([0, 2])
                else:
                    raise ValueError(f"Unknown term label {term.label}")

            for alstr, atomic in self.atomic_dict.items():
                head_name = atomic.head.name
                tail_name = atomic.tail.name

                negative_indicator = alstr in self.negative_alstr

                head_id = term_name_to_node_id[head_name]
                tail_id = term_name_to_node_id[tail_name]

                # the graph here does not contain the reverse edge.
                # the edge is added in the dataloader
                edge_index.append([head_id, tail_id])
                edge_attr.append(
                    [atomic.interp_list[_instance_index], negative_indicator]
                )

            x = torch.tensor(x, dtype=torch.long)
            edge_index = (
                torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            )
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
            data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_obj.num_vars = torch.sum(x[:, 1] > 0)
            # data_obj.lstr = self.lstr()
            pyg_data_list.append(data_obj)
        return pyg_data_list

    def get_bfs_variable_ordering(self, source_var_name="f"):
        """
        get variable ordering by a topological sort
        """
        visited_vars = set(source_var_name)
        var_name_levels = [[(source_var_name, 0)]]
        while True:
            for var_name, order in var_name_levels[-1]:
                next_var_name_level = []
                for atomic_name in self.term_name2atomic_name_list[var_name]:
                    atomic = self.atomic_dict[atomic_name]
                    for term in atomic.get_terms():
                        if term.label == Term.CONSTANT:
                            continue

                        if term.name not in visited_vars:
                            visited_vars.add(term.name)
                        else:
                            continue

                        next_var_name_level.append((term.name, order + 1))

            if len(next_var_name_level) == 0:
                break
            else:
                var_name_levels.append(next_var_name_level)

        return var_name_levels
