# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import logging
import os.path as osp
import re
import threading
from copy import deepcopy
from itertools import chain
from subprocess import PIPE, STDOUT, Popen, TimeoutExpired
from typing import Callable, Optional, Union

import networkx as nx
import numpy as np
import pydot
import pyparsing as pp

import entangle.ops as tgops
import entangle.sgraph.sexpr as tgsexpr
from entangle.sgraph.sexpr import SExpr
from entangle.sgraph.sgraph import SGraph
from entangle.utils import ENODE_SPLIT
from entangle.utils.print_utils import BYELLOW, RST, print_ft


class CannotFindPostconditions(Exception):
    pass


ELEMENT = (
    pp.Word(pp.alphanums + "_@.?-")
    | pp.Word("+")
    | pp.Word("-")
    | pp.Word("*")
    | pp.Word("/")
    | pp.Word("=")
    | pp.Word("!=")
    | pp.Word(">")
    | pp.Word("<")
    | pp.Word(">=")
    | pp.Word("<=")
)
RECURSIVE_EXPRESSION = pp.Forward()
RECURSIVE_EXPRESSION <<= (
    pp.Literal("(").suppress()
    + pp.Group(pp.OneOrMore(ELEMENT ^ RECURSIVE_EXPRESSION))
    + pp.Literal(")").suppress()
) | ELEMENT
EXPRESSION = ELEMENT ^ RECURSIVE_EXPRESSION

pyparsing_lock = threading.Lock()


def parse_expression_with_lock(s: str) -> pp.ParseResults:
    with pyparsing_lock:
        return EXPRESSION.parse_string(s)


def get_names_from_expr_str(s: str, inputs=True, scalars=True) -> set[str]:
    if inputs:
        input_names = list(re.findall(r"\(input [-\.\w\d_@]+\)", s))
        input_names = {n[7:-1].strip(" ").split("@")[0] for n in input_names}
    if scalars:
        scalar_names = {t[1] for t in re.findall(r"(\(| )(Sym[\w\d_@]+)", s)}
    if inputs and not scalars:
        return input_names
    elif not inputs and scalars:
        return scalar_names
    elif inputs and scalars:
        return input_names | scalar_names
    else:
        raise ValueError("Either inputs or scalars should be True.")


def get_input_names_from_expr_str(s: str) -> set[str]:
    return get_names_from_expr_str(s, inputs=True, scalars=False)


def get_scalar_names_from_expr_str(s: str) -> set[str]:
    if "(" not in s and ")" not in s and s.startswith("Sym"):
        return {s}
    else:
        return get_names_from_expr_str(s, inputs=False, scalars=True)


class EGraph:
    """
    `EGraph` is a list of `SExpr`s that represent pre/post-conditions
    in the form of e-graph.

    We need an `sgraph` and an `eclasses`: list[list[SExpr]] to represent what are
    equivalent.

    FIXME: There can be duplication under current design. We should think a
    better representation for this.
    """

    def __init__(self, sgraph: SGraph, eclasses: list[list[SExpr]]):
        self.sgraph = sgraph
        self.eclasses = eclasses


class SExprECondition:
    def __init__(
        self,
        inputs: list[SExpr],
        eclasses: list[list[SExpr]],
        intermediate_sexprs: list[SExpr] = None,
    ):
        if any(s.op.scalar for s in inputs):
            assert all(
                s.op.scalar for s in inputs
            ), "All inputs should be scalar if any is scalar."
            self.all_scalar = True
        else:
            self.all_scalar = False
        for eclass in eclasses:
            assert type(eclass) in (list, tuple), "eclass should be a list or tuple."
        self.inputs = inputs
        self.eclasses = eclasses
        self.intermediate_sexprs = intermediate_sexprs

    def __str__(self):
        return "\n".join(
            [ENODE_SPLIT.join(map(str, eclass)) for eclass in self.eclasses]
        )

    @staticmethod
    def just_map(
        origin: SExpr | str,
        targets: list[SExpr | str],
        placeholder: Callable[[str], SExpr] = None,
    ) -> SExpr:
        sexprs = [origin, *targets]
        if placeholder is not None:
            sexprs = [
                sexpr if type(sexpr) is SExpr else placeholder(sexpr)
                for sexpr in sexprs
            ]
        return SExprECondition(inputs=sexprs, eclasses=[])

    @staticmethod
    def all_eq(
        origin: SExpr | str,
        targets: list[SExpr | str],
        placeholder: Callable[[str], SExpr] = None,
    ) -> SExpr:
        sexprs = [origin, *targets]
        if placeholder is not None:
            sexprs = [
                sexpr if type(sexpr) is SExpr else placeholder(sexpr)
                for sexpr in sexprs
            ]
        shape = sexprs[0].shape
        assert all(
            s.shape == shape for s in sexprs
        ), f"Got inconsistent shapes {[(s.name, s.shape) for s in sexprs]}"
        return SExprECondition(inputs=sexprs, eclasses=[[*sexprs]])

    @staticmethod
    def eq_zeros(
        sexpr: Union[str, SExpr], placeholder: Callable[[str], SExpr] = None
    ) -> "SExprECondition":
        if type(sexpr) is str:
            assert placeholder is not None
            sexpr = placeholder(sexpr)
        return SExprECondition(
            inputs=[s := sexpr, z := tgsexpr.make_fill(sexpr.shape, 0)],
            eclasses=[[s, z]],
        )

    @staticmethod
    def concat_target(
        origin: SExpr | str,
        targets: list[SExpr | str],
        dim: int,
        placeholder: Callable[[str], SExpr] = None,
        label_result: bool = False,
    ) -> "SExprECondition":
        if placeholder is not None:
            origin = origin if type(origin) is SExpr else placeholder(origin)
            targets = [
                sexpr if type(sexpr) is SExpr else placeholder(sexpr)
                for sexpr in targets
            ]
        result = tgsexpr.concat(targets, dim=dim)
        if label_result:
            result = result.clone_with(name=f"EXPECTED__{origin.name}")
        return SExprECondition(
            inputs=[origin, *targets],
            eclasses=[[origin, result]],
        )

    @staticmethod
    def sum_target(
        origin: SExpr | str,
        targets: list[SExpr | str],
        placeholder: Callable[[str], SExpr] = None,
        label_result: bool = False,
    ) -> "SExprECondition":
        if placeholder is not None:
            origin = origin if type(origin) is SExpr else placeholder(origin)
            targets = [
                sexpr if type(sexpr) is SExpr else placeholder(sexpr)
                for sexpr in targets
            ]
        result = tgsexpr.sum(targets)
        if label_result:
            result = result.clone_with(name=f"EXPECTED__{origin.name}")
        return SExprECondition(
            inputs=[origin, *targets],
            eclasses=[[origin, result]],
        )

    @staticmethod
    def mesh_cond(
        origin: SExpr | str,
        targets: list[SExpr | str],
        transformations: list[str | tuple[str, int]],
        mesh_shape: list[int],
        placeholder: Callable[[str], SExpr] = None,
    ) -> "SExprECondition":
        """
        transformations: a list of
            "r": replicate
            "sum": sum targets to be origin
            ("s", dim): shard along `dim`
            ("o", offsets: list[int]): concat targets with offsets to be origin.
        """
        if placeholder is not None:
            origin = origin if type(origin) is SExpr else placeholder(origin)
            targets = [
                sexpr if type(sexpr) is SExpr else placeholder(sexpr)
                for sexpr in targets
            ]
        assert len(mesh_shape) == len(transformations)
        real_targets = targets
        targets = np.array(targets).reshape(mesh_shape)
        eclasses = []
        cur_mesh_shape = copy.deepcopy(mesh_shape)
        for trans in reversed(transformations):
            targets = targets.reshape(-1, cur_mesh_shape[-1])
            if type(trans) is str:
                if trans.lower() == "r":
                    if len(cur_mesh_shape) == 1:
                        eclasses.append([origin, *targets.flatten()])
                    else:
                        eclasses.extend([ts.tolist() for ts in targets])
                    targets = targets[:, 0]
                else:
                    assert trans.lower() == "sum"
                    targets = [
                        tgsexpr.sum(targets_in_group.tolist())
                        for targets_in_group in targets
                    ]
                    targets = np.array(targets)
                    if len(cur_mesh_shape) == 1:
                        eclasses.append([origin, *targets.flatten()])
            else:
                assert trans[0].lower() == "s"
                dim = trans[1]
                targets = [
                    tgsexpr.concat(targets_in_group.tolist(), dim=dim)
                    for targets_in_group in targets
                ]
                targets = np.array(targets)
                if len(cur_mesh_shape) == 1:
                    eclasses.append([origin, *targets.flatten()])
            cur_mesh_shape.pop(-1)

        return SExprECondition(inputs=[origin, *real_targets], eclasses=eclasses)


class ECondition:
    def __init__(
        self,
        input_names: list[str] = None,
        eclasses: list[list[str]] = None,
        used_names: list[list[str]] = None,
    ):
        """
        `eclasses` and `used_names` share the same structure.
        The element of `eclasses` is a str condition, while the element of
        `used_names` is a set of input names that are used in the corresponding
        condition.
        """
        assert eclasses is not None
        self.eclasses = eclasses
        self.pure_scalar = all(self.is_pure_scalar_eclass(e) for e in eclasses)
        if used_names is None:
            if self.pure_scalar:
                self.used_names = [
                    [get_scalar_names_from_expr_str(s) for s in eclass]
                    for eclass in eclasses
                ]
            else:
                self.used_names = [
                    [get_input_names_from_expr_str(s) for s in eclass]
                    for eclass in eclasses
                ]
        else:
            self.used_names = used_names
        if input_names is None:
            self.input_names_set = set()
            for eclass_names in self.used_names:
                for eexpr_names in eclass_names:
                    self.input_names_set.update(eexpr_names)
            self.input_names = list(self.input_names_set)
        else:
            self.input_names = input_names
            self.input_names_set = set(input_names)

        # Setup for scalar conditions.
        # `eq_eclasses` ignores those inequality conditinos for smtlib.
        if any(self.is_scalar_eclass(e) for e in self.eclasses):
            assert all(
                self.is_scalar_eclass(e) for e in self.eclasses
            ), f"All eclass should be scalar eclass if any is. Got {self.eclasses=}"
            self.all_scalar = True
            self.eq_eclasses = [
                eclass
                for eclass in eclasses
                if not (len(eclass) == 2 and eclass[1] in ("true", "false"))
            ]
        else:
            self.all_scalar = False
            self.eq_eclasses = eclasses

        if (
            len(self.eclasses) == 1
            and len(self.eclasses[0]) == 1
            and self.eclasses[0][0] == ""
        ):
            raise ValueError("Empty eclass")

    @staticmethod
    def is_pure_scalar_eclass(eclass: list[str]) -> bool:
        for s in eclass:
            if s.find("(input") != -1:
                return False
        return True

    @staticmethod
    def is_scalar_eclass(eclass: list[str]) -> bool:
        # FIXME: This is not a good judger method.
        for s in eclass:
            if re.search(r"\((\>|\<|\>=|\<=|=|!=)|true|false", s):
                return True
            elif s.startswith(SExpr.SCALAR_PREFIX):
                return True
        return False

    def split(self) -> list["ECondition"]:
        if len(self.eclasses) == 1:
            return [self]
        else:
            return [
                ECondition(None, [eclass], [used_names])
                for eclass, used_names in zip(self.eclasses, self.used_names)
            ]

    def merge(*econds: "ECondition") -> "ECondition":
        new_input_names = set.union(*[set(econd.input_names) for econd in econds])
        new_eclasses = list(chain(*[econd.eclasses for econd in econds]))
        new_used_names = list(chain(*[econd.used_names for econd in econds]))
        return ECondition(list(new_input_names), new_eclasses, new_used_names)

    def extract(
        self, input_sexpr_names: set[str], output_replacements: dict[str, str] = {}
    ) -> Union["ECondition", None]:
        new_input_names = set()
        new_eclasses = []
        for eclass, eclass_names in zip(self.eclasses, self.used_names):
            new_eclass = []
            for eexpr, eexpr_names in zip(eclass, eclass_names):
                if eexpr_names.issubset(input_sexpr_names):
                    replaced_eexpr_names = set()
                    for name in eexpr_names:
                        if name in output_replacements:
                            replacement = output_replacements[name]
                            eexpr = eexpr.replace(f"{name}@", f"{replacement}@")
                            replaced_eexpr_names.add(replacement)
                        else:
                            replaced_eexpr_names.add(name)
                    new_eclass.append(eexpr)
                    new_input_names.update(replaced_eexpr_names)
            if len(new_eclass) > 1:
                # There need to be at least two eexpr to form a condition.
                new_eclasses.append(new_eclass)
        if len(new_eclasses) == 0:
            return None
        else:
            return ECondition(list(new_input_names), new_eclasses)

    def to_smtlib_strs(self) -> list[str]:
        condition_strs = []
        for eclass in self.eclasses:
            assert len(eclass) == 2, "Only support binary condition for now."
            condition_strs.append(f"(= {eclass[0]} {eclass[1]})")
        return condition_strs

    @staticmethod
    def max_parenthesis_depth(s: str) -> int:
        depth = 0
        max_depth = 0
        for c in s:
            if c == "(":
                depth += 1
                max_depth = max(max_depth, depth)
            elif c == ")":
                depth -= 1
                assert depth >= 0
        return max_depth

    def prune_self_provable(self, ids: list[int]) -> "ECondition":
        # enode with same ids are equivalent.
        assert len(self.eclasses) == 1, "Only support single eclass for now."
        grouped_enodes: dict[int, list[tuple[str, int]]] = {}
        for idx, (eq_id, enode) in enumerate(zip(ids, self.eclasses[0])):
            if eq_id not in grouped_enodes:
                grouped_enodes[eq_id] = []
            grouped_enodes[eq_id].append((enode, idx))
        new_eclass = []
        new_cur_used_names = []
        for enodes in grouped_enodes.values():
            min_depth = None
            min_idx = None
            for enode, idx in enodes:
                depth = self.max_parenthesis_depth(enode)
                if min_depth is None or depth < min_depth:
                    min_depth = depth
                    min_enode = enode
                    min_idx = idx
                if enode.strip(" ").startswith("(input"):
                    min_enode = enode
                    min_idx = idx
                    break
            if min_idx is None:
                continue
            new_eclass.append(min_enode)
            new_cur_used_names.append(self.used_names[0][min_idx])
        new_eclasses = [new_eclass]
        new_used_names = [new_cur_used_names]
        return ECondition(None, new_eclasses, new_used_names)

    @staticmethod
    def from_formats(
        input_names: dict[str, str],
        eclasses_formats: list[list[str]],
        name_mapper: Callable[[str], str] = None,
    ) -> "ECondition":
        mapped_input_names = input_names
        if name_mapper is not None:
            mapped_input_names = {k: name_mapper(v) for k, v in input_names.items()}
        eclasses = [
            [s.format(**mapped_input_names) for s in eclass_formats]
            for eclass_formats in eclasses_formats
        ]

        return ECondition(list(input_names.values()), eclasses)

    @staticmethod
    def from_sexpr_econdition(
        sexpr_econdition: SExprECondition,
    ) -> "ECondition":
        input_names = []
        for sexpr in sexpr_econdition.inputs:
            name = sexpr.name
            assert (
                name is not None or sexpr.op == tgops.fill
            ), f"Invalid sexpr={sexpr!r}"
            if name is not None:
                input_names.append(name)
            else:
                assert sexpr.op == tgops.fill
                input_names.append(sexpr.to_egg_str())
        eclasses = [
            [sexpr if type(sexpr) is str else sexpr.to_egg_str() for sexpr in eclass]
            for eclass in sexpr_econdition.eclasses
        ]
        return ECondition(input_names, eclasses)

    @staticmethod
    def from_str(s: str, ignore_scalar=True) -> "ECondition":
        eclasses = []
        for line in s.split("\n"):
            if line.strip("\n\r\t ") == "":
                continue
            eclass = line.split(ENODE_SPLIT)
            if ignore_scalar and ECondition.is_scalar_eclass(eclass):
                print(f"Skipeed due to ignore_scalar: {eclass=}")
                continue
            eclasses.append(eclass)
        return ECondition(None, eclasses)

    def to_egg_str(self, eq_only: bool = False):
        used = self.eq_eclasses if eq_only else self.eclasses
        return "\n".join(
            [ENODE_SPLIT.join(sorted(map(str, eclass))) for eclass in used]
        )

    def __repr__(self):
        return (
            "ECondition("
            + "\n".join(
                [ENODE_SPLIT.join(map(str, eclass)) for eclass in self.eclasses]
            )
            + ")"
        )

    def __hash__(self):
        return hash(self.to_egg_str())

    def to_str_eq_only(self):
        return "\n".join(
            [ENODE_SPLIT.join(map(str, eclass)) for eclass in self.eq_eclasses]
        )


RANK_UNSET = -1
RANK_CONSTANT = -2
RANK_ACROSS = -3


class ENode:
    def __init__(self, node_id: int, eclass_id: int, children: list[int], data):
        self.node_id: int = node_id
        self.eclass_id: int = eclass_id
        self.children: list[int] = children
        self.data = data

        self.valid = False

    def is_leaf(self):
        return len(self.children) == 0

    def is_numeric(self) -> bool:
        return self.is_leaf() and str.isnumeric(self.data)

    def is_shape_literal(self) -> bool:
        # This is an ugly hack, corresponding to the lemma <inverse-expand>,
        # where we have a name shape with only one dim, making egraph regarding
        # it as a number.
        data = self.data.strip('"')
        if data.startswith("[") and data.endswith("]"):
            return True
        else:
            return False

    def is_symval(self) -> bool:
        return self.data.startswith("Sym")

    def is_tensor_name_leaf(self) -> bool:
        return self.is_leaf() and "@" in self.data

    def is_non_tensor_leaf(self) -> bool:
        if self.is_leaf() and (
            self.is_numeric() or self.is_shape_literal() or self.is_symval()
        ):
            return True
        else:
            try:
                float(self.data)
                return True
            except:
                return False

    def __repr__(self):
        arg_oks_str = (
            "" if "arg_oks" not in self.__dict__ else f", oks={str(self.arg_oks)}"
        )

        if self.is_leaf():
            return f"ENode[{self.node_id}]({self.data}, eclass={self.eclass_id}{arg_oks_str})"
        else:
            return f"ENode[{self.node_id}]({self.data}({self.children}), eclass={self.eclass_id}{arg_oks_str})"


class EClass:
    def __init__(self, eclass_id, nodes: list[int], shape: Optional[str] = None):
        self.eclass_id: int = eclass_id
        self.nodes: list[int] = nodes

        self.representative: ENode = None
        self.candidates: set[ENode] = set()
        self.valid = False

        shape = shape.strip('"')
        assert (
            shape == "" or shape.startswith("[") and shape.endswith("]")
        ), f"Invalid shape: {shape}"
        self.shape: str = shape

    def set_representative_if_not_set(self, node: ENode, egraph: EGraph):
        if self.valid:
            if node.data == self.representative.data == "input":
                node_name = egraph.get_enode(egraph.get_eclass(node.children[0]).nodes[0]).data
                repr_name = egraph.get_enode(egraph.get_eclass(self.representative.children[0]).nodes[0]).data
                if node_name < repr_name:
                    self.representative = node
            elif node.data == "input" and self.representative.data != "input":
                # If the representative is not an input, we should set it to the input node.
                self.representative = node
            else:
                if len(self.representative.children) > 0:
                    if len(node.data) == 0:
                        self.representative = node
                    else:
                        node_op = tgops.Op.get(node.data)
                        repr_op = tgops.Op.get(self.representative.data)
                        if node_op.leaf and not repr_op.leaf:
                            self.representative = node
            return
        self.representative = node
        self.valid = True

    def add_candidate(self, enode: ENode):
        assert self.valid
        self.candidates.add(enode)

    def __repr__(self):
        if self.valid:
            return f"EClass[{self.eclass_id}]({self.nodes}, valid={self.valid}, representative={self.representative.node_id})"
        else:
            return f"EClass[{self.eclass_id}]({self.nodes}, valid={self.valid})"


class EGraph:

    def __init__(
        self,
        path: str,
        y_expr: str = None,
        yi_exprs: list[str] = None,
        ignore_size=False,
        verbose=False,
        logger=None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose
        self.json_path = path
        with open(path) as f:
            self.data = json.load(f)
        self.nodes: list[ENode] = [
            ENode(node_id, eclass_id, children, data)
            for node_id, (data, eclass_id, children) in enumerate(self.data["nodes"])
        ]
        self.node_id_to_node: dict[int, ENode] = {
            node.node_id: node for node in self.nodes
        }
        self.eclasses: list[EClass] = sorted(
            [
                EClass(eclass_id, nodes, shape)
                for eclass_id, nodes, shape in self.data["classes"]
            ],
            key=lambda eclass: eclass.eclass_id,
        )
        self.eclass_id_to_eclass: dict[int, EClass] = {
            eclass.eclass_id: eclass for eclass in self.eclasses
        }

        # Setup `used_by`
        # NOTE: will also add the root `y_eclass_id` if available.
        self.used_by: dict[int, list[tuple[int, int]]] = {}
        for node in self.nodes:
            for arg_idx, eclass_id in enumerate(node.children):
                if eclass_id not in self.used_by:
                    self.used_by[eclass_id] = []
                self.used_by[eclass_id].append((node.node_id, arg_idx))

        # Setup y_expr if available
        self.y_expr = y_expr or self.data["y"]
        if self.y_expr == "":
            self.y_expr = None
        if self.y_expr is not None:
            self.y_id: int = self.lookup_expr_root(
                self.y_expr, ignore_size=ignore_size
            ).node_id
            self.y_eclass_id: int = self.get_enode(self.y_id).eclass_id
            self.y_name_id = self.get_eclass(
                self.get_enode(self.y_id).children[0]
            ).nodes[0]
            if self.y_eclass_id not in self.used_by:
                self.used_by[self.y_eclass_id] = []
        self.yi_exprs = yi_exprs or self.data["yis"]

        self.yi_ids: int = [
            self.lookup_expr_root(yi_expr, ignore_size=ignore_size).node_id
            for yi_expr in self.yi_exprs
        ]
        self.yi_eclass_ids: list[int] = [
            self.get_enode(yi_id).eclass_id for yi_id in self.yi_ids
        ]
        for yi_eclass_id in self.yi_eclass_ids:
            if yi_eclass_id not in self.used_by:
                self.used_by[yi_eclass_id] = []

        assert all(
            self.get_enode(yi_id).data in ("input", "weight") for yi_id in self.yi_ids
        ), "We have to know the yi tensor names from yi_exprs."
        self.yi_name_ids: int = [
            self.get_eclass(self.get_enode(yi_id).children[0]).nodes[0]
            for yi_id in self.yi_ids
        ]

        self.logger.info(f"y_expr: {[self.y_expr]}")
        self.logger.info(f"yi_exprs: {self.yi_exprs}")

        self.mark_representative()

        self.logger.info(f"Number of eclasses: {len(self.eclasses)}")
        self.logger.info(f"Number of enodes: {len(self.nodes)}")
        self.logger.info(
            f"After mark representative\n{self}",
        )

        self.post_condition_eclasses: list[EClass] = None

    def visualize_saturated(self):
        self.to_dot(osp.join(osp.splitext(self.json_path)[0] + ".dot"))

    def compute_post_condition_eclasses(self):
        self.post_condition_eclasses = self.get_post_condition_eclasses()

    def get_enode(self, node_id: int) -> ENode:
        return self.node_id_to_node[node_id]

    def get_eclass(self, eclass_id: int) -> EClass:
        return self.eclass_id_to_eclass[eclass_id]

    @staticmethod
    def remove_size(varname: str) -> str:
        assert varname.count("@") <= 1
        return varname.split("@")[0]

    def lookup_expr_root(self, expr: str, ignore_size=False) -> ENode:
        parsed_expr = parse_expression_with_lock(expr)[0]

        def lookup_expr_ids_sub(
            root: ENode, parsed_expr: list | str
        ) -> ENode | list[tuple[str | ENode]]:
            if type(parsed_expr) is str:
                if root.data == parsed_expr:
                    return root
                elif ignore_size and EGraph.remove_size(root.data) == parsed_expr:
                    return root
                else:
                    return None
            else:
                ret = [(root.data, root)]
                for child_eclass_id, child_expr in zip(root.children, parsed_expr[1:]):
                    child_eclass = self.get_eclass(child_eclass_id)
                    child_ret = None
                    for child_node_id in child_eclass.nodes:
                        child_node = self.get_enode(child_node_id)
                        child_ret = lookup_expr_ids_sub(child_node, child_expr)
                        if child_ret is not None:
                            break
                    if child_ret is None:
                        return None
                    ret.append(child_ret)
                return ret

        for node in self.nodes:
            if (
                (type(parsed_expr) is not str and node.data == parsed_expr[0])
                or (node.data == parsed_expr)
                or (ignore_size and EGraph.remove_size(node.data) == parsed_expr)
            ):
                root = node
                result = lookup_expr_ids_sub(root, parsed_expr)
                if result is not None:
                    if type(result) is ENode:
                        return result
                    else:
                        assert type(result) is list and type(result[0]) is tuple
                        assert type(result[0][1]) == ENode
                        return result[0][1]

        raise RuntimeError(f"Cannot find expr: {expr}")

    def mark_representative(self):
        new_oked: set[int] = set()

        for enode in self.nodes:
            enode.arg_oks = [False] * len(enode.children)

        for eclass in self.eclasses:
            eclass.used_yi_eclass_ids = set()

        # Mark constants
        for eclass in self.eclasses:
            for enode_id in eclass.nodes:
                enode = self.get_enode(enode_id)
                if enode.is_non_tensor_leaf():
                    enode.valid = True
                    eclass.set_representative_if_not_set(enode, self)
                    eclass.add_candidate(enode)
                    new_oked.add(eclass.eclass_id)
                    continue
                elif enode.is_tensor_name_leaf() and enode.node_id in self.yi_name_ids:
                    enode.valid = True
                    eclass.set_representative_if_not_set(enode, self)
                    eclass.add_candidate(enode)
                    new_oked.add(eclass.eclass_id)
                    continue

        for yi_eclass_id in self.yi_eclass_ids:
            yi_eclass = self.get_eclass(yi_eclass_id)
            yi_eclass.used_yi_eclass_ids = {yi_eclass.eclass_id}

        while len(new_oked) > 0:
            updated_user_nodes: set[ENode] = set()
            for oked_eclass_id in new_oked:
                # Check its user nodes.
                if oked_eclass_id not in self.used_by:
                    # No one used this eclass, skip.
                    continue
                oked_eclass = self.get_eclass(oked_eclass_id)
                for user_node_id, user_arg_idx in self.used_by[oked_eclass_id]:
                    user_node = self.get_enode(user_node_id)
                    assert (
                        type(user_node.data) == str
                    ), f"got {user_node=}, {user_node.data=}"
                    # setup_rank(user_node)
                    if user_node.data not in tgops.Op.OPS:
                        continue
                    else:
                        user_op = tgops.Op.get(user_node.data)
                        if user_op == tgops.matadd:
                            ec0 = self.get_eclass(user_node.children[0])
                            ec1 = self.get_eclass(user_node.children[1])
                            if (
                                len(ec1.nodes) == 1
                                and self.get_enode(ec1.nodes[0]).is_non_tensor_leaf()
                            ):
                                # Allow matadd when adding a constant or both are scalar tensor
                                pass
                            elif ec0.shape == "s" and ec1.shape == "s":
                                pass
                            else:
                                continue
                        elif not user_op.relation:
                            continue
                    if user_node.arg_oks[user_arg_idx] != True:
                        user_node.arg_oks[user_arg_idx] = True
                        user_eclass = self.get_eclass(user_node.eclass_id)
                        user_eclass.used_yi_eclass_ids = (
                            user_eclass.used_yi_eclass_ids.union(
                                oked_eclass.used_yi_eclass_ids
                            )
                        )
                        # XXX: This prunes some unnecessary nodes.
                        # if not user_eclass.valid or (
                        #     user_eclass.valid
                        #     and not user_eclass.eclass_id
                        #     in oked_eclass.used_yi_eclass_ids
                        # ):
                        # XXX: Also some heuristics to filter out unnecessary nodes.
                        # if user_node.data == "slice":
                        #     should_skip = False
                        #     for n in oked_eclass.candidates:
                        #         if n.data == "slice" and n.children[1] >= user_node.children[1]:
                        #             should_skip = True
                        #             break
                        #         if n.data == "concat":
                        #             should_skip = True
                        #             break
                        #     if should_skip:
                        #         continue
                        # print(f"Updated user node: {user_node.node_id}")
                        updated_user_nodes.add(user_node)

            # If some node has all its args valid, then the eclass is valid.
            new_new_oked = set()
            for user_node in updated_user_nodes:
                if all(user_node.arg_oks):
                    if (
                        user_node.data == "slice"
                        and user_node.children[0] == user_node.eclass_id
                    ):
                        continue
                    # print(f"User node all ok: {user_node.node_id}")
                    user_node.valid = True
                    eclass = self.get_eclass(user_node.eclass_id)
                    eclass.set_representative_if_not_set(user_node, self)
                    eclass.add_candidate(user_node)
                    new_new_oked.add(user_node.eclass_id)
            new_oked = new_new_oked

        if self.verbose:
            self.logger.info(self)

        for enode in self.nodes:
            del enode.arg_oks

    def extract_to_valid_only(self):
        egraph = deepcopy(self)
        new_eclasses = []
        for eclass in egraph.eclasses:
            new_eclass_nodes = []
            for enode_id in eclass.nodes:
                enode = egraph.get_enode(enode_id)
                if enode.valid:
                    new_eclass_nodes.append(enode.node_id)
            eclass.nodes = new_eclass_nodes
            if len(new_eclass_nodes) > 0:
                new_eclasses.append(eclass)
        egraph.eclasses = new_eclasses

        for ec in egraph.eclasses:
            print(ec)

        return egraph

    def get_post_condition_eclasses(self) -> list[EClass]:
        """
        Returns eclass list that forms the post condition by collecting
        the root `y_eclass`'s representative descendants.
        """
        y_eclass = self.get_eclass(self.y_eclass_id)
        post_condition_eclasses = [y_eclass]
        pending = [y_eclass]

        if not y_eclass.valid:
            self.logger.info("After mark representative")
            self.logger.info(self)
            raise CannotFindPostconditions("Validity not marked for y_eclass.")

        for eclass in self.eclasses:
            eclass.visited = False
        y_eclass.visited = True

        # Basically, we recursively visit the children of root y_eclass
        # until we meet any yi_eclass.
        while len(pending) > 0:
            next_pending = []
            for eclass in pending:
                # If the representative of the eclass is an expected leaf
                for child_eclass_id in eclass.representative.children:
                    if self.get_eclass(child_eclass_id).visited:
                        continue
                    self.get_eclass(child_eclass_id).visited = True
                    assert self.get_eclass(child_eclass_id).valid
                    node = self.get_eclass(child_eclass_id).representative
                    post_condition_eclasses.append(self.get_eclass(child_eclass_id))
                    if not node.is_numeric() and node.node_id not in self.yi_name_ids:
                        next_pending.append(self.get_eclass(child_eclass_id))
            pending = next_pending

        for eclass in self.eclasses:
            del eclass.visited

        return post_condition_eclasses

    def get_representative(self, eclass_id: int) -> ENode:
        representative = self.get_eclass(eclass_id).representative
        if representative is None:
            raise RuntimeError(f"Cannot find representative for {eclass_id=}")
        return representative

    def clone(self) -> "EGraph":
        egraph = deepcopy(self)
        egraph.eclass_id_to_eclass = {
            eclass.eclass_id: eclass for eclass in egraph.eclasses
        }
        egraph.node_id_to_node = {node.node_id: node for node in egraph.nodes}

    def get_valid_descendants(
        self, eclass_id: int, representative_only=False, stop_at_yi_name=False
    ) -> list[EClass]:
        """
        `stop_at_yi_name`: If True, we stop dfs at yi enode. This is for when we have
        multiple yi along a path (e.g., a path including an yi and its yi child, which
        might happen using explorative way).
        """
        eclass = self.get_eclass(eclass_id)
        visited: set[EClass] = set()
        descendants: list[EClass] = []

        def dfs(eclass: EClass):
            if eclass in visited:
                return
            if (
                stop_at_yi_name
                and len(nodes := eclass.nodes) == 1
                and nodes[0] in self.yi_name_ids
            ):
                return
            visited.add(eclass)
            descendants.append(eclass)
            if representative_only:
                for child_eclass_id in eclass.representative.children:
                    dfs(self.get_eclass(child_eclass_id))
            else:
                for candidate in eclass.candidates:
                    for child_eclass_id in candidate.children:
                        dfs(self.get_eclass(child_eclass_id))

        dfs(eclass)
        return descendants

    def extract_to_postcondition(
        self,
        all_candidates=False,
        representative_only=False,
        including_yis=False,
    ) -> "EGraph":
        """
        - default: traverse from root until meeting yis.
        - all_candidates:
            EClass-wise, collect all candidates from root y_eclass.
        - representative_only:
            ENode-wise, collect only representative nodes.
        - representative_only and including_yis:
            Based on representative_only, try to connect the yis to the sub-graph by
            dfs (from bottom to up) from missing yis.
        """
        assert (all_candidates, representative_only, including_yis) in {
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, True, True),
        }, f"{all_candidates=}, {representative_only=}, {including_yis=}"

        egraph = deepcopy(self)
        y_node = egraph.get_enode(egraph.y_id)
        y_eclass = egraph.get_eclass(egraph.y_eclass_id)
        y_name_eclass = egraph.get_eclass(y_node.children[0])

        if all_candidates:
            used_eclasses = egraph.get_valid_descendants(egraph.y_eclass_id)
        else:
            used_eclasses = egraph.post_condition_eclasses

        def set_eclass_nodes(eclass: EClass, representative_only=representative_only):
            """
            Set the nodes of the eclass based on the condition.
            """
            if representative_only:
                eclass.nodes = [eclass.representative.node_id]
            else:
                eclass.nodes = [candidate.node_id for candidate in eclass.candidates]

        eclasses: set[EClass] = set()
        self.logger.info(used_eclasses)
        for eclass in used_eclasses:
            if not eclass.valid:
                continue
            set_eclass_nodes(eclass)
            if eclass.eclass_id == y_eclass.eclass_id:
                eclass.nodes.append(egraph.y_id)
            eclasses.add(eclass)
            egraph.eclass_id_to_eclass[eclass.eclass_id] = eclass
        eclasses.add(y_name_eclass)

        if including_yis:

            def include_representative_descendants(eclass_id: int):
                eclass = egraph.get_eclass(eclass_id)
                if eclass in eclasses:
                    return
                set_eclass_nodes(eclass, representative_only=True)
                eclasses.add(eclass)
                for enode in eclass.candidates:
                    for child_eclass_id in enode.children:
                        child_eclass = egraph.get_eclass(child_eclass_id)
                        if child_eclass in eclasses:
                            continue
                        include_representative_descendants(child_eclass_id)

            def dfs_until_valid_desc(
                enode_id: int, path: list[int], visited: set[int]
            ) -> bool:
                """
                This function is to find a path from enode_id to the current sub-graph.
                """
                assert path[-1] == enode_id
                enode = egraph.get_enode(enode_id)
                if egraph.get_eclass(enode.eclass_id) in eclasses:
                    # Met the representative sub-graph.
                    return True
                if enode.eclass_id not in egraph.used_by:
                    return False
                for user_id, _ in egraph.used_by[enode.eclass_id]:
                    user_node = egraph.get_enode(user_id)
                    if not user_node.valid:
                        continue
                    user_eclass = egraph.get_eclass(user_node.eclass_id)
                    if user_eclass.eclass_id in visited:
                        continue
                    if not user_eclass.valid:
                        # We only want to traverse valid eclasses.
                        continue
                    visited.add(user_eclass.eclass_id)
                    path.append(user_node.node_id)
                    if dfs_until_valid_desc(user_node.node_id, path, visited):
                        return True
                    path.pop()
                return False

            for yi_name_id in egraph.yi_name_ids:
                yi_name_enode = egraph.get_enode(yi_name_id)
                yi_name_eclass = egraph.get_eclass(yi_name_enode.eclass_id)
                if yi_name_eclass in eclasses:
                    continue
                path = [yi_name_id]
                if dfs_until_valid_desc(yi_name_id, path, {yi_name_eclass.eclass_id}):
                    for node_id in path:
                        enode = egraph.get_enode(node_id)
                        eclass = egraph.get_eclass(enode.eclass_id)
                        if eclass not in eclasses:
                            eclass.nodes = [node_id]
                            eclasses.add(eclass)
                        else:
                            if node_id not in eclass.nodes:
                                eclass.nodes.append(node_id)

                    # Include all the representative descendants from the root.
                    path_root_enode = egraph.get_enode(path[-1])
                    include_representative_descendants(path_root_enode.eclass_id)
                else:
                    print(
                        f"{BYELLOW}Failed to find path to minimum post condition subgraph from {egraph.get_enode(yi_name_id)}{RST}"
                    )

        # NOTE: Modifying egraph.eclasses is good enough to represent a sub-graph.
        egraph.eclasses = eclasses
        return egraph

    def extract_to_sexpr_str(self) -> str:
        self.logger.info("Extracting to sexpr str:")
        self.logger.info(self.eclasses)
        node_id_to_sexpr_str: dict[int, str] = {
            self.y_id: self.y_expr,
        }

        def extract_to_sexpr_str_sub(node_id) -> str:
            if node_id in node_id_to_sexpr_str:
                return node_id_to_sexpr_str[node_id]
            enode = self.get_enode(node_id)
            if len(enode.children) == 0:
                result = enode.data
            else:
                args_str = " ".join(
                    extract_to_sexpr_str_sub(
                        self.get_representative(arg_eclass_id).node_id
                    )
                    for arg_eclass_id in enode.children
                )
                result = f"({enode.data} {args_str})"
            node_id_to_sexpr_str[node_id] = result
            return result

        eclass_str_list = []
        y_eclass_id = self.y_eclass_id
        y_name_eclass_id = self.get_enode(self.y_name_id).eclass_id
        for eclass in self.eclasses:
            assert eclass.valid or eclass.eclass_id in (y_eclass_id, y_name_eclass_id)
            enodes = list(map(self.get_enode, eclass.nodes))
            if len(enodes) <= 1:
                # Doesn't form an eclass.
                continue
            eclass_str_list.append(
                ENODE_SPLIT.join(
                    sorted(
                        [extract_to_sexpr_str_sub(enode.node_id) for enode in enodes]
                    )
                )
            )
        eclasses_str = "\n".join(sorted(eclass_str_list))
        return eclasses_str

    def all_yi_included(self) -> bool:
        node_ids: set[int] = set()
        for eclass in self.eclasses:
            node_ids.update(eclass.nodes)
        return all(yi_id in node_ids for yi_id in self.yi_ids)

    def to_dot(
        self,
        path: str,
        allowed: dict[int, set[int]] = None,
        allowed_edges: dict[int, bool] = None,
        add_dotdotdot=False,
    ):
        """
        Generate dot based on eclasses's `valid`, `representative` and `candidates`.

        The allowed xxx are for interactive visualization.
        allowed: dict[eclass_id, set[node_id]]
        allowed_edges: dict[node_id, bool]
        """

        def get_eclass_vis_attr(eclass: EClass):
            return {
                "label": f"{eclass.eclass_id}",
                "style": "dashed",
                "color": "green2" if eclass.valid else "gray",
                "bgcolor": "white",
                "penwidth": 2 if eclass.valid else 1,
            }

        def create_pydot_node_from_enode(enode: ENode, eclass: EClass, node_idx: int):
            """
            node_idx: the idx of this `enode` in this `eclass`.
            """
            node_id = enode.node_id
            args = []
            params = []
            for child_eclass_id in enode.children:
                arg_enode = self.get_enode(self.get_eclass(child_eclass_id).nodes[0])
                if arg_enode.is_non_tensor_leaf():
                    params.append(arg_enode.data)
                else:
                    args.append(str(child_eclass_id))
            args_str = ",".join(args)
            params_str = ",".join(params)
            label = f"{node_id}:{enode.data}"
            if args_str != "" and params_str != "":
                label += f"({args_str},\\n{params_str})"
            elif args_str != "" and params_str == "":
                label += f"({args_str})"
            if eclass.shape != "":
                label += f"\n{eclass.shape}"
            if enode.data == "input" or enode.is_tensor_name_leaf():
                color = "gold"
            else:
                color = "black"
            pydot_node = pydot.Node(
                f"{enode.eclass_id}.{node_idx}",
                label=label,
                color=color,
                style="filled",
                fillcolor="ghostwhite",
                penwidth=(
                    3.5
                    if enode == eclass.representative
                    else (2.5 if enode in eclass.candidates else 1)
                ),
            )
            return pydot_node

        def create_pydot_node_ellipsis(eclass: EClass):
            return pydot.Node(
                f'"{eclass.eclass_id}...."',
                label='"..."',
                color="black",
                style="filled",
                fillcolor="gainsboro",
                penwidth=1,
            )

        enode_to_pydot_node = {}
        eclass_to_pydot_cluster = {}
        finally_used_enodes_per_eclass: dict[EClass, list[ENode]] = {}
        eclass_to_pydot_dot_node: dict[EClass, pydot.Node] = {}
        graph = pydot.Dot(graph_type="digraph", fontname="Verdana", compound="true")

        if allowed_edges is not None:
            assert allowed is not None
        if allowed is not None:
            for eclass_id, allowed_node_ids in allowed.items():
                if allowed_node_ids is None:
                    allowed[eclass_id] = set(self.get_eclass(eclass_id).nodes)

        # Add Cluster and Node
        for eclass in self.eclasses:
            eclass_id = eclass.eclass_id
            if all(
                self.get_enode(node_id).is_non_tensor_leaf() for node_id in eclass.nodes
            ):
                continue
            if all(self.get_enode(node_id).data == "noop" for node_id in eclass.nodes):
                continue
            if allowed is not None and eclass_id not in allowed:
                continue
            cluster = pydot.Cluster(f"{eclass_id}", **get_eclass_vis_attr(eclass))
            eclass_to_pydot_cluster[eclass] = cluster
            graph.add_subgraph(cluster)
            if eclass not in finally_used_enodes_per_eclass:
                finally_used_enodes_per_eclass[eclass] = []
            for node_idx, node_id in enumerate(eclass.nodes):
                if allowed is not None and node_id not in allowed[eclass_id]:
                    continue
                enode = self.get_enode(node_id)
                pydot_node = create_pydot_node_from_enode(enode, eclass, node_idx)
                enode_to_pydot_node[enode] = pydot_node
                cluster.add_node(pydot_node)
                finally_used_enodes_per_eclass[eclass].append(enode)
            if (
                add_dotdotdot
                and allowed is not None
                and any(node_id not in allowed[eclass_id] for node_id in eclass.nodes)
            ):
                # Add an `...` node to represent the hidden nodes.
                pydot_node = create_pydot_node_ellipsis(eclass)
                cluster.add_node(pydot_node)
                eclass_to_pydot_dot_node[eclass] = pydot_node

        # Add Edge
        for eclass in self.eclasses:
            eclass_id = eclass.eclass_id
            if allowed is not None and eclass_id not in allowed:
                continue
            for node_id in eclass.nodes:
                if allowed is not None and node_id not in allowed[eclass_id]:
                    continue
                if allowed_edges is not None and node_id not in allowed_edges:
                    continue
                node = self.get_enode(node_id)
                if node not in enode_to_pydot_node:
                    continue
                arg_idx = 0
                for arg_eclass_id in node.children:
                    arg_eclass = self.get_eclass(arg_eclass_id)
                    if arg_eclass not in eclass_to_pydot_cluster:
                        continue
                    if allowed is not None and arg_eclass.eclass_id not in allowed:
                        continue
                    if arg_eclass.nodes[0] in enode_to_pydot_node:
                        dst_pydot_node = enode_to_pydot_node[arg_eclass.nodes[0]]
                    elif len(finally_used_enodes_per_eclass[arg_eclass]) > 0:
                        dst_pydot_node = enode_to_pydot_node[
                            finally_used_enodes_per_eclass[arg_eclass][0]
                        ]
                    elif arg_eclass in eclass_to_pydot_dot_node:
                        dst_pydot_node = eclass_to_pydot_dot_node[arg_eclass]
                    else:
                        raise RuntimeError(
                            f"Cannot find a node for {arg_eclass_id} in {arg_eclass.nodes}"
                        )

                    pydot_edge = pydot.Edge(
                        enode_to_pydot_node[node],
                        dst_pydot_node,
                        label=(
                            ""
                            if len(node.children) <= 1
                            else f'"{arg_idx}:{arg_eclass.eclass_id}"'
                        ),
                        lhead=f"cluster_{arg_eclass.eclass_id}",
                    )
                    arg_idx += 1
                    graph.add_edge(pydot_edge)

        graph.write(path)
        svg_path = osp.splitext(path)[0] + ".svg"
        if len(self.eclasses) >= 500:
            print("Too many eclasses, using newrank.")
            p = Popen(
                f"dot -Tsvg {path} -o {svg_path} -Gnewrank=true 2>&1 > /dev/null",
                shell=True,
                stdout=PIPE,
                stderr=STDOUT,
            )
        else:
            p = Popen(
                f"dot -Tsvg {path} -o {svg_path} 2>&1 > /dev/null || dot -Tsvg {path} -o {svg_path} -Gnewrank=true 2>&1 > /dev/null",
                shell=True,
                stdout=PIPE,
                stderr=STDOUT,
            )
        try:
            p.wait(timeout=10)
            print(f"Pretty svg generated to {osp.symabspath(svg_path)}.")
            return True
        except TimeoutExpired as e:
            print("dot timeouts.")
            log_path = osp.splitext(path)[0] + ".timeout.log"
            try:
                out, _ = p.communicate(timeout=10)
                with open(log_path, "w") as f:
                    f.write(out.decode())
                print(f"Timeout log written to {log_path}.")
            except TimeoutExpired as e:
                print(f"Communicate timeouts.")
            return False
        finally:
            p.kill()

    def __repr__(self):
        return f"EGraph(\n{self.nodes}\n{self.eclasses}\n)"

    def __str__(self):
        ret = ""
        for eclass in self.eclasses:
            ret += f"EClass[{eclass.eclass_id}] valid={eclass.valid}, representative={None if eclass.representative is None else eclass.representative.node_id}, candidates={[n.node_id for n in eclass.candidates]}\n"
            for node_id in eclass.nodes:
                ret += f"\t{str(self.get_enode(node_id))}\n"
        return ret

    def extract_to_digraph(self) -> nx.DiGraph:
        raise DeprecationWarning("This method is deprecated.")
        graph = nx.DiGraph()
        for eclass in self.post_condition_eclasses:
            enode = eclass.representative
            graph.add_node(
                enode.node_id,
                **{
                    "enode": enode,
                    "id": enode.node_id,
                    "op": enode.data,
                    "color": "lime" if enode.node_id in self.yi_ids else "cyan",
                },
            )
        for _, attr in graph.nodes(data=True):
            attr["label"] = f"[{attr['id']}] {attr['op']}"
        for eclass in self.post_condition_eclasses:
            enode = eclass.representative
            if enode.node_id not in self.yi_name_ids:
                for arg_idx, child_eclass_id in enumerate(enode.children):
                    child = self.get_eclass(child_eclass_id).representative
                    graph.add_edge(enode.node_id, child.node_id, **{"arg_idx": arg_idx})

        return graph

    def extract_to_recexpr_list(self) -> str:
        raise DeprecationWarning("This method is deprecated.")
        digraph = self.extract_to_digraph()

        result_list = []
        """
        Each element is in the below format:
        (op, [arg1, arg2, ...], [param1, param2, ...], name: str|None)
        """
        result_idx_map: dict[ENode, int] = {}
        for node_id in nx.dfs_postorder_nodes(digraph):
            enode: ENode = self.get_enode(node_id)
            if enode.is_leaf():
                continue
            if enode.data in ("input", "weight"):
                name = self.get_enode(self.get_eclass(enode.children[0]).nodes[0]).data
                result = (enode.data, [], [], name)
            else:
                tensor_children = []
                param_children = []
                for child_eclass_id in enode.children:
                    child = self.get_representative(child_eclass_id)
                    if child.is_numeric():
                        param_children.append(int(child.data))
                    elif child.is_shape_literal():
                        param_children.append(child.data)
                    else:
                        tensor_children.append(result_idx_map[child])
                result = (enode.data, tensor_children, param_children, None)
            result_idx_map[enode] = len(result_list)
            result_list.append(result)
        return result_list
