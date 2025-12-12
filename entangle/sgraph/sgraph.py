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

from dataclasses import dataclass
from itertools import chain
from typing import Callable

import networkx as nx

import entangle.ops as tgops
from entangle.sgraph.sexpr import SExpr, ShapeLike
from entangle.utils import ENODE_SPLIT
from entangle.utils.print_utils import BRED, RST


def sexprs_to_nx_graph(outputs: list[tuple[int, SExpr]]) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    # Reset node_ids and add nodes to `nx_graph`
    for output in outputs:
        for sexpr in output.post_order_dfs():
            nx_graph.add_node(
                sexpr.sexpr_id,
                sexpr=sexpr,
                # TODO: FIXME: Maybe decide this by out degree?
                is_output=sexpr is output,
            )
    # Add edges
    for sexpr_id, attr in nx_graph.nodes(data=True):
        sexpr: SExpr = attr["sexpr"]
        for arg_idx, arg in enumerate(sexpr.args):
            nx_graph.add_edge(sexpr_id, arg.sexpr_id, arg_idx=arg_idx)
    return nx_graph


@dataclass
class SGraphStats:
    num_nodes: int
    num_edges: int
    depth: int

    def to_dict(self):
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "depth": self.depth,
        }


class SGraph:
    """
    SGraph uses `SExpr` as root (output) nodes, but also pre-collect more information
    for ease of graph traversal and manipulation.
    """

    def __init__(
        self,
        outputs: list[SExpr],
        sexpr_order: dict[SExpr, int] = None,
    ):
        self.inputs: list[SExpr] = []  # Will initialize later.
        self.scalars = []  # for sexpr that has op == tgops.scalar
        self.shape_scalars = []  # for scalar sexpr shape in all the sexprs
        self.outputs: list[SExpr] = list(outputs)
        self.sexprs: set[SExpr] = set()
        self.sexpr_order: dict[SExpr, int] = sexpr_order
        # Pre-collected information 1: name_to_sexpr and inputs
        # A map from name to `SExpr`
        self.name_to_sexpr: dict[str, SExpr] = {}
        for output in outputs:
            for sexpr, is_leaf in output.post_order_dfs(
                visit_sexpr_param=True, return_is_leaf=True
            ):
                sexpr: SExpr
                if sexpr.name is not None:
                    self.name_to_sexpr[sexpr.name] = sexpr
                if is_leaf:
                    self.inputs.append(sexpr)
                if sexpr.op == tgops.scalar:
                    self.scalars.append(sexpr)
                self.shape_scalars.extend(sexpr.get_shape_scalars())
                self.sexprs.add(sexpr)
        # Pre-collected information 2: nx_graph
        # NOTE!!!! nx_graph doesn't include those scalar or SymInt.
        self.nx_graph = sexprs_to_nx_graph(outputs)
        # Pre-collected information 3: used_by
        self.used_by: dict[SExpr, list[SExpr]] = {}
        for _, attr in self.nx_graph.nodes(data=True):
            sexpr = attr["sexpr"]
            for arg in sexpr.args:
                self.used_by.get(sexpr, []).append(arg)

    @staticmethod
    def from_nx_graph(
        nx_graph: nx.DiGraph,
        sexpr_order: dict[SExpr, int] = None,
        outputs: set[SExpr] = None,
    ) -> "SGraph":
        if outputs is not None:
            roots = outputs
        else:
            roots = []
            for sexpr_id, attr in nx_graph.nodes(data=True):
                if nx_graph.in_degree(sexpr_id) == 0:
                    roots.append(attr["sexpr"])
        if sexpr_order is None:
            sexpr_order = {}
            try:
                for i, sexpr_id in enumerate(nx.topological_sort(self.nx_graph)):
                    sexpr = nx_graph.nodes[sexpr_id]["sexpr"]
                    sexpr_order[sexpr] = i
            except nx.exception.NetworkXUnfeasible as e:
                print(f"{BRED}Failed to produce a topo-order.{RST}")
                raise e
        return SGraph(roots, sexpr_order)

    def get_sexpr(self, name=None, sexpr_id=None):
        """
        Get the `SExpr` by name or `sexpr_id`.
        """
        if name is not None:
            return self.name_to_sexpr[name]
        else:
            return self.nx_graph.nodes[sexpr_id]["sexpr"]

    def get_sexprs_by(
        self, condition: Callable[[SExpr], bool], sort=False
    ) -> list[SExpr]:
        non_sorted = [
            attr["sexpr"]
            for _, attr in self.nx_graph.nodes(data=True)
            if condition(attr["sexpr"])
        ]
        if sort:
            return sorted(non_sorted, key=lambda e: self.sexpr_order[e])
        else:
            return non_sorted

    def get_input_sexprs(self, sort=False) -> list[SExpr]:
        return self.get_sexprs_by(lambda sexpr: sexpr.op == tgops.inpt, sort=sort)

    def get_constant_sexprs(self, sort=False) -> list[SExpr]:
        return self.get_sexprs_by(lambda sexpr: sexpr.op.constant, sort=sort)

    def get_scalar_sexprs(self) -> list[SExpr]:
        return self.scalars

    def get_shape_scalars(self) -> list[SExpr]:
        return self.shape_scalars

    def get_skeleton_sexprs(self, sort=False) -> list[SExpr]:
        return self.get_sexprs_by(
            lambda sexpr: type(sexpr.op) is tgops.Op and sexpr.op.skeleton,
            sort=sort,
        )

    def save(self, path, intermediate_name: bool = False):
        with open(path, "a") as f:
            f.write(
                f"# SGraph (intermediate_name={intermediate_name}): {self.outputs[0].name}\n"
            )
            if intermediate_name:
                writen: set[SExpr] = set()
                for output in self.outputs:
                    for sexpr, is_leaf in output.post_order_dfs(return_is_leaf=True):
                        # It makes no sense to write leaf because it will be x equals x.
                        if not is_leaf and sexpr not in writen:
                            s = sexpr.to_egg_str(recursive=False)
                            f.write(f"{s}{ENODE_SPLIT}{sexpr.to_egg_str_as_inpt()}\n")
                            writen.add(sexpr)
            else:
                for sexpr in self.outputs:
                    s = sexpr.to_egg_str()
                    f.write(f"{s}{ENODE_SPLIT}{sexpr.to_egg_str_as_inpt()}\n")
            f.write("\n")

    def get_name_to_shape_mappings(self) -> dict[str, ShapeLike]:
        mappings: dict[str, ShapeLike] = {}
        for sexpr in chain(self.sexprs):
            if sexpr in mappings:
                continue
            mappings[f"{sexpr.name}@"] = sexpr.shape
        return mappings

    def get_stats(self) -> SGraphStats:
        # Get information from nx
        num_nodes = self.nx_graph.number_of_nodes()
        num_edges = self.nx_graph.number_of_edges()
        depth = nx.dag_longest_path_length(self.nx_graph)
        return SGraphStats(num_nodes, num_edges, depth)

    def __repr__(self):
        return f"SGraph({self.outputs[0]!r})"
