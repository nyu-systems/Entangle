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

from itertools import chain
from typing import Generator, Iterable

import networkx as nx
import rich

import entangle.ops as tgops
from entangle.sgraph.sexpr import SExpr
from entangle.sgraph.sgraph import SGraph
from entangle.utils.print_utils import BRED, RST


class CutGroup:
    """
    CutGroup describes a group of cut sexprs that will be used to partition the sgraph.
    We assume every cut_group should either be a pass-through group or it will have a
    condition.

    We use SExpr to represent a cut, which is natural and beneficial.
    Using SExpr, we can easily extract its producers recursively to get a sub-sgraph that
    produces this cut.
    """

    def __init__(self, origin_cut: SExpr, target_cuts: list[SExpr]):
        self.origin_cut = origin_cut
        self.target_cuts = target_cuts
        self.cuts = [self.origin_cut] + self.target_cuts

    @staticmethod
    def extract_sgraph(outputs: list[SExpr], begin_names: set[str]) -> SGraph:
        """
        Extract the subgraph that produces `output` and begins with SExprs in `begin_cuts`.
        Note that we use `begin_names` instead of just input+skeleton ops so that
        we can consider `pass-through` (See tg infer args)
        """
        if not isinstance(outputs, Iterable):
            outputs = [outputs]
        # A dict mapping original SExpr to the cloned SExpr.
        sexpr_clone_map: dict[SExpr, SExpr] = {}
        terminate_cb = lambda e: e.name in begin_names
        next_order = 0
        sexpr_order: dict[SExpr, int] = {}
        for output in outputs:
            for sexpr, is_leaf, is_term in output.post_order_dfs(
                return_is_leaf=True, terminate_callback=terminate_cb
            ):
                sexpr: SExpr
                next_order += 1
                if is_leaf or is_term:
                    cloned = sexpr.get_placeholderized()
                elif sexpr in sexpr_clone_map:
                    continue
                else:
                    cloned_args = [sexpr_clone_map[arg] for arg in sexpr.args]
                    cloned = sexpr.clone_with(args=cloned_args)
                sexpr_order[cloned] = next_order
                sexpr_clone_map[sexpr] = cloned
        return SGraph([sexpr_clone_map[output] for output in outputs], sexpr_order)

    def extract_sgraphs(self, begin_names: set[str]) -> tuple[SGraph, list[SGraph]]:
        origin_sub_sgraph = CutGroup.extract_sgraph(self.origin_cut, begin_names)
        ranked_cuts: dict[int, set[SExpr]] = {}

        for cut in self.target_cuts:
            assert cut.rank is not None, f"Rank not set for {cut}."
            ranked_cuts.setdefault(cut.rank, set()).add(cut)
        target_sub_sgraphs = []
        for rank, cuts in sorted(ranked_cuts.items(), key=lambda x: x[0]):
            target_sub_sgraphs.append(CutGroup.extract_sgraph(cuts, begin_names))
        return origin_sub_sgraph, target_sub_sgraphs

    def __str__(self):
        return f"CutGroup({self.origin_cut!r}, {self.target_cuts!r})"

    def __repr__(self):
        return f"CutGroup({self.origin_cut!r}, {self.target_cuts!r})"

    def is_only_input(self):
        return all(cut.op == tgops.inpt for cut in self.cuts)

    def is_only_constant(self):
        return all(cut.op.constant for cut in self.cuts)


class SSkeleton:
    """
    `SSkeleton` describes a list of CutGroup with dependencies.
    It also provides a topological order for the CutGroup base on these
    dependencies.

    Definition of the **Dependency**: a CutGroup `cg`'s dependencies are the set of
    CutGroups whose outputs set intersects with `cg`'s inputs. Note that our
    condition extraction algorithm will only extract useful parts, so it is safe
    to use "intersection" here.
    """

    def __init__(
        self,
        origin_sgraph: SGraph,
        target_sgraphs: list[SGraph],
        cut_groups: list[CutGroup],
    ):
        self.origin_sgraph = origin_sgraph
        self.target_sgraphs = target_sgraphs
        self.sgraphs = [origin_sgraph, *target_sgraphs]

        self.cut_groups: list[CutGroup] = cut_groups
        self.cuts: list[SExpr] = list(
            chain(*[cut_group.cuts for cut_group in self.cut_groups])
        )

        self.sorted_cut_groups: list[CutGroup] = None
        self.cut_group_to_id: dict[CutGroup, int] = None
        # self.compute_topological_order()
        # self.compute_origin_topological_order_then_verify_consistency()
        self.compute_tolopogical_order_by_partial_order()

    def compute_tolopogical_order_by_partial_order(self):
        """
        This method can only be called in __init__.
        """
        assert self.sorted_cut_groups is None, "Topological order already computed."
        print(
            f"{BRED}Warning: using `compute_tolopogical_order_by_partial_order` to compute. This is a temporary solution and can be very slow.{RST}"
        )

        cut_to_group: dict[SExpr, CutGroup] = {}
        for cut_group in self.cut_groups:
            for cut in cut_group.cuts:
                cut_to_group[cut] = cut_group

        def compute_partial_order_sub(
            sexpr: SExpr, cuts: set[SExpr]
        ) -> tuple[set[tuple[CutGroup, CutGroup]], set[SExpr]]:
            """
            Compute the partial order of cuts.
            Return: (partial_order, descendants)
            """
            descendants = []
            partial_order = set()
            for child in sexpr.args:
                if child in cuts:
                    descendants.append(child)
                child_partial_order, child_descendants = compute_partial_order_sub(
                    child, cuts
                )
                descendants.extend(child_descendants)
                partial_order.update(child_partial_order)
            if sexpr in cuts:
                for desc in descendants:
                    partial_order.add((cut_to_group[sexpr], cut_to_group[desc]))
            return partial_order, descendants

        def compute_partial_order(
            sgraph: SGraph, cuts: set[SExpr]
        ) -> set[tuple[SExpr, SExpr]]:
            partial_order = set()
            for output in sgraph.outputs:
                partial_order.update(compute_partial_order_sub(output, cuts)[0])
            return partial_order

        origin_po = compute_partial_order(self.origin_sgraph, set(self.cuts))
        target_pos = [
            compute_partial_order(sg, set(self.cuts)) for sg in self.target_sgraphs
        ]

        for target_po in target_pos:
            for po in origin_po:
                assert (
                    po in target_po or (po[1], po[0]) not in target_po
                ), f"Got origin_po={po}, but inversed not in target_po."

        po_graph = nx.DiGraph()
        cg_to_idx = {}
        for i, cg in enumerate(self.cut_groups):
            po_graph.add_node(i, cg=cg)
            cg_to_idx[cg] = i
        for po_set in [origin_po, *target_pos]:
            idx_po = [(cg_to_idx[po[0]], cg_to_idx[po[1]]) for po in po_set]
            for po in sorted(idx_po):
                if not po_graph.has_edge(*po):
                    po_graph.add_edge(*po)
        try:
            self.sorted_cut_groups = [
                self.cut_groups[idx]
                for idx in reversed(list(nx.topological_sort(po_graph)))
            ]
        except Exception as e:
            print("Failed to get topological order.")
            raise e
        self.cut_group_to_id = {cg: i for i, cg in enumerate(self.sorted_cut_groups)}

    def get_group_id(self, cut_group: CutGroup) -> int:
        return self.cut_group_to_id[cut_group]

    def __getitem__(self, idx) -> CutGroup | list[CutGroup]:
        if type(idx) is int:
            return self.sorted_cut_groups[idx]
        elif type(idx) is list:
            return [self.__getitem__(i) for i in idx]
        else:
            raise RuntimeError(f"Invalid index {idx}.")

    def __iter__(self) -> Generator[CutGroup, None, None]:
        for idx in range(len(self)):
            yield self[idx]

    def __len__(self):
        return len(self.sorted_cut_groups)


def default_group_sgraph_cuts(
    origin_sgraph: SGraph,
    target_sgraphs: list[SGraph],
) -> list[CutGroup]:
    assert origin_sgraph.sexpr_order is not None
    assert all(sg.sexpr_order is not None for sg in target_sgraphs)

    origin_input_sexprs = origin_sgraph.get_input_sexprs(sort=True)
    target_input_sexprs_list = [sg.get_input_sexprs(sort=True) for sg in target_sgraphs]
    lengths = [len(l) for l in [origin_input_sexprs, *target_input_sexprs_list]]
    assert len(set(lengths)) == 1, f"Requiring same number of inputs, got {lengths}."
    origin_skeleton_sexprs = origin_sgraph.get_skeleton_sexprs(sort=True)
    target_skeleton_sexprs_list = [
        sg.get_skeleton_sexprs(sort=True) for sg in target_sgraphs
    ]

    origin_cuts = origin_input_sexprs + origin_skeleton_sexprs
    target_cuts_list: list[list[SExpr]] = [
        input_sexprs + skeleton_sexprs
        for input_sexprs, skeleton_sexprs in zip(
            target_input_sexprs_list, target_skeleton_sexprs_list
        )
    ]

    assert (
        len(set([len(l) for l in [origin_cuts, *target_cuts_list]])) == 1
    ), "Requiring same number of skeletons."

    visited_sexpr = set()
    cut_groups = []
    for origin_cut, *target_cuts in zip(origin_cuts, *target_cuts_list):
        cuts = [origin_cut, *target_cuts]
        num_args = len(origin_cut.args)
        assert all(
            len(c.args) == num_args for c in cuts
        ), f"Inconsistent cuts: {list(map(repr, cuts))}"
        for i in range(num_args):
            origin_arg = origin_cut.args[i]
            target_args = [c.args[i] for c in target_cuts]
            if origin_arg in visited_sexpr:
                if all(arg in visited_sexpr for arg in target_args):
                    continue
                else:
                    # This happens when a skeleton op1 using another skeleton op2 as arg,
                    # but the distributed version uses a viewed skeleton op2 as arg.
                    # For this case, we should still add this as cut group.
                    rich.print(
                        f"Duplicated origin: {repr(origin_arg)}, {[repr(arg) for arg in target_args]}"
                    )
                    pass
            cut_group = CutGroup(origin_arg, target_args)
            visited_sexpr.update([origin_arg, *target_args])
            cut_groups.append(cut_group)
        if origin_cut in visited_sexpr:
            assert all(arg in visited_sexpr for arg in target_cuts), "Inconsistent."
            continue
        cut_group = CutGroup(origin_cut, target_cuts)
        visited_sexpr.update(cuts)
        cut_groups.append(cut_group)

    return cut_groups
