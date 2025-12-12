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

import re
from copy import copy
from typing import Union

import networkx as nx
import numpy as np
import rich
import entangle.ops as tgops
import entangle.utils
from entangle.convert.convert import *
from entangle.sgraph.sgraph import *
from entangle.utils.print_utils import BYELLOW, RST


class CannotMatchAllDistOps(Exception): ...


class SGraphTransformer:
    """ "
    !!!!!!! README before modifying !!!!!!!
    This class contains a series of transformations on SGraphs.
    Since the graph can be expressed either by SExpr itself or by the
    corresponding networkx graph, it can be troublesome to transforming both.
    Given that the networkx graph provide great facility to manipulate a
    graph, we always transform the networkx graph and then rebuild the SExprs.

    However, to add a new node in networkx, we usually need to create a new
    SExpr, you still need to maintain the correct args when adding any
    transformation.
    """

    DIST_ID_GEN = entangle.utils.get_id_generator()
    # User provided group_id to size mapping.
    oracle_group_id_to_size: Callable[[str | int], int] = None

    def __init__(self, sgraphs: Union[list[SGraph], SGraph]):
        if type(sgraphs) is SGraph:
            sgraphs = [sgraphs]
        self.sgraphs: list[SGraph] = sgraphs

        # `self.replacements` record all node replacement by name.
        self.replacements: dict[str, str] = {}

        self.nx_graphs: list[nx.DiGraph] = [g.nx_graph.copy() for g in self.sgraphs]

        # `self.nx_graph` maintains current graph in networkx format.
        self.nx_graph: nx.DiGraph = nx.union_all(self.nx_graphs)
        self.sexpr_order = {}
        for g in self.sgraphs:
            if g.sexpr_order is not None:
                self.sexpr_order.update(g.sexpr_order)
            else:
                rich.print("Skipped sgraph:", g)
        self.outputs = set.union(*[set(g.outputs) for g in self.sgraphs])

    def record_replacement(self, old_sexpr: SExpr, new_sexpr: SExpr):
        if old_sexpr.name != new_sexpr.name:
            self.replacements[old_sexpr.name] = new_sexpr.name
        if old_sexpr in self.sexpr_order:
            self.sexpr_order[new_sexpr] = self.sexpr_order[old_sexpr]
        if old_sexpr in self.outputs:
            self.outputs.remove(old_sexpr)
            self.outputs.add(new_sexpr)

    def collapsed_replacements(self) -> dict[str, str]:
        collapsed_replacements = {}
        for old_name, new_name in self.replacements.items():
            while new_name in self.replacements:
                new_name = self.replacements[new_name]
            collapsed_replacements[old_name] = new_name
        return collapsed_replacements

    def force_leaf(self, force_leaf_set: set[str]) -> "SGraphTransformer":
        """
        Force a node to be a leaf node.
        """
        assert len(self.sgraphs) == 1, "Only support single graph now."
        sgraph = self.sgraphs[0]
        nx_graph = self.nx_graph
        name_to_sexpr: dict[str, SExpr] = {}
        for sg in self.sgraphs:
            name_to_sexpr.update(sg.name_to_sexpr)

        for leaf_name in force_leaf_set:
            if leaf_name not in name_to_sexpr:
                continue
            leaf_sexpr = name_to_sexpr[leaf_name]
            if leaf_sexpr.op != tgops.inpt:
                print(f"{BYELLOW}WARNING{RST}: Forcing {leaf_sexpr!r} to be a leaf.")
                leaf_sexpr.op = tgops.inpt
                leaf_sexpr.args = []
                leaf_sexpr.params = []
                nx_graph.remove_nodes_from(
                    list(nx_graph.predecessors(leaf_sexpr.sexpr_id))
                )
        return self

    @staticmethod
    def get_dist_op_nodes(nx_graph: nx.DiGraph, return_waits=False):
        dist_op_nodes = []
        dist_wait_nodes = []
        for sexpr_id, attr in nx_graph.nodes(data=True):
            sexpr = attr["sexpr"]
            if sexpr.op == tgops.dist_wait:
                dist_wait_nodes.append(sexpr_id)
            elif sexpr.op.dist:
                dist_op_nodes.append(sexpr_id)
        if return_waits:
            return dist_op_nodes, dist_wait_nodes
        else:
            return dist_op_nodes

    @staticmethod
    def get_dist_op_group_id(sexpr: SExpr) -> str:
        assert sexpr.op.dist
        res = sexpr.params[-1]
        assert type(res) is str, f"Invalid group id: {res=}, {sexpr!r}"
        return res

    @staticmethod
    def get_dist_op_group_size(sexpr: SExpr) -> int:
        assert sexpr.op.dist
        if sexpr.op in (tgops.all_gather, tgops.reduce_scatter):
            if len(sexpr.params) == 1:
                print(f"{BYELLOW}WARNING{RST}Maybe a vLLM collective op")
                return None
            else:
                res = sexpr.params[-2]
        elif sexpr.op in (tgops.all_to_all_single,):
            assert len(sexpr.params[0]) == len(
                sexpr.params[1]
            ), f"Invalid group size: {sexpr.params=}"
            res = len(sexpr.params[0])
        elif SGraphTransformer.oracle_group_id_to_size is not None:
            assert sexpr.op in (
                tgops.dist_broadcast,
                tgops.all_reduce,
            ), "No other ops should use oracle."
            group_id = SGraphTransformer.get_dist_op_group_id(sexpr)
            res = SGraphTransformer.oracle_group_id_to_size(group_id)
        if res is None:
            print(
                f"Failed to get group_size for {sexpr.op}, please configure `oracle_group_id_to_size` in your Config"
            )
        if res is None:
            print(
                f"{BYELLOW}WARNING{RST} Got None for group size, maybe a vLLM collective op"
            )
        assert (
            res is None or type(res) is int
        ), f"Invalid group size for {sexpr!r}: {res}, {sexpr.params=}"
        return res

    def merge_dist_ops(self, just_mark: bool = False) -> "SGraphTransformer":
        """
        FIXME: This method can be more efficient using topological traverse.

        `just_mark`: If True, we just mark the sexprs that should be merged with one unique id.
        """
        if len(self.sgraphs) == 1:
            # We don't need to merge anything for a single graph.
            return self
        world_size = len(self.nx_graphs)

        dist_op_nodes_per_rank: list[list[SExpr]] = []

        for nx_graph in self.nx_graphs:
            dist_op_nodes = SGraphTransformer.get_dist_op_nodes(nx_graph)
            sexprs = [nx_graph.nodes[n]["sexpr"] for n in dist_op_nodes]
            dist_op_nodes_per_rank.append(sexprs)
        dist_op_nodes_per_rank = [
            sorted(
                dist_op_nodes,
                key=lambda s: (
                    s.dist_id if s.dist_id is not None else self.sexpr_order[s]
                ),
            )
            for dist_op_nodes in dist_op_nodes_per_rank
        ]

        pended_list: list[bool] = [False] * world_size
        group_id_to_size: dict[int, int] = {}
        group_id_to_sexprs: dict[int, list[SExpr]] = {}
        while True:
            changed = False
            for rank in range(len(pended_list)):
                pended = pended_list[rank]
                if pended or len(dist_op_nodes_per_rank[rank]) == 0:
                    # When this rank is pended (waiting for other ranks to form a group),
                    # or there is no dist op left for this rank, we skip it.
                    continue
                # If there is a dist op that can be pended, we try it.
                changed = True
                sexpr: SExpr = dist_op_nodes_per_rank[rank][0]
                assert type(sexpr.rank) is int, f"Invalid rank: {sexpr.rank=}"
                if just_mark:
                    group_id = SGraphTransformer.get_dist_op_group_id(sexpr)
                else:
                    assert sexpr.dist_id is not None, f"Must have dist_id: {sexpr!r}"
                    group_id = sexpr.dist_id
                group_size = SGraphTransformer.get_dist_op_group_size(sexpr)
                if self.oracle_group_id_to_size is not None:
                    real_group_id = SGraphTransformer.get_dist_op_group_id(sexpr)
                    if oracle_size := self.oracle_group_id_to_size(real_group_id):
                        assert (
                            oracle_size == group_size
                        ), f"{oracle_size=}, {group_size=}"
                assert (
                    group_id not in group_id_to_size
                    or group_id_to_size[group_id] == group_size
                )
                group_id_to_size[group_id] = group_size
                if group_id not in group_id_to_sexprs:
                    group_id_to_sexprs[group_id] = []
                found_sexprs: list[SExpr] = group_id_to_sexprs[group_id]
                found_sexprs.append(sexpr)
                pended_list[rank] = True

            if changed:
                continue

            # Check for complete groups
            for group_id in list(group_id_to_sexprs):
                found_sexprs = group_id_to_sexprs[group_id]
                group_size = group_id_to_size[group_id]
                if group_size is None or len(found_sexprs) == group_size:
                    # We found a complete group to execute the dist op.
                    # We can now handle this.

                    # Sort by rank
                    found_sexprs = sorted(found_sexprs, key=lambda x: x.rank)
                    # Mark edge arg_idx
                    # in0 ---0---\                   /---0--- out_wait0 ---
                    # in1 ---1---\\                 //---1--- out_wait1 ---
                    #             ++--- dist_op ---++
                    # in2 ---2---//                 \\---2--- out_wait2 ---
                    # in3 ---3---/                   \---3--- out_wait3 ---
                    for arg_idx, sexpr in enumerate(found_sexprs):
                        # For predecessor `wait`s.
                        wait_sexpr_id = list(
                            self.nx_graph.predecessors(sexpr.sexpr_id)
                        )[0]
                        self.nx_graph.edges[wait_sexpr_id, sexpr.sexpr_id][
                            "arg_idx"
                        ] = arg_idx
                        # For successor `input`s.
                        input_sexpr_id = list(self.nx_graph.successors(sexpr.sexpr_id))[
                            0
                        ]
                        self.nx_graph.edges[sexpr.sexpr_id, input_sexpr_id][
                            "arg_idx"
                        ] = arg_idx
                    if just_mark:
                        dist_id = next(self.DIST_ID_GEN)
                        for sexpr in found_sexprs:
                            sexpr.dist_id = dist_id
                    else:
                        if (
                            # Should be marked
                            any(s.dist_id is None for s in found_sexprs)
                            # Should be same mark
                            or len(set(s.dist_id for s in found_sexprs)) > 1
                        ):
                            raise CannotMatchAllDistOps(
                                f"Cannot match all dist ops.\n"
                                f"sexpr_ids={[s.sexpr_id for s in found_sexprs]}\n"
                                f"sexpr_dist_ids={[s.dist_id for s in found_sexprs]}\n"
                            )
                        # Only really merge when not `just_mark`.
                        first_sexpr = found_sexprs[0]
                        for sexpr in found_sexprs[1:]:
                            self.nx_graph = nx.contracted_nodes(
                                self.nx_graph, first_sexpr.sexpr_id, sexpr.sexpr_id
                            )
                    group_id_to_size.pop(group_id)
                    group_id_to_sexprs.pop(group_id)
                    for sexpr in found_sexprs:
                        r = sexpr.rank
                        dist_op_nodes_per_rank[r] = dist_op_nodes_per_rank[r][1:]
                        pended_list[r] = False
                    changed = True
            all_done = all(len(ns) == 0 for ns in dist_op_nodes_per_rank)
            if all_done:
                break
            elif not changed:
                print()
                rich.print(f"We have {group_size=}")
                rich.print(f"and {dist_op_nodes_per_rank=}")
                rich.print(f"but {found_sexprs=}")
                rich.print("group_id_to_sexprs=", group_id_to_sexprs)
                raise CannotMatchAllDistOps(f"Cannot match all dist ops.")
        return self

    def add_dist_wait_if_not_exists(self) -> "SGraphTransformer":
        dist_op_nodes = SGraphTransformer.get_dist_op_nodes(self.nx_graph)
        for dist_op_node_id in dist_op_nodes:
            dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
            dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
            user_ids = list(self.nx_graph.predecessors(dist_op_node_id))
            user_sexprs: list[SExpr] = [
                self.nx_graph.nodes[u]["sexpr"] for u in user_ids
            ]
            if len(user_ids) == 1 and user_sexprs[0].op == tgops.dist_wait:
                continue
            assert all(s.op != tgops.dist_wait for s in user_sexprs)
            dist_wait_sexpr = SExpr(
                tgops.dist_wait,
                [dist_op_sexpr],
                [],
                name=f"{dist_op_sexpr.name}_wait",
                shape=dist_op_sexpr.shape,
                rank=dist_op_sexpr.rank,
            )
            self.nx_graph.add_node(dist_wait_sexpr.sexpr_id, sexpr=dist_wait_sexpr)
            self.nx_graph.add_edge(dist_wait_sexpr.sexpr_id, dist_op_node_id, arg_idx=0)
            for user_id, user_sexpr in zip(user_ids, user_sexprs):
                edge_attr = self.nx_graph.edges[user_id, dist_op_node_id]
                self.nx_graph.remove_edge(user_id, dist_op_node_id)
                self.nx_graph.add_edge(user_id, dist_wait_sexpr.sexpr_id, **edge_attr)
                assert "arg_idx" in edge_attr, f"Invalid edge attr: {edge_attr=}"
                self.record_replacement(dist_op_sexpr, dist_wait_sexpr)
        return self.rebuild_sepxrs()

    def merge_duplicated_dist_wait(self) -> "SGraphTransformer":
        dist_op_nodes, dist_wait_nodes = SGraphTransformer.get_dist_op_nodes(
            self.nx_graph, return_waits=True
        )
        # For dist_wait with a dist_wait child, we just merge this nodes into its only child.
        # FIXME This is caused by my hacks that making all async ops to be sync (i.e., wait before
        # returning). So it may have two waits (one inside dist call and one at the original wait call)
        for dist_wait_node in dist_wait_nodes:
            children = list(self.nx_graph.successors(dist_wait_node))
            assert len(children) == 1, "dist_wait should have only one child"
            child = children[0]
            child_sexpr: SExpr = self.nx_graph.nodes[child]["sexpr"]
            if child_sexpr.op == tgops.dist_wait:
                dist_wait_sexpr: SExpr = self.nx_graph.nodes[dist_wait_node]["sexpr"]
                self.nx_graph = nx.contracted_nodes(
                    self.nx_graph, child, dist_wait_node, self_loops=False
                )
                self.record_replacement(dist_wait_sexpr, child_sexpr)
        return self

    def lower_all_gather(self, dist_op_node_id, sorted_input_ids, sorted_output_ids):
        # Lower all_gather_into_tensor to concat.
        dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
        dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
        # Remove edges around the dist_op.
        self.nx_graph.remove_node(dist_op_node_id)

        concat_sexpr: SExpr = self.nx_graph.nodes[sorted_input_ids[0]]["sexpr"]
        dim = 0  # all_gather only supports dim=0 now.
        # Concat inputs
        for input_id in sorted_input_ids[1:]:
            input_sexpr: SExpr = self.nx_graph.nodes[input_id]["sexpr"]
            new_shape = copy(concat_sexpr.shape)
            new_shape[dim] += input_sexpr.shape[dim]
            new_concat_sexpr = SExpr(
                tgops.concat,
                [concat_sexpr, input_sexpr],
                [0],
                name=f"ag_concat_{concat_sexpr.sexpr_id}_{input_sexpr.sexpr_id}",
                shape=new_shape,
            )
            self.nx_graph.add_node(
                new_concat_sexpr.sexpr_id,
                sexpr=new_concat_sexpr,
            )
            self.nx_graph.add_edge(
                new_concat_sexpr.sexpr_id, concat_sexpr.sexpr_id, arg_idx=0
            )
            self.nx_graph.add_edge(
                new_concat_sexpr.sexpr_id, input_sexpr.sexpr_id, arg_idx=1
            )
            self.record_replacement(dist_op_sexpr, new_concat_sexpr)
            concat_sexpr = new_concat_sexpr
        # Replace outputs with simple clone.
        relabel_mapping = {}
        for arg_idx, output_id in enumerate(sorted_output_ids):
            output_attr = self.nx_graph.nodes[output_id]
            clone_sexpr = SExpr(
                tgops.clone,
                [concat_sexpr],
                [],
                name=f"ag_clone_{concat_sexpr.sexpr_id}",
                shape=concat_sexpr.shape,
            )
            old_output_sexpr: SExpr = output_attr["sexpr"]
            output_attr["sexpr"] = clone_sexpr
            self.nx_graph.add_edge(output_id, concat_sexpr.sexpr_id, arg_idx=arg_idx)
            self.record_replacement(old_output_sexpr, clone_sexpr)
            relabel_mapping[output_id] = clone_sexpr.sexpr_id
        self.nx_graph = nx.relabel_nodes(self.nx_graph, relabel_mapping)

    def lower_reduce_scatter(
        self, dist_op_node_id, sorted_input_ids, sorted_output_ids
    ):
        # Lower reduce_scatter into sum and slices.
        dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
        dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
        # Remove edges around the dist_op.
        self.nx_graph.remove_node(dist_op_node_id)

        assert dist_op_sexpr.params[0] in (
            "sum",
            "max",
        ), "Only sum/max is supported now."
        reduce_op = (
            tgops.reduce_add if dist_op_sexpr.params[0] == "sum" else tgops.maximum
        )
        reduced_sexpr: SExpr = self.nx_graph.nodes[sorted_input_ids[0]]["sexpr"]
        n_dim = len(reduced_sexpr.shape)
        dim = 0  # reduce_scatter only supports dim=0 now.
        for input_id in sorted_input_ids[1:]:
            input_sexpr: SExpr = self.nx_graph.nodes[input_id]["sexpr"]
            new_shape = copy(reduced_sexpr.shape)
            new_shape[dim] += input_sexpr.shape[dim]
            new_reduced_sexpr = SExpr(
                reduce_op,
                [reduced_sexpr, input_sexpr],
                [],
                name=f"rs_add_{reduced_sexpr.sexpr_id}_{input_sexpr.sexpr_id}",
                shape=reduced_sexpr.shape,
            )
            self.nx_graph.add_node(new_reduced_sexpr.sexpr_id, sexpr=new_reduced_sexpr)
            self.nx_graph.add_edge(
                new_reduced_sexpr.sexpr_id, reduced_sexpr.sexpr_id, arg_idx=0
            )
            self.nx_graph.add_edge(
                new_reduced_sexpr.sexpr_id, input_sexpr.sexpr_id, arg_idx=1
            )
            reduced_sexpr = new_reduced_sexpr
            self.record_replacement(dist_op_sexpr, new_reduced_sexpr)
        # Replace the output with a slice.
        relabel_mapping = {}
        assert (
            reduced_sexpr.shape[dim] % len(sorted_output_ids) == 0
        ), f"Invalid shape, dim 0 of {reduced_sexpr.shape} should be divisible by {len(sorted_output_ids)}."
        size_per_rank = reduced_sexpr.shape[dim] // len(sorted_output_ids)
        for arg_idx, output_id in enumerate(sorted_output_ids):
            begin = size_per_rank * arg_idx
            end = begin + size_per_rank
            new_shape = copy(reduced_sexpr.shape)
            new_shape[dim] = size_per_rank
            slice_sexpr = SExpr(
                tgops.slice,
                [reduced_sexpr],
                [0, begin, end, 1],
                name=self.nx_graph.nodes[output_id]["sexpr"].name,
                shape=new_shape,
            )
            output_attr = self.nx_graph.nodes[output_id]
            old_output_sexpr = output_attr["sexpr"]
            if old_output_sexpr.shape != slice_sexpr.shape:
                raise CannotMatchAllDistOps(
                    f"Failed to lower reduce scatter. "
                    f"Expecting [{old_output_sexpr.sexpr_id}].shape={old_output_sexpr.shape}, got shape={slice_sexpr.shape}.\n"
                    f"Involved sexprs: {sorted_output_ids}"
                )
            output_attr["sexpr"] = slice_sexpr
            self.nx_graph.add_edge(output_id, reduced_sexpr.sexpr_id, arg_idx=0)
            self.record_replacement(old_output_sexpr, slice_sexpr)
            relabel_mapping[output_id] = slice_sexpr.sexpr_id
        self.nx_graph = nx.relabel_nodes(self.nx_graph, relabel_mapping)

    def lower_all_to_all_single(
        self, dist_op_node_id, sorted_input_ids, sorted_output_ids
    ):
        # Lower all_to_all_single into slice and concat.
        dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
        dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
        # Remove edges around the dist_op.
        self.nx_graph.remove_node(dist_op_node_id)

        # 1. Collect splits information
        if "contraction" in dist_op_attr:
            all_dist_op_sexprs: list[SExpr] = [dist_op_sexpr] + [
                n["sexpr"] for n in dist_op_attr["contraction"].values()
            ]
        else:
            assert (
                len(self.sgraphs) == 1
            ), 'If len(sgraph) > 1, there should be "contraction"'
            all_dist_op_sexprs = [dist_op_sexpr]
        all_dist_op_sexprs = sorted(all_dist_op_sexprs, key=lambda s: s.rank)
        group_size = len(all_dist_op_sexprs)
        # All the lengths of splits should be equal to group_size
        assert all(len(s.params[0]) == group_size for s in all_dist_op_sexprs)
        assert all(len(s.params[1]) == group_size for s in all_dist_op_sexprs)
        output_split_lists = np.array([s.params[0] for s in all_dist_op_sexprs])
        input_split_lists = np.array([s.params[1] for s in all_dist_op_sexprs])
        assert np.all(
            output_split_lists.T == input_split_lists
        ), f"Invalid splits: {output_split_lists=}, {input_split_lists=}"

        # Compute output shapes
        input_nodes = [self.nx_graph.nodes[i] for i in sorted_input_ids]
        input_sexprs: list[SExpr] = [n["sexpr"] for n in input_nodes]
        input_shapes = [s.shape for s in input_sexprs]
        output_shape0s = [sum(s) for s in output_split_lists]
        output_shapes = [
            [s, *shape[1:]] for s, shape in zip(output_shape0s, input_shapes)
        ]
        relabel_mapping = {}
        joining_names = "_".join([s.name[-7:] for s in input_sexprs])
        for i, (output_id,) in enumerate(zip(sorted_output_ids)):
            cur_input_name_prefix = re.search(
                r"\w+__r\d+__", input_sexprs[i].name
            ).group(0)
            concat_sexpr = SExpr(
                tgops.a2a.getitem(i),
                input_sexprs,
                [],  # FIXME: Should provide partitioning.
                name=f"{cur_input_name_prefix}__{joining_names}__a2a_{i}",
                shape=output_shapes[i],
                rank=input_sexprs[i].rank,
            )
            output_attr = self.nx_graph.nodes[output_id]
            old_output_sexpr = output_attr["sexpr"]
            assert (
                old_output_sexpr.shape == concat_sexpr.shape
            ), f"{old_output_sexpr.shape=}, {concat_sexpr.shape=}"
            output_attr["sexpr"] = concat_sexpr
            for a in concat_sexpr.args:
                self.nx_graph.add_edge(output_id, a.sexpr_id, arg_idx=i)
            self.record_replacement(old_output_sexpr, concat_sexpr)
            relabel_mapping[output_id] = concat_sexpr.sexpr_id
        self.nx_graph = nx.relabel_nodes(self.nx_graph, relabel_mapping)

    # def lower_all_to_all_single(
    #     self, dist_op_node_id, sorted_input_ids, sorted_output_ids
    # ):
    #     # Lower all_to_all_single into slice and concat.
    #     dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
    #     dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
    #     # Remove edges around the dist_op.
    #     self.nx_graph.remove_node(dist_op_node_id)

    #     # 1. Collect splits information
    #     if "contraction" in dist_op_attr:
    #         all_dist_op_sexprs: list[SExpr] = [dist_op_sexpr] + [
    #             n["sexpr"] for n in dist_op_attr["contraction"].values()
    #         ]
    #     else:
    #         assert (
    #             len(self.sgraphs) == 1
    #         ), 'If len(sgraph) > 1, there should be "contraction"'
    #         all_dist_op_sexprs = [dist_op_sexpr]
    #     all_dist_op_sexprs = sorted(all_dist_op_sexprs, key=lambda s: s.rank)
    #     group_size = len(all_dist_op_sexprs)
    #     # All the lengths of splits should be equal to group_size
    #     assert all(len(s.params[0]) == group_size for s in all_dist_op_sexprs)
    #     assert all(len(s.params[1]) == group_size for s in all_dist_op_sexprs)
    #     output_split_lists = np.array([s.params[0] for s in all_dist_op_sexprs])
    #     input_split_lists = np.array([s.params[1] for s in all_dist_op_sexprs])
    #     assert np.all(
    #         output_split_lists.T == input_split_lists
    #     ), f"Invalid splits: {output_split_lists=}, {input_split_lists=}"

    #     # 2. Split inputs
    #     split_input_lists: list[list[SExpr]] = [[] for _ in range(group_size)]
    #     for input_idx in range(group_size):
    #         begin = 0
    #         cur_input_sexpr_id = sorted_input_ids[input_idx]
    #         cur_input_sexpr_node = self.nx_graph.nodes[cur_input_sexpr_id]
    #         cur_input_sexpr: SExpr = cur_input_sexpr_node["sexpr"]
    #         cur_input_name_prefix = re.search(
    #             r"\w+__r\d+__", cur_input_sexpr.name
    #         ).group(0)
    #         for input_split in input_split_lists[input_idx]:
    #             sliced_shape = copy(cur_input_sexpr.shape)
    #             sliced_shape[0] = input_split
    #             sliced_sexpr = SExpr(
    #                 tgops.slice,
    #                 [cur_input_sexpr],
    #                 [0, begin, begin + input_split, 1],
    #                 name=cur_input_name_prefix
    #                 + f"a2a_slice_{cur_input_sexpr_id}_{begin}_{begin+input_split}",
    #                 shape=sliced_shape,
    #                 rank=cur_input_sexpr.rank,
    #             )
    #             self.nx_graph.add_node(sliced_sexpr.sexpr_id, sexpr=sliced_sexpr)
    #             self.nx_graph.add_edge(
    #                 sliced_sexpr.sexpr_id, cur_input_sexpr.sexpr_id, arg_idx=0
    #             )
    #             split_input_lists[input_idx].append(sliced_sexpr)
    #             begin += input_split

    #     # 3. Concat split inputs into outputs
    #     relabel_mapping = {}
    #     for arg_idx, output_id in enumerate(sorted_output_ids):
    #         sexprs_to_concat = [
    #             split_input_lists[i][arg_idx] for i in range(group_size)
    #         ]
    #         concat_sexpr = sexprs_to_concat[0]
    #         for s in sexprs_to_concat[1:]:
    #             assert s.shape[1:] == concat_sexpr.shape[1:]
    #             new_shape = copy(concat_sexpr.shape)
    #             new_shape[0] += s.shape[0]
    #             new_concat_sexpr = SExpr(
    #                 tgops.concat,
    #                 [concat_sexpr, s],
    #                 [0],
    #                 name=f"a2a_concat_{concat_sexpr.sexpr_id}_{s.sexpr_id}",
    #                 shape=new_shape,
    #             )
    #             self.nx_graph.add_node(
    #                 new_concat_sexpr.sexpr_id, sexpr=new_concat_sexpr
    #             )
    #             self.nx_graph.add_edge(
    #                 new_concat_sexpr.sexpr_id, concat_sexpr.sexpr_id, arg_idx=0
    #             )
    #             self.nx_graph.add_edge(new_concat_sexpr.sexpr_id, s.sexpr_id, arg_idx=1)
    #             concat_sexpr = new_concat_sexpr
    #         output_attr = self.nx_graph.nodes[output_id]
    #         old_output_sexpr = output_attr["sexpr"]
    #         assert (
    #             old_output_sexpr.shape == concat_sexpr.shape
    #         ), f"{old_output_sexpr.shape=}, {concat_sexpr.shape=}"
    #         output_attr["sexpr"] = concat_sexpr
    #         for a in concat_sexpr.args:
    #             self.nx_graph.add_edge(output_id, a.sexpr_id, arg_idx=arg_idx)
    #         self.record_replacement(old_output_sexpr, concat_sexpr)
    #         relabel_mapping[output_id] = concat_sexpr.sexpr_id
    #     self.nx_graph = nx.relabel_nodes(self.nx_graph, relabel_mapping)

    def lower_all_reduce(self, dist_op_node_id, sorted_input_ids, sorted_output_ids):
        # Lower reduce_scatter into reduce_add/maximum and slices.
        dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
        dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
        # Remove edges around the dist_op.
        self.nx_graph.remove_node(dist_op_node_id)

        assert dist_op_sexpr.params[0] in (
            "sum",
            "max",
        ), "Only sum/max is supported now."
        reduce_op = (
            tgops.reduce_add if dist_op_sexpr.params[0] == "sum" else tgops.maximum
        )
        reduced_sexpr: SExpr = self.nx_graph.nodes[sorted_input_ids[0]]["sexpr"]
        n_dim = len(reduced_sexpr.shape)
        dim = 0  # reduce_scatter only supports dim=0 now.
        for input_id in sorted_input_ids[1:]:
            input_sexpr: SExpr = self.nx_graph.nodes[input_id]["sexpr"]
            new_shape = copy(reduced_sexpr.shape)
            new_shape[dim] += input_sexpr.shape[dim]
            new_reduced_sexpr = SExpr(
                reduce_op,
                [reduced_sexpr, input_sexpr],
                [],
                name=f"rs_add_{reduced_sexpr.sexpr_id}_{input_sexpr.sexpr_id}",
                shape=reduced_sexpr.shape,
            )
            self.nx_graph.add_node(new_reduced_sexpr.sexpr_id, sexpr=new_reduced_sexpr)
            self.nx_graph.add_edge(
                new_reduced_sexpr.sexpr_id, reduced_sexpr.sexpr_id, arg_idx=0
            )
            self.nx_graph.add_edge(
                new_reduced_sexpr.sexpr_id, input_sexpr.sexpr_id, arg_idx=1
            )
            reduced_sexpr = new_reduced_sexpr
            self.record_replacement(dist_op_sexpr, new_reduced_sexpr)
        # Replace outputs with simple clone.
        relabel_mapping = {}
        for arg_idx, output_id in enumerate(sorted_output_ids):
            output_attr = self.nx_graph.nodes[output_id]
            clone_sexpr = SExpr(
                tgops.clone,
                [reduced_sexpr],
                [],
                name=f"ag_clone_{reduced_sexpr.sexpr_id}",
                shape=reduced_sexpr.shape,
            )
            old_output_sexpr: SExpr = output_attr["sexpr"]
            output_attr["sexpr"] = clone_sexpr
            self.nx_graph.add_edge(output_id, reduced_sexpr.sexpr_id, arg_idx=arg_idx)
            self.record_replacement(old_output_sexpr, clone_sexpr)
            relabel_mapping[output_id] = clone_sexpr.sexpr_id
        self.nx_graph = nx.relabel_nodes(self.nx_graph, relabel_mapping)

    def lower_dist_ops(self) -> "SGraphTransformer":
        dist_op_nodes, dist_wait_nodes = SGraphTransformer.get_dist_op_nodes(
            self.nx_graph, return_waits=True
        )
        dist_op_nodes = sorted(
            dist_op_nodes, key=lambda x: self.nx_graph.nodes[x]["sexpr"].dist_id
        )

        # For dist ops, we lower the operation using only concat/split/slice/reduce-ops.
        for dist_op_node_id in dist_op_nodes:
            dist_op_attr = self.nx_graph.nodes[dist_op_node_id]
            assert (
                self.nx_graph.out_degree(dist_op_node_id) >= 1
            ), 'dist op should have at least one output, which is "wait".'
            dist_op_sexpr: SExpr = dist_op_attr["sexpr"]
            sorted_input_ids = sorted(
                self.nx_graph.successors(dist_op_node_id),
                key=lambda x: self.nx_graph.edges[dist_op_node_id, x]["arg_idx"],
            )
            sorted_output_ids = sorted(
                self.nx_graph.predecessors(dist_op_node_id),
                key=lambda x: self.nx_graph.edges[x, dist_op_node_id]["arg_idx"],
            )
            for output_id in sorted_output_ids:
                assert self.nx_graph.nodes[output_id]["sexpr"].op == tgops.dist_wait

            if len(sorted_input_ids) == 1:
                assert len(sorted_output_ids) == 1
                # If there is only one input and one output, we just remove it, also the waits.
                dist_op_sexpr: SExpr = self.nx_graph.nodes[dist_op_node_id]["sexpr"]
                wait_user_ids = list(self.nx_graph.predecessors(sorted_output_ids[0]))
                for wait_user_id in wait_user_ids:
                    edge_attr = self.nx_graph.edges[wait_user_id, sorted_output_ids[0]]
                    arg_idx = edge_attr["arg_idx"]
                    self.nx_graph.add_edge(
                        wait_user_id, sorted_input_ids[0], arg_idx=arg_idx
                    )
                if len(wait_user_ids) == 0:
                    # If the wait is the output node, we should not just remove it.
                    # We should also record replacement for it.
                    input_sexpr = self.nx_graph.nodes[sorted_input_ids[0]]["sexpr"]
                    wait_sexpr = self.nx_graph.nodes[sorted_output_ids[0]]["sexpr"]
                    self.record_replacement(wait_sexpr, input_sexpr)
                self.nx_graph.remove_node(dist_op_node_id)  # remove the dist op itself.
                self.nx_graph.remove_node(sorted_output_ids[0])  # remove the wait node.
            elif dist_op_sexpr.op == tgops.all_gather:
                self.lower_all_gather(
                    dist_op_node_id, sorted_input_ids, sorted_output_ids
                )
            elif dist_op_sexpr.op == tgops.reduce_scatter:
                self.lower_reduce_scatter(
                    dist_op_node_id, sorted_input_ids, sorted_output_ids
                )
            elif dist_op_sexpr.op == tgops.all_to_all_single:
                self.lower_all_to_all_single(
                    dist_op_node_id, sorted_input_ids, sorted_output_ids
                )
            elif dist_op_sexpr.op == tgops.all_reduce:
                self.lower_all_reduce(
                    dist_op_node_id, sorted_input_ids, sorted_output_ids
                )
            else:
                raise NotImplementedError(
                    f"Warning: dist op {dist_op_sexpr.op} is not handled yet."
                )

        return self

    def merge_clones(self) -> "SGraphTransformer":
        """
        Eliminate `clone` nodes by merging them into their children
        because they don't change the semantics.
        """
        for node_id, attr in self.nx_graph.nodes(data=True):
            sexpr = attr["sexpr"]
            if sexpr.op != tgops.clone:
                continue
            child_id = list(self.nx_graph.successors(node_id))[0]
            child_sexpr = self.nx_graph.nodes[child_id]["sexpr"]
            self.nx_graph = nx.contracted_nodes(
                self.nx_graph, child_id, node_id, self_loops=False
            )
            self.record_replacement(sexpr, child_sexpr)
        return self

    def sanity_check(self) -> "SGraphTransformer":
        try:
            # arg_idx in edge attr
            for u, v, attr in self.nx_graph.edges(data=True):
                u_sexpr = self.nx_graph.nodes[u]["sexpr"]
                v_sexpr = self.nx_graph.nodes[v]["sexpr"]
                assert (
                    "arg_idx" in attr
                ), f"Invalid edge attr: {attr=}, {u_sexpr!r}, {v_sexpr!r}"
            # No cycle
            cycle = nx.find_cycle(self.nx_graph)
            for idx, (u, v) in enumerate(cycle):
                u_sexpr = self.nx_graph.nodes[u]["sexpr"]
                v_sexpr = self.nx_graph.nodes[v]["sexpr"]
                if idx == 0:
                    print(f"{u_sexpr!r} --> ", end="")
                else:
                    print(f"{v_sexpr!r} --> ", end="")

            raise RuntimeError(
                "Cycle detected in the graph. There might be bug during transformation."
            )
        except nx.exception.NetworkXNoCycle:
            return self

    def rebuild_sepxrs(self) -> "SGraphTransformer":
        """
        During transformations, it is possible that we messed up the sexpr nodes' args
        referencings. But since we guarantee the reference in nx_graph is correct, we
        can always call this function to rebuild the affected sexprs to ensure the
        correctness.
        """
        self.sanity_check()
        relabel_mapping = {}
        for node_id, attr in self.nx_graph.nodes(data=True):
            sexpr: SExpr = attr["sexpr"]
            sorted_input_ids = sorted(
                self.nx_graph.successors(node_id),
                key=lambda x: self.nx_graph.edges[node_id, x]["arg_idx"],
            )
            if all(
                sexpr.args[arg_idx].sexpr_id == input_id
                for arg_idx, input_id in enumerate(sorted_input_ids)
            ):
                # If all the inputs are not changed, then it is not affected.
                continue

            # This sexpr is affected, rebuild it.
            # rich.print(f"Rebuilding affected sexpr({sexpr.name})", sexpr)
            new_sexpr = sexpr.clone_with(
                args=[
                    self.nx_graph.nodes[input_id]["sexpr"]
                    for input_id in sorted_input_ids
                ]
            )
            self.record_replacement(sexpr, new_sexpr)
            # rich.print(f"\tinto sexpr({new_sexpr.name})", new_sexpr)
            assert (
                sexpr.name == new_sexpr.name
            ), "Otherwise we need to update replacement."
            attr["sexpr"] = new_sexpr
            used_by_ids = list(self.nx_graph.predecessors(node_id))
            try:
                for used_by_id in used_by_ids:
                    arg_idx = self.nx_graph.edges[used_by_id, node_id]["arg_idx"]
                    self.nx_graph.nodes[used_by_id]["sexpr"].args[arg_idx] = new_sexpr
                relabel_mapping[node_id] = new_sexpr.sexpr_id
            except Exception as e:
                print("\n\n\n")
                sexpr = self.nx_graph.nodes[used_by_id]["sexpr"]
                rich.print(arg_idx)
                rich.print(sexpr)
                print(f"{sexpr.sexpr_id=}")
                raise e

        self.nx_graph = nx.relabel_nodes(self.nx_graph, relabel_mapping)
        return self

    def to_sgraph(self) -> SGraph:
        return SGraph.from_nx_graph(self.nx_graph, self.sexpr_order, self.outputs)
