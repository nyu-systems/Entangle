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

import itertools
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import networkx as nx

OP_COLOR = "#007CBE"
VALUE_COLOR = "#FFA3AF"
INPUT_COLOR = "#00AF54"
OUTPUT_COLOR = "#FBAF00"


NODE_LABEL_ATTR_EXCLUDE = ["rank", "label", "kind", "vtype", "is_op", "color", "font"]


def rank_graph(graph: nx.DiGraph, rank: int) -> nx.DiGraph:
    ranked_graph = nx.DiGraph()
    for node, attr in graph.nodes(data=True):
        attr = attr.copy()
        attr["label"] = f"{rank}: {attr['label']}"
        ranked_graph.add_node(f"{rank}: {node}", **attr)
    for edge in graph.edges:
        ranked_edge_attr = graph.edges[edge]
        ranked_edge_attr["idx"] = f'{rank}: {ranked_edge_attr["idx"]}'
        ranked_graph.add_edge(
            f"{rank}: {edge[0]}", f"{rank}: {edge[1]}", **ranked_edge_attr
        )
    return ranked_graph


def set_node_label(node, node_attr):
    node_attr["label"] = "\n".join(
        [node]
        + [
            f"{k}: {v}"
            for k, v in node_attr.items()
            if k not in NODE_LABEL_ATTR_EXCLUDE
        ]
    )


def load_graph(file_path: str, rank: int = None) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph = pickle.load(open(file_path, "rb"))
    # node color
    topo_order = list((nx.topological_generations(graph)))
    for node in topo_order[0]:
        graph.nodes[node]["color"] = INPUT_COLOR
    for node in topo_order[-1]:
        graph.nodes[node]["color"] = OUTPUT_COLOR
    for nodes in topo_order[1:-1]:
        for node in nodes:
            node_attr = graph.nodes[node]
            node_attr["color"] = OP_COLOR if node_attr["is_op"] else VALUE_COLOR

    # node label, rank
    for n in graph:
        node_attr = graph.nodes[n]
        set_node_label(n, node_attr)
        if rank is not None:
            node_attr["rank"] = rank

    # edge label
    for u, v, attr in graph.edges(data=True):
        attr["label"] = str(attr["idx"])

    if rank is None:
        return graph
    else:
        return rank_graph(graph, rank)


def composite_collectives(graphs: list[nx.DiGraph], group_id_to_name: dict[int, str]):
    assert type(graphs) == list
    Collective = namedtuple("Collective", ["rank", "node", "group_name", "tag"])

    def find_collectives(graph, rank) -> list[Collective]:
        collectives = []
        for node in graph:
            if node.find("dist::") != -1:
                attr = graph.nodes[node]
                collectives.append(
                    Collective(
                        rank, node, group_id_to_name[attr["group_id"]], attr["tag"]
                    )
                )
        return collectives

    collectives_per_rank = []
    for rank, graph in enumerate(graphs):
        collectives_per_rank.append(find_collectives(graph, rank))
    collectives = list(itertools.chain(*collectives_per_rank))
    collective_id_set = set(
        [(collective.group_name, collective.tag) for collective in collectives]
    )
    composited_collectives = {k: [] for k in collective_id_set}
    for collective in collectives:
        composited_collectives[(collective.group_name, collective.tag)].append(
            collective
        )
    for k in composited_collectives:
        composited_collectives[k] = sorted(
            composited_collectives[k], key=lambda x: x.rank
        )
    return composited_collectives


def merge_ranked_graphs(
    ranked_graphs: list[nx.DiGraph], group_id_to_name: dict[int, str]
) -> nx.DiGraph:
    composited_collectives = composite_collectives(
        ranked_graphs, group_id_to_name=group_id_to_name
    )

    merged_graph = nx.DiGraph()
    for graph in ranked_graphs:
        merged_graph = nx.compose(merged_graph, graph)

    for composite in composited_collectives:
        # merged_graph = nx.contracted_nodes(merged_graph, '0: dist::allgather_base__360511942', '1: dist::allgather_base__360511942')
        merged_graph = nx.contracted_nodes(
            merged_graph,
            *[collective.node for collective in composited_collectives[composite]],
        )

    return merged_graph


def collapse_leaf_accessors(graph: nx.DiGraph, inplace=True):
    if not inplace:
        graph = graph.copy()
    frontiers = next(nx.topological_generations(graph))
    while len(frontiers) > 0:
        next_frontiers = []
        for parent in frontiers:
            nodes_to_remove = []
            all_children_remove = True
            ops = list(graph.successors(parent))
            for op in ops:
                op_attr = graph.nodes[op]
                if op_attr["is_op"] and op_attr["kind"] == "prim::GetAttr":
                    assert graph.out_degree(op) == 1
                    child = next(graph.successors(op))
                    child_attr = graph.nodes[child]
                    new_child = f'{parent}->{op_attr["name"]}'
                    graph.add_node(new_child, **child_attr)
                    set_node_label(new_child, graph.nodes[new_child])
                    for _, succ, edge_attr in graph.edges(child, data=True):
                        graph.add_edge(new_child, succ, **edge_attr)
                    graph.add_edges_from(
                        [(new_child, succ) for succ in graph.successors(child)]
                    )
                    graph.remove_node(child)
                    next_frontiers.append(new_child)
                    nodes_to_remove.append(op)
                else:
                    all_children_remove = False
            if all_children_remove:
                nodes_to_remove.append(parent)
            graph.remove_nodes_from(nodes_to_remove)
        frontiers = next_frontiers
    return graph


def draw_nx(graph):
    reversed_graph = graph.reverse(copy=True)

    def get_dag_pos(graph, return_max_layer=False):
        max_layer = 0
        for layer, nodes in enumerate(nx.topological_generations(graph)):
            for node in nodes:
                graph.nodes[node]["layer"] = layer
            max_layer = max(layer, max_layer)

        pos = nx.multipartite_layout(graph, subset_key="layer")
        if return_max_layer:
            return pos, max_layer
        else:
            return pos

    reversed_pos = get_dag_pos(reversed_graph)

    plt.figure(figsize=(64, 16))
    pos = {node: [-p[0] * 3, p[1]] for node, p in reversed_pos.items()}
    color_map = [
        OP_COLOR if graph.nodes[node]["is_op"] else VALUE_COLOR for node in graph
    ]
    labels = {}
    for node in graph:
        labels[node] = graph.nodes[node]["label"]

    nx.draw(graph, pos=pos, ax=plt.gca(), labels=labels, node_color=color_map)
    nx.draw_networkx_edge_labels(
        graph, pos=pos, ax=plt.gca(), edge_labels=nx.get_edge_attributes(graph, "label")
    )
