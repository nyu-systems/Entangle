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

import os.path as osp

import matplotlib.pyplot as plt
import networkx as nx

import entangle.ops as tgops
from entangle.sgraph.sexpr import SExpr
from entangle.utils import visual_utils
from entangle.utils.print_utils import BRI, RST

PYVIS_JS_OPTIONS = """
const options = {
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "LR",
      "sortMethod": "directed"
    }
  }
}
"""

DEFAULT_COLOR = "crimson"
INPUT_COLOR = "gold"
CONSTANT_COLOR = "gold"
VIEW_COLOR = "darkgrey"
SKELETON_COLOR = "skyblue"
DIST_COLOR = "lightcoral"
OUTPUT_COLOR = "darkviolet"


def setup(graph: nx.DiGraph, inplace=False):
    if not inplace:
        graph = graph.copy()

    for node in graph:
        graph.nodes[node]["font"] = {"size": 18}

    for u, v, attr in graph.edges(data=True):
        attr["font"] = {"size": 16}

    for node, attr in graph.nodes(data=True):
        sexpr: SExpr = attr["sexpr"]
        op = sexpr.op  # if type(sexpr.op) is str else sexpr.op.value
        if sexpr.op.dist:
            attr["color"] = DIST_COLOR
        elif op.noncompute:
            attr["color"] = VIEW_COLOR
        elif op.skeleton:
            attr["color"] = SKELETON_COLOR
        elif op.constant:
            attr["color"] = CONSTANT_COLOR
        elif op in (tgops.inpt, tgops.weight):
            attr["color"] = INPUT_COLOR
        else:
            attr["color"] = DEFAULT_COLOR
        if graph.in_degree(node) == 0:
            attr["shape"] = "star"
            attr["color"] = OUTPUT_COLOR
        if sexpr.log_name is not None:
            attr["label"] = sexpr.name + "\n" + sexpr.log_name + "\n" + repr(sexpr.op)
        else:
            if sexpr.name is None:
                print(sexpr, sexpr.sexpr_id)
            attr["label"] = sexpr.name + "\n" + repr(sexpr.op)
    for _, attr in graph.nodes(data=True):
        sexpr = attr["sexpr"]
        if sexpr.dist_id is not None:
            id_str = f"[{sexpr.sexpr_id}|{sexpr.dist_id}]"
        else:
            id_str = f"[{sexpr.sexpr_id}]"
        if sexpr.shape != None:
            attr["label"] = (
                attr["label"] + f"\n{id_str}|{sexpr.shape.parentheses_str()}"
            )
        attr.pop("sexpr")
        if "contraction" in attr:
            attr.pop("contraction")
    for u, v, attr in graph.edges(data=True):
        if graph.out_degree(u) > 1:
            attr["label"] = "" if "arg_idx" not in attr else str(attr["arg_idx"])
    return graph


def draw_nx(graph: nx.DiGraph):
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

    plt.figure(figsize=(8, 128))
    pos = {node: [p[1], -p[0] * 128] for node, p in reversed_pos.items()}
    import rich

    rich.print(pos)
    labels = {}
    for node in graph:
        labels[node] = graph.nodes[node]["label"]

    nx.draw(graph, pos=pos, ax=plt.gca(), labels=labels)
    nx.draw_networkx_edge_labels(
        graph, pos=pos, ax=plt.gca(), edge_labels=nx.get_edge_attributes(graph, "label")
    )
    plt.show()
    plt.pause(0)


def visualize_sgraph_to_infer(
    origin_sgraph, target_sgraphs, output_dir, graph_prefix: str = None
):
    """
    Helper to visualize sgraph to infer in a single web page.
    `graph_prefix`: the prefix of the graph file name.
    """
    import entangle.sgraph.visualization as sgraph_viz

    html_body = ""
    height_percent = 100 // (1 + len(target_sgraphs))
    filenames = []
    for idx, sgraph in enumerate([origin_sgraph, *target_sgraphs]):
        file_prefix = "origin" if idx == 0 else f"target"
        rank = 0 if idx == 0 else idx - 1
        if graph_prefix is None or graph_prefix == "":
            filename = f"{file_prefix}.r{rank}.html"
        else:
            filename = f"{file_prefix}.{graph_prefix}.r{rank}.html"
        filenames.append(filename)
        output_path = osp.join(output_dir, filename)
        nx_graph = sgraph_viz.setup(sgraph.nx_graph)
        visual_utils.draw_pyvis(nx_graph, output_path=output_path, reverse=True)
        html_body += f"""<h3>{filename.removesuffix('.html')}</h3><iframe src="{filename}" id="origin" style="width: 95%; height: {height_percent}%;"></iframe>\n"""
    html = f"<html><head></head><body>{html_body}</body></html>\n"
    index_html_path = osp.join(output_dir, "index.html")
    with open(index_html_path, "w") as f:
        f.write(html)
    print(f"{BRI}Collected visualization{RST} into {index_html_path}, {filenames=}")
