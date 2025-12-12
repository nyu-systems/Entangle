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

import matplotlib.pyplot as plt
import networkx as nx

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

DEFAULT_COLOR = "skyblue"
INPUT_COLOR = "gold"
VIEW_COLOR = "darkgrey"
DIST_COLOR = "lightcoral"


def setup(graph: nx.DiGraph, inplace=False):
    if not inplace:
        graph = graph.copy()

    for _, attr in graph.nodes(data=True):
        if "tensor_shape" in attr and attr["tensor_shape"] != None:
            attr["label"] = attr["label"] + "\n" + str(tuple(attr["tensor_shape"]))

    for node in graph:
        graph.nodes[node]["font"] = {"size": 18}

    for u, v, attr in graph.edges(data=True):
        attr["font"] = {"size": 16}

    for node, attr in graph.nodes(data=True):
        op = attr["op"]
        target = attr["target"]
        if op == "call_function":
            if target.startswith("_c10d_functional"):
                attr["color"] = DIST_COLOR
            elif target in (
                "<built-in function getitem>",
                "aten.clone.default",
                "aten.expand.default",
                "aten.permute.default",
                "aten.repeat.default",
                "aten.slice.Tensor",
                "aten.split_with_sizes.default",
                "aten.t.default",
                "aten.transpose.int",
                "aten.view.default",
                "aten.constant_pad_nd.default",
            ):
                attr["color"] = VIEW_COLOR
            elif target in ("aten.empty.memory_format",):
                attr["color"] = INPUT_COLOR
        elif op == "placeholder":
            attr["color"] = INPUT_COLOR
        else:
            attr["color"] = DEFAULT_COLOR
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
