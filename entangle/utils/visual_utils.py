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

import networkx as nx
import rich
from pyvis.network import Network

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


def draw_pyvis(graph: nx.DiGraph, output_path: str, reverse: bool = True):
    simplified_graph = nx.DiGraph()
    for node_id, attr in graph.nodes(data=True):
        simplified_graph.add_node(
            node_id,
            **{k: attr[k] for k in ("label", "color", "font", "shape") if k in attr}
        )
    for u, v, attr in graph.edges(data=True):
        simplified_graph.add_edge(u, v, **attr)
    net = Network(directed=True)
    # net.show_buttons(filter_=["layout","nodes"])
    net.set_options(PYVIS_JS_OPTIONS)
    if reverse:
        simplified_graph = simplified_graph.reverse()
    net.from_nx(simplified_graph)
    net.save_graph(output_path)
