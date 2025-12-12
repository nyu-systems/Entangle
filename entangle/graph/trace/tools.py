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
import pickle
import random

import matplotlib.pyplot as plt
import networkx as nx
import rich
import torch


def print_graph(graph, indent=0, file=None):
    print(f"nested graph: indent*{indent}")
    for node in graph.nodes():
        print(indent, "\t" * indent, node, end="", file=file)
        if node.hasAttribute("Subgraph"):
            subgraph = node.g("Subgraph")
            print_graph(subgraph, indent + 1, file=file)
    print(
        "\t" * indent,
        f"return ({','.join(['%'+output.debugName() for output in graph.outputs()])})",
        file=file,
    )


def add_value_node(graph, node) -> str:
    node_label = node.debugName()
    vtype = node.type().str()
    attributes = {"is_op": False, "vtype": vtype}
    print(f"added input node {node_label}, {attributes=}")
    graph.add_node(node_label, **attributes)
    return node_label


def dump_graph(traced_graph, path):
    graph = nx.DiGraph()
    for input_ in traced_graph.inputs():
        add_value_node(graph, input_)
    for node in list(traced_graph.nodes()):
        # FIXME: make it unique
        nid = random.randint(0, 2 << 31)
        node_label = f"{node.kind()}__{nid}"
        node_attr = {"is_op": True, "kind": node.kind()}
        for name in node.attributeNames():
            kindOfAttr = node.kindOf(name)
            node_attr[name] = eval(f"node.{kindOfAttr}(name)")
        graph.add_node(node_label, **node_attr)
        print(f"added op node {node_label}, {node_attr=}")
        # Connect inputs to this op
        for idx, input_ in enumerate(node.inputs()):
            edge_attr = {"idx": idx, "dir": "in"}
            print(f"added edge {input_.debugName()} -> {node_label}, {edge_attr=}")
            graph.add_edge(input_.debugName(), node_label, **edge_attr)
        # Connect this op to outputs
        for idx, output in enumerate(node.outputs()):
            output_label = add_value_node(graph, output)
            edge_attr = {"idx": idx, "dir": "out"}
            print(f"added edge {node_label} -> {output_label}, {edge_attr=}")
            graph.add_edge(node_label, output_label, **edge_attr)

    for i, node in enumerate(graph):
        print(i, graph.nodes[node], graph.nodes[node])

    with open(path, "wb") as f:
        pickle.dump(graph, f, pickle.HIGHEST_PROTOCOL)
        print(f"graph saved to {path}")


def trace_and_dump(model, example_inputs, dirname, rank=None):
    torch._C._jit_set_inline_everything_mode(True)
    torch.onnx._globals.GLOBALS.autograd_inlining = True
    traced_model = torch.jit.trace(model, example_inputs, check_trace=False)
    with open(osp.join(dirname, f"{rank}.log"), "w") as file:
        print("----------------------------------- model", file=file)
        print(traced_model.graph)
    dump_graph(traced_model.graph, osp.join(dirname, f"{rank}.gpickle"))
    return traced_model
