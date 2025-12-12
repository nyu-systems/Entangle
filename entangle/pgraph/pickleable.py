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

import logging
import os
import os.path as osp
import pickle
import re
import sys
from typing import Any, Callable

import networkx as nx
import rich
import sympy
import torch
import torch.fx
import torch.fx.immutable_collections
import torch.fx.traceback
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.graph import CodeGen
from torch.fx.passes.shape_prop import TensorMetadata

from entangle.sym.sym_manager import SymManager


TORCH_VERSION = tuple(map(int, torch.__version__.split(".")[:2]))


def assert_torch_version():
    return TORCH_VERSION[0] == 2 and TORCH_VERSION[1] >= 2


def make_arg_to_pickleable(node_dict):
    def arg_to_pickleable(arg, node_dict):
        type_arg = type(arg)
        if type_arg in (bool, int, float, complex, type(None)):
            return arg
        elif type_arg is str:
            assert (
                not arg.startswith("n__")
                and 'Should not start with "n__" because we use it mark nodes'
            )
            return arg
        elif type_arg in (list, tuple, torch.fx.immutable_collections.immutable_list):
            return type_arg([arg_to_pickleable(a) for a in arg])
        elif type_arg in (dict, torch.fx.immutable_collections.immutable_dict):
            for k in arg.keys():
                assert type(k) is str
            return {k: arg_to_pickleable(v) for k, v in arg.items()}
        elif type_arg is torch.fx.Node:
            return f"n__{arg.name}"
        elif type_arg in (torch.dtype, torch.device, torch.memory_format):
            return arg
        elif type_arg is torch.layout:
            return str(arg)
        else:
            raise ValueError(f"Unknown type: {type_arg}, {arg=}")

    return arg_to_pickleable


class PickleableNode:
    META_SPECIAL_KEYS = (
        # These three need to be handled specially.
        "eager_input_vals",
        "example_value",
        "from_node",
        "fwd_nn_module_stack",
        "fwd_source_fn_stack",
        "nn_module_stack",
        "original_aten",  # `Target` can cover the information
        "source_fn_stack",
        "tensor_dict",
        "grapharg",  # TODO: This is example/concrete arguments. It can be interesting to analyze. But ignore for now.
    )

    @staticmethod
    def pickleable_meta(meta: dict[str, Any]) -> dict[str, Any]:
        pmeta = {
            k: v for k, v in meta.items() if k not in PickleableNode.META_SPECIAL_KEYS
        }
        if "nn_module_stack" in meta:
            nn_module_stack = meta["nn_module_stack"]
            for k, v in nn_module_stack.items():
                assert type(k) is str
                assert type(v) is tuple and len(v) == 2
            pmeta["nn_module_stack"] = {
                k: (v[0], repr(v[1])) for k, v in nn_module_stack.items()
            }
        if "source_fn_stack" in meta:
            source_fn_stack = meta["source_fn_stack"]
            for frame in source_fn_stack:
                assert type(frame) is tuple and type(frame[0]) is str
            pmeta["source_fn_stack"] = [
                (frame[0], repr(frame[1])) for frame in source_fn_stack
            ]
        if "from_node" in meta:
            from_node = meta["from_node"]
            if TORCH_VERSION[0] == 2 and TORCH_VERSION[1] < 7:
                # Legacy `from_node`
                for frame in from_node:
                    assert type(frame) is tuple and type(frame[0]) is str, str(frame)
                pmeta["from_node"] = [(frame[0], repr(frame[1])) for frame in from_node]
            elif TORCH_VERSION[0] == 2 and TORCH_VERSION[1] >= 7:
                for node_src in from_node:
                    assert type(node_src) is torch.fx.traceback.NodeSource, str(
                        node_src
                    )
                pmeta["from_node"] = node_src
            else:
                raise RuntimeError(f"Unsupported torch version: {torch.__version__}")
        if "val" in meta:
            val = meta["val"]
            if isinstance(val, py_sym_types):
                val_str = CodeGen._sym_repr(val)
                maybe_type_annotation = f"Sym({val_str})"
                pmeta["val"] = maybe_type_annotation
            else:
                pmeta["val"] = str(val)
        return pmeta

    def __init__(self, rank, op, name, target, args, kwargs, meta=None):
        self.rank: int = rank
        self.op: str = op
        self.name: str = name
        self.target: str = target
        self.args: list[PickleableNode | list | tuple] = args
        self.kwargs: dict = kwargs
        self.meta: dict = meta or {}
        if "tensor_meta" in self.meta:
            tmeta = self.meta["tensor_meta"]
            sym_to_str = lambda x: (
                SymManager.rank_expr(x.node.expr, self.rank)
                if type(x) is torch.SymInt
                else x
            )
            self.meta["tensor_meta"] = TensorMetadata(
                shape=tuple(map(sym_to_str, tmeta.shape)),
                dtype=tmeta.dtype,
                requires_grad=tmeta.requires_grad,
                stride=(
                    None
                    if tmeta.stride is None
                    else tuple(map(sym_to_str, tmeta.stride))
                ),
                memory_format=tmeta.memory_format,
                is_quantized=tmeta.is_quantized,
                qparams=tmeta.qparams,
            )

    @staticmethod
    def from_fx(node: torch.fx.Node, rank: int, owner_graph: "PickleableGraph"):
        return PickleableNode(
            rank,
            str(node.op),
            node.name,
            str(node.target),
            owner_graph.arg_to_pickleable(node.args),
            owner_graph.arg_to_pickleable(node.kwargs),
            PickleableNode.pickleable_meta(node.meta),
        )

    def remap(self, owner_graph: "PickleableGraph") -> "PickleableNode":
        """
        When transforming, we may want to remap some args.
        """
        return PickleableNode(
            self.rank,
            self.op,
            self.name,
            self.target,
            owner_graph.arg_to_pickleable(self.args, remap=True),
            owner_graph.arg_to_pickleable(self.kwargs, remap=True),
            self.meta,
        )

    @staticmethod
    def flatten_args_of_node(args):
        if type(args) is PickleableNode:
            return [args]
        elif type(args) in (list, tuple):
            return [a for arg in args for a in PickleableNode.flatten_args_of_node(arg)]
        elif type(args) in (dict,):
            return [
                a
                for arg in args.values()
                for a in PickleableNode.flatten_args_of_node(arg)
            ]
        else:
            return []

    def direct_dependents(self) -> list["PickleableNode"]:
        return PickleableNode.flatten_args_of_node(
            self.args
        ) + PickleableNode.flatten_args_of_node(self.kwargs)

    @property
    def repr(self):
        return self.__repr__()

    def __repr__(self):
        return PickleableNode.repr_node(self.rank, self.name)

    @staticmethod
    def _stringfy_args(args):
        args_type = type(args)
        if args_type in (bool, int, float, complex, type(None), str):
            return args
        elif args_type is PickleableNode:
            return args.repr
        elif args_type in (tuple, list, torch.fx.immutable_collections.immutable_list):
            return args_type([PickleableNode._stringfy_args(a) for a in args])
        elif args_type in (dict, torch.fx.immutable_collections.immutable_dict):
            return {k: PickleableNode._stringfy_args(v) for k, v in args.items()}
        else:
            return str(args)

    def __str__(self):
        ret = f"Node[rank={self.rank}](op={self.op}, name={self.name}, target='{self.target}', args={PickleableNode._stringfy_args(self.args)}, kwargs={PickleableNode._stringfy_args(self.kwargs)}"
        keys = set(self.meta.keys())
        if "tensor_meta" in self.meta:
            tmeta = self.meta["tensor_meta"]
            type_and_shape = (
                f"{str(tmeta.dtype).removeprefix('torch.')}{list(tmeta.shape)}"
            )
            ret += f", tensor_meta={type_and_shape}"
            keys.remove("tensor_meta")
        if "from_node" in self.meta:
            from_node_str = str(self.meta["from_node"]).strip(" \n")
            ret += f", from_node={from_node_str}"
            keys.remove("from_node")
        if "nn_module_stack" in self.meta:
            nn_stack = [frame[0] for frame in self.meta["nn_module_stack"].values()]
            ret += f", nn_stack={nn_stack}"
            keys.remove("nn_module_stack")
        if "source_fn_stack" in self.meta:
            fn_stack = [frame[0] for frame in self.meta["source_fn_stack"]]
            ret += f", fn_stack={fn_stack}"
            keys.remove("source_fn_stack")
        filtered_meta = {k: v for k, v in self.meta.items() if k in keys}
        ret += f", other_meta={filtered_meta}"
        ret += ")"
        return ret

    @property
    def tensor_meta(self):
        if "tensor_meta" not in self.meta:
            raise RuntimeError(
                f"No tensor_meta for node {self}, available meta keys: {self.meta.keys()}"
            )
        return self.meta["tensor_meta"]

    def has_tensor_meta(self):
        return "tensor_meta" in self.meta

    @property
    def tensor_shape(self):
        return self.tensor_meta.shape

    def get_tensor_shape(self):
        if self.has_tensor_meta():
            return self.tensor_shape
        else:
            return None

    def is_sym_scalar(self):
        # it can be either a placeholder or converting a tensor to sym scalar.
        return (
            "_local_scalar_dense" in self.target
            or self.target == "aten.sym_size.int"
            or self.is_sym_bool()
            or self.name.startswith("_tensor_constant")
        )

    def get_scalar_name(self) -> str:
        if (val := self.meta.get("val", None)) and val.find("Sym") != -1:
            val: str
            assert not self.is_sym_bool(), f"SymBool is not a name: {self}"
            return val.removeprefix("Sym(").removesuffix(")")
        if (
            (from_node := self.meta.get("from_node", None))
            and TORCH_VERSION[0] == 2
            and TORCH_VERSION[1] >= 7
            and type(from_node) is torch.fx.traceback.NodeSource
        ):
            # At least since torch 2.7
            name = from_node.name
        else:
            unbacked_bindings = self.meta.get("unbacked_bindings", None)
            assert (
                unbacked_bindings is not None
            ), f"{self}\nkeys={list(self.meta.keys())}"
            # TODO: Remove this debug code when I find documentation for unbacked_bindings.
            # print(f"{self.name}")
            # for k, v in unbacked_bindings.items():
            #     print(f"\t{k}<{type(k)}>: {v}<{type(v)}>")
            from entangle.utils.print_utils import BRED, RST

            print(f"{BRED}HACKED {self.name} value{RST}")
            assert type(unbacked_bindings) is dict
            assert (
                len(unbacked_bindings) == 1
            ), f"Expected 1 binding, but got {unbacked_bindings=}"
            k = list(unbacked_bindings.keys())[0]
            v = unbacked_bindings[k]
            assert len(v) == 0, "otherwise unknown case."
            assert "_" not in str(
                k
            ), f"got name {k} with underscore, which hinders shape representation."
            name = str(k)

        name = SymManager.repr_scalar_name(self.rank, name)
        return name

    @staticmethod
    def repr_node(rank: int, name: str):
        return f"n__r{rank}__{name}"

    @staticmethod
    def is_node_repr_name(name: str):
        return name.startswith("n__r")

    @staticmethod
    def remove_node_repr_prefix(name: str) -> str:
        assert PickleableNode.is_node_repr_name(name)
        return re.match(r"n__r\d+__(\w+)", name).group(1)

    def is_sym_bool(self):
        if (val := self.meta.get("val", None)) and val.find("Sym") != -1:
            val: str
            return ">" in val or "<" in val or "=" in val
        return False


class PickleableGraph:
    def __init__(
        self, rank=None, direction=None, gid=None, readable_str: str = None, **kwargs
    ):
        self.nodes: list[PickleableNode] = []
        self.node_dict = {}
        self.direction = direction
        self.gid = gid
        self.sanity_check()
        self.rank = rank
        self.readable_str: str = readable_str

    def add_node(self, node: PickleableNode):
        self.nodes.append(node)
        self.node_dict[repr(node)] = node

    def arg_to_pickleable(self, arg, remap: bool = False):
        type_arg = type(arg)
        if type_arg in (bool, int, float, complex, type(None)):
            return arg
        elif type_arg is str:
            if PickleableNode.is_node_repr_name(arg):
                return self.node_dict[
                    PickleableNode.repr_node(
                        self.rank, PickleableNode.remove_node_repr_prefix(arg)
                    )
                ]
            else:
                return arg
        elif type_arg in (list, tuple, torch.fx.immutable_collections.immutable_list):
            if type_arg == torch.fx.immutable_collections.immutable_list:
                # Force converting to list
                type_arg = list
            return type_arg([self.arg_to_pickleable(a, remap) for a in arg])
        elif type_arg in (dict, torch.fx.immutable_collections.immutable_dict):
            for k in arg.keys():
                assert type(k) is str
            return {k: self.arg_to_pickleable(v, remap) for k, v in arg.items()}
        elif type_arg is torch.fx.Node:
            return self.node_dict[PickleableNode.repr_node(self.rank, arg.name)]
        elif type_arg in (torch.dtype, torch.device, torch.memory_format):
            return arg
        elif type_arg is torch.layout:
            return str(arg)
        elif str(type_arg) in ("<class 'torch._ops.OpOverload'>", "<class 'ellipsis'>"):
            return str(arg)
        elif type_arg in (slice,):
            return str(type_arg)
        elif remap and type_arg == PickleableNode:
            return self.node_dict[repr(arg)]
        else:
            raise ValueError(f"Unknown type: {type_arg}, {arg=}")

    def sanity_check(self):
        # All nodes used are defined before
        defined = set()
        for n in self.nodes:
            for arg in n.args:
                if type(arg) is str and PickleableNode.is_node_repr_name(arg):
                    if arg not in defined:
                        raise RuntimeError(f"arg {arg} not in defined set: {defined}")
            defined.add(repr(n))

    def __str__(self):
        return (
            f"Graph[rank={self.rank}]({self.direction}, gid={self.gid}) ["
            + "".join(["\n    " + str(n).replace("\n", "\n    ") for n in self.nodes])
            + "\n]"
        )

    def save(self, path: str):
        # Check pickleable for each node.
        recur_limit = sys.getrecursionlimit()
        print("Setting recursion limit to", len(self.nodes) * 2)
        sys.setrecursionlimit(len(self.nodes) * 2)
        for node in self.nodes:
            try:
                pickle.dumps(node)
            except Exception as e:
                rich.print(node)
                logging.error(f"Error when pickling node: {node}")
                raise e
        with open(path, "wb") as f:
            pickle.dump(self, f)
        sys.setrecursionlimit(recur_limit)

    @staticmethod
    def load(path: str) -> "PickleableGraph":
        with open(path, "rb") as f:
            obj = pickle.load(f)
            assert type(obj) is PickleableGraph, f"{type(obj)=}"
            return obj

    @staticmethod
    def from_text(text: str) -> "PickleableGraph":
        lines = text.strip("\n").split("\n")
        matched = re.match(r"Graph\[rank=(\d+)\]\((\w+), gid=(\d+)\)", lines[0])
        assert (
            matched is not None
        ), "First line should be Graph[rank=<rank>](<direction>, <gid>)"
        assert lines[-1] == "]", "Last line should be ']'"
        rank = int(matched.group(1))
        direction = matched.group(2)
        gid = int(matched.group(3))

        from torch import device  # The function `empty` requires device.

        pgraph = PickleableGraph(rank=rank, direction=direction, gid=gid)
        for line in lines[1:-1]:
            matched = re.match(
                r" *Node\[rank=(\d+)\]\(op=(\w+), name=(\w+), target='(.+)', args=(\(.*\)), kwargs=(\{.*\})(, tensor_meta=(\w+)(\[[\w\d, ]+\]))?",
                line,
            )
            assert matched is not None, f"Unable to parse line: {line}"
            rank = int(matched.group(1))
            op = matched.group(2)
            name = matched.group(3)
            target = matched.group(4)
            args = pgraph.arg_to_pickleable(eval(matched.group(5)))
            kwargs = pgraph.arg_to_pickleable(eval(matched.group(6)))
            meta = {}
            if matched.group(7) is not None:
                tensor_type = matched.group(8)
                tensor_shape = eval(matched.group(9))
                tensor_meta = TensorMetadata(
                    tensor_shape,
                    tensor_type,
                    requires_grad=False,
                    stride=None,
                    memory_format=None,
                    is_quantized=False,
                    qparams={},
                )
                meta["tensor_meta"] = tensor_meta
            node = PickleableNode(rank, op, name, target, args, kwargs, meta)
            pgraph.add_node(node)
        pgraph.sanity_check()
        return pgraph

    def to_nx(self, ignore_output=True) -> nx.DiGraph:
        arg_repr_to_node = {}
        g = nx.DiGraph()
        get_node_attr = lambda node: {
            "label": node.repr,
            "op": node.op,
            "target": node.target,
            "tensor_shape": node.get_tensor_shape(),
        }
        for node in self.nodes:
            if node.op == "output" and ignore_output:
                if len(node.args[0]) > 0:
                    output_node_repr = node.args[0][0].repr
                continue
            arg_repr_to_node[node.repr] = node
            node_attr = get_node_attr(node)
            g.add_node(node.repr, **node_attr)
            for arg in node.direct_dependents():
                if type(arg) is PickleableNode and PickleableNode.is_node_repr_name(
                    arg.repr
                ):
                    arg_attr = get_node_attr(arg)
                    g.add_node(arg.repr, **arg_attr)
                    g.add_edge(arg.repr, node.repr)

        # leaves_to_remove = []
        # for node in g.nodes:
        #     if g.out_degree(node) == 0:
        #         if node != output_node_repr:
        #             leaves_to_remove.append(node)
        # g.remove_nodes_from(leaves_to_remove)
        return g


def to_pickleable_graph(graph: torch.fx.Graph, rank=None, **kwargs):
    pgraph = PickleableGraph(rank=rank, **kwargs)
    for n in graph.nodes:
        pnode = PickleableNode.from_fx(n, rank=rank, owner_graph=pgraph)
        pgraph.add_node(pnode)
    pgraph.sanity_check()
    return pgraph


def collapse_log_tensor(
    pg: PickleableGraph, helper_pg: PickleableGraph = None
) -> PickleableGraph:
    """
    helper_pg: The fw and bw graph are actually auto partitioned from `both` graph.
        And some of the log_tensor in bw will actually appear in fw. This parameter
        is the `both` graph that can provide log_tensor information when collapsing
        the bw graph.
    """
    # FIXME: Using __dict__ because the dumped gpickles may not be the same version of
    # current codes.

    # 1. Collapsing the log_tensor nodes
    new_pg = PickleableGraph(**pg.__dict__)
    for n in pg.nodes:
        if (
            n.target == "<built-in function getitem>"
            and n.args[0].target.startswith("auto_functionalized")
            and n.args[0].args[0] == "tg.inplace_log_tensor.default"
        ):
            # No need to add this node, instead, re-map node_dict to original tensor
            if n.args[0].target == "auto_functionalized":
                t = n.args[0].kwargs["t"]
            else:
                assert n.args[0].target == "auto_functionalized_v2"
                t = n.args[0].kwargs["_all_bases"][0]
            s = n.args[0].kwargs["s"]
            new_t = new_pg.node_dict[repr(t)]
            new_pg.node_dict[repr(n)] = new_t
            assert type(t) is PickleableNode
        elif (
            n.target == "<built-in function getitem>"
            and n.args[0].target.startswith("auto_functionalized")
            and n.args[0].args[0] == "tg.inplace_log_grad.default"
        ):
            if n.args[0].target == "auto_functionalized":
                t = n.args[0].kwargs["t"]
                grad = n.args[0].kwargs["grad"]
            else:
                assert n.args[0].target == "auto_functionalized_v2"
                t = n.args[0].kwargs["_all_bases"][0]
                grad = n.args[0].kwargs["_all_bases"][1]
            new_grad = new_pg.node_dict[repr(grad)]
            new_pg.node_dict[repr(n)] = new_grad
            assert type(t) is PickleableNode
        elif n.target.startswith("auto_functionalized") and n.args[0] in (
            "tg.inplace_log_tensor.default",
            "tg.inplace_log_grad.default",
        ):
            # Just don't add this node.
            pass
        else:
            new_n = n.remap(new_pg)
            new_pg.add_node(new_n)

    # 2. Rename nodes if name available.
    # A map from original tensor's name to the log name of the tensor.
    name_mapper: dict[str, str] = {}

    # We should set the very first node's name that is not a clone-like op.
    def get_first_non_clone_like_node(node: PickleableNode):
        if node.target == "aten.copy.default":
            return get_first_non_clone_like_node(node.args[1])
        elif node.target == "aten.detach.default":
            return get_first_non_clone_like_node(node.args[0])
        elif node.target == "aten.clone.default":
            return get_first_non_clone_like_node(node.args[0])
        else:
            return node

    def setup_name_mapper(pg: PickleableGraph):
        for n in pg.nodes:
            if n.target == "<built-in function getitem>" and n.args[
                0
            ].target.startswith("auto_functionalized"):
                if n.args[0].args[0] == "tg.inplace_log_tensor.default":
                    if n.args[0].target == "auto_functionalized":
                        t = n.args[0].kwargs["t"]
                    else:
                        assert n.args[0].target == "auto_functionalized_v2"
                        t = n.args[0].kwargs["_all_bases"][0]
                    res_t = get_first_non_clone_like_node(t)
                    t = res_t
                    s = n.args[0].kwargs["s"]
                    assert type(t) is PickleableNode
                    if t.name not in name_mapper:
                        name_mapper[t.name] = s
                    name_mapper[n.name] = s
                elif n.args[0].args[0] == "tg.inplace_log_grad.default":
                    if n.args[0].target == "auto_functionalized":
                        t = n.args[0].kwargs["t"]
                        grad = n.args[0].kwargs["grad"]
                    else:
                        assert n.args[0].target == "auto_functionalized_v2"
                        t = n.args[0].kwargs["_all_bases"][0]
                        grad = n.args[0].kwargs["_all_bases"][1]
                    t = get_first_non_clone_like_node(t)
                    assert type(t) is PickleableNode
                    assert t.name in name_mapper, f"{t.name} not in name_mapper"
                    name_mapper[grad.name] = name_mapper[t.name] + ".grad"

    # We use helper_pg as backup name mapper, so setup it first.
    if helper_pg is not None:
        setup_name_mapper(helper_pg)
    setup_name_mapper(pg)

    for n in new_pg.nodes:
        if n.name in name_mapper:
            n.name = name_mapper[n.name]

    new_pg.sanity_check()
    return new_pg


def load_pgraph(
    path: str,
    collapse: bool = True,
    try_use_helper: bool = True,
    get_lift_fresh_copy_constant_value: Callable[[str], int] = None,
    force_no_lift=False,
) -> PickleableGraph:
    helper_pg = None
    basename = osp.basename(path)
    if try_use_helper and basename.startswith("bw"):
        matched = re.match(r"(fw|bw|both)(\.g\d+\.r\d+\.gpickle)", basename)
        assert matched is not None, f"Invalid basename: {basename}"
        both_path = osp.join(osp.dirname(path), f"both{matched.group(2)}")
        if osp.exists(both_path):
            helper_pg = PickleableGraph.load(both_path)
    pg = PickleableGraph.load(path)
    if collapse:
        pg = collapse_log_tensor(pg, helper_pg)
    for n in pg.nodes:
        if n.target == "aten.lift_fresh_copy.default" and not force_no_lift:
            if get_lift_fresh_copy_constant_value is None:
                raise RuntimeError(
                    "get_lift_fresh_copy_constant_value is required for loading pgraph with aten.lift_fresh_copy.default. \n"
                    "Please provide it with the your Config class; or set `force_no_lift` if this is not expected."
                )
            n.kwargs["value"] = get_lift_fresh_copy_constant_value(path, n.name)
    return pg
