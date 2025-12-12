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

from functools import reduce
from operator import mul
from typing import Callable, Sequence

from tqdm import tqdm
import rich
from entangle.convert.mappings import OpMapping
from entangle.ops import Op
from entangle.pgraph.pickleable import PickleableGraph, PickleableNode
from entangle.sgraph.sexpr import SExpr
from entangle.sgraph.sgraph import SGraph


def to_sexpr_str(sexpr: SExpr) -> str:
    """
    Converting to s-expression string.
    """
    return sexpr.__str__()


def pgraph_to_sgraph(
    pgraph: PickleableGraph,
    name_prefix=None,
    filter_output: Callable[[list[SExpr]], list[str | SExpr]] = None,
) -> SGraph:
    name_prefix = name_prefix or ""
    pnode_to_sexpr: dict[PickleableNode, SExpr] = {}
    unreferenced_sexpr: set[SExpr] = set()
    output_pnode = None
    sexpr_order: dict[SExpr, int] = {}

    for pnode_idx, pnode in enumerate(tqdm(pgraph.nodes, leave=False)):
        if pnode.target == "output":
            output_pnode = pnode
            break
        elif pnode.op == "get_attr":
            # These are for `self.xxx`. They should be lifted using `lift_fresh_copy` later.
            # We will interprete `lift_fresh_copy` as input.
            # See `entangle.convert.mappings.misc` for how this is done.
            continue
        elif pnode.is_sym_bool():
            continue
        elif pnode.target == "aten._assert_scalar.default":
            # TODO: It is possible we can convert such nodes into scalar conditions.
            # But we manually do it for now and skip them.
            continue

        op_mapping = OpMapping.get(pnode)
        if op_mapping.op.multi_out or op_mapping.auto_functionalized:
            # We don't put this node directly, instead we add a node when
            # we process the `getitem`s.
            # For auto_functionalized op, no matter the number of outputs,
            # it always returns a tuple, so we will need getitem.
            continue

        try:
            sexpr = op_mapping.convert(pnode, pnode_to_sexpr, name_prefix)
        except Exception as e:
            rich.print(
                f"Error when converting {pnode} to SExpr. {pnode=}\nop_mapping={op_mapping!s}"
            )
            raise e
        sexpr_order[sexpr] = pnode_idx

        pnode_to_sexpr[pnode] = sexpr
        for arg in sexpr.args:
            if arg in unreferenced_sexpr:
                unreferenced_sexpr.remove(arg)
        unreferenced_sexpr.add(sexpr)

        # print(f"Converting {pnode!r} to {sexpr!r}: args={[repr(arg) for arg in sexpr.args]}, params={sexpr.params}")

    assert (
        output_pnode is not None
    ), "There should be a output node in the end of pgraph."
    print(f"There are still {len(unreferenced_sexpr)} unreferenced nodes.")
    rich.print(unreferenced_sexpr)

    # Some sym int outputs are just ignored.
    # 1. starting with _tensor_constant
    # 2. Sym Int with meta["val"] like "Sym(xxx > 0)"
    outputs = [
        pnode_to_sexpr[pnode]
        for pnode in output_pnode.args[0]
        if pnode is not None and not pnode.is_sym_scalar()
    ]
    print(f"Using outputs: {outputs=}")
    if filter_output is not None:
        outputs_or_names = filter_output(outputs)
        name_to_sexpr = {
            sexpr.name: sexpr
            for sexpr in pnode_to_sexpr.values()
            if not sexpr.op.scalar
        }
        outputs = [
            out if type(out) is SExpr else name_to_sexpr[out]
            for out in outputs_or_names
            if type(out) is SExpr or out in name_to_sexpr
        ]
    sgraph = SGraph(outputs, sexpr_order)
    return sgraph


def from_rec_expr_list(raw: str) -> SExpr:
    rec_expr_list = eval(raw)
    sexpr_list = []
    sexpr_idx_map: dict[SExpr, int] = {}
    unreferenced_idx: set[int] = set()
    for idx, (op, tensor_children, numeric_children, name) in enumerate(rec_expr_list):
        for child_idx in tensor_children:
            if child_idx in unreferenced_idx:
                unreferenced_idx.remove(child_idx)
        tensor_children = [sexpr_idx_map[child_idx] for child_idx in tensor_children]
        if type(name) is str and "@" in name:
            shape = [int(n) for n in name.split("@")[1].split("_")]
        else:
            shape = None
        sexpr = SExpr(
            SExpr.Op[str(op).upper()],
            tensor_children,
            numeric_children,
            name=name,
            shape=shape,
        )
        sexpr_idx_map[idx] = sexpr
        sexpr_list.append(sexpr)
        unreferenced_idx.add(idx)

    assert len(unreferenced_idx) == 1, "There should be only one root node."
    return sexpr_list[unreferenced_idx.pop()]
