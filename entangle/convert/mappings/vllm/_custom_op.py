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

import entangle.ops as tgops
from entangle.convert.mappings.mapping import *
from entangle.sgraph import sexpr
from entangle.sgraph.sexpr import ShapeLike


def rotary_embedding_converter(pnode: PickleableNode):
    args = pnode.kwargs
    return [args["positions"], args["query"], args["key"], args["cos_sin_cache"]], [
        args["head_size"]
    ]


def unified_attention_with_output_converter(pnode: PickleableNode):
    args = pnode.kwargs
    return [args["query"], args["key"], args["value"]], []


def rms_norm_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    af_node = pnode.args[0]
    args = af_node.kwargs
    print(args.keys())
    input_pnode: PickleableNode = args["input"]
    result_pnode: PickleableNode = args["result"]
    if result_pnode.target != "aten.empty_like.default" or (
        result_shape := result_pnode.get_tensor_shape()
    ) != (input_shape := input_pnode.get_tensor_shape()):
        raise ValueError(
            f"result must be an empty_like with same shape of input, got {result_pnode.target=}, {result_shape=}, {input_shape=}"
        )
    input_sexpr = pnode_to_sexpr[input_pnode]
    weight_sexpr = pnode_to_sexpr[args["weight"]]
    return (
        tgops.apex.rms_forward_affine.getitem(0),
        [input_sexpr, weight_sexpr],
        [args["epsilon"]],
        None,
        None,
    )


def fused_add_rms_norm_converter(pnode: PickleableNode):
    args = pnode.kwargs
    return [args["input"], args["residual"], args["weight"]], [args["epsilon"]]


def silu_and_mul_converter(pnode: PickleableNode):
    args = pnode.kwargs
    return [args["input"]], []


def all_reduce_converter(pnode: PickleableNode):
    args = pnode.args
    return [args[0]], ["sum", None, args[1]]


OpMapping(
    "_C.rms_norm.default",
    tgops.vllm._custom_ops.rms_norm,
    auto_functionalized=True,
    converter=rms_norm_converter,
)

OpMapping(
    "_C.rotary_embedding.default",
    tgops.vllm._custom_ops.rotary_embedding,
    auto_functionalized=True,
    arg_converter=rotary_embedding_converter,
)

OpMapping(
    "vllm.unified_attention_with_output.default",
    tgops.vllm._custom_ops.unified_attention_with_output,
    auto_functionalized=True,
    arg_converter=unified_attention_with_output_converter,
)

OpMapping(
    "_C.fused_add_rms_norm.default",
    tgops.vllm._custom_ops.fused_add_rms_norm,
    auto_functionalized=True,
    arg_converter=fused_add_rms_norm_converter,
)

OpMapping(
    "_C.silu_and_mul.default",
    tgops.vllm._custom_ops.silu_and_mul,
    auto_functionalized=True,
    arg_converter=silu_and_mul_converter,
)

OpMapping(
    "vllm.all_reduce.default", tgops.all_reduce, arg_converter=all_reduce_converter
)
