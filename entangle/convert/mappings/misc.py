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
from entangle.pgraph.pickleable import PickleableNode


def placeholder_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
):
    if pnode.get_tensor_shape() is not None:
        return tgops.inpt, [], [], None, pnode.get_tensor_shape()
    else:
        # This should be a scalar. But we have no information to know that.
        # We temporarily use assertion of its name to ensure correctness.
        assert (
            pnode.name.find("scalar") != -1
            or pnode.name.find("tensor_constant") != -1
        ), f"The pnode should be a scalar because it doesn't have shape. But the name doesn't contain 'scalar' or 'tensor_constant'. {pnode!s}"
        shape = []
        name = pnode.get_scalar_name()
        return tgops.scalar, [], [], name, shape


def lift_fresh_copy_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
):
    shape = pnode.get_tensor_shape()
    assert shape is not None
    assert len(shape) == 0, f"got {shape} for {pnode=}"
    shape = ShapeLike(shape)
    # NOTE: the kwargs["value"] is actually added when calling `entangle.pgraph.pickleable.load_pgraph`
    return tgops.fill, [], [shape, pnode.kwargs["value"]], None, shape


OpMapping("placeholder", tgops.inpt, converter=placeholder_converter)

# The target `aten.lift_fresh_copy.default` is usually used after `get_attr` op that
# get an attribute from `self`. And `get_attr` returns something with no shape. Thus,
# we use `aten.lift_fresh_copy.default` as an input instead.
OpMapping(
    "aten.lift_fresh_copy.default", tgops.inpt, converter=lift_fresh_copy_converter
)
OpMapping("mylib.fused_layernorm_affine.default", tgops.fused_layernorm_affine)

OpMapping("common.ewmul_sliced.default", tgops.ewmul_sliced)
