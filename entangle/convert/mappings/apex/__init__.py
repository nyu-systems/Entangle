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


"""
apex::fused_layer_norm_affine_bwd

    grad_output: torch.Tensor,
    mean: torch.Tensor,
    invvar: torch.Tensor,
    input_or_output: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    memory_efficient: bool = False,
"""
def fused_layer_norm_affine_bwd_arg_converter(pnode: PickleableNode) ->ARG_CONVERTER_RETURN_TYPE:
    normalized_shape = pnode.args[4]
    if len(normalized_shape) != 1:
        raise NotImplementedError(
            f"Only support normalized_shape with length 1, got {normalized_shape}"
        )
    args = [pnode.args[0], pnode.args[1], pnode.args[2], pnode.args[3], pnode.args[5], pnode.args[6]]
    params = [pnode.args[7]]
    return args, params

OpMapping(
    "apex.fused_layer_norm_affine_fwd.default",
    tgops.fused_layernorm_affine,
)

OpMapping(
    "apex.fused_layer_norm_affine_bwd.default",
    tgops.apex.fused_layernorm_affine_bwd,
    arg_converter=fused_layer_norm_affine_bwd_arg_converter,
)
