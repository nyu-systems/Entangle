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

from entangle.ops.op import Op
from entangle.ops.tg import *
from entangle.ops.torch import *

rms_forward_affine = Op(
    "rms_forward_affine", multi_in=True, multi_out=True, skeleton=True
)

rms_backward_affine = Op(
    "rms_backward_affine", multi_in=True, multi_out=True, skeleton=True
)

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
fused_layernorm_affine_bwd = Op(
    "fused_layernorm_affine_bwd",
    multi_in=True,
    multi_out=True,
    skeleton=True,
)
