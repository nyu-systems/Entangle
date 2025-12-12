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

fused_add_rms_norm = Op("vllm_fused_add_rms_norm", multi_in=True, skeleton=True)

rms_norm = Op("vllm_rms_norm", multi_in=True, skeleton=True)

rotary_embedding = Op(
    "vllm_rotary_embedding", multi_in=True, multi_out=True, skeleton=True
)

unified_attention_with_output = Op(
    "vllm_unified_attention_with_output", multi_in=True, skeleton=True
)

silu_and_mul = Op("vllm_silu_and_mul", skeleton=True)
