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

import torch
from torch import device

b8: type
i32: type
i64: type
bf16: type
f32: type
f64: type
fx_pytree: type
Sym: type


# Graph[rank=0](both, gid=1)
class joint_helper(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "f32[1]"; primals_2: "f32[1]"; tangents_1: "f32[1]"; 
    
        primals_1, primals_2, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/optimizer/optimizer.py:236 in scale_loss, code: return self.get_loss_scale() * loss
        mul: "f32[1]" = torch.ops.aten.mul.Tensor(primals_2, primals_1);  primals_1 = None
        mul_1: "f32[1]" = torch.ops.aten.mul.Tensor(tangents_1, primals_2);  tangents_1 = primals_2 = None
        return pytree.tree_unflatten([mul, mul_1, None], self._out_spec)
        

