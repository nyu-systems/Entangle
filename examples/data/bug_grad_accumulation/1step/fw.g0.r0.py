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


# Graph[rank=0](fw, gid=0)
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "f32[8]", arg3_1: "f32[8]", arg4_1: "f32[]"):
        # File: /opt/tiger/transformers/tests/trainer/test_trainer_simple.py:94 in forward, code: y = input_x * self.a + self.b
        mul: "f32[8]" = torch.ops.aten.mul.Tensor(arg3_1, arg0_1);  arg3_1 = arg0_1 = None
        add: "f32[8]" = torch.ops.aten.add.Tensor(mul, arg1_1);  mul = arg1_1 = None
        
        # File: /opt/tiger/transformers/tests/trainer/test_trainer_simple.py:103 in my_loss, code: loss = ((y - labels) ** 2).sum()
        sub: "f32[8]" = torch.ops.aten.sub.Tensor(add, arg2_1);  add = arg2_1 = None
        pow_1: "f32[8]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(pow_1);  pow_1 = None
        
        # File: /opt/tiger/transformers/tests/trainer/test_trainer_simple.py:104 in my_loss, code: loss = loss / num_items_in_batch
        div: "f32[]" = torch.ops.aten.div.Tensor(sum_1, 8);  sum_1 = None
        
        # File: /opt/tiger/transformers/src/transformers/trace_trainer.py:3612 in training_step, code: return loss.detach()
        detach: "f32[]" = torch.ops.aten.detach.default(div);  div = None
        detach_1: "f32[]" = torch.ops.aten.detach.default(detach);  detach = None
        detach_2: "f32[]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        
        # File: /opt/tiger/transformers/src/transformers/trace_trainer.py:2487 in fn, code: tr_loss = tr_loss + tr_loss_step
        add_1: "f32[]" = torch.ops.aten.add.Tensor(arg4_1, detach_2);  arg4_1 = detach_2 = None
        return (add_1,)
        

