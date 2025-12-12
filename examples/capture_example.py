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

import os
import os.path as osp

import entangle
import torch
import entangle.graph.export.tools
from entangle.graph.dynamo.tools import *


# Enable log_tensor operator for capturing autograd.Function parameters.
entangle.graph.export.tools.LOG_TENSOR = True


class Lin(torch.autograd.Function):
    """
    An example to ability of capturing autograd function.
    NOTE: the mathematics logics here is NOT correct. It is complicated for showcasing purpose.
    """

    @staticmethod
    def forward(ctx, x, w, b):
        ctx.shape = w.shape
        # This is an example to set your customized tensor name of the traced tensor.
        # For example, here we want to give the tensor a name `weight`. Then when
        # you provide the input relations, you can directly use the name.
        w = entangle.log_tensor(w, "weight")
        w = w.clone()
        ctx.save_for_backward(x, w, b)
        res = x @ w.t() + b
        return res

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx.shape
        x, w, b = ctx.saved_tensors
        w = torch.ones(shape)
        grad_x = grad_output.mm(w)
        grad_w = grad_output.t().mm(x)
        grad_b = grad_output.sum(0) + 5
        return grad_x, grad_w, grad_b


class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc = torch.nn.Linear(4, 4)
        self.weight = torch.nn.Parameter(torch.randn(4, 4))
        self.bias = torch.nn.Parameter(torch.randn(4))

        self.torch_linear = torch.nn.Linear(4, 4)

    def forward(self, x):
        y = Lin.apply(x, self.weight, self.bias)
        z = self.torch_linear(x)
        sum = y + z
        return sum.mean()


model = MyModule()
sample_inputs = torch.randn(4, 4)


def fn(model):
    loss = model(sample_inputs)
    loss.backward()
    return loss

# Run it first to check the validity of the definition.
fn(model)

print("=======================================================================")

dirname = osp.join(tempfile.gettempdir(), "capture_example")
os.makedirs(dirname, exist_ok=True)

dynamo_and_dump(
    model,
    fn,
    compile_model_or_fn="fn",
    dirname=dirname,
    formats=["code"],
    rank=0,
    logs=False,
)
