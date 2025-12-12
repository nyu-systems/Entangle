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

import entangle.graph.export.tools as export_tools


@torch.library.custom_op("tg::slice", mutates_args=())
def slice(
    t: torch.Tensor, dim: int, start: int, end: int, step: int = 1
) -> torch.Tensor:
    t_clone = t.clone()
    return torch.ops.aten.slice(t_clone, dim, start, end, step)


@slice.register_fake
def fake_slice(
    t: torch.Tensor, dim: int, start: int, end: int, step: int = 1
) -> torch.Tensor:
    # assert 0 <= start <= end <= t.shape[dim]
    shape = list(t.shape)
    s = end - start
    torch._check_is_size(s)
    shape[dim] = s
    return torch.empty(shape, dtype=t.dtype, device=t.device)


@torch.library.custom_op("tg::slice_scatter", mutates_args=())
def slice_scatter(
    inpt: torch.Tensor, src: torch.Tensor, dim: int, start: int, end: int, step: int = 1
) -> torch.Tensor:
    inpt_clone = inpt.clone()
    src_clone = src.clone()
    return torch.slice_scatter(inpt_clone, src_clone, dim, start, end, step)


@slice_scatter.register_fake
def fake_slice_scatter(
    inpt: torch.Tensor, src: torch.Tensor, dim: int, start: int, end: int, step: int = 1
) -> torch.Tensor:
    return torch.empty_like(inpt)


TENSOR_NAME_MAP: dict[torch.Tensor, str] = {}


def map_tensor(t: torch.Tensor, name: str):
    TENSOR_NAME_MAP[t] = name


def get_tensor_name(t: torch.Tensor) -> str:
    TENSOR_NAME_MAP[t]


def inplace_log_tensor(t: torch.Tensor, s: str) -> None:
    if export_tools.LOG_TENSOR:
        _inplace_log_tensor(t, s)


@torch.library.custom_op("tg::inplace_log_tensor", mutates_args=("t",))
def _inplace_log_tensor(t: torch.Tensor, s: str) -> None: 
    map_tensor(t, s)


@_inplace_log_tensor.register_fake
def fake_inplace_log_tensor(t: torch.Tensor, s: str) -> None:
    map_tensor(t, s)


def log_tensor(t: torch.Tensor, s: str) -> torch.Tensor:
    if export_tools.LOG_TENSOR:
        return _log_tensor(t, s)
    else:
        return t


@torch.library.custom_op("tg::log_tensor", mutates_args=())
def _log_tensor(t: torch.Tensor, s: str) -> torch.Tensor:
    TENSOR_NAME_MAP[t] = s
    return t.clone()


@_log_tensor.register_fake
def fake_log_tensor(t: torch.Tensor, s: str) -> torch.Tensor:
    return torch.empty_like(t)


def inplace_log_grad(t: torch.Tensor, grad: torch.Tensor) -> None:
    return _inplace_log_grad(t, grad)


@torch.library.custom_op("tg::inplace_log_grad", mutates_args=("grad",))
def _inplace_log_grad(t: torch.Tensor, grad: torch.Tensor) -> None: ...


@_inplace_log_grad.register_fake
def fake_inplace_log_grad(t: torch.Tensor, grad: torch.Tensor) -> None:
    s = f"{get_tensor_name(t)}.grad"
    map_tensor(grad, s)


class LogTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, s):
        ctx.s = s
        return log_tensor(t, s)

    @staticmethod
    def backward(ctx, grad):
        grad = log_tensor(grad, f"{ctx.s}.grad")
        return grad, None
