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

import importlib
import logging
import os
import os.path as osp
import pickle
import sys
from itertools import chain
from types import ModuleType
from typing import Callable, Iterable, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fdist
from functorch.compile import aot_module_simplified, make_boxed_func
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.library import device_types_t

from entangle.pgraph.pickleable import *
from entangle.utils.module_utils import load_module

USE_RNG = os.environ.get("TG_USE_RNG", "0") == "1"
LOG_TENSOR = os.environ.get("TG_LOG_TENSOR", "0") == "1"
USE_CUSTOM_OP = os.environ.get("TG_USE_CUSTOM_OP", "0") == "1"
USE_COMPILER_DISABLE = os.environ.get("TG_USE_COMPILER_DISABLE", "0") == "1"

if USE_CUSTOM_OP and USE_COMPILER_DISABLE:
    raise RuntimeError("Cannot use both custom op and compiler disable")


class CustomOpArgData:
    def __init__(
        self,
        real: Callable,
        fake: Callable,
        /,
        *,
        mutates_args: Union[str, Iterable[str]] = (),
        device_types: device_types_t = None,
        schema: Optional[str] = None,
    ):
        self.real = real
        self.fake = fake
        self.mutates_args = mutates_args
        self.device_types = device_types
        self.schema = schema

    def clone_with(
        self,
        real: Callable = None,
        fake: Callable = None,
        mutates_args: Union[str, Iterable[str]] = None,
        device_types: device_types_t = None,
        schema: Optional[str] = "",
    ):
        return CustomOpArgData(
            real if real is not None else self.real,
            fake if fake is not None else self.fake,
            mutates_args=(
                mutates_args if mutates_args is not None else self.mutates_args
            ),
            device_types=(
                device_types if device_types is not None else self.device_types
            ),
            schema=schema if schema != "" else self.schema,
        )


def export(model, sample_inputs, strict=False):
    g = torch.export(model, sample_inputs, strict=strict)
    print(g)
    return g


def wrap_custom_op(custom_op_arg_datas: Union[CustomOpArgData, list[CustomOpArgData]]):
    if type(custom_op_arg_datas) is CustomOpArgData:
        custom_op_arg_datas = [custom_op_arg_datas]
    if USE_CUSTOM_OP:
        for custom_op_arg_data in custom_op_arg_datas:
            real = custom_op_arg_data.real
            if type(real) is torch.library.CustomOpDef:
                continue
            fake = custom_op_arg_data.fake
            m = sys.modules[real.__module__]
            ops_module_name = real.__module__.split(".")[0]
            ops_func_name = "__".join([*real.__module__.split(".")[1:], real.__name__])
            try:
                custom = m.__dict__[real.__name__] = torch.library.custom_op(
                    ops_module_name + "::" + ops_func_name,
                    mutates_args=custom_op_arg_data.mutates_args,
                    device_types=custom_op_arg_data.device_types,
                    schema=custom_op_arg_data.schema,
                )(real)
                custom.register_fake(fake)
            except Exception as e:
                print("Error registering", ops_module_name + "::" + ops_func_name)
                raise e
    elif USE_COMPILER_DISABLE:
        for custom_op_arg_data in custom_op_arg_datas:
            real = custom_op_arg_data.real
            m = sys.modules[real.__module__]
            ops_module_name = real.__module__.split(".")[0]
            ops_func_name = "__".join([*real.__module__.split(".")[1:], real.__name__])
            try:
                custom = m.__dict__[real.__name__] = torch.compiler.disable(real)
            except Exception as e:
                print("Error disabling", ops_module_name + "::" + ops_func_name)
                raise e


def wrap_modules(path: str, module_name: str, recursive=True):
    if osp.isdir(path):
        for filename in os.listdir(path):
            if not filename.startswith("_"):
                sub_path = osp.join(path, filename)
                if osp.isdir(sub_path):
                    sub_module_name = f"{module_name}.{filename}"
                else:
                    assert osp.isfile(sub_path)
                    sub_module_name = f"{module_name}.{filename.strip('.py')}"
                wrap_modules(sub_path, sub_module_name, recursive)
    else:
        assert osp.isfile(path), f"{path=}"
        if (
            osp.isfile(path)
            and path.endswith(".py")
            and not osp.split(path)[1].startswith("_")
        ):
            module = load_module(path, module_name)
            if "custom_op_arg_datas" in module.__dict__:
                if type(module.custom_op_arg_datas) in (list, tuple):
                    wrap_custom_op(module.custom_op_arg_datas)
                elif type(module.custom_op_arg_datas) is dict:
                    wrap_custom_op(list(module.custom_op_arg_datas.values()))


def is_fdist_handle(obj: Any) -> bool:
    return type(obj) in (fdist.AsyncCollectiveTensor, FunctionalTensor)


def wait_or_id(handle, t):
    # Here `id` means no change, assigning back.
    if is_fdist_handle(handle):
        return fdist.wait_tensor(handle)
    elif type(handle) is not torch.Tensor:
        handle.wait()
        return t
    else:
        return t
