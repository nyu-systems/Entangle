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
from typing import Iterable

import entangle.ops as tgops
from entangle.convert.mappings.mapping import *
from entangle.sgraph import sexpr
from entangle.sgraph.sexpr import ShapeLike


def shapelike_param_arg_converter(
    pnode: PickleableNode,
) -> ARG_CONVERTER_RETURN_TYPE:
    args, params = OpMapping.default_arg_converter(pnode)
    new_params = []
    for param in params:
        if isinstance(param, Iterable):
            param = list(param)
            if all(isinstance(x, int) for x in param):
                param = ShapeLike(param)
        new_params.append(param)
    return args, new_params


OpMapping(
    "hlo.broadcast", tgops.hlo.broadcast, arg_converter=shapelike_param_arg_converter
)
OpMapping("hlo.dot", tgops.hlo.dot, arg_converter=shapelike_param_arg_converter)
OpMapping("hlo.gather", tgops.hlo.gather, arg_converter=shapelike_param_arg_converter)
OpMapping("hlo.logistic", tgops.hlo.logistic)
OpMapping("hlo.reduce_max", tgops.hlo.reduce_max)
OpMapping(
    "hlo.AwsNeuronRmsNorm",
    tgops.hlo.rms_norm,
    arg_converter=shapelike_param_arg_converter,
)  # NOTE: original rms_forward_affine returns output and mean, we only need output
OpMapping("hlo.select", tgops.hlo.select)
