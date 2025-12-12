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


def gemm_inf_arg_converter(pnode: PickleableNode):
    if pnode.args[2] is None:
        return pnode.args[:2], ()
    else:
        raise NotImplementedError("tex_gemm_inf with bias not implemented yet.")


OpMapping(
    "tex.gemm_inf.default", tgops.te.tex_gemm_inf, arg_converter=gemm_inf_arg_converter
)

OpMapping("flash_attn._flash_attn_forward.default", tgops.te.flash_attn_forward)
