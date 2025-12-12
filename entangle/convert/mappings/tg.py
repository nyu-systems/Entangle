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

import entangle.ops as tgops
from entangle.convert.mappings.mapping import *
from entangle.pgraph.pickleable import PickleableNode


def inplace_log_tensor_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    parent_pnode = pnode.args[0]
    logged_tensor_sexpr = pnode_to_sexpr[parent_pnode.kwargs["t"]]
    logged_tensor_sexpr.set_log_name(parent_pnode.kwargs["s"])
    return tgops.clone, (logged_tensor_sexpr,), (), None, None


OpMapping(
    "tg.inplace_log_tensor.default",
    tgops.clone,
    auto_functionalized=True,
    converter=inplace_log_tensor_converter,
)
OpMapping(
    "tg.log_tensor.default",
    tgops.clone,
    arg_converter=lambda pnode: (pnode.args[:1], ()),
)
OpMapping("tg.slice.default", tgops.slice)
OpMapping("tg.slice_scatter.default", tgops.slice_scatter)
