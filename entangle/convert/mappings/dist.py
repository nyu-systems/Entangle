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

OpMapping("_c10d_functional.broadcast.default", tgops.dist_broadcast)
OpMapping("_c10d_functional.all_gather_into_tensor.default", tgops.all_gather)
OpMapping("_c10d_functional.all_reduce.default", tgops.all_reduce)
OpMapping("_c10d_functional.all_to_all_single.default", tgops.all_to_all_single)
OpMapping("_c10d_functional.reduce_scatter_tensor.default", tgops.reduce_scatter)

OpMapping("_c10d_functional.wait_tensor.default", tgops.dist_wait)
