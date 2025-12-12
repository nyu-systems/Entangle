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

inpt = Op("input", leaf=True, relation=True)
empty = Op("empty", leaf=True, constant=True, relation=True)  # Although empty is not really a constant, we treat it as a somewhat constant.
weight = Op("weight", leaf=True, relation=True)
fill = Op("fill", leaf=True, constant=True, relation=True)
arange = Op("arange", leaf=True, constant=True, relation=True)
scalar = Op("scalar", leaf=True, scalar=True, relation=True)
boolean = Op("boolean", leaf=True, boolean=True, relation=True)

# Relation ops
## Moving relation ops
bitwise_and = Op("bitwise_and", multi_in=True, noncompute=True, relation=True)
bitwise_not = Op("bitwise_not", noncompute=True, relation=True)
bitwise_or = Op("bitwise_or", multi_in=True, noncompute=True, relation=True)
concat = Op("concat", multi_in=True, noncompute=True, relation=True)
clone = Op("clone", noncompute=True, relation=True)
embedding = Op("embedding", multi_in=True, noncompute=True, relation=True)
expand = Op("expand", noncompute=True, relation=True)
ge_scalar = Op("ge_scalar", noncompute=True, relation=True)
ge_tensor = Op("ge_tensor", noncompute=True, relation=True)
index = Op("index", multi_in=True, noncompute=True, relation=True)
# 1 indices arg
index_put = Op("index_put", multi_in=True, noncompute=True, relation=True)
index_put_acc = Op("index_put_acc", multi_in=True, noncompute=True, relation=True)
# 2 indices args
index_put_2 = Op("index_put_2", multi_in=True, noncompute=True, relation=True)
index_put_acc_2 = Op("index_put_acc_2", multi_in=True, noncompute=True, relation=True)
index_fill_scalar = Op(
    "index_fill_scalar", multi_in=False, noncompute=True, relation=True
)
index_select = Op("index_select", multi_in=True, noncompute=True, relation=True)
le_tensor = Op("le_tensor", noncompute=True, relation=True)
lt_scalar = Op("lt_scalar", noncompute=True, relation=True)
mask_fill_scalar = Op("mask_fill_scalar", multi_in=True, skeleton=True, relation=True)
maximum = Op("maximum", multi_in=True, noncompute=True, relation=True)
nonzero = Op("nonzero", noncompute=True, relation=True)
pad = Op("pad", noncompute=True, relation=True)
repeat = Op("repeat", noncompute=True, relation=True)
reshape = Op("reshape", noncompute=True, relation=True)
select = Op("select", noncompute=True, relation=True)
select_scatter = Op("select_scatter", multi_in=True, noncompute=True, relation=True)
masked_select = Op("masked_select", noncompute=True, relation=True)
slice = Op("slice", noncompute=True, relation=True)
slice_backward = Op("slice_backward", noncompute=True, relation=True)
slice_scatter = Op("slice_scatter", multi_in=True, noncompute=True, relation=True)
split = Op("split", multi_out=True, noncompute=True, relation=True)
split_with_sizes = Op(
    "split_with_sizes", multi_out=True, noncompute=True, relation=True
)
squeeze = Op("squeeze", noncompute=True, relation=True)
transpose = Op("transpose", noncompute=True, relation=True)
unsqueeze = Op("unsqueeze", noncompute=True, relation=True)
## Dist Relation ops
reduce_add = Op(
    "reduce_add", multi_in=True, relation=True
)  # With broadcast. NOTE: when matadd is across devices, it is relation, othewise it is no

# Skeleton ops
add_scalar = Op("add_scalar", multi_in=True, skeleton=True)
baddbmm = Op("baddbmm", multi_in=True, skeleton=True)
bmm = Op("bmm", multi_in=True, skeleton=True)
cos = Op("cos", skeleton=True)
matdiv = Op("matdiv", multi_in=True, skeleton=True, relation=True)
embedding_dense_backward = Op("embedding_dense_backward", multi_in=True, skeleton=True)
ewmul = Op("ewmul", multi_in=True, skeleton=True, relation=True)
ewmul_sliced = Op("ewmul_sliced", multi_in=True, skeleton=True)  # Slice for SymInt
exp = Op("exp", skeleton=True)
fused_layernorm_affine = Op(
    "fused_layernorm_affine", multi_in=True, multi_out=True, skeleton=True
)
gelu = Op("gelu", skeleton=True)
gelu_backward = Op("gelu_backward", skeleton=True)
log = Op("log", skeleton=True)
native_layer_norm = Op("native_layer_norm", multi_in=True, multi_out=True, skeleton=True)
matadd = Op("matadd", multi_in=True, skeleton=True)  # With broadcast
max_dim = Op("max_dim", skeleton=True, multi_out=True)
addmm = Op("addmm", multi_in=True, skeleton=True)
mean_default = Op("mean_default", skeleton=True)
matmul = Op("matmul", multi_in=True, skeleton=True)
mul_scalar = Op("mul_scalar", multi_in=True, skeleton=True)
addcmul = Op("addcmul", multi_in=True, skeleton=True)
matsub = Op("matsub", multi_in=True, skeleton=True)  # With broadcast
native_dropout = Op("native_dropout", skeleton=True, multi_out=True)
rsub_scalar = Op("rsub_scalar", multi_in=True, skeleton=True)
scatter_src = Op("scatter_src", multi_in=True, skeleton=True)
sigmoid = Op("sigmoid", skeleton=True)
sigmoid_backward = Op("sigmoid_backward", multi_in=True, skeleton=True)
silu = Op("silu", skeleton=True)
sin = Op("sin", skeleton=True)
softmax = Op("softmax", skeleton=True)
softmax_backward_data = Op("softmax_backward_data", skeleton=True)
sum_dim_int_list = Op("sum_dim_int_list", skeleton=True)
sum_dim_int_list_keep = Op("sum_dim_int_list_keep", skeleton=True)
sum_default = Op("sum_default", skeleton=True)
cumsum = Op("cumsum", skeleton=True)
topk = Op("topk", multi_in=False, multi_out=True, skeleton=True)
argsort = Op("argsort", skeleton=True)
sort = Op("sort", multi_out=True, skeleton=True)
eq_scalar = Op("eq_scalar", skeleton=True)
clamp = Op("clamp", skeleton=True)
pow_tensor_scalar = Op("pow_tensor_scalar", skeleton=True)

# Dist ops
dist_broadcast = Op("dist_broadcast", dist=True)
all_gather = Op("all_gather", dist=True)
all_reduce = Op("all_reduce", dist=True)
all_to_all_single = Op("all_to_all_single", dist=True)
reduce_scatter = Op("reduce_scatter", dist=True, multi_in=True)

a2a = Op("a2a", multi_in=True, multi_out=True, dist=True)

dist_wait = Op("wait", dist=True)

noop = Op("noop")
