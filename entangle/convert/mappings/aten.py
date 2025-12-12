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

MAX_I64 = (1 << 63) - 1


def pow_tensor_scalar_converter(pnode: PickleableNode):
    return pnode.args[:1], [pnode.args[1]]


def addmm_arg_converter(pnode: PickleableNode):
    inpt, batch1, batch2 = pnode.args
    beta = pnode.kwargs.get("beta", 1.0)
    alpha = pnode.kwargs.get("alpha", 1.0)
    return [inpt, batch1, batch2], [beta, alpha]


def arange_arg_converter(pnode: PickleableNode):
    if len(pnode.args) == 1:
        return [], [0, pnode.args[0]]
    elif len(pnode.args) == 2:
        start = pnode.args[0]
        end = pnode.args[1]
        return [], [start, end]
    else:
        raise NotImplementedError("arange with step is not implemented.")


def baddbmm_arg_converter(pnode: PickleableNode):
    inpt, batch1, batch2 = pnode.args
    beta = pnode.kwargs.get("beta", 1.0)
    alpha = pnode.kwargs.get("alpha", 1.0)
    return [inpt, batch1, batch2], [beta, alpha]


def cat_default_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    shape = ShapeLike(pnode.get_tensor_shape())
    assert shape is not None, f"Cannot get shape from {pnode!s}"
    args = []
    for arg in pnode.args[0]:
        args.append(pnode_to_sexpr[arg])
    if len(pnode.args) == 1:
        dim = 0
    else:
        dim = positivate_dim(pnode.args[1], shape)
    if len(args) == 2:
        return tgops.concat, args, [dim], None, shape
    elif len(args) == 1:
        return tgops.clone, args, [], None, shape
    else:
        concat_sexpr = sexpr.concat(args, dim=dim, name=name_prefix + repr(pnode))
        assert concat_sexpr.shape == shape, f"got {concat_sexpr.shape=}, {shape=}"
        return tgops.concat, concat_sexpr.args, concat_sexpr.params, None, shape


def constant_pad_nd_default_arg_converter(pnode: PickleableNode):
    params = [*pnode.args[1], pnode.args[2]]
    assert params[-1] == 0.0
    params = params[:-1]
    params += [0] * (6 - len(params))
    return pnode.args[:1], params


def empty_arg_converter(*_):
    return [], []


def index_put_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    inpt, indices, value = pnode.args[:3]
    if len(pnode.args) == 3:
        accumulate = False
    else:
        assert len(pnode.args) == 4, "Otherwise, never encountered.."
        accumulate = pnode.args[3]
    inpt_shape = inpt.get_tensor_shape()
    assert inpt_shape is not None
    if accumulate:
        if len(indices) == 1:
            op = tgops.index_put_acc
        elif len(indices) == 2:
            op = tgops.index_put_acc_2
        else:
            raise NotImplementedError(
                f"index_put with {len(indices)} indices is not implemented."
            )
    else:
        if len(indices) == 1:
            op = tgops.index_put
        elif len(indices) == 2:
            op = tgops.index_put_2
        else:
            raise NotImplementedError(f"index_put with {len(indices)} indices.")
    args = [inpt, *indices, value]
    args = [pnode_to_sexpr[a] for a in args]
    return op, args, [], None, None


def index_fill_scalar_arg_converter(pnode: PickleableNode):
    inpt = pnode.args[0]
    dim = pnode.args[1]
    indices = pnode.args[2]
    value = pnode.args[3]
    assert type(dim) == int
    assert len(indices.get_tensor_shape()) == 1
    assert type(value) == int
    return [inpt, indices], [dim, value]


def max_dim_arg_converter(pnode: PickleableNode):
    inpt = pnode.args[0]
    inpt_shape = inpt.get_tensor_shape()
    assert inpt_shape is not None
    if len(pnode.args) > 2:
        raise NotImplementedError(f"max_dim with keepdim is not implemented. {pnode!s}")
    dim = pnode.args[1]
    dim = positivate_dim(dim, inpt_shape)
    if dim < len(inpt_shape) - 1:
        raise NotImplementedError(
            f"max_dim with dim={dim} is not implemented. {pnode!s}"
        )
    return [inpt], [dim]


def make_empty_converter(shape_arg_type: str, shape_arg_index: int = 0):
    def empty_converter(
        op_mapping: "OpMapping",
        pnode: PickleableNode,
        pnode_to_sexpr: CONVERT_MEMO_TYPE,
        name_prefix: str,
    ) -> CONVERTER_RETURN_TYPE:
        try:
            if shape_arg_type == "tensor":
                shape = pnode.args[shape_arg_index].get_tensor_shape()
            else:
                assert shape_arg_type == "shape"
                shape = pnode.args[shape_arg_index]
            assert shape is not None
            shape = ShapeLike(shape)
            return tgops.empty, (), [shape], None, shape
        except Exception as e:
            raise RuntimeError(f"Failed for {op_mapping=}, {pnode!s}") from e

    return empty_converter


def make_fill_converter(
    shape_arg_type: str, value=None, shape_arg_index: int = 0, value_arg_index: int = 1
):
    def fill_converter(
        op_mapping: "OpMapping",
        pnode: PickleableNode,
        pnode_to_sexpr: CONVERT_MEMO_TYPE,
        name_prefix: str,
        value=value,
    ) -> CONVERTER_RETURN_TYPE:
        try:
            if shape_arg_type == "tensor":
                sym_shape = shape = pnode.args[shape_arg_index].get_tensor_shape()
            else:
                assert shape_arg_type == "shape"
                shape = pnode.args[shape_arg_index]
                sym_shape = pnode.get_tensor_shape()
            assert shape is not None
            shape = ShapeLike(shape)
            if value is None and len(pnode.args) >= value_arg_index + 1:
                value_pnode = pnode.args[value_arg_index]
                if type(value_pnode) is PickleableNode and value_pnode.is_sym_scalar():
                    value = value_pnode.kwargs["value"]
                    assert type(value) in (int, float), f"{value_pnode=}"
                else:
                    assert type(value_pnode) in (int, float), f"{value_pnode=}"
                    value = pnode.args[value_arg_index]
            assert (
                value is not None
            ), "Should provide value or args should contain value."
            return tgops.fill, (), [shape, value], None, sym_shape
        except Exception as e:
            raise RuntimeError(f"Failed for {op_mapping=}, {pnode!s}")

    return fill_converter


def permute_default_arg_converter(pnode: PickleableNode):
    params = pnode.args[1]
    return pnode.args[:1], [ShapeLike(params)]


def scatter_src_arg_converter(pnode: PickleableNode):
    inpt = pnode.args[0]
    dim = pnode.args[1]
    index = pnode.args[2]
    src = pnode.args[3]
    index_shape = index.get_tensor_shape()
    assert index_shape is not None
    dim = positivate_dim(dim, index_shape)
    return [inpt, index, src], [dim]


def slice_tensor_arg_converter(pnode: PickleableNode):
    args = pnode.args[:1]
    params = pnode.args[1:]
    if len(params) == 4:
        if params[-1] != 1:
            raise NotImplementedError("slice with stride is not implemented.")
        else:
            params = params[:-1]
    shape = pnode.get_tensor_shape()
    assert (
        len(args) == 1 and len(params) == 3
    ), f"Got {args=}, {params=} from {str(pnode)=}"
    # (dim, start, end) ==> (dim, start, end, step)
    dim, start, end = params
    inpt_shape = pnode.args[0].get_tensor_shape()
    if type(start) is int:
        if start < 0:
            start += inpt_shape[dim]
    else:
        assert (
            start.is_sym_scalar()
        ), f"{start=}, {start.name=}, {start.op=}, {start.target=}"
    if type(end) is int:
        if end < 0:
            end += inpt_shape[dim]
        if end == MAX_I64:
            adjusted_end = start + shape[dim]
            assert (
                inpt_shape[dim] == adjusted_end
            ), f"{adjusted_end=}, but {inpt_shape[dim]=}"
            end = start + shape[dim]  # Adjust the end to the actual end.
    else:
        assert end.is_sym_scalar(), f"{end=}, {end.target=}"
    params = [dim, start, end, 1]
    return args, params


def slice_scatter_arg_converter(pnode: PickleableNode):
    inpt, src, dim, begin, end = pnode.args
    inpt_shape = inpt.get_tensor_shape()
    assert inpt_shape is not None
    if type(begin) is int:
        if begin < 0:
            begin += inpt_shape[dim]
        elif begin == MAX_I64:
            begin = inpt_shape[dim]
    if type(end) is int:
        if end < 0:
            end += inpt_shape[dim]
        elif end == MAX_I64:
            end = inpt_shape[dim]
    return [inpt, src], [dim, begin, end]


def split_tensor_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    assert op_mapping.idx is not None, f"split must have idx, got {op_mapping.idx}."
    shape = pnode.get_tensor_shape()
    parent_pnode = pnode.args[0]
    args = [pnode_to_sexpr[parent_pnode.args[0]]]
    original_shape = args[0].shape
    idx = op_mapping.idx
    split_size = parent_pnode.args[1]
    if len(parent_pnode.args) == 3:
        dim = positivate_dim(parent_pnode.args[2], original_shape)
    else:
        assert len(parent_pnode.args) == 2
        dim = 0
    begin = split_size * idx
    end = split_size * (idx + 1)
    assert (
        end <= original_shape[dim]
    ), f"{end=} should be less than {original_shape[dim]=}"
    params = [dim, begin, end, 1]
    assert shape[dim] == end - begin, f"{shape[dim]=} should be {end - begin=}"
    return tgops.slice, args, params, None, shape


def split_with_sizes_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    assert (
        op_mapping.idx is not None
    ), f"split_with_sizes must have idx, got {op_mapping.idx}."
    op = tgops.slice
    idx = op_mapping.idx
    parent_pnode = pnode.args[0]
    shape = pnode.get_tensor_shape()
    args = [pnode_to_sexpr[parent_pnode.args[0]]]
    split_sizes = parent_pnode.args[1]
    begin = sum(split_sizes[:idx])
    dim = 0 if len(parent_pnode.args) == 2 else parent_pnode.args[2]
    dim = positivate_dim(dim, shape)
    params = [dim, begin, begin + split_sizes[idx], 1]
    return op, args, params, None, shape


def squeeze_arg_converter(pnode: PickleableNode):
    inpt = pnode.args[0]
    dim = pnode.args[1]
    inpt_shape = inpt.get_tensor_shape()
    assert inpt_shape is not None
    assert type(dim) is int
    dim = positivate_dim(dim, inpt_shape, neg_plus_one=True)
    return [inpt], [dim]


def sum_dim_int_list_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    inpt = pnode.args[0]
    shape = inpt.get_tensor_shape()
    assert shape is not None, f"Cannot get shape from {inpt}."
    dims = [d if d >= 0 else len(shape) + d for d in pnode.args[1]]
    if len(pnode.args) == 2:
        op = tgops.sum_dim_int_list
    else:
        assert len(pnode.args) == 3
        assert type(pnode.args[2]) is bool
        if pnode.args[2] is True:
            op = tgops.sum_dim_int_list_keep
        else:
            op = tgops.sum_dim_int_list
    return op, [pnode_to_sexpr[inpt]], [ShapeLike(dims)], None, None


def cumsum_converter(pnode: PickleableNode):
    inpt = pnode.args[0]
    dim = pnode.args[1]
    return [inpt], [dim]


def sum_default_converter(pnode: PickleableNode):
    return pnode.args[:1], []


def transpose_int_arg_converter(pnode: PickleableNode):
    params = pnode.args[1:]
    shape = pnode.get_tensor_shape()
    assert len(params) == 2
    transpose_params = list(range(len(shape)))
    transpose_params[params[0]], transpose_params[params[1]] = (
        params[1],
        params[0],
    )
    params = [ShapeLike(transpose_params)]
    return pnode.args[:1], params


def t_default_arg_converter(pnode: PickleableNode):
    shape = pnode.get_tensor_shape()
    assert len(shape) == 2
    # Transpose dim 1 and 0, and another 0 for `shuffle`.
    params = [1, 0]
    return pnode.args[:1], [ShapeLike(params)]


def topk_default_arg_converter(pnode: PickleableNode):
    shape = pnode.args[0].get_tensor_shape()
    k, dim = pnode.args[1], pnode.args[2]
    if dim < 0:
        dim += len(shape)
    largest, sorted = pnode.args[3], pnode.args[4]
    assert type(k) == int
    assert type(largest) == bool
    assert type(sorted) == bool
    return [pnode.args[0]], [k, dim, largest, sorted]


def sort_default_converter(pnode: PickleableNode):
    return pnode.args, []


def argsort_stable_converter(pnode: PickleableNode):
    assert pnode.kwargs["stable"] is True, "Cannot handle unstable argsort."
    return pnode.args, []


def addcmul_arg_converter(pnode: PickleableNode):
    v = pnode.kwargs["value"]
    assert type(v) in (int, float)
    return pnode.args, [v]


def mask_fill_scalar_converter(pnode: PickleableNode):
    assert type(pnode.args[2]) in (int, float)
    return pnode.args[:2], [pnode.args[2]]


def native_dropout_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
):
    assert op_mapping.idx is not None
    real_pnode = pnode.args[0]
    inpt_pnode = real_pnode.args[0]
    if op_mapping.idx == 0:
        return tgops.clone, [pnode_to_sexpr[inpt_pnode]], [], None, None
    else:
        shape = inpt_pnode.get_tensor_shape()
        assert shape is not None
        shape = ShapeLike(shape)
        return tgops.fill, [], [shape, 0], None, shape
        raise NotImplementedError(
            f"native_dropout with {op_mapping.idx} is not implemented."
        )


def eq_scalar_converter(pnode: PickleableNode):
    assert type(pnode.args[1]) in (int, float)
    return pnode.args[:1], [pnode.args[1]]


def clamp_converter(pnode: PickleableNode):
    assert len(pnode.args) == 2
    return pnode.args[:1], [pnode.args[1]]


def sym_int_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
):
    shape = []
    name = pnode.get_scalar_name()
    return tgops.scalar, [], [], name, shape
    

def unsqueeze_arg_converter(pnode: PickleableNode):
    arg_shape = pnode.args[0].get_tensor_shape()
    assert arg_shape is not None
    dim = pnode.args[1]
    dim = positivate_dim(dim, arg_shape, neg_plus_one=True)
    return pnode.args[:1], [dim]


def view_default_arg_converter(pnode: PickleableNode):
    params = pnode.args[1]
    shape = pnode.get_tensor_shape()
    # Fix all the -1 in the shape.
    if -1 in params:
        idx = params.index(-1)
        assert (
            params.count(-1) <= 1 and len([p for p in params if p <= 0]) <= 1
        ), f"At most one -1 is allowed in view, got {params} ."
        shape_mul = reduce(mul, shape)
        param_mul = reduce(mul, params)
        param_mul = -param_mul
        assert shape_mul % param_mul == 0, f"Cannot reshape {shape=} into {params=}"
        params[idx] = shape_mul // param_mul
    return pnode.args[:1], [ShapeLike(params)]


def second_arg_as_shape_converter(pnode: PickleableNode):
    shape = pnode.args[1]
    assert all(p > 0 for p in shape), f"Only positive factors are allowed, got {shape}."
    return pnode.args[:1], [ShapeLike(pnode.args[1])]


def make_arg_as_shape_input_converter(
    idx: int, force_inpt: bool = True, shape_as_param: bool = False
) -> CONVERTER_TYPE:
    def arg_as_shape_input_converter(
        op_mapping: "OpMapping",
        pnode: PickleableNode,
        pnode_to_sexpr: CONVERT_MEMO_TYPE,
        name_prefix: str,
        idx=idx,
    ) -> CONVERTER_RETURN_TYPE:
        shape = ShapeLike(pnode.args[idx])
        if shape_as_param:
            params = [shape]
        else:
            params = []
        if force_inpt:
            op = tgops.inpt
        else:
            op = op_mapping.op
        return op, (), params, None, shape

    return arg_as_shape_input_converter


def local_scalar_dense_converter(
    op_mapping: "OpMapping",
    pnode: PickleableNode,
    pnode_to_sexpr: CONVERT_MEMO_TYPE,
    name_prefix: str,
) -> CONVERTER_RETURN_TYPE:
    # This should be a scalar. But we have no information to know that.
    # We temporarily use assertion of its name to ensure correctness.
    assert (
        pnode.name.find("scalar") != -1
    ), f"The pnode should be a scalar because it doesn't have shape. But the name doesn't contain 'scalar'. {pnode!s}"
    shape = []
    name = pnode.get_scalar_name()
    return tgops.scalar, [], [], name, shape


def softmax_arg_converter(pnode: PickleableNode):
    inpt_shape = pnode.args[0].get_tensor_shape()
    assert inpt_shape is not None
    return pnode.args[:1], [positivate_dim(pnode.args[1], inpt_shape)]


"""
List of IR is available from https://pytorch.org/docs/main/torch.compiler_ir.html
Note that some of the ops are missing from the page but will be manually added here.
"""
# aten._adaptive_avg_pool2d: _adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
# aten._adaptive_avg_pool2d_backward: _adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
# aten._adaptive_avg_pool3d: _adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
# aten._cdist_forward: _cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
# aten._embedding_bag: _embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)

# aten._local_scalar_dense: _local_scalar_dense(Tensor self) -> Scalar
OpMapping(
    "aten._local_scalar_dense.default",
    tgops.scalar,
    converter=local_scalar_dense_converter,
)
# aten._log_softmax: _log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
# aten._native_batch_norm_legit: _native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# aten._native_batch_norm_legit.no_stats: _native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# aten._native_batch_norm_legit_no_training: _native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
# aten._pdist_forward: _pdist_forward(Tensor self, float p=2) -> Tensor

# aten._softmax: _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
OpMapping("aten._softmax.default", tgops.softmax, arg_converter=softmax_arg_converter)

# aten._softmax_backward_data
OpMapping(
    "aten._softmax_backward_data.default",
    tgops.softmax_backward_data,
    arg_converter=lambda pnode: (
        pnode.args[:2],
        [positivate_dim(pnode.args[2], pnode.args[0].get_tensor_shape())],
    ),
)

# aten._to_copy: _to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
OpMapping("aten._to_copy.default", tgops.clone)

# aten.abs: abs(Tensor self) -> Tensor
# aten.acos: acos(Tensor self) -> Tensor
# aten.acosh: acosh(Tensor self) -> Tensor
# aten.adaptive_avg_pool1d: adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor

# aten.add.Scalar: add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
OpMapping("aten.add.Scalar", tgops.add_scalar)

# aten.add.Tensor: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
OpMapping("aten.add.Tensor", tgops.matadd)

# aten.addmm: addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
OpMapping("aten.addmm.default", tgops.addmm, arg_converter=addmm_arg_converter)

# aten.alias: alias(Tensor(a) self) -> Tensor(a)
OpMapping("aten.alias.default", tgops.clone)

# aten.amax: amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
# aten.amin: amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
# aten.any: any(Tensor self) -> Tensor
# aten.any.dim: any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
# aten.any.dims: any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor

# aten.arange.default
OpMapping("aten.arange.default", tgops.arange, arg_converter=arange_arg_converter)

# aten.arange.start
OpMapping("aten.arange.start", tgops.arange, arg_converter=arange_arg_converter)

# aten.arange.start_step: arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# aten.argmax: argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
# aten.argmin: argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
# aten.as_strided: as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
# aten.asin: asin(Tensor self) -> Tensor
# aten.asinh: asinh(Tensor self) -> Tensor
# aten.atan: atan(Tensor self) -> Tensor
# aten.atan2: atan2(Tensor self, Tensor other) -> Tensor
# aten.atan2.out: atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
# aten.atanh: atanh(Tensor self) -> Tensor
# aten.avg_pool1d: avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
# aten.avg_pool2d: avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
# aten.avg_pool2d_backward: avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
# aten.avg_pool3d: avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor

# aten.baddbmm
OpMapping("aten.baddbmm.default", tgops.baddbmm, arg_converter=baddbmm_arg_converter)

# aten.bitwise_and.Scalar: bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
# aten.bitwise_and.Tensor: bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
OpMapping("aten.bitwise_and.Tensor", tgops.bitwise_and)
# aten.bitwise_not: bitwise_not(Tensor self) -> Tensor
OpMapping("aten.bitwise_not.default", tgops.bitwise_not)
# aten.bitwise_or.Scalar: bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
# aten.bitwise_or.Tensor: bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
OpMapping("aten.bitwise_or.Tensor", tgops.bitwise_or)
# aten.bitwise_xor.Scalar: bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
# aten.bitwise_xor.Tensor: bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor

# aten.bmm: bmm(Tensor self, Tensor mat2) -> Tensor
OpMapping("aten.bmm.default", tgops.bmm)

# aten.cat: cat(Tensor[] tensors, int dim=0) -> Tensor
OpMapping("aten.cat.default", tgops.concat, converter=cat_default_converter)

# aten.ceil: ceil(Tensor self) -> Tensor

# aten.clamp: clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
OpMapping("aten.clamp.default", tgops.clamp, arg_converter=clamp_converter)

# aten.clamp.Tensor: clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor

# aten.clone: clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
OpMapping("aten.clone.default", tgops.clone)

# aten.col2im: col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor

# aten.constant_pad_nd: constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
OpMapping(
    "aten.constant_pad_nd.default",
    tgops.pad,
    arg_converter=constant_pad_nd_default_arg_converter,
)
# aten.convolution: convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
# aten.convolution_backward: convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)

# aten.copy: copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
OpMapping(
    "aten.copy.default", tgops.clone, arg_converter=lambda pnode: (pnode.args[1:2], [])
)

# aten.cos: cos(Tensor self) -> Tensor
OpMapping("aten.cos.default", tgops.cos)
# aten.cosh: cosh(Tensor self) -> Tensor
# aten.cumsum: cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor

# aten.detach.default: detach(Tensor self) -> Tensor
OpMapping("aten.detach.default", tgops.clone)

# aten.diagonal: diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
# aten.div.Scalar: div.Scalar(Tensor self, Scalar other) -> Tensor
# aten.div.Scalar_mode: div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor

# aten.div.Tensor: div.Tensor(Tensor self, Tensor other) -> Tensor
OpMapping("aten.div.Tensor", tgops.matdiv)

# aten.div.Tensor_mode: div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
# aten.embedding: embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
OpMapping("aten.embedding.default", tgops.embedding)

# aten.embedding_dense_backward: embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor
OpMapping("aten.embedding_dense_backward.default", tgops.embedding_dense_backward)

# aten.empty_like
OpMapping(
    "aten.empty_like.default", tgops.empty, converter=make_empty_converter("tensor")
)

# aten.empty.memory_format: empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
OpMapping(
    "aten.empty.memory_format", tgops.empty, converter=make_empty_converter("shape")
)

# aten.empty_strided: empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

# aten.eq.Scalar: eq.Scalar(Tensor self, Scalar other) -> Tensor
OpMapping("aten.eq.Scalar", tgops.eq_scalar, arg_converter=eq_scalar_converter)

# aten.eq.Tensor: eq.Tensor(Tensor self, Tensor other) -> Tensor
# aten.erf: erf(Tensor self) -> Tensor

# aten.exp: exp(Tensor self) -> Tensor
OpMapping("aten.exp.default", tgops.exp)

# aten.expand: expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
OpMapping(
    "aten.expand.default", tgops.expand, arg_converter=second_arg_as_shape_converter
)

# aten.expm1: expm1(Tensor self) -> Tensor

# aten.fill.Scalar: fill.Scalar(Tensor self, Scalar value) -> Tensor
OpMapping(
    "aten.fill.Scalar",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="tensor"),
)

# aten.fill.Tensor: fill.Tensor(Tensor self, Tensor value) -> Tensor
OpMapping(
    "aten.fill.Tensor",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="tensor"),
)

# aten.flip: flip(Tensor self, int[] dims) -> Tensor
# aten.floor: floor(Tensor self) -> Tensor
# aten.fmod.Scalar: fmod.Scalar(Tensor self, Scalar other) -> Tensor
# aten.fmod.Tensor: fmod.Tensor(Tensor self, Tensor other) -> Tensor
# aten.full: full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# aten.gather: gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
# aten.ge.Scalar: ge.Scalar(Tensor self, Scalar other) -> Tensor
OpMapping("aten.ge.Scalar", tgops.ge_scalar)
# aten.ge.Tensor: ge.Tensor(Tensor self, Tensor other) -> Tensor
OpMapping("aten.ge.Tensor", tgops.ge_tensor)

# aten.gelu: gelu(Tensor self, *, str approximate=’none’) -> Tensor
OpMapping("aten.gelu.default", tgops.gelu)

# aten.gelu_backward.default
OpMapping("aten.gelu_backward.default", tgops.gelu_backward)

# aten.grid_sampler_2d: grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
# aten.gt.Scalar: gt.Scalar(Tensor self, Scalar other) -> Tensor
# aten.gt.Tensor: gt.Tensor(Tensor self, Tensor other) -> Tensor
# aten.hardtanh: hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor

# aten.index.Tensor: index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
OpMapping("aten.index.Tensor", tgops.index)

# aten.index_put: index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
OpMapping("aten.index_put.default", tgops.index_put, converter=index_put_converter)

# aten.index_fill: index_fill(Tensor self, dim int, indices Tensor, value int/float) -> Tensor
OpMapping(
    "aten.index_fill.int_Scalar",
    tgops.index_fill_scalar,
    arg_converter=index_fill_scalar_arg_converter,
)

# aten.index_select: index_select(Tensor self, int dim, Tensor index) -> Tensor
OpMapping("aten.index_select.default", tgops.index_select)

# aten.isinf: isinf(Tensor self) -> Tensor
# aten.isnan: isnan(Tensor self) -> Tensor
# aten.le.Scalar: le.Scalar(Tensor self, Scalar other) -> Tensor
# aten.le.Tensor: le.Tensor(Tensor self, Tensor other) -> Tensor
OpMapping("aten.le.Tensor", tgops.le_tensor)

# aten.leaky_relu: leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor

# aten.log: log(Tensor self) -> Tensor
OpMapping("aten.log.default", tgops.log)

# aten.log10: log10(Tensor self) -> Tensor
# aten.log1p: log1p(Tensor self) -> Tensor
# aten.log2: log2(Tensor self) -> Tensor
# aten.logical_and: logical_and(Tensor self, Tensor other) -> Tensor
# aten.logical_not: logical_not(Tensor self) -> Tensor
# aten.logical_or: logical_or(Tensor self, Tensor other) -> Tensor
# aten.logical_xor: logical_xor(Tensor self, Tensor other) -> Tensor

# aten.lt.Scalar: lt.Scalar(Tensor self, Scalar other) -> Tensor
OpMapping("aten.lt.Scalar", tgops.lt_scalar)

# aten.lt.Tensor: lt.Tensor(Tensor self, Tensor other) -> Tensor

# aten.max.dim: max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
OpMapping("aten.max.dim", tgops.max_dim, arg_converter=max_dim_arg_converter)

# aten.max_pool2d_with_indices: max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
# aten.max_pool2d_with_indices_backward: max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
# aten.max_pool3d_with_indices: max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)

# aten.maximum: maximum(Tensor self, Tensor other) -> Tensor
OpMapping("aten.maximum", tgops.maximum)

# aten.mean.default(Tensor) -> Tensor
OpMapping(
    "aten.mean.default",
    tgops.mean_default,
    arg_converter=lambda pnode: (pnode.args, ()),
)

# aten.mean: mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
# aten.mean.dim: mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
# aten.min.dim: min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
# aten.minimum: minimum(Tensor self, Tensor other) -> Tensor

# aten.mm: mm(Tensor self, Tensor mat2) -> Tensor
OpMapping("aten.mm.default", tgops.matmul, arg_converter=lambda pnode: (pnode.args, ()))

# aten.mul.Scalar: mul.Scalar(Tensor self, Scalar other) -> Tensor
OpMapping("aten.mul.Scalar", tgops.mul_scalar)

# aten.mul.Tensor: mul.Tensor(Tensor self, Tensor other) -> Tensor
OpMapping("aten.mul.Tensor", tgops.ewmul)

# aten.addcmul.default
OpMapping("aten.addcmul.default", tgops.addcmul, arg_converter=addcmul_arg_converter)

# aten.masked_fill.Scalar
OpMapping(
    "aten.masked_fill.Scalar",
    tgops.mask_fill_scalar,
    arg_converter=mask_fill_scalar_converter,
)

# aten.native_dropout: native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
OpMapping(
    "aten.native_dropout.default",
    tgops.native_dropout,
    converter=native_dropout_converter,
)

# aten.native_group_norm: native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
# aten.native_group_norm_backward: native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
# aten.native_layer_norm: native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)

OpMapping("aten.native_layer_norm.default", tgops.native_layer_norm)

# aten.native_layer_norm_backward: native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
# aten.ne.Scalar: ne.Scalar(Tensor self, Scalar other) -> Tensor
# aten.ne.Tensor: ne.Tensor(Tensor self, Tensor other) -> Tensor
# aten.neg: neg(Tensor self) -> Tensor

# aten.new_empty_strided.default
OpMapping(
    "aten.new_empty_strided.default",
    tgops.empty,
    converter=make_arg_as_shape_input_converter(idx=1),
)


# aten.nonzero: nonzero(Tensor self) -> Tensor
OpMapping("aten.nonzero.default", tgops.nonzero)

# aten.ones_like.default
# FIXME: Use it as a placeholder for now. But the semantic of ones might matter.
OpMapping(
    "aten.ones_like.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="tensor", value=1),
)

OpMapping(
    "aten.ones.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="shape", value=1),
)

# aten.new_ones.default
OpMapping(
    "aten.new_ones.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="shape", value=1, shape_arg_index=1),
)

# aten.permute: permute(Tensor(a) self, int[] dims) -> Tensor(a)
OpMapping(
    "aten.permute.default",
    tgops.transpose,
    arg_converter=permute_default_arg_converter,
)
# aten.pow.Scalar: pow.Scalar(Scalar self, Tensor exponent) -> Tensor

# aten.pow.Tensor_Scalar: pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
OpMapping(
    "aten.pow.Tensor_Scalar",
    tgops.pow_tensor_scalar,
    arg_converter=pow_tensor_scalar_converter,
)

# aten.pow.Tensor_Tensor: pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
# aten.prod: prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
# aten.prod.dim_int: prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
# aten.rand: rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# aten.randn: randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# aten.randperm: randperm(SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
# aten.reciprocal: reciprocal(Tensor self) -> Tensor
# aten.reflection_pad1d: reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor
# aten.reflection_pad2d: reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor
# aten.reflection_pad3d: reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor
# aten.relu: relu(Tensor self) -> Tensor
# aten.remainder.Scalar: remainder.Scalar(Tensor self, Scalar other) -> Tensor
# aten.remainder.Tensor: remainder.Tensor(Tensor self, Tensor other) -> Tensor

# aten.repeat: repeat(Tensor self, SymInt[] repeats) -> Tensor
OpMapping(
    "aten.repeat.default",
    tgops.repeat,
    arg_converter=second_arg_as_shape_converter,
)

# aten.replication_pad2d: replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor
# aten.replication_pad3d: replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor
# aten.resize_: resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
# aten.round: round(Tensor self) -> Tensor
# aten.rsqrt: rsqrt(Tensor self) -> Tensor

# aten.rsub.Scalar: rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
OpMapping("aten.rsub.Scalar", tgops.rsub_scalar)

# aten.scalar_tensor: scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

# aten.scatter.src: scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
OpMapping(
    "aten.scatter.src", tgops.scatter_src, arg_converter=scatter_src_arg_converter
)

# aten.scatter.value: scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
# aten.scatter_add: scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
# aten.scatter_reduce.two: scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor

# aten.select.int: select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
OpMapping("aten.select.int", tgops.slice)

# aten.select_scatter: select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
OpMapping("aten.select_scatter.default", tgops.select_scatter)

# aten.sigmoid: sigmoid(Tensor self) -> Tensor
OpMapping("aten.sigmoid.default", tgops.sigmoid)

# aten.sigmoid_backward.default
OpMapping("aten.sigmoid_backward.default", tgops.sigmoid_backward)

# aten.sign: sign(Tensor self) -> Tensor

# aten.silu.default: silu(Tensor self) -> Tensor
OpMapping("aten.silu.default", tgops.silu)

# aten.sin: sin(Tensor self) -> Tensor
OpMapping("aten.sin.default", tgops.sin)
# aten.sinh: sinh(Tensor self) -> Tensor

# aten.slice.Tensor: slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
OpMapping("aten.slice.Tensor", tgops.slice, arg_converter=slice_tensor_arg_converter)

# aten.slice_backward.default: at::Tensor at::slice_backward(const at::Tensor &grad_output, at::IntArrayRef input_sizes, int64_t dim, int64_t start, int64_t end, int64_t step)
# See https://pytorch.org/cppdocs/api/function_namespaceat_1a756675fa9bbe312abbbf80811ef256c3.html.
# XXX Zhanghan: Looks like this is some kind of padding.
OpMapping("aten.slice_backward.default", tgops.slice_backward)

# aten.slice_scatter: slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
OpMapping(
    "aten.slice_scatter.default",
    tgops.slice_scatter,
    arg_converter=slice_scatter_arg_converter,
)

# aten.sort: sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
OpMapping("aten.sort.default", tgops.sort, arg_converter=sort_default_converter)

# aten.argsort.stable(Tensor self, int dim=-1, bool stable=False) -> (Tensor indices)
OpMapping("aten.argsort.stable", tgops.argsort, arg_converter=argsort_stable_converter)


# aten.split.Tensor: split.Tensor(Tensor self, int split_size, int dim=0) -> Tensor[]
OpMapping(
    "aten.split.Tensor",
    tgops.split,
    converter=split_tensor_converter,
)

# aten.split_with_sizes: split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
OpMapping(
    "aten.split_with_sizes.default",
    tgops.split_with_sizes,
    converter=split_with_sizes_converter,
)
# aten.sqrt: sqrt(Tensor self) -> Tensor
# aten.squeeze.dim: squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
OpMapping("aten.squeeze.dim", tgops.squeeze, arg_converter=squeeze_arg_converter)
# aten.squeeze.dims: squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
# aten.sub.Scalar: sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor

# aten.sub.Tensor: sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
OpMapping("aten.sub.Tensor", tgops.matsub)

# aten.sum.dim_IntList: sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
OpMapping(
    "aten.sum.dim_IntList", tgops.sum_dim_int_list, converter=sum_dim_int_list_converter
)

# aten.sum.default: sum(Tensor self) -> Tensor
OpMapping("aten.sum.default", tgops.sum_default, arg_converter=sum_default_converter)

# aten.cumsum.default: cumsum(Tensor self, int dim) -> Tensor
OpMapping("aten.cumsum.default", tgops.cumsum, arg_converter=cumsum_converter)

# aten.sym_numel: sym_numel(Tensor self) -> SymInt
# aten.sym_size.int: sym_size.int(Tensor self, int dim) -> SymInt
OpMapping("aten.sym_size.int", tgops.scalar, converter=sym_int_converter)

# aten.sym_storage_offset: sym_storage_offset(Tensor self) -> SymInt
# aten.sym_stride.int: sym_stride.int(Tensor self, int dim) -> SymInt

# aten.t: t(Tensor self) -> Tensor
OpMapping("aten.t.default", tgops.transpose, arg_converter=t_default_arg_converter)

# aten.tan: tan(Tensor self) -> Tensor
# aten.tanh: tanh(Tensor self) -> Tensor
# aten.topk: topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
OpMapping("aten.topk.default", tgops.topk, arg_converter=topk_default_arg_converter)

# aten.transpose.int: transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
OpMapping(
    "aten.transpose.int",
    tgops.transpose,
    arg_converter=transpose_int_arg_converter,
)

# aten.trunc: trunc(Tensor self) -> Tensor

OpMapping(
    "aten._unsafe_view.default", tgops.reshape, arg_converter=view_default_arg_converter
)

# aten.unsqueeze: unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
OpMapping(
    "aten.unsqueeze.default", tgops.unsqueeze, arg_converter=unsqueeze_arg_converter
)

# aten.upsample_bilinear2d.vec: upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
# aten.upsample_nearest2d.vec: upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
# aten.var.correction: var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
# aten.var.dim: var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor

# aten.view: view(Tensor(a) self, SymInt[] size) -> Tensor(a)
OpMapping("aten.view.default", tgops.reshape, arg_converter=view_default_arg_converter)

# aten.where.self: where.self(Tensor condition, Tensor self, Tensor other) -> Tensor

# aten.zero.default
OpMapping(
    "aten.zero.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="tensor", value=0, shape_arg_index=0),
)

# aten.zeros.default
OpMapping(
    "aten.zeros.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="shape", value=0),
)

OpMapping(
    "aten.new_zeros.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="shape", value=0, shape_arg_index=1),
)

# aten.zeros_like.default
# FIXME: Use it as a placeholder for now. But the semantic of zeros might matter.
OpMapping(
    "aten.zeros_like.default",
    tgops.fill,
    converter=make_fill_converter(shape_arg_type="tensor", value=0),
)

# aten.bernoulli.p
OpMapping("aten.bernoulli.p", tgops.inpt, arg_converter=empty_arg_converter)
