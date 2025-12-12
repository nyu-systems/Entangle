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

from typing import Union

import numpy as np
from torch_neuronx.pyhlo import xla_data_pb2
from torch_neuronx.pyhlo.service.hlo_pb2 import (
    HloComputationProto,
    HloInstructionProto,
    HloModuleProto,
)
from torch_neuronx.pyhlo.xla_data_pb2 import LiteralProto

from entangle.pgraph.pickleable import *

TO_TORCH_DTYPE = {
    xla_data_pb2.PRED: torch.bool,
    xla_data_pb2.F32: torch.float32,
    xla_data_pb2.F16: torch.float16,
    xla_data_pb2.S32: torch.int32,
    xla_data_pb2.S64: torch.int64,
}

TO_FX_TARGET_NAME = {
    "add": "aten.add.Tensor",
    "all_reduce": "_c10d_functional.all_reduce.default",
    "and": "aten.bitwise_and.Tensor",
    "arange": "aten.arange.start",
    "clone": "aten.clone.default",
    "concatenate": "aten.cat.default",
    "cosine": "aten.cos.default",
    "divide": "aten.div.Tensor",
    "empty": "aten.empty.memory_format",
    "exponential": "aten.exp.default",
    "maximum": "aten.maximum",
    "multiply": "aten.mul.Tensor",
    "ge.Tensor": "aten.ge.Tensor",
    "le.Tensor": "aten.le.Tensor",
    "ones": "aten.ones.default",
    "reduce_sum": "aten.sum.dim_IntList",
    "reshape": "aten.view.default",
    "sine": "aten.sin.default",
    "slice": "aten.slice.Tensor",
    "subtract": "aten.sub.Tensor",
    "transpose": "aten.permute.default",
    "zeros": "aten.zeros.default",
}


def get_reduce_op(computation: HloComputationProto) -> str:
    if len(computation.instructions) != 3:
        raise NotImplementedError(f"{computation.instructions=} not implemented")
    assert computation.instructions[0].opcode == "parameter"
    assert computation.instructions[1].opcode == "parameter"
    assert computation.instructions[2].opcode in ("add", "maximum")
    op = computation.instructions[2].opcode
    return op


def convert_reduce(
    instr: HloInstructionProto,
    args: list[PickleableNode],
    id_to_pnode: dict[int, PickleableNode],
    id_to_computation: dict[int, HloComputationProto],
) -> tuple[str, list[int], dict]:
    """
    Returns (target, shape, args, kwargs)
    """
    dims = instr.dimensions
    if len(dims) != 1:
        raise NotImplementedError(f"{dims=} not implemented for reduce")
    dim = dims[0]
    inputs = id_to_pnode[instr.operand_ids[0]]
    init_values = id_to_pnode[instr.operand_ids[1]]

    assert len(instr.called_computation_ids) == 1
    computation = id_to_computation[instr.called_computation_ids[0]]
    op = get_reduce_op(computation)
    if op == "add":
        if init_values == 0.0:
            # 'aten.sum.dim_IntList', args=('n__r0__fw_values', [-1], keep_dims=False)
            target = "reduce_sum"
            args = [inputs, [dim], False]
            return (target, args, {})
        else:
            raise NotImplementedError(
                f"{op=} with {init_values=} not implemented for reduce"
            )
    elif op == "maximum":
        if init_values == -float("inf"):
            # 'aten.max.dim', args=('n__r0__fw_values', [-1], keep_dims=False)
            target = "reduce_max"
            args = [inputs, dim]
            return (target, args, {})
        else:
            raise NotImplementedError(
                f"{op=} with {init_values=} not implemented for reduce"
            )
    else:
        raise NotImplementedError(f"{op=} not implemented for reduce")


def convert_all_reduce(
    instr: HloInstructionProto,
    args: list[PickleableNode],
    id_to_pnode: dict[int, PickleableNode],
    id_to_computation: dict[int, HloComputationProto],
) -> tuple[str, list[int], dict]:
    assert len(instr.called_computation_ids) == 1
    computation = id_to_computation[instr.called_computation_ids[0]]
    op = get_reduce_op(computation)
    inputs = id_to_pnode[instr.operand_ids[0]]
    assert len(instr.replica_groups) == 1
    replica_ids = [str(i) for i in instr.replica_groups[0].replica_ids]
    replica_ids_str = "g" + "_".join(sorted(replica_ids))
    target = "all_reduce"
    if op == "add":
        args = [inputs, "sum", replica_ids_str]
        return (target, args, {})
    elif op == "maximum":
        args = [inputs, "max", replica_ids_str]
        return (target, args, {})
    else:
        raise NotImplementedError(f"{op=} not implemented for all-reduce")


def convert_literal(instr: HloInstructionProto) -> tuple[str, list[int], dict]:
    """
    Returns (target, shape, args, kwargs)
    """
    literal: "LiteralProto" = instr.literal
    shape = list(literal.shape.dimensions)
    dtype = literal.shape.element_type

    if dtype == xla_data_pb2.F32:
        values = list(literal.f32s)
    # The F16s, BF16s, U16s and S16s are encoded in little endian byte order
    elif dtype == xla_data_pb2.F16:
        values = np.frombuffer(literal.f16s, dtype=np.float16).tolist()
    else:
        raise NotImplementedError(f"{dtype=} not implemented")
    if len(shape) == 0:
        return ("scalar", [values[0]], {})
    elif len(set(values)) == 1:
        value = values[0]
        if value == 0:
            return ("zeros", [shape], {})
        elif value == 1:
            return ("ones", [shape], {})
        else:
            raise NotImplementedError(
                f"literal value {value} not implemented, {instr=}"
            )
    else:
        return ("empty", [shape], {})


def convert_hlo_proto_to_pnode(
    instr: HloInstructionProto,
    rank: int,
    id_to_pnode: dict[int, PickleableNode],
    id_to_computation: dict[int, HloComputationProto],
) -> Union[PickleableNode, float, int]:
    get_pnode = lambda instr_id: id_to_pnode[instr_id]
    name = instr.name
    opcode = instr.opcode
    if opcode == "parameter":
        op = "placeholder"
        target = name
    elif opcode == "custom-call":
        op = "call_function"
        target = instr.custom_call_target
    else:
        op = "call_function"
        target = opcode
    # Default args and kwargs
    args = []
    kwargs = {}

    shape = tuple(instr.shape.dimensions)

    # Now, use op and target below.
    if op == "placeholder":
        pass
    else:
        # Check https://www.tensorflow.org/mlir/hlo_ops and
        # https://github.com/openxla/stablehlo/blob/main/docs/spec.md for ops definitions.
        assert op == "call_function", f"{op=}, {target=} not implemented"
        args = [get_pnode(instr_id) for instr_id in instr.operand_ids]
        if target == "AwsNeuronRmsNorm":
            pass
        elif target == "AwsNeuronTransferWithStaticRing":
            target = "clone"
        elif target in (
            "add",
            "and",
            "cosine",
            "divide",
            "exponential",
            "logistic",
            "maximum",
            "multiply",
            "sine",
            "subtract",
        ):
            pass
        elif target == "all-reduce":
            target, args, kwargs = convert_all_reduce(
                instr, args, id_to_pnode, id_to_computation
            )
        elif target == "broadcast":
            dims = instr.dimensions
            args = [*args, list(dims), shape]
        elif target == "compare":
            target = f"{instr.comparison_direction.lower()}.Tensor"
        elif target == "concatenate":
            dims = instr.dimensions
            if len(dims) != 1:
                raise NotImplementedError(f"{dims=} not implemented for concatenate")
            dim = dims[0]
            args = [args, dim]
        elif target == "constant":
            target, args, kwargs = convert_literal(instr)
        elif target == "convert":
            # Treat `convert` as no-op for now, because we currently don't care about the dtype.
            target = "clone"
        elif target == "dot":
            lhs_contracting_dims = list(
                instr.dot_dimension_numbers.lhs_contracting_dimensions
            )
            rhs_contracting_dims = list(
                instr.dot_dimension_numbers.rhs_contracting_dimensions
            )
            lhs_batch_dims = list(instr.dot_dimension_numbers.lhs_batch_dimensions)
            rhs_batch_dims = list(instr.dot_dimension_numbers.rhs_batch_dimensions)
            args = args + [
                lhs_contracting_dims,
                rhs_contracting_dims,
                lhs_batch_dims,
                rhs_batch_dims,
            ]
        elif target == "gather":
            operand = args[0]
            start_indices = args[1]
            offset_dims = instr.gather_dimension_numbers.offset_dims
            collapsed_slice_dims = instr.gather_dimension_numbers.collapsed_slice_dims
            start_index_map = instr.gather_dimension_numbers.start_index_map
            index_vector_dim = instr.gather_dimension_numbers.index_vector_dim
            # operand_batching_dims = instr.gather_dimension_numbers.operand_batching_dims
            # start_indices_batching_dims = (
            #     instr.gather_dimension_numbers.start_indices_batching_dims
            # )
            slice_sizes = instr.gather_slice_sizes
            args = [
                operand,
                start_indices,
                list(offset_dims),
                list(collapsed_slice_dims),
                list(start_index_map),
                index_vector_dim,
                # list(operand_batching_dims),
                # list(start_indices_batching_dims),
                list(slice_sizes),
            ]
        elif target == "iota":
            if len(shape) != 1:
                raise NotImplementedError(f"{shape=} not implemented for iota")
            dims = instr.dimensions
            assert len(dims) == 1 and dims[0] == 0, f"{dims=} not implemented for iota"
            target = "arange"
            args = [0, shape[0]]
        elif target == "reduce":
            target, args, kwargs = convert_reduce(
                instr, args, id_to_pnode, id_to_computation
            )
        elif target == "reshape":
            shape = list(instr.shape.dimensions)
            args = args + [shape]
        elif target == "select":
            assert len(args) == 3
        elif target == "slice":
            arg_shape = args[0].get_tensor_shape()
            assert arg_shape is not None, f"{arg_shape=} should not be None"
            slice_dims = [
                (i, s.start, s.limit, s.stride)
                for i, s in enumerate(instr.slice_dimensions)
            ]
            useful_slice_dims = []
            for size, (dim, start, end, stride) in zip(arg_shape, slice_dims):
                if end - start > 0 and end - start != size:
                    useful_slice_dims.append((dim, start, end, stride))
            if len(useful_slice_dims) == 0:
                target = "clone"
                args = args[:1]
            elif len(useful_slice_dims) == 1:
                dim, start, end, stride = useful_slice_dims[0]
                args = args + [dim, start, end, stride]
            else:
                raise NotImplementedError(
                    f"{useful_slice_dims=} for more than 1 dimension not implemented"
                )
        elif target == "transpose":
            dims = list(instr.dimensions)
            assert len(args) == 1
            args = args + [dims]
        elif target == "tuple":
            raise RuntimeError("We only consider tuple as outputs.")
        else:
            print(instr)
            raise NotImplementedError(f"{op=}, {target=} not implemented")
        if target in TO_FX_TARGET_NAME:
            target = TO_FX_TARGET_NAME[target]
        else:
            target = f"hlo.{target}"
    if "\n----\n" not in instr.metadata.source_file:
        raise RuntimeError(
            f"Meta source_file should contain \\n----\\n, but not found in {instr=}"
        )
    stack = eval(instr.metadata.source_file.split("\n----\n")[1])
    meta = {
        "tensor_meta": TensorMetadata(
            shape=shape,
            dtype=TO_TORCH_DTYPE[instr.shape.element_type],
            requires_grad=False,
            stride=None,
            memory_format=None,
            is_quantized=False,
            qparams=None,
        ),
        "stack": stack,
    }
    pnode = PickleableNode(
        rank=rank, op=op, name=name, target=target, args=args, kwargs=kwargs, meta=meta
    )
    return pnode


def from_hlo_module(hlo_module: "HloModuleProto", rank: int) -> PickleableGraph:
    # https://github.com/openxla/xla/blob/main/xla/service/hlo.proto
    # https://github.com/openxla/xla/blob/main/xla/xla.proto
    # https://github.com/openxla/xla/blob/main/xla/xla_data.proto

    # hlo_module = HloModuleProto()
    # hlo_module.ParseFromString(open(path, "rb").read())
    id_to_computation = {}
    name_to_computation = {}
    for computation in hlo_module.computations:
        id_to_computation[computation.id] = computation
        name_to_computation[computation.name] = computation
    entry_computation = name_to_computation[hlo_module.entry_computation_name]

    id_to_pnode: dict[int, PickleableNode] = {}
    pgraph = PickleableGraph(rank=rank)
    for instr in entry_computation.instructions:
        if instr.id == entry_computation.root_id:
            assert instr.opcode == "tuple"
            pnode = PickleableNode(
                rank,
                name="output",
                op="output",
                target="output",
                args=[[id_to_pnode[i] for i in instr.operand_ids]],
                kwargs={},
            )
            pgraph.add_node(pnode)
            id_to_pnode[instr.id] = pnode
        else:
            pnode = convert_hlo_proto_to_pnode(
                instr, rank, id_to_pnode, id_to_computation
            )
            if pnode.target == "hlo.scalar":
                # We force inline all scalars.
                value = pnode.args[0]
                id_to_pnode[instr.id] = value
            else:
                pgraph.add_node(pnode)
                id_to_pnode[instr.id] = pnode
    pgraph.sanity_check()
    return pgraph
