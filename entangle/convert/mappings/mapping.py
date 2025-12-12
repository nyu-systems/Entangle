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

from typing import Callable, Optional, Union

import sympy

import entangle.ops as tgops
from entangle.pgraph import PickleableNode
from entangle.sgraph import SExpr, ShapeLike
from entangle.sym.sym_manager import SymManager

CONVERT_MEMO_TYPE = dict[PickleableNode, SExpr]
ARG_CONVERTER_RETURN_TYPE = tuple[list[PickleableNode], list[int | float | bool | str]]

# For arg_converter, the input pnode is the pnode or the getitem target pnode.
ARG_CONVERTER_TYPE = Callable[[PickleableNode], ARG_CONVERTER_RETURN_TYPE]

# For converter, the input pnode is always current pnode (i.e., getitem itself if any).
# [op, args, params, name, shape]
CONVERTER_RETURN_TYPE = tuple[
    Optional[tgops.Op],  # op can be None for default
    list[PickleableNode | SExpr],
    list[int | float | bool | str],
    Optional[str],  # name can be None for default
    Optional[list[int]],  # shape can be None for default
]  # or returning op, args, params, name, shape

# Params are [op_mapping, pnode, pnode_to_sexpr, name_prefix]
CONVERTER_TYPE = Callable[
    ["OpMapping", PickleableNode, CONVERT_MEMO_TYPE, str],
    CONVERTER_RETURN_TYPE,
]


def positivate_dim(dim: int, shape: list[int], neg_plus_one=False) -> int:
    assert shape is not None
    if dim >= 0:
        return dim
    else:
        return len(shape) + dim + neg_plus_one



class OpMapping:
    MAPPINGS: dict[str, "OpMapping"] = {}

    def __init__(
        self,
        fx_name,
        op: tgops.Op,
        auto_functionalized: bool = False,
        arg_converter: ARG_CONVERTER_TYPE = None,
        converter: CONVERTER_TYPE = None,
        idx: int = None,
    ):
        if fx_name in OpMapping.MAPPINGS:
            raise ValueError(
                f"OpMapping {fx_name} already exists in MAPPINGS: {OpMapping.MAPPINGS[fx_name]}"
            )
        self.fx_name = fx_name
        self.op = op
        self.auto_functionalized = auto_functionalized
        self.arg_converter: ARG_CONVERTER_TYPE = arg_converter

        assert not (
            (arg_converter is not None and arg_converter != self.default_arg_converter)
            and (converter is not None and converter != self.default_converter)
        ), "Only one of arg_converter and converter can be set if not default."
        self.arg_converter = arg_converter or self.default_arg_converter
        self.converter = converter or self.default_converter
        self.idx = idx

        OpMapping.MAPPINGS[fx_name] = self

    def getitem(self, idx: int):
        """
        Get the getitem version of OpMapping for self.
        """
        assert (
            self.op.multi_out or self.auto_functionalized
        ), f"Only multi-out ops or auto_functionalized op mapping can have getitem subops, got {str(self)}"
        name = f"{self.fx_name}_{idx}"
        if name in OpMapping.MAPPINGS:
            return OpMapping.MAPPINGS[name]
        elif self.auto_functionalized and not self.op.multi_out:
            # In the case of single-output auto_functionalized function,
            # we don't need to get the getitem version of the op.
            assert idx == 0
            return OpMapping(
                name,
                self.op,
                auto_functionalized=False,
                arg_converter=self.arg_converter,
                converter=self.converter,
                idx=idx,
            )
        else:
            return OpMapping(
                name,
                self.op.getitem(idx),
                auto_functionalized=False,
                arg_converter=self.arg_converter,
                converter=self.converter,
                idx=idx,
            )

    def __str__(self):
        return f"OpMapping({self.fx_name}, {self.op}, auto_functionalized={self.auto_functionalized})"

    def __repr__(self):
        return f"OpMapping({self.fx_name})"

    @staticmethod
    def get(pnode: PickleableNode) -> "OpMapping":
        """
        name:
            1. For op in ("placeholder", "get_attr"), name should be "placeholder"
            2. For target == "<built-in function getitem>", name should be parent's name.
                2.1.
                    But if parent is auto_functionalized, the name depends on if the op
                    is multi_out or not.
                2.2.
                    otherwise, return the getitem version of parent's op mapping.
            3.1.
                For op in ("call_function") and target != "auto_functionalized",
                name should be the `target` name.
            3.2.
                For op in ("call_function") and target == "auto_functionalized",
                name should be args[0]
        """
        if pnode.op in ("placeholder", "get_attr"):
            return OpMapping.MAPPINGS["placeholder"]
        elif pnode.target == "<built-in function getitem>":
            parent_pnode = pnode.args[0]
            parent_op_mapping = OpMapping.get(parent_pnode)
            if parent_pnode.target == "auto_functionalized":
                if parent_op_mapping.op.multi_out:
                    # We need the index from 0 (auto_functionalized uses index starting from 1)
                    return parent_op_mapping.getitem(pnode.args[1] - 1)
                else:
                    return parent_op_mapping.getitem(0)
            elif parent_pnode.target == "auto_functionalized_v2":
                # This was added since [Jan-06-2025](https://dev-discuss.pytorch.org/t/a-new-strategy-for-automatic-custom-operators-functionalization/2733)
                if parent_op_mapping.op.multi_out:
                    # We need the index from 0 (auto_functionalized uses index starting from 1)
                    return parent_op_mapping.getitem(pnode.args[1] - 1)
                else:
                    return parent_op_mapping.getitem(0)
            else:
                return parent_op_mapping.getitem(pnode.args[1])
        elif pnode.op == "call_function" and not pnode.target.startswith("auto_functionalized"):
            return OpMapping.MAPPINGS[pnode.target]
        elif pnode.op == "call_function" and pnode.target.startswith("auto_functionalized"):
            return OpMapping.MAPPINGS[pnode.args[0]]
        else:
            raise ValueError(f"Unsupported op={pnode.op}, target={pnode.target}")

    @staticmethod
    def default_arg_converter(pnode):
        args = []
        params = []
        if pnode.target == "auto_functionalized":
            used_args = pnode.kwargs.values()
        elif pnode.target == "auto_functionalized_v2":
            if pnode.args[0] != "tg.inplace_log_tensor.default":
                raise NotImplementedError("auto_functionalized_v2 is not supported for operators other than log_tensor yet.")
            used_args = (pnode.kwargs["s"], pnode.kwargs["_all_bases"][0])
        else:
            used_args = pnode.args

        # Convert the `used_args`
        for arg in used_args:
            if type(arg) is PickleableNode and arg.get_tensor_shape() is not None:
                # If arg.get_tensor_shape() is None, then it is a scalar input.
                args.append(arg)
            else:
                if type(arg) is PickleableNode and arg.get_tensor_shape() in (
                    None,
                    [],
                ):
                    assert arg.name.find("scalar") != -1
                if type(arg) is list:
                    if all(type(a) is int for a in arg):
                        # ShapeLike
                        params.append(ShapeLike(arg))
                    else:
                        for a in arg:
                            if type(a) is PickleableNode:
                                args.append(a)
                            else:
                                raise RuntimeError(
                                    f"Got {a=}, {type(a)=}\n{args=}\n{str(pnode)=}"
                                )
                else:
                    params.append(arg)
        # Convert the `kwargs` from `pnode`
        if not pnode.target.startswith("auto_functionalized") and pnode.kwargs != {}:
            if pnode.target in (
                "aten.clone.default",
                "aten.empty.memory_format",
                "aten.new_ones.default",
                "aten._to_copy.default",
                "aten.empty_like.default",
                "aten.argsort.stable",
                "aten.arange.start",
                "aten.ones_like.default",
                "aten.zeros_like.default",
                "aten.zeros.default",
            ):
                pass
            elif pnode.target == "aten.baddbmm.default":
                # {"beta": 0.0, "alpha": 0.0}
                # FIXME: We may need to allow float in Egg.
                # Ignored beta and alpha here. But we should add them back later.
                pass
            elif pnode.target == "aten.addcmul.default":
                assert pnode.kwargs == {"value": 1.0}
                params.append(1.0)
            elif pnode.target == "aten.add.Tensor":
                params.append(pnode.kwargs["alpha"])
            else:
                assert False, f"kwargs are not supported, got {str(pnode)=}"
        return args, params

    @staticmethod
    def default_converter(
        op_mapping: "OpMapping",
        pnode: PickleableNode,
        pnode_to_sexpr: CONVERT_MEMO_TYPE,
        name_prefix: str,
    ) -> CONVERTER_RETURN_TYPE:
        assert (
            not op_mapping.op.multi_out
        ), "cannot convert multi-out op to SExpr, must use its getitem version."

        if op_mapping.idx is not None:
            arg_param_source_pnode = pnode.args[0]
        else:
            arg_param_source_pnode = pnode
        args, params = op_mapping.arg_converter(arg_param_source_pnode)

        args = [
            arg if type(arg) is not PickleableNode else pnode_to_sexpr[arg]
            for arg in args
        ]
        params = [
            param if type(param) is not PickleableNode else pnode_to_sexpr[param]
            for param in params
        ]

        return (
            op_mapping.op,
            args,
            params,
            name_prefix + repr(pnode),
            pnode.get_tensor_shape(),
        )

    def convert(
        self,
        pnode: PickleableNode,
        pnode_to_sexpr: CONVERT_MEMO_TYPE,
        name_prefix: Optional[str] = "",
    ) -> SExpr:
        """
        This method `convert` will check the None returns and fill those fields with default methods.
        """
        op, args, params, name, shape = self.converter(
            self, pnode, pnode_to_sexpr, name_prefix
        )
        assert all(
            not isinstance(a, PickleableNode) for a in args
        ), f"args should not contain PickleableNode, got {args}"
        if op is None:
            op = self.op
        if name is None:
            name = name_prefix + repr(pnode)
        if op == tgops.scalar:
            name = SymManager.make_scalar_name(name, name_prefix)
            sym_expr = SymManager.convert_sympy_expr(sympy.Symbol(name), "")[1]
        else:
            sym_expr = None
            assert name.startswith(
                name_prefix
            ), f"{name=} should start with {name_prefix}, check your converter."
        if shape is None:
            shape = pnode.get_tensor_shape()
        rank = pnode.rank

        # Convert any `sympy.Expr` in shape
        assert all(
            any(isinstance(s, t) for t in (int, sympy.Expr)) for s in shape
        ), f"shape should be a list of int or sympy.Expr, got { {s: type(s) for s in shape} }"

        shape = [
            (s if type(s) is int else SymManager.convert_sympy_expr(s, name_prefix)[0])
            for s in shape
        ]

        # Convert any `sympy.Expr` in params
        for i in range(len(params)):
            param = params[i]
            if isinstance(param, sympy.Expr):
                params[i] = SymManager.convert_sympy_expr(param, name_prefix)[0]

        sexpr = SExpr(
            op, args, params, name=name, shape=shape, rank=rank, sym_expr=sym_expr
        )
        return sexpr
