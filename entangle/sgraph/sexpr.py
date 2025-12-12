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

import copy
import re
from enum import Enum
from math import prod
from typing import Callable, Iterable, Optional, Union

import sympy

import entangle.ops as tgops
import entangle.sym as sym
import entangle.utils
from entangle.ops import Op


def indent_str(s: str, indent: int):
    indent_str = indent * " "
    return indent_str + s.replace("\n", "\n" + indent_str)


class ShapeLike(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for s in self:
            if type(s) == sympy.Symbol:
                raise RuntimeError(
                    "There may be bug. Symbolic element in ShapeLike should be in form of SExpr."
                )

    def __repr__(self):
        s_list = list(map(SExpr.any_to_egg_str, self))
        return f'''"[{','.join(s_list)}]"'''
        # for char in ("_", " ", "(", ")"):
        #     assert all(
        #         s.find(char) == -1 for s in s_list
        #     ), f"found s containing '{char}' in {s_list=}, this will confuse the shape parser."
        # return "s" + "_".join(map(SExpr.any_to_egg_str, self))

    def parentheses_str(self):
        return f"({','.join(map(SExpr.any_to_egg_str, self))})"

    def to_egg_str(self):
        return "[{}]".format(
            ",".join([s.to_egg_str() if type(s) is SExpr else str(s) for s in self])
        )


class SExpr:
    SCALAR_PREFIX = "Sym"

    class FORMAT(Enum):
        SEXPR = 0
        RECEXPR_LIST = 1

        def extension(self):
            if self == SExpr.FORMAT.SEXPR:
                return "sexpr"
            elif self == SExpr.FORMAT.RECEXPR_LIST:
                return "recexpr"

    SEXPR_ID_GEN = entangle.utils.get_id_generator()

    @staticmethod
    def get_rank_from_name(name: str) -> int:
        matched = re.search(r"__r(\d+)__|R(\d+)\w+\d+", name)
        if matched is not None:
            return int(matched.group(1) or matched.group(2))
        else:
            return None

    def __init__(
        self,
        op: Op,
        args: list["SExpr"] = None,
        params: list[int] = None,
        name: Optional[str] = None,
        shape: Optional[ShapeLike | list[Union[int, "SExpr"]]] = None,
        rank: Optional[int] = None,
        sym_expr: Optional[sympy.Expr] = None,
        dist_id: int = None,
    ):
        self.op: Op = op
        self.args: list[SExpr] = list(args) if args is not None else []
        self.params: list[int] = list(params) if params is not None else []
        self.name: Optional[str] = name
        self.shape: Optional[ShapeLike] = None if shape is None else ShapeLike(shape)
        # Check rank matches name (skip scalar)
        if rank is not None and name is not None and not op.scalar:
            assert self.get_rank_from_name(name) == rank, f"{name=}, {rank=}"
        self.rank: Optional[int] = rank

        # `sexpr_id` is only used in SGraph and assigned by SGraph
        self.sexpr_id = next(SExpr.SEXPR_ID_GEN)

        if sym_expr is not None:
            assert (
                self.op.scalar
            ), f"Only tgops.scalar can have sym expr, got {self.op=}, {sym_expr=}"
        self.sym_expr: sympy.Expr = sym_expr
        self.dist_id = dist_id
        # if self.sexpr_id == 1075:
        #     raise RuntimeError(f"{[repr(a) for a in args]}")

        # `log_name` is the only field that can be changed without cloning.
        self.log_name = None

        self.placeholderized_sexpr: SExpr = None

    def get_unranked_name(self) -> str:
        name = self.name
        assert name is not None, f"got {self.op}: {self}"
        assert name.startswith(("Sn__r", "Dn__r")), f"got {self.op}: {self}"
        return re.sub(r"[S|D]n__r\d+__", "", name)

    def set_log_name(self, log_name: str) -> "SExpr":
        self.log_name = log_name
        return self

    def clone_with(
        self, op=None, name: str = None, args: list["SExpr"] = None
    ) -> "SExpr":
        return SExpr(
            op or self.op,
            args or copy.copy(self.args),
            copy.copy(self.params),
            name or self.name,
            copy.copy(self.shape),
            self.rank,
            self.sym_expr,
            self.dist_id,
        )

    def is_leaf(self):
        return len(self.args) == 0

    def get_placeholderized(self, keep_constant=True) -> "SExpr":
        assert (
            self.name is not None
        ), "The SExpr must have a name to be placeholderized."
        assert (
            self.shape is not None
        ), "The SExpr must have a shape to be placeholderized."
        if self.op.constant and keep_constant or self.op == tgops.inpt:
            # For ops having semantics like zeros, ones, arange, no need to placeholderize
            return self
        elif self.placeholderized_sexpr is not None:
            return self.placeholderized_sexpr
        else:
            placeholderized = SExpr(
                tgops.inpt,
                args=[],
                params=[],
                name=self.name,
                shape=self.shape,
                rank=self.rank,
            )
            self.placeholderized_sexpr = placeholderized
            return placeholderized

    def get_shape_scalars(self) -> set["SExpr"]:
        """
        Return leaf scalars from self.shape.
        """
        results: set[SExpr] = set()
        for s in self.shape:
            if isinstance(s, SExpr) and s.op.scalar:
                results.update(s.get_leaf_scalar_sexprs())
        return results

    def get_leaf_scalar_sexprs(self) -> set["SExpr"]:
        """
        Return leaf scalars from self.args.
        """
        assert self.op.scalar
        if len(self.args) == 0:
            if self.name is not None:
                return [self]
        results: set[SExpr] = set()
        for arg in self.args:
            assert arg.op.scalar
            results.update(arg.get_leaf_scalar_sexprs())
        return results

    def post_order_dfs(
        self,
        visit_sexpr_param=False,
        return_is_leaf=False,
        terminate_callback: Callable[["SExpr"], bool] = None,
    ) -> Iterable["SExpr"] | Iterable[Union["SExpr", bool]]:
        """
        terminate_callback: A callback function that decide if we want to terminate
            dfs from the current node. When this is provided, this method will return
            a tuple of (node, [is_leaf, ], is_term) instead of just node.
        """

        def create_ret(node, is_leaf, is_term):
            ret = [node]
            if return_is_leaf:
                ret.append(is_leaf)
            if terminate_callback is not None:
                ret.append(is_term)
            if len(ret) == 1:
                return ret[0]
            else:
                return tuple(ret)

        visited = set()
        # The `stack` stores (SExpr to visit, child_idx). Here, `child_idx`
        # indicates which child this sexpr is regarding to its parent.
        stack = [(self, 0)]
        while len(stack) > 0:
            obj = stack[-1]
            node = obj[0]
            child_idx = obj[1]
            assert type(node) is SExpr, f"{node=}"
            is_leaf = node.is_leaf()
            is_term = False if terminate_callback is None else terminate_callback(node)
            if is_leaf or is_term:
                visited.add(node)
                yield create_ret(node, is_leaf, is_term)
                stack.pop()
            else:
                end_child_idx = (
                    len(node.args)
                    if not visit_sexpr_param
                    else len(node.args) + len(node.params)
                )
                if child_idx < len(node.args):
                    child = node.args[child_idx]
                else:
                    while child_idx < end_child_idx:
                        child = node.params[child_idx - len(node.args)]
                        if type(child) is SExpr:
                            break
                        child_idx += 1
                if child_idx == end_child_idx:
                    # We have visited all children of `node`
                    visited.add(node)
                    yield create_ret(node, False, is_term)
                    stack.pop()
                    continue
                else:
                    stack[-1] = (node, child_idx + 1)
                    if child not in visited:
                        # Only put the child that hasn't been put into the result RecExpr list.
                        stack.append((child, 0))

    @staticmethod
    def any_to_egg_str(obj, recursive: bool = True) -> str:
        # This is for some sexpr appears in params...
        # FIXME: This should be unified.
        if type(obj) is SExpr:
            return obj.to_egg_str()
        else:
            return str(obj)

    def to_egg_str_as_inpt(self) -> str:
        # return f"(input {self.name}@{ShapeLike(self.shape)})"
        return f"(input {self.name}@)"  # NOTE: shape should be dumped in another file.

    def to_egg_str(self, recursive: bool = True) -> str:
        """
        Different from `__str__`, this is a special version for dumping to Rust Egg.
        """
        op_name = self.op.name
        op_name = op_name.lower()
        if self.op in (tgops.inpt, tgops.weight):
            # return f"({op_name} {self.name}@{self.shape})"
            return f"({op_name} {self.name}@)"  # NOTE: shape should be dumped in another file.
        elif self.op == tgops.scalar:
            assert self.sym_expr is not None, "Scalar must have sym_expr."
            if isinstance(self.sym_expr, sympy.Integer):
                return str(int(self.sym_expr))
            else:
                assert self.name is not None
                return self.to_smtlib_str()
                # return self.name
        elif self.op == tgops.boolean:
            return self.name
        else:
            if recursive:
                if self.op in (
                    tgops.reshape,
                    tgops.repeat,
                    tgops.expand,
                    tgops.transpose,
                ):
                    assert len(self.params) == 1, f"{self}"
                    assert type(self.params[0]) is ShapeLike, f"{self}"
                    return f"({op_name} {self.args[0].to_egg_str()} {self.params[0]})"
                else:
                    params_str = "".join(
                        [f" {self.any_to_egg_str(param)}" for param in self.params]
                    )
                    args_str = "".join([f" {arg.to_egg_str()}" for arg in self.args])
                    return f"({op_name}{args_str}{params_str})"
            else:
                if self.op in (
                    tgops.reshape,
                    tgops.repeat,
                    tgops.expand,
                    tgops.transpose,
                ):
                    assert len(self.params) == 1, f"{self}"
                    assert type(self.params[0]) is ShapeLike, f"{self}"
                    return f"({op_name} {self.args[0].to_egg_str_as_inpt()} {self.params[0]})"
                else:
                    params_str = "".join(
                        [
                            f" {self.any_to_egg_str(param, False)}"
                            for param in self.params
                        ]
                    )
                    args_str = "".join(
                        [f" {arg.to_egg_str_as_inpt()}" for arg in self.args]
                    )
                    return f"({op_name}{args_str}{params_str})"

    def to_smtlib_str(self) -> str:
        assert self.op.scalar, f"Can only convert scalar, got {self.op=}"
        assert len(self.params) == 0
        if len(self.args) == 0:
            assert self.sym_expr is not None
            return str(self.sym_expr)
        else:
            args_str = "".join([f" {arg.to_smtlib_str()}" for arg in self.args])
            return f"({self.op.name}{args_str})"

    def pretty(self, indent=4):
        op_name = self.op.name
        op_name = op_name.lower()
        if self.op in (tgops.inpt, tgops.weight):
            return f"({op_name} {self.name}@{self.shape})"
        else:
            params_str = " ".join([f"{param}" for param in self.params])
            args_str = "\n".join([arg.pretty() for arg in self.args])
            return f"({op_name}\n{' '*indent}{params_str}\n{indent_str(args_str, indent)}\n)"

    def __str__(self):
        op_name = self.op.name
        op_name = op_name.lower()
        if self.op in (tgops.inpt, tgops.weight):
            return f"({op_name} {self.name}@{self.shape})"
        elif self.op == tgops.scalar:
            if self.name:
                return self.name
            else:
                assert self.sym_expr is not None, "Scalar must have sym_expr."
                return str(self.sym_expr)
        else:
            params_str = "".join([f" {param}" for param in self.params])
            args_str = "".join([f" {str(arg)}" for arg in self.args])
            return f"({op_name}{params_str}{args_str})"

    def __repr__(self):
        op_name = self.op.name
        op_name = op_name.lower()
        ids = (
            f"{self.sexpr_id}"
            if self.dist_id is None
            else f"{self.sexpr_id}|{self.dist_id}"
        )
        if self.name is not None:
            return f"({op_name} {self.name} {ids})"
        else:
            return f"({op_name} {ids})"

    def __add__(self, other: "SExpr"):
        assert self.op.scalar, f"Only scalar can be added, got {self.op=}"
        assert other.op.scalar, f"Only scalar can be added, got {other.op=}"
        new_sym_expr = self.sym_expr + other.sym_expr
        new_sexpr, new_sym_expr = sym.SymManager.convert_sympy_expr(new_sym_expr)
        return new_sexpr


def get_placeholder_maker(get_sexpr: Callable[[str], SExpr]):
    def placeholder(name: str, shape: list[int] = None) -> SExpr:
        sexpr = get_sexpr(name)
        shape = shape or sexpr.shape
        rank = SExpr.get_rank_from_name(name)
        assert rank is not None, f"Cannot extract rank from {name=}"
        return SExpr(tgops.inpt, args=[], params=[], name=name, shape=shape, rank=rank)

    return placeholder


def get_scalar_maker(get_sexpr: Callable[[str], SExpr]):
    def scalar(name: str) -> SExpr:
        if not name.startswith(SExpr.SCALAR_PREFIX):
            name = SExpr.SCALAR_PREFIX + name
        sexpr = get_sexpr(name)
        assert sexpr.op == tgops.scalar
        return sexpr

    return scalar


def concat2(sexpr1, sexpr2, dim: int, name: str = None) -> SExpr:
    assert len(sexpr1.shape) == len(sexpr2.shape), "Concatenation requires same n_dim."
    assert (
        sexpr1.shape[:dim] == sexpr2.shape[:dim]
        and sexpr1.shape[dim + 1 :] == sexpr2.shape[dim + 1 :]
    ), f"Concatenation requires same shape except dim, got {sexpr1.shape} and {sexpr2.shape} for dim={dim}."
    new_shape = (
        copy.copy(sexpr1.shape[:dim])
        + [sexpr1.shape[dim] + sexpr2.shape[dim]]
        + copy.copy(sexpr1.shape[dim + 1 :])
    )
    rank = sexpr1.rank if sexpr1.rank == sexpr2.rank else None
    return SExpr(
        tgops.concat,
        args=[sexpr1, sexpr2],
        params=[dim],
        name=name,
        shape=new_shape,
        rank=rank,
    )


def concat(sexprs: list[SExpr], dim: int, name: str = None) -> SExpr:
    if len(sexprs) == 1:
        return sexprs[0]
    if dim < 0:
        dim += len(sexprs[0].shape)
    assert len(sexprs) >= 2, "Concat takes at least 2 inputs."
    res = sexprs[0]
    for i, sexpr in enumerate(sexprs[1:]):
        if name is not None:
            used_name = name + f"_with_{i}"
        else:
            used_name = None
        res = concat2(res, sexpr, dim, name=used_name)
    return res


def reshape(sexpr: SExpr, new_shape: ShapeLike) -> SExpr:
    assert prod(sexpr.shape) == prod(
        new_shape
    ), "Reshape requires same number of elements."
    return SExpr(
        tgops.reshape,
        args=[sexpr],
        params=[new_shape],
        shape=new_shape,
        rank=sexpr.rank,
    )


def slice(sexpr: SExpr, dim: int, begin: int, end: int, step: int = 1) -> SExpr:
    if dim < 0:
        dim += len(sexpr.shape)
    if begin < 0:
        begin += sexpr.shape[dim]
    if end < 0:
        end += sexpr.shape[dim]
    assert (
        0 <= begin <= end <= sexpr.shape[dim]
    ), f"Invalid slice params: {dim=}, {begin=}, {end=}, {step=}, {sexpr.shape=}"
    new_shape = sexpr.shape.copy()
    new_shape[dim] = (end - begin) // step
    return SExpr(
        tgops.slice,
        args=[sexpr],
        params=[dim, begin, end, step],
        shape=new_shape,
        rank=sexpr.rank,
    )


def matadd(sexpr1: SExpr, sexpr2: SExpr) -> SExpr:
    assert (
        sexpr1.shape == sexpr2.shape
    ), f"Matadd requires same shape, got {sexpr1.shape} and {sexpr2.shape}."
    rank = sexpr1.rank if sexpr1.rank == sexpr2.rank else None
    return SExpr(tgops.matadd, args=[sexpr1, sexpr2], shape=sexpr1.shape, rank=rank)


def sum(sexprs: list[SExpr]) -> SExpr:
    assert len(sexprs) >= 2, "Sum takes at least 2 inputs."
    res = matadd(sexprs[0], sexprs[1])
    for sexpr in sexprs[2:]:
        res = matadd(res, sexpr)
    return res


def repeat(sexpr: SExpr, repeats: ShapeLike) -> SExpr:
    assert len(sexpr.shape) == len(repeats), "Repeat requires same n_dim."
    new_shape = copy.copy(sexpr.shape)
    new_shape = [new_shape[i] * repeats[i] for i in range(len(new_shape))]
    return SExpr(
        tgops.repeat, args=[sexpr], params=[repeats], shape=new_shape, rank=sexpr.rank
    )


def transpose(sexpr: SExpr, perm: ShapeLike = None) -> SExpr:
    if perm is None:
        perm = ShapeLike([1, 0])
    assert len(sexpr.shape) == len(perm), "Transpose requires same n_dim."
    new_shape = [sexpr.shape[i] for i in perm]

    name = sexpr.name + f"_permute_{'_'.join(map(str, perm))}"
    return SExpr(
        tgops.transpose,
        args=[sexpr],
        params=[perm],
        name=name,
        shape=new_shape,
        rank=sexpr.rank,
    )


def matmul(sexpr1: SExpr, sexpr2: SExpr) -> SExpr:
    assert (
        len(sexpr1.shape) == 2 and len(sexpr2.shape) == 2
    ), "Matmul requires 2D inputs."
    assert (
        sexpr1.shape[1] == sexpr2.shape[0]
    ), "Matmul requires shape[1] of input1 to be equal to shape[0] of input2."
    rank = sexpr1.rank if sexpr1.rank == sexpr2.rank else None
    return SExpr(
        tgops.matmul,
        args=[sexpr1, sexpr2],
        shape=[sexpr1.shape[0], sexpr2.shape[1]],
        rank=rank,
    )


def make_fill(shape: Union[list[int], ShapeLike], value: Union[int, float]) -> SExpr:
    return SExpr(tgops.fill, args=[], params=[shape, value], shape=shape)


ZERO = SExpr(tgops.scalar, sym_expr=sympy.Integer(0), shape=ShapeLike())
TRUE = SExpr(tgops.boolean, name="true", shape=ShapeLike())
FALSE = SExpr(tgops.boolean, name="false", shape=ShapeLike())


def const_scalar(value: int) -> SExpr:
    return SExpr(tgops.scalar, sym_expr=sympy.Integer(value), shape=ShapeLike())


def gt(x: SExpr, y: SExpr) -> SExpr:
    assert x.op.scalar and y.op.scalar, "Only scalar use this."
    rank = x.rank or y.rank
    return SExpr(tgops.s_gt, args=[x, y], shape=[], rank=rank)


def ge(x: SExpr, y: SExpr) -> SExpr:
    assert x.op.scalar and y.op.scalar, "Only scalar use this."
    rank = x.rank or y.rank
    return SExpr(tgops.s_ge, args=[x, y], shape=[], rank=rank)


def lt(x: SExpr, y: SExpr) -> SExpr:
    assert x.op.scalar and y.op.scalar, "Only scalar use this."
    rank = x.rank or y.rank
    return SExpr(tgops.s_lt, args=[x, y], shape=[], rank=rank)


def le(x: SExpr, y: SExpr) -> SExpr:
    assert x.op.scalar and y.op.scalar, "Only scalar use this."
    rank = x.rank or y.rank
    return SExpr(tgops.s_le, args=[x, y], shape=[], rank=rank)


def eq(sexpr1: SExpr, sexpr2: SExpr) -> SExpr:
    assert sexpr1.op.scalar and sexpr2.op.scalar, "Eq requires scalar inputs."
    return SExpr(tgops.s_eq, args=[sexpr1, sexpr2], shape=[], rank=None)
