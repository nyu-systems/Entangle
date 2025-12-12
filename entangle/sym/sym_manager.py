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

from typing import Optional

import rich
import sympy

import entangle.ops as tgops
from entangle.sgraph.sexpr import SExpr

SYMPY_TO_OP = {
    sympy.Add: tgops.s_add,
    sympy.Mul: tgops.s_mul,
}


def get_id_generator(begin=1000):
    i = begin
    while True:
        yield i
        i += 1


class SymManager:
    SYM_ID_GEN = get_id_generator(begin=1000)

    # The `NAME` is a name for Symbol.
    INSTANTIATION_NAME_TO_VAL: dict[str, int] = {}

    @staticmethod
    def setup_sym_instantiation(name_to_val: dict[str, int]):
        SymManager.INSTANTIATION_NAME_TO_VAL = name_to_val

    @staticmethod
    def repr_scalar_name(rank: int, name: str):
        return f"R{rank}{name}"

    @staticmethod
    def rank_expr(expr: sympy.Expr, rank: int) -> sympy.Expr:
        if type(expr) is sympy.Symbol:
            return sympy.Symbol(SymManager.repr_scalar_name(rank, expr.name))
        ranked_args = []
        for arg in expr.args:
            ranked_args.append(SymManager.rank_expr(arg, rank))
        return type(expr)(*ranked_args)

    def get_next_name():
        return f"u{next(SymManager.SYM_ID_GEN)}"

    @staticmethod
    def make_scalar_name(name, name_prefix, check_ready=False):
        if (half := name_prefix + name).startswith("Sym") and check_ready:
            return half
        else:
            return SExpr.SCALAR_PREFIX + half

    @staticmethod
    def is_leaf_sym_expr(expr: sympy.Expr) -> bool:
        return type(expr) == sympy.Symbol or isinstance(expr, sympy.Number)

    @staticmethod
    def convert_sympy_expr(
        expr: sympy.Expr, name_prefix: Optional[str] = ""
    ) -> tuple[SExpr, sympy.Expr]:
        assert isinstance(expr, sympy.Expr), f"{expr} is not a sympy.Expr."
        if isinstance(expr, sympy.Integer):
            sexpr = SExpr(tgops.scalar, [], [], name=None, shape=[], sym_expr=expr)
            return sexpr, expr
        if type(expr) is sympy.Symbol:
            name = SymManager.make_scalar_name(
                repr(expr), name_prefix, check_ready=True
            )
            if name in SymManager.INSTANTIATION_NAME_TO_VAL:
                # If provided, use the concrete value.
                new_sym_expr = sympy.Integer(SymManager.INSTANTIATION_NAME_TO_VAL[name])
                new_sexpr = SExpr(
                    tgops.scalar, [], [], name=None, shape=[], sym_expr=new_sym_expr
                )
            else:
                new_sym_expr = sympy.Symbol(name)
                new_sexpr = SExpr(
                    tgops.scalar, [], [], name=name, shape=[], sym_expr=new_sym_expr
                )
        else:
            assert len(expr.args) >= 2, f"{expr=}"
            new_sym_expr = None
            new_sexpr = None
            for arg in expr.args:
                arg_sexpr, arg_sym_expr = SymManager.convert_sympy_expr(
                    arg, name_prefix
                )
                if new_sym_expr is None:
                    assert new_sexpr is None
                    new_sym_expr = arg_sym_expr
                    new_sexpr = arg_sexpr
                else:
                    new_sym_expr = sympy.simplify(
                        type(expr)(new_sym_expr, arg_sym_expr)
                    )
                    if isinstance(new_sym_expr, sympy.Integer):
                        new_sexpr = SExpr(tgops.scalar, shape=[], sym_expr=new_sym_expr)
                    else:
                        new_sexpr = SExpr(
                            SYMPY_TO_OP[type(expr)],
                            [new_sexpr, arg_sexpr],
                            [],
                            name=None,
                            shape=[],
                            sym_expr=new_sym_expr,
                        )

            if type(expr) not in SYMPY_TO_OP:
                raise NotImplementedError(
                    f"Support for {type(expr)} is not implemented."
                )
        return new_sexpr, new_sym_expr
