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

import os
import os.path as osp
import pickle

from entangle.custom_op import (
    LogTensor,
    inplace_log_grad,
    inplace_log_tensor,
    log_tensor,
    slice,
    slice_scatter,
)
from entangle.graph import get_group_rank, DYNAMO_TRACING, HACK_FOR_DYNAMO, USING_DYNAMO


def get_global_states():
    import entangle.sgraph.sexpr

    return {"SExpr.SEXPR_ID_GEN": next(entangle.sgraph.sexpr.SExpr.SEXPR_ID_GEN)}


def resume_global_states(global_states: dict):
    import entangle.sgraph.sexpr
    import entangle.sym.sym_manager
    import entangle.utils

    entangle.sgraph.sexpr.SExpr.SEXPR_ID_GEN = entangle.utils.get_id_generator(
        global_states["SExpr.SEXPR_ID_GEN"]
    )


def save_global_states(path: str):
    import entangle.sgraph.sexpr
    import entangle.sym.sym_manager

    objs = {
        "entangle.sgraph.SExpr.next_SEXPR_ID_GEN": next(
            entangle.sgraph.sexpr.SExpr.SEXPR_ID_GEN
        ),
        "next(entangle.sym.sym_manager.SymManager.SYM_ID_GEN)": next(
            entangle.sym.sym_manager.SymManager.SYM_ID_GEN
        ),
        "entangle.sym.sym_manager.SymManager.INSTANTIATION_NAME_TO_VAL": entangle.sym.sym_manager.SymManager.INSTANTIATION_NAME_TO_VAL,
    }
    with open(path, "wb") as f:
        pickle.dump(objs, f)


def load_and_resume_global_states(path: str):
    import entangle.sgraph.sexpr
    import entangle.sym.sym_manager
    import entangle.utils

    with open(path, "rb") as f:
        objs = pickle.load(f)
    entangle.sgraph.sexpr.SExpr.SEXPR_ID_GEN = entangle.utils.get_id_generator(
        objs["entangle.sgraph.SExpr.next_SEXPR_ID_GEN"]
    )
    entangle.sym.sym_manager.SymManager.SYM_ID_GEN = (
        entangle.utils.get_id_generator(
            objs["next(entangle.sym.sym_manager.SymManager.SYM_ID_GEN)"]
        )
    )
    entangle.sym.sym_manager.SymManager.INSTANTIATION_NAME_TO_VAL = objs[
        "entangle.sym.sym_manager.SymManager.INSTANTIATION_NAME_TO_VAL"
    ]


def symabspath(path: str) -> str:
    if osp.isabs(path):
        return path
    else:
        pwd = os.getcwd()
        return osp.join(pwd, path)


osp.symabspath = symabspath
