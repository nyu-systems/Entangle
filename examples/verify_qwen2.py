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

import argparse
from typing import Callable, Sequence

from entangle.convert.convert import *
from entangle.convert.mappings.vllm._custom_op import *
from entangle.sgraph import sexpr
from entangle.sgraph.egraph import SExprECondition
from entangle.sgraph.sskeleton import *
from entangle.tools.config import ExplorativeConfig


def make_qkv_precondition(s, d, q_nh, kv_nh, head_size, world_size, dim=0):
    q_nh_per_rank = q_nh // world_size
    kv_nh_per_rank = max(1, kv_nh // world_size)
    q_size_per_rank = head_size * q_nh_per_rank
    kv_size_per_rank = head_size * kv_nh_per_rank
    assert world_size % kv_nh == 0, f"{world_size=} must be divisible by {kv_nh=}"
    gap = world_size // kv_nh
    query_group_size = q_nh // kv_nh
    q = sexpr.concat([sexpr.slice(di, dim, 0, q_size_per_rank) for di in d], dim=dim)
    ks = [sexpr.slice(di, dim, q_size_per_rank, q_size_per_rank + kv_size_per_rank) for di in d]
    vs = [
        sexpr.slice(
            di,
            dim,
            q_size_per_rank + kv_size_per_rank,
            q_size_per_rank + 2 * kv_size_per_rank,
        )
        for di in d
    ]
    # fmt: off
    # print(world_size, kv_nh, gap, [str(s) for s in ks])
    # print([[str(s) for s in ks[i * gap : (1 + i) * gap]] for i in range(len(ks) // gap)])
    # print([[str(s) for s in vs[i * gap : (i + 1) * gap]] for i in range(len(vs) // gap)])
    # fmt: on
    k = sexpr.concat([ki for ki in ks[::gap]], dim=dim)
    v = sexpr.concat([vi for vi in vs[::gap]], dim=dim)
    return SExprECondition(
        inputs=[s, *d],
        eclasses=[
            [s, sexpr.concat([q, k, v], dim=dim)],
            *[ks[i * gap : (i + 1) * gap] for i in range(len(ks) // gap)],
            *[vs[i * gap : (i + 1) * gap] for i in range(len(vs) // gap)],
        ],
    )


class MyConfig(ExplorativeConfig):
    def __init__(self, args: Sequence[str]):
        super().__init__(args)
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_layers", type=int, required=True)
        parser.add_argument("--tp", type=int, required=True)
        args = parser.parse_args(args)
        self.num_layers = args.num_layers
        self.tp = args.tp
        self.world_size = self.tp

    def build_preconditions(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(self.get_sexpr)
        world_size = self.world_size
        num_layers = self.num_layers
        head_size = 128
        q_nh = 12
        kv_nh = 2
        # fmt: off
        return [
            # Vocab Embedding
            SExprECondition.all_eq(f"Sn__r0__arg0_1", [f"Dn__r{i}__arg0_1" for i in range(world_size)], placeholder),
            SExprECondition.concat_target(f"Sn__r0__arg1_1", [f"Dn__r{i}__arg1_1" for i in range(world_size)], 0, placeholder),
            SExprECondition.all_eq(f"Sn__r0__embedding", [f"Dn__r{i}__all_reduce_wait" for i in range(world_size)], placeholder),  # HACK
            SExprECondition.all_eq(f"Sn__r0__arg2_1", [f"Dn__r{i}__arg2_1" for i in range(world_size)], placeholder),
            # QKV bias
            make_qkv_precondition(placeholder(f"Sn__r0__arg3_1"), [placeholder(f"Dn__r{i}__arg3_1") for i in range(world_size)], q_nh, kv_nh, head_size, world_size),
             # QKV weight
            make_qkv_precondition(placeholder(f"Sn__r0__arg4_1"), [placeholder(f"Dn__r{i}__arg4_1") for i in range(world_size)], q_nh, kv_nh, head_size, world_size),
            SExprECondition.all_eq(f"Sn__r0__arg5_1", [f"Dn__r{i}__arg5_1" for i in range(world_size)], placeholder),  # cos_sin_cache
            SExprECondition.all_eq(f"Sn__r0__arg6_1", [f"Dn__r{i}__arg6_1" for i in range(world_size)], placeholder),  # positions
            SExprECondition.concat_target(f"Sn__r0__arg7_1", [f"Dn__r{i}__arg7_1" for i in range(world_size)], 1, placeholder),  # Large Linear
            SExprECondition.all_eq(f"Sn__r0__arg8_1", [f"Dn__r{i}__arg8_1" for i in range(world_size)], placeholder),  # rms_norm weight
            SExprECondition(
                inputs=[
                    s:=placeholder(f"Sn__r0__arg9_1"), 
                    *(d := [placeholder(f"Dn__r{i}__arg9_1") for i in range(world_size)])
                ],
                eclasses=[
                    [
                        s, 
                        sexpr.concat(
                            [
                                sexpr.concat([sexpr.slice(di, 0, 0, di.shape[0] // 2) for di in d], dim=0), 
                                sexpr.concat([sexpr.slice(di, 0, di.shape[0] // 2, di.shape[0]) for di in d], dim=0)
                            ], 
                            dim=0)
                    ]
                ]
            ),  # ffn weight 1 and 3
            SExprECondition.concat_target(f"Sn__r0__arg10_1", [f"Dn__r{i}__arg10_1" for i in range(world_size)], 1, placeholder),  # ffn weight 2
            SExprECondition.all_eq(f"Sn__r0__arg11_1", [f"Dn__r{i}__arg11_1" for i in range(world_size)], placeholder),  # rms_norm weight
        ]
        # fmt: on

    def oracle_group_id_to_size(self, group_id: str | int) -> int:
        return self.tp

    def build_expected(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(self.get_sexpr)
        # fmt: off
        return [
            SExprECondition(
                inputs=[s := placeholder("Sn__r0__getitem_34"), d := placeholder(f"Dn__r{i}__getitem_34")],
                eclasses=[[s, d]],
            )  
            for i in range(self.world_size)
        ]
        # fmt: on
