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

import numpy as np
from entangle.convert.convert import *
from entangle.convert.mappings.vllm._custom_op import *
from entangle.sgraph import sexpr
from entangle.sgraph.egraph import SExprECondition
from entangle.sgraph.sskeleton import *
from entangle.tools.config import ExplorativeConfig


class MyConfig(ExplorativeConfig):
    def __init__(self, args: Sequence[str]):
        super().__init__(args)
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_layers", type=int, default=1, choices=[1, 2, 4, 8, 12, 16])
        parser.add_argument("--tp", type=int, default=2, choices=[2, 4, 8])
        args = parser.parse_args(args)
        self.num_layers = args.num_layers
        self.tp = args.tp
        self.world_size = self.tp

    def build_preconditions(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(self.get_sexpr)
        num_layers = self.num_layers
        world_size = self.world_size
        # fmt: off
        q_range = [i//world_size for i in [0, 2048]]
        k_range = [i//world_size for i in [2048, 2560]]
        v_range = [i//world_size for i in [2560, 3072]]
        base = 6 + num_layers * 2
        bc_base1 = 77 + num_layers*9
        bc_base2 = 84 + num_layers*9 
        bc_s_gap = 102
        bc_t_gap = 114
        layers = [
            [
                # rms_norm weight
                SExprECondition.all_eq(f"Sn__r0__p{base+7*l}.{base+1+7*l}", [f"Dn__r{i}__p{base+7*l}.{base+1+7*l}" for i in range(world_size)], placeholder),
                # qkv weight
                SExprECondition(
                    inputs=[
                        s:=placeholder(f"Sn__r0__p{base+1+7*l}.{base+2+7*l}"), 
                        *(d:=[placeholder(f"Dn__r{i}__p{base+1+7*l}.{base+2+7*l}") for i in range(world_size)])],
                    eclasses=[
                        [
                            s, 
                            sexpr.concat(
                                [
                                    sexpr.concat([sexpr.slice(di, 1, *q_range) for di in d], dim=1),
                                    sexpr.concat([sexpr.slice(di, 1, *k_range) for di in d], dim=1),
                                    sexpr.concat([sexpr.slice(di, 1, *v_range) for di in d], dim=1),
                                ], 
                                dim=1,
                            ),
                        ]
                    ]
                ),
                # A broadcasted constant
                SExprECondition.concat_target(f"Sn__r0__broadcast.{bc_base1+bc_s_gap*l}", [f"Dn__r{i}__broadcast.{bc_base1+bc_t_gap*l}" for i in range(world_size)], 2, placeholder),
                # A broadcasted constant: -30000
                SExprECondition.concat_target(f"Sn__r0__broadcast.{bc_base2+bc_s_gap*l}", [f"Dn__r{i}__broadcast.{bc_base2+bc_t_gap*l}" for i in range(world_size)], 1, placeholder),
                # Output projection weight at src/transformers_neuronx/layers/attention.py:886
                SExprECondition.concat_target(f"Sn__r0__p{base+2+7*l}.{base+3+7*l}", [f"Dn__r{i}__p{base+2+7*l}.{base+3+7*l}" for i in range(world_size)], 1, placeholder),
                # rms_norm weight
                SExprECondition.all_eq(f"Sn__r0__p{base+3+7*l}.{base+4+7*l}", [f"Dn__r{i}__p{base+3+7*l}.{base+4+7*l}" for i in range(world_size)], placeholder),
                # gated mlp weight 1: src/transformers_neuronx/llama/hlo.py:312
                SExprECondition.concat_target(f"Sn__r0__p{base+4+7*l}.{base+5+7*l}", [f"Dn__r{i}__p{base+4+7*l}.{base+5+7*l}" for i in range(world_size)], 1, placeholder),
                # gated mlp weight 2: src/transformers_neuronx/llama/hlo.py:312
                SExprECondition.concat_target(f"Sn__r0__p{base+5+7*l}.{base+6+7*l}", [f"Dn__r{i}__p{base+5+7*l}.{base+6+7*l}" for i in range(world_size)], 1, placeholder),
                # gated mlp weight 3: src/transformers_neuronx/llama/hlo.py:312
                SExprECondition.concat_target(f"Sn__r0__p{base+6+7*l}.{base+7+7*l}", [f"Dn__r{i}__p{base+6+7*l}.{base+7+7*l}" for i in range(world_size)], 1, placeholder),
                # rms_norm weight
                SExprECondition.all_eq(f"Sn__r0__p{base+7+7*l}.{base+8+7*l}", [f"Dn__r{i}__p{base+7+7*l}.{base+8+7*l}" for i in range(world_size)], placeholder),
                # Weight for computing logits
                SExprECondition.concat_target(f"Sn__r0__p{base+8+7*l}.{base+9+7*l}", [f"Dn__r{i}__p{base+8+7*l}.{base+9+7*l}" for i in range(world_size)], 1, placeholder),
            ]
            for l in range(num_layers)
        ]

        return [
            # Hidden
            SExprECondition.all_eq(f"Sn__r0__p0.1", [f"Dn__r{i}__p0.1" for i in range(world_size)], placeholder),
            # inv_freq
            SExprECondition.all_eq(f"Sn__r0__constant.{13+9*num_layers}", [f"Dn__r{i}__constant.{13+9*num_layers}" for i in range(world_size)], placeholder),
            # cache id? src/transformers_neuronx/layers/transformer.py:86
            SExprECondition.all_eq(f"Sn__r0__p1.2", [f"Dn__r{i}__p1.2" for i in range(world_size)], placeholder),
            # `start_ids` at src/transformers_neuronx/layers/transformer.py:97
            SExprECondition.all_eq(f"Sn__r0__p2.3", [f"Dn__r{i}__p2.3" for i in range(world_size)], placeholder),
            SExprECondition.just_map(f"Sn__r0__iota.{20+9*num_layers}", [f"Dn__r{i}__iota.{20+9*num_layers}" for i in range(world_size)], placeholder),
            # index select when done: src/transformers_neuronx/layers/transformer.py:331
            SExprECondition.all_eq(f"Sn__r0__p3.4", [f"Dn__r{i}__p3.4" for i in range(world_size)], placeholder),
        ] + list(chain(*layers))
        # fmt: on

    def oracle_group_id_to_size(self, group_id: str | int) -> int:
        return self.tp

    def build_expected(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(self.get_sexpr)
        world_size = self.world_size
        num_layers = self.num_layers
        return [
            # fmt: off
            SExprECondition(
                inputs=[
                    s:=placeholder(f"Sn__r0__reshape.{38+111*num_layers}"), 
                    *(d := [placeholder(f"Dn__r{i}__reshape.{38+123*num_layers}") for i in range(world_size)])
                ], 
                eclasses=[
                    [s, sexpr.concat(d, dim=0).clone_with(name="EXPECTED_Sn__r0__reshape.149")],
                ],
            ),
            # fmt: on
        ]
