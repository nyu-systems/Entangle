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
from copy import copy
from typing import Callable, Sequence

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
        parser.add_argument("--num_layers", type=int, required=True)
        parser.add_argument("--tp", type=int, required=True)
        args = parser.parse_args(args)
        self.num_layers = args.num_layers
        self.tp = args.tp
        self.world_size = args.tp

    def build_preconditions(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(self.get_sexpr)
        num_layers = self.num_layers
        world_size = self.world_size

        # fmt: off
        n_per_layer = 12
        transformer_layers = [
            [
                SExprECondition.concat_target(f"Sn__r0__primals_{4+layer*n_per_layer}", [f"Dn__r{i}__primals_{4+layer*n_per_layer}" for i in range(world_size)], 0, placeholder),
                SExprECondition.concat_target(f"Sn__r0__primals_{5+layer*n_per_layer}", [f"Dn__r{i}__primals_{5+layer*n_per_layer}" for i in range(world_size)], 0, placeholder),
                SExprECondition.concat_target(f"Sn__r0__primals_{6+layer*n_per_layer}", [f"Dn__r{i}__primals_{6+layer*n_per_layer}" for i in range(world_size)], 1, placeholder),
                SExprECondition.all_eq(f"Sn__r0__primals_{7+layer*n_per_layer}", [f"Dn__r{i}__primals_{7+layer*n_per_layer}" for i in range(world_size)], placeholder),
                SExprECondition.all_eq(f"Sn__r0__primals_{8+layer*n_per_layer}", [f"Dn__r{i}__primals_{8+layer*n_per_layer}" for i in range(world_size)], placeholder),
                SExprECondition.all_eq(f"Sn__r0__primals_{9+layer*n_per_layer}", [f"Dn__r{i}__primals_{9+layer*n_per_layer}" for i in range(world_size)], placeholder),
                SExprECondition.concat_target(f"Sn__r0__primals_{10+layer*n_per_layer}", [f"Dn__r{i}__primals_{10+layer*n_per_layer}" for i in range(world_size)], 0, placeholder),
                SExprECondition.concat_target(f"Sn__r0__primals_{11+layer*n_per_layer}", [f"Dn__r{i}__primals_{11+layer*n_per_layer}" for i in range(world_size)], 0, placeholder),
                SExprECondition.concat_target(f"Sn__r0__primals_{12+layer*n_per_layer}", [f"Dn__r{i}__primals_{12+layer*n_per_layer}" for i in range(world_size)], 1, placeholder),
                SExprECondition.all_eq(f"Sn__r0__primals_{13+layer*n_per_layer}", [f"Dn__r{i}__primals_{13+layer*n_per_layer}" for i in range(world_size)], placeholder),
                SExprECondition.all_eq(f"Sn__r0__primals_{14+layer*n_per_layer}", [f"Dn__r{i}__primals_{14+layer*n_per_layer}" for i in range(world_size)], placeholder),
                SExprECondition.all_eq(f"Sn__r0__primals_{15+layer*n_per_layer}", [f"Dn__r{i}__primals_{15+layer*n_per_layer}" for i in range(world_size)], placeholder),
                (SExprECondition.concat_target(f"Sn__r0__empty", [f"Dn__r{i}__empty" for i in range(world_size)], 0, placeholder)
                 if layer == 0 else SExprECondition.concat_target(f"Sn__r0__empty_{layer}", [f"Dn__r{i}__empty_{layer*3}" for i in range(world_size)], 0, placeholder) ),
            ]
            for layer in range(num_layers)
        ]
        return list(chain(*transformer_layers)) + [
            # This is for vocab embedding.
            SExprECondition.concat_target(f"Sn__r0__primals_1", [f"Dn__r{i}__primals_1" for i in range(world_size)], 0, placeholder),
            # This is initial input vocab ids.
            SExprECondition.all_eq(f"Sn__r0__primals_{6+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{6+num_layers*n_per_layer}" for i in range(world_size)], placeholder),
            # These are for position embedding.
            SExprECondition.all_eq(f"Sn__r0__primals_{5+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{5+num_layers*n_per_layer}" for i in range(world_size)], placeholder),
            SExprECondition.all_eq(f"Sn__r0__primals_{8+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{8+num_layers*n_per_layer}" for i in range(world_size)], placeholder),
            # This is ground truth labels.
            SExprECondition.all_eq(f"Sn__r0__primals_{7+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{7+num_layers*n_per_layer}" for i in range(world_size)], placeholder),
            # These are for label for cross-entropy.
            SExprECondition.all_eq(f"Sn__r0__primals_{10+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{10+num_layers*n_per_layer}" for i in range(world_size)], placeholder),
            SExprECondition.all_eq(f"Sn__r0__primals_{9+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{9+num_layers*n_per_layer}" for i in range(world_size)], placeholder),

            SExprECondition.all_eq(f"Sn__r0__primals_2", [f"Dn__r{i}__primals_2" for i in range(world_size)], placeholder),
            SExprECondition.all_eq(f"Sn__r0__primals_3", [f"Dn__r{i}__primals_3" for i in range(world_size)], placeholder),
            SExprECondition.concat_target(f"Sn__r0__primals_{4+num_layers*n_per_layer}", [f"Dn__r{i}__primals_{4+num_layers*n_per_layer}" for i in range(world_size)], 0, placeholder)
        ]
        # fmt: on

    def oracle_group_id_to_size(self, group_id: str | int) -> int:
        return self.tp

    def build_expected(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(self.get_sexpr)
        world_size = self.world_size
        return [
            # fmt: off
            SExprECondition(
                inputs=[
                    s := placeholder("Sn__r0__div_2"),
                    *(d := [placeholder(f"Dn__r{i}__div_2") for i in range(world_size)]),
                ],
                eclasses=[[s, *d]],
            )
            # fmt: on
        ]

    def get_lift_fresh_copy_constant_value(self, graph_path: str, s: str) -> int:
        assert s.find("lift_fresh_copy") != -1, f"got {s}"
        return 0
