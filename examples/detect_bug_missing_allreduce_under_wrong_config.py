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
        self.num_layers = 1
        self.tp = 2
        self.world_size = 2

    def build_preconditions(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(
            self.get_sexpr
        )
        num_layers = self.num_layers
        world_size = self.world_size
        # fmt: off
        return [
            SExprECondition.concat_target(f"Sn__r0__mul_4", [f"Dn__r{i}__mul_4" for i in range(world_size)], 2, placeholder),
            SExprECondition.concat_target(f"Sn__r0__module.module.embedding.word_embeddings.weight", [f"Dn__r{i}__module.module.embedding.word_embeddings.weight" for i in range(world_size)], 0, placeholder),
            SExprECondition.concat_target(f"Sn__r0__module.module.decoder.layers.0.mlp.linear_fc2.weight", [f"Dn__r{i}__module.module.decoder.layers.0.mlp.linear_fc2.weight" for i in range(world_size)], 1, placeholder),
        ]
        # fmt: on

    def oracle_group_id_to_size(self, group_id: str | int) -> int:
        return self.tp

    def build_expected(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(
            self.get_sexpr
        )
        world_size = self.world_size
        return []

    def get_force_leaf_set(self) -> set[str]:
        # fmt: off
        return {
            "Sn__r0__tangents_21", *{f"Dn__r{i}__tangents_21" for i in range(self.world_size)},  # loss
            "Sn__r0__div", *{f"Dn__r{i}__div" for i in range(self.world_size)},  # forward cross entropy output
            "Sn__r0__mul_4", *{f"Dn__r{i}__mul_4" for i in range(self.world_size)},  # FIXME: TESTING
        }
        # fmt: on

    def get_lift_fresh_copy_constant_value(self, graph_path: str, s: str) -> int:
        assert s.find("lift_fresh_copy") != -1, f"got {s}"
        return 0

    def filter_output(self, default_outputs: list[SExpr]) -> list:
        return [f"Sn__r0__mm_7", *[f"Dn__r{i}__mm_7" for i in range(self.world_size)]]
