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

from abc import ABC
from typing import Callable, Sequence, Type

import entangle.ops as tgops
from entangle.sgraph.egraph import SExprECondition
from entangle.sgraph.sexpr import SExpr
from entangle.sgraph.sgraph import SGraph
from entangle.sgraph.sskeleton import CutGroup, default_group_sgraph_cuts


class Config(ABC):
    def __init__(self, args: Sequence[str]):
        self.get_sexpr = None
        self.args = args

    def set_get_sexpr(self, get_sexpr: Callable[[str], SExpr]):
        self.get_sexpr = get_sexpr

    def build_preconditions(self) -> list[SExprECondition]:
        """
        This method is usually required to implement. Because all computational graphs are
        likely to have some inputs that need to be specified some preconditions.
        """
        return []

    def convert_str_cut_group(self, str_cut_group: tuple[str, list[str]]) -> CutGroup:
        """
        E.g., convert
                ("Sn__r0__getitem_317", ["Dn__r0__cat_10", "Dn__r1__cat_10"])
        into CutGroup using `self.get_sexpr` to map from name to SExpr.
        """
        convert_to_sexpr = lambda x: x if type(x) is SExpr else self.get_sexpr(x)
        origin, targets = str_cut_group
        origin_sexpr = convert_to_sexpr(origin)
        targets = [convert_to_sexpr(t) for t in targets]
        return CutGroup(origin_sexpr, targets)

    def convert_str_cut_groups(
        self, str_cut_groups: list[tuple[str, list[str]]]
    ) -> list[CutGroup]:
        """
        E.g., convert a list of str cut groups like
            [
                ("Sn__r0__getitem_317", ["Dn__r0__cat_10", "Dn__r1__cat_10"]),
                ("Sn__r0__getitem_319", ["Dn__r0__getitem_326", "Dn__r1__getitem_326"]),
            ]
        into CutGroup using `self.get_sexpr` to map from name to SExpr.
        """
        return [self.convert_str_cut_group(cg) for cg in str_cut_groups]

    def get_default_cut_groups(
        self, origin_sgraph: SGraph, target_sgraphs: list[SGraph]
    ) -> list[CutGroup]:
        return default_group_sgraph_cuts(origin_sgraph, target_sgraphs)

    def get_cut_groups(
        self,
        origin_sgraph: SGraph,
        target_sgraphs: list[SGraph],
    ) -> list[CutGroup]:
        """
        If you want to further partition the graph by some cut points other than the default
        ones, you can implement this method.
        """
        return default_group_sgraph_cuts(origin_sgraph, target_sgraphs)

    def build_expected(self) -> list[SExprECondition]:
        """
        When this is implemented, the expected conditions will be checked after all post-conditions been inferred.
        """
        return []

    def get_symval_instantiation(self) -> dict[str, int]:
        return {}

    def get_force_leaf_set(self) -> set[str]:
        """
        Returns a set of node names that should be forced to be leaf nodes.
        """
        return set()

    def get_lift_fresh_copy_constant_value(self, graph_path: str, s: str) -> int:
        raise RuntimeError(
            f"Got {s} in {graph_path=} here. It is like you did not implemented `get_lift_fresh_copy_constant_value` in the Config subclass, or you missed {s}."
        )

    def filter_output(self, default_outputs: list[SExpr]) -> list:
        return default_outputs[:1]

    def oracle_group_id_to_size(self, group_id: str | int) -> int:
        """
        Return the size of the group with the given ID.
        None means unknown
        """
        return None


class ExplorativeConfig(Config):
    def get_through_sexpr_cb(self) -> Callable[[SExpr], bool]:
        """
        This method only works in `explorative` mode (see `entangle.sgraph.infer.ExplorativeInferManager`).
        By specifying some sexprs that should not be considered as a valid postcondition point, we can
        acheive something similar to the `through` in the `sskeleton` mode.
        """
        return lambda _: False

    def get_cut_groups(self, origin_sgraph: SGraph, target_sgraphs: list[SGraph]):
        """
        For Explorative mode, the cut_groups just serve as a hint.
        """
        origin_sgraph
        target_sgraphs
        return []

    def get_hint_ops(self) -> dict[tgops.Op, set[tgops.Op]]:
        """
        When using `entangle.sgraph.infer.ExplorativeInferManager`, you can specify some ops that
        the target ops should in the allowed ops set.

        E.g.,
            If you assume a sum should always matches sum or mean, then you can write
            ```
            return {
                tgops.sum: {tgops.sum, tgops.mean},
            }
            ```
        """
        return {}


def load_config_class(module) -> Type[Config]:
    for key, value in module.__dict__.items():
        if (
            isinstance(value, type)
            and value is not Config
            and issubclass(value, Config)
            and value not in (Config, ExplorativeConfig)
        ):
            print("Loading Config class: ", value)
            return value
    raise RuntimeError("Cannot find a class inherited from Config in the module")
