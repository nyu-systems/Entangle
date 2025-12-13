# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Modifications Copyright (c) 2025 [Zhanghan Wang]
# Note: Support better logging.
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
from datetime import datetime

from entangle.sgraph.infer.infer import InferManager
from entangle.sgraph.sgraph import SGraph
from entangle.sgraph.sskeleton import CutGroup, SSkeleton
from entangle.tools.egg import EggRunner
from entangle.utils.print_utils import BRI, BYELLOW, RST, print_ft, get_global_logger


LOGGER = None


class SSkeletonInferManager(InferManager):
    def __init__(
        self,
        origin_sgraph: SGraph,
        target_sgraphs: list[SGraph],
        cut_groups: list[CutGroup],
        egg_runner: EggRunner,
        save_group: bool = False,
    ):
        super().__init__(
            origin_sgraph, target_sgraphs, cut_groups, egg_runner, save_group
        )
        global LOGGER
        LOGGER = get_global_logger()
        # This cut_groups is not ordered.
        # To get ordered cut_groups, use `self.sskeleton`.
        self.sskeleton = SSkeleton(origin_sgraph, target_sgraphs, cut_groups)
        # visual_utils.draw_pyvis(
        #     sskeleton.setup_to_visualize(), osp.join(args.output, "sskeleton.html")
        # )

    def run(
        self,
        root_dirname: str,
        begin: int,
        end: int,
        through: set[int] = None,
    ):
        LOGGER.info_ft(f"{BRI}All CutGroups to Solve (#={len(self.sskeleton)}){RST}", c="=")
        for idx, cg in enumerate(self.sskeleton):
            print(idx, cg)
        LOGGER.info_ft("", c="=")

        if through is None:
            through = set()
        if begin < 0:
            begin += len(self.sskeleton)
        if end == -1:
            end = len(self.sskeleton)
        LOGGER.info_ft(f"{BRI}Infer Postcondition{RST}", c="=")
        os.makedirs(root_dirname, exist_ok=True)
        for cut_group in self.sskeleton:
            begin_date = datetime.now()
            group_id = self.sskeleton.get_group_id(cut_group)
            group_name = f"group{group_id}"
            is_only_input = cut_group.is_only_input()
            is_pass_through = group_id in through
            dirname = osp.join(root_dirname, f"group{group_id}")

            LOGGER.info_ft(f"{BRI}{group_name}{RST} input?({is_only_input})")

            if group_id >= end:
                LOGGER.info_ft(f"{BRI}Reached end group (group_id={group_id}), exiting.{RST}")
                break

            if group_id >= begin:
                self.run_one(dirname, cut_group, is_pass_through, group_name)
            else:
                print(
                    f"{BYELLOW}Skipped group{group_id} due to range [{begin}, {end}).{RST}"
                )

            self.post_run_process(cut_group, is_pass_through, dirname)
            LOGGER.info_ft(
                f"Done with {BRI}{group_name}{RST} in {datetime.now() - begin_date}"
            )
            print()
