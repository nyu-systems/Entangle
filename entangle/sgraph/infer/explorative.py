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

import logging
import os
import os.path as osp
import pickle
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from itertools import chain
from typing import Callable, Unpack

import networkx as nx
import rich
import rich.progress

import entangle
import entangle.ops as tgops
from entangle.sgraph.egraph import CannotFindPostconditions, ECondition
from entangle.sgraph.infer.infer import InferManager
from entangle.sgraph.sexpr import SExpr
from entangle.sgraph.sgraph import SGraph
from entangle.sgraph.sskeleton import CutGroup
from entangle.sgraph.transform import CannotMatchAllDistOps
from entangle.tools.egg import EggRunner
from entangle.utils.print_utils import BGREEN, BRED, BRI, BYELLOW, RST, print_ft, get_global_logger

LOGGER = None


class CannotFindPotentialTargetOutputs(Exception): ...


class ExplorativeInferManager(InferManager):
    def __init__(
        self,
        origin_sgraph: SGraph,
        target_sgraphs: list[SGraph],
        cut_groups: list[CutGroup],
        egg_runner: EggRunner,
        save_group: bool = False,
        through_sexpr_cb: Callable[[str], bool] = None,
        way: Callable[[Unpack[set]], set] = set.union,
    ):
        """
        `through_sexpr_cb`: a callback function that returns True if the sexpr should not be considered
        as a cut. This is useful when we want to skip exploring some nodes and hint the explorer to be
        more efficient.
        `way`: either `set.union` or `set.intersection`. In the method `get_common_users`, we use `way`
        for the users of each initial sexpr. `set.union` is a safer choice.
        """
        super().__init__(
            origin_sgraph, target_sgraphs, cut_groups, egg_runner, save_group
        )
        global LOGGER
        LOGGER = get_global_logger()

        if through_sexpr_cb is None:
            through_sexpr_cb = lambda _: False
        self.through_sexpr_cb = through_sexpr_cb
        assert way in (set.union, set.intersection), "Only allow union or intersection."
        self.way = way

        # Allow user defined cut mapping (without exploring) This is a lazy way to handle some corner cases.
        # Meanwhile, those cuts are served as barriers, preventing explore beyond them.
        self.barriers: set[str] = set()
        self.manual_cut_mapping: dict[str, tuple[str]] = {}
        for cg in cut_groups:
            self.manual_cut_mapping[cg.origin_cut.name] = [
                cut.name for cut in cg.target_cuts
            ]
            self.barriers.update([cut.name for cut in cg.target_cuts])
        # `origin_to_targets` maps from origin_name to a set of "tuples containing representale target name",
        # Each tuple can be used to represent the origin.
        self.origin_to_targets: dict[str, set[tuple[str]]] = {}
        self.origin_to_econditions: dict[str, set[ECondition]] = {}

    def get_mapped_targets(self, origin_name: str) -> list[list[str]]:
        if origin_name not in self.origin_to_targets:
            LOGGER.info(
                f"{BYELLOW}Warning: No mapped targets found for {origin_name}.{RST} "
                f"This is only valid for those empty tensors for inplace ops."
            )
            return [[]]
        return self.origin_to_targets[origin_name]

    def get_commom_users(
        self,
        sgraph: SGraph,
        sexprs: list[SExpr],
        limit: int = 4,
        hint_ops: set[tgops.Op] = None,
        way: Callable[[Unpack[set]], set] = set.union,
    ) -> set[SExpr]:
        """
        This method seaerches at most `limit` layers from the initial `sexprs` and find the intersection
        or union of users of each sexpr in `sexprs` (dependends on `self.way`).
        During searching, all explored sexpr will be collected into `explored`, so that at the end, we
        can check if the found output sexprs are valid by checking if their args are all explored.

        `self.way`: set during initializing the ExplorativeInferManager. See __init__ docstring for more details.

        `limit`: default as 4. The maximum layers to explore. The initial `sexprs` considered as layer 0.
        `at_least`: default as 0. If specified, we will filter out those sexprs that would be valid even
        if we only explored `at_least - 1` layers.
        `hint_ops`: to reduce searching space, we allow user to provide a `hint_ops` set. Only
        sexprs with op in this set will be considered.

        """
        if len(sexprs) == 0:
            return set()

        init_sexprs: set[SExpr] = set(sexprs)
        heads = sexprs

        in_degrees: dict[SExpr, int] = {}
        min_steps_map: dict[SExpr, int] = {}

        for sexpr in sexprs:
            in_degrees[sexpr] = 0
            min_steps_map[sexpr] = 0

        results = set()

        while len(heads) > 0:
            new_heads = set()
            for sexpr in heads:
                if sexpr in init_sexprs:
                    min_steps_map[sexpr] = min_steps = 0
                    results.add(sexpr)
                else:
                    assert sexpr not in min_steps_map
                    arg_min_steps = [
                        min_steps_map[s] for s in sexpr.args if not s.op.constant
                    ]
                    if len(arg_min_steps) == 0:
                        prev_min_steps = 0
                    elif len(arg_min_steps) == 1:
                        prev_min_steps = arg_min_steps[0]
                    else:
                        prev_min_steps = min(*arg_min_steps)
                    if sexpr.op.dist and sexpr.op != tgops.dist_wait:
                        min_steps = prev_min_steps
                    elif self.through_sexpr_cb(sexpr):
                        min_steps = prev_min_steps
                    else:
                        min_steps = prev_min_steps + 1
                        results.add(sexpr)
                    min_steps_map[sexpr] = min_steps
                if min_steps >= limit or (
                    min_steps > 0 and sexpr.name in self.barriers
                ):
                    continue
                for suc in sgraph.nx_graph.predecessors(sexpr.sexpr_id):
                    suc_sexpr: SExpr = sgraph.nx_graph.nodes[suc]["sexpr"]
                    if suc_sexpr not in in_degrees:
                        # First time visiting.
                        # FIXME: This only considers the constant as direct input.
                        # Ideally, we should propogate a `constatnt` property along
                        # SExprs.
                        constant_count = sum(s.op.constant for s in suc_sexpr.args)
                        in_degrees[suc_sexpr] = len(suc_sexpr.args) - constant_count - 1
                    else:
                        in_degrees[suc_sexpr] -= 1
                    if sexpr.op.constant:
                        # No need to descrease, because we exclude the constants
                        # when initializing. So add back.
                        in_degrees[suc_sexpr] += 1
                    if in_degrees[suc_sexpr] == 0:
                        new_heads.add(suc_sexpr)
            heads = new_heads

        return results

    def map_origin_to_targets(self, econdition: ECondition):
        origin_inpt = None
        target_inpts: list[str] = []
        pure_sym = True
        for inpt in econdition.input_names:
            if inpt.startswith("Sym"):
                continue
            elif inpt.startswith("Sn"):
                assert origin_inpt is None, "Only one origin input is allowed."
                origin_inpt = inpt
            else:
                target_inpts.append(inpt)
            pure_sym = False
        if pure_sym:
            return
        if origin_inpt not in self.origin_to_targets:
            self.origin_to_targets[origin_inpt] = set()
        new = tuple(sorted([t for t in target_inpts]))
        LOGGER.info(f"origin={origin_inpt} <---> targets={new}")
        self.origin_to_targets[origin_inpt].add(new)
        if origin_inpt not in self.origin_to_econditions:
            self.origin_to_econditions[origin_inpt] = set()
        self.origin_to_econditions[origin_inpt].add(econdition)

    def get_mapped_args(self, output: SExpr) -> set[SExpr]:
        mapped_args: set[SExpr] = set()
        visited: set[SExpr] = set()
        terminate_cb = lambda x: x.name in self.origin_to_targets
        for sexpr, is_leaf, is_term in output.post_order_dfs(
            return_is_leaf=True, terminate_callback=terminate_cb
        ):
            sexpr: SExpr
            if is_leaf or is_term:
                mapped_args.add(sexpr)
            elif sexpr in visited:
                continue
            visited.add(sexpr)
        return mapped_args

    def post_run_process(self, cut_group, is_pass_through, dirname):
        postcondition_str = self.egg_runner.get_postcondition(dirname)
        postcondition = ECondition.from_str(postcondition_str)
        self.add_econdition(postcondition)
        self.conditioned_names.update([c.name for c in cut_group.cuts])
        if postcondition is not None:
            self.map_origin_to_targets(postcondition)

    def run(
        self,
        root_dirname: str,
        begin: int = None,
        explore_limit: int = 4,
        hint_ops: dict[tgops.Op, set[tgops.Op]] = None,
        max_workers: int = 1,
    ):
        all_begin_date = datetime.now()
        egg_runner = self.egg_runner
        topo = reversed(list(nx.topological_sort(self.origin_sgraph.nx_graph)))
        async_run = True

        if begin is None:
            begin = 0

        for topo_idx, origin_sexpr_id in enumerate(topo):
            if topo_idx < begin:
                continue
            begin_date = datetime.now()
            group_name = f"group{topo_idx}"
            dirname = osp.join(root_dirname, group_name)
            os.makedirs(dirname, exist_ok=True)

            LOGGER.info_ft(f"{BRI}{group_name}{RST}")

            origin_attr = self.origin_sgraph.nx_graph.nodes[origin_sexpr_id]
            origin_sexpr: SExpr = origin_attr["sexpr"]
            LOGGER.info(f"{BRI}Origin Sexpr{RST}: {origin_sexpr!r}")
            if self.through_sexpr_cb(origin_sexpr):
                LOGGER.info(f"{BYELLOW}Pass through for {origin_sexpr!r}{RST}")
                os.system(f"touch {dirname}/through.log")
            elif origin_sexpr.name in self.origin_to_targets:
                for target_names in self.origin_to_targets[origin_sexpr.name]:
                    target_sexprs = [self.name_to_sexpr[name] for name in target_names]
                    cut_group = CutGroup(origin_sexpr, target_sexprs)
                    self.run_one(
                        dirname, cut_group, False, group_name, assume_provided=True
                    )
            elif origin_sexpr.op == tgops.inpt:
                # For input, we don't need to infer postconditions.
                LOGGER.info(f"{BRED}Skipped{RST}: Not provided.")
            elif origin_sexpr.op.constant:
                # For constants, we don't need to infer postconditions if not provided.
                LOGGER.info(f"{BRED}Skipped{RST}: Not provided.")
            elif origin_sexpr.op.dist and origin_sexpr.op != tgops.dist_wait:
                LOGGER.info(f"{BYELLOW}Skipped{RST}: DistOp for origin.")
            elif origin_sexpr.name in self.manual_cut_mapping:
                LOGGER.info(f"{BGREEN}Found user-specified CutGroup.{RST}")
                target_sexprs = [
                    self.name_to_sexpr[name]
                    for name in self.manual_cut_mapping[origin_sexpr.name]
                ]
                cut_group = CutGroup(origin_sexpr, target_sexprs)
                self.run_one(dirname, cut_group, False, group_name)
                self.post_run_process(cut_group, False, dirname)
            else:
                egg_runner.clean(dirname, keep_root=True)
                origin_args = list(self.get_mapped_args(origin_sexpr))
                raw_econditions: set[ECondition] = set()
                for origin_arg in origin_args:
                    if origin_arg.name in self.origin_to_econditions:
                        raw_econditions.update(
                            self.origin_to_econditions[origin_arg.name]
                        )
                LOGGER.info(f"{BRI}Origin Args{RST}:", end="")
                LOGGER.info(origin_args)
                assert (
                    len(origin_args) > 0
                ), "No mapped args found, maybe this is input and you forget providing preconditions?"
                # Args of an origin_sexpr must have been conditioned due to topologic.
                # `potential_target_args_list_per_origin_arg` maps each origin_arg into a list of potential target args.
                potential_target_args_list_per_origin_arg: list[list[str]] = []
                for arg in origin_args:
                    target_args_list = self.get_mapped_targets(arg.name)
                    potential_target_args_list_per_origin_arg.append(
                        list(target_args_list)
                    )
                LOGGER.info(f"{BRI}Potential Target Args{RST}:", end="")
                LOGGER.info(potential_target_args_list_per_origin_arg)
                LOGGER.info()

                hint_ops_for_this = hint_ops.get(origin_sexpr.op, None)

                # Usually when origin progress one step, target progresses at least one step.
                # So we start from 1. But at the end, we will still try 0.

                all_target_args = set()

                by_sg: dict[SGraph, list[list[SExpr]]] = {}
                for o_arg_idx, t_args in enumerate(
                    potential_target_args_list_per_origin_arg
                ):
                    t_args = set.union(
                        *[set(t) for t in t_args]
                    )  # The element is now a list of tuple, need to flatten
                    for t_arg in t_args:
                        t_sg = self.name_to_sgraph[t_arg]
                        if t_sg not in by_sg:
                            by_sg[t_sg] = [[] for _ in range(len(origin_args))]
                        t_arg = self.name_to_sexpr[t_arg]
                        by_sg[t_sg][o_arg_idx].append(t_arg)
                        all_target_args.add(t_arg)
                LOGGER.info(f"{BRI}by_sg{RST}=", end="")
                LOGGER.info(by_sg)
                if len(by_sg) == 0:
                    raise CannotFindPotentialTargetOutputs(
                        f"{BRED}len(by_sg) is 0, forget providing preconditions for some inputs?{RST}"
                    )

                zero_succeed = False
                for limit in chain(range(1, 8), [0]):
                    LOGGER.info(f"------- {BRI}g{topo_idx}-{limit=}{RST}")
                    succeed = False
                    t_outs_per_sg = []
                    for sg, t_args_per_o_arg in by_sg.items():
                        t_args_all_o_arg = set.union(
                            *[set(t) for t in t_args_per_o_arg]
                        )
                        common_t_outs = self.get_commom_users(
                            sg,
                            t_args_all_o_arg,
                            limit=limit,
                            hint_ops=hint_ops_for_this,
                            way=set.union,
                        )
                        t_outs_per_sg.append(set(common_t_outs))
                        LOGGER.info(f"For {sg=}, we found {t_outs_per_sg[-1]}")
                    t_outs: set[SExpr] = set.union(*t_outs_per_sg)
                    # Filter t_outs by skeleton nodes
                    if origin_sexpr.op.skeleton:
                        t_outs = set(filter(lambda s: s.op == origin_sexpr.op, t_outs))
                    LOGGER.info(f"{BRI}t_outs{RST}=", end="")
                    LOGGER.info(sorted(t_outs, key=lambda s: s.name))

                    if len(t_outs) == 0:
                        continue

                    jobs = []
                    if limit >= 1:
                        actual_t_outs = t_outs - all_target_args
                        if len(actual_t_outs) != 0:
                            jobs = [(limit, 0, 0, actual_t_outs, all_target_args)]
                    else:
                        jobs = [(0, 0, 0, t_outs, all_target_args)]
                    LOGGER.info(f"---------- {BRI}Found {len(jobs)} for {limit=}{RST}")
                    if len(jobs) == 0:
                        continue
                    # Run all jobs
                    callback_and_infos: list[tuple[Future, str, CutGroup]] = []
                    # Collect info for successful trials.
                    collected_post_info: list[tuple[ECondition, CutGroup, str]] = []

                    if async_run:
                        executor = ThreadPoolExecutor(max_workers=max_workers)
                    for (
                        limit,
                        args_trial_idx,
                        sexprs_trial_idx,
                        target_sexprs,
                        target_args,
                    ) in jobs:
                        trial_id = f"{limit}.{args_trial_idx}.{sexprs_trial_idx}"
                        trial_name = f"trial{trial_id}"
                        trial_dirname = osp.join(dirname, trial_id)
                        LOGGER.info(f"------- Begin {BRI}{trial_name} | {limit=}{RST}")
                        LOGGER.info(f"{BRI}Trying target_sexprs:{RST}", end="")
                        LOGGER.info(sorted(target_sexprs, key=lambda s: s.name))
                        LOGGER.info(f"{BRI}Target Args{RST}:", end="")
                        LOGGER.info(sorted(target_args, key=lambda s: s.name))

                        target_sexprs = list(target_sexprs)
                        cut_group = CutGroup(origin_sexpr, target_sexprs)
                        is_only_input = cut_group.is_only_input()
                        begin_names = set(
                            [a.name for a in origin_args]
                            + [s.name for s in target_args]
                        )
                        args = (trial_dirname, cut_group, is_only_input, trial_dirname)
                        kwargs = {
                            "begin_names": begin_names,
                            "intermediate_name": True,
                            "raw_econditions": self.econditions + raw_econditions,
                            "additional_input_names": [s.name for s in target_args],
                        }

                        if async_run:
                            res = executor.submit(self.run_one, *args, **kwargs)
                            status = f"{BYELLOW}Async running{RST}"
                        else:
                            try:
                                self.run_one(*args, **kwargs)
                                res = None
                            except (
                                CannotFindPostconditions,
                                CannotMatchAllDistOps,
                            ) as e:
                                res = e
                                status = f"{BRED}Failed: {type(e).__name__}{RST}"
                            else:
                                status = f"{BGREEN}Success{RST}"
                        callback_and_infos.append((res, trial_dirname, cut_group))
                        LOGGER.info(f"------- {BRI}g{topo_idx}-{trial_name}{RST}: {status}")
                    if async_run:
                        executor.shutdown(wait=True)

                    # Wait for processes at the end of `limit` iteration.
                    LOGGER.info(f"----------------- {BRI}Callbacks for {limit=}{RST}")
                    for run_one_res, trial_dirname, cut_group in callback_and_infos:
                        # assert process is not None
                        try:
                            if async_run:
                                assert run_one_res.done()
                                if (e := run_one_res.exception()) is not None:
                                    raise e
                            else:
                                if run_one_res is not None:
                                    raise run_one_res
                        except (
                            CannotFindPostconditions,
                            CannotMatchAllDistOps,
                        ) as e:
                            status = f"{BRED}Failed: {type(e).__name__}{RST}"
                            continue
                        else:
                            status = f"{BGREEN}Success{RST}"
                        finally:
                            LOGGER.info(
                                f"------- {BRI}g{topo_idx}-{trial_name}{RST}: {status}"
                            )
                        postcondition_str = egg_runner.get_postcondition(trial_dirname)
                        postcondition = ECondition.from_str(postcondition_str)
                        collected_post_info.append(
                            (postcondition, cut_group, trial_dirname)
                        )

                    # if len(collected_post_info) > 0:
                    #     # Early stop if already found and progress at least one.
                    #     # NOTE: XXX: This is not safe. To ensure safety, we should explore
                    #     # a large enough `limit`. But in most cases, early return is fine.
                    #     break

                    # Add postconditions and update conditioned names.
                    for postcondition, cut_group, trial_dirname in collected_post_info:
                        kept_econditions = self.add_econdition(
                            postcondition,
                            required_name=cut_group.origin_cut.name,
                            do_add=False,
                        )
                        if len(kept_econditions) > 0:
                            succeed = True
                        else:
                            continue
                        self.conditioned_names.update([c.name for c in cut_group.cuts])
                        if postcondition is not None:
                            self.map_origin_to_targets(
                                ECondition.merge(*kept_econditions)
                            )
                    LOGGER.info(
                        f"------------------------- Done {BRI}g{topo_idx}-{trial_name}{RST}: {succeed=}\n"
                    )
                    if limit == 0:
                        zero_succeed = succeed
                    if limit > 0 and succeed:
                        break

                if not (succeed or zero_succeed):
                    # Log conditions for debugging.
                    LOGGER.info("========================================================")
                    for ec in self.econditions:
                        LOGGER.info(ec)
                    LOGGER.info("--------------------------------------------------------")
                    LOGGER.info(self.origin_to_targets)
                    elapsed = datetime.now() - all_begin_date
                    LOGGER.info(f"{BRED}Stopped, time elapsed: {elapsed}{RST}")
                    # Save state.
                    self.through_sexpr_cb = None
                    self.save(osp.join(dirname, "checkpoint.pkl"))
                    raise CannotFindPostconditions("Check the conditions above.")

            elapsed = datetime.now() - begin_date
            all_elapsed = datetime.now() - all_begin_date
            LOGGER.info_ft(f"Done with {BRI}{group_name}{RST} in {elapsed}/{all_elapsed}")
            LOGGER.info()

    def save(self, path):
        self.global_states = entangle.get_global_states()
        pickle.dump(self, open(path, "wb"))

    def resume(self, resumed: "ExplorativeInferManager"):
        self.way: Callable[[Unpack[set]], set] = resumed.way
        self.origin_to_targets: dict[str, set[tuple[Unpack[str]]]] = (
            resumed.origin_to_targets
        )
        self.origin_to_econditions: dict[str, set[ECondition]] = (
            resumed.origin_to_targets
        )
        self.origin_sgraph: SGraph = resumed.origin_sgraph
        self.target_sgraphs: list[SGraph] = resumed.target_sgraphs
        self.cut_groups: list[CutGroup] = resumed.cut_groups
        self.sgraphs: list[SGraph] = resumed.sgraphs
        self.name_to_sexpr: dict[str, SExpr] = resumed.name_to_sexpr
        self.name_to_sgraph: dict[str, SGraph] = resumed.name_to_sgraph
        self.egg_runner: EggRunner = resumed.egg_runner
        self.save_group: bool = resumed.save_group
        self.econditions: list[ECondition] = resumed.econditions
        self.added_econdition_strs: set[str] = resumed.added_econdition_strs
        self.scalar_econditions: list[ECondition] = resumed.scalar_econditions
        self.conditioned_names: set[str] = resumed.conditioned_names

        entangle.resume_global_states(resumed.global_states)
        self.conditioned_names: set[str] = resumed.conditioned_names

        entangle.resume_global_states(resumed.global_states)
