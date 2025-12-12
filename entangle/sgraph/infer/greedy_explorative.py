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

import logging
import os
import os.path as osp
import pickle
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from itertools import chain
from typing import Callable, Generator, Iterable, Unpack

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
from entangle.utils.print_utils import BGREEN, BRED, BRI, BYELLOW, RST, print_ft

logger = logging.getLogger(__name__)


class CannotFindPotentialTargetOutputs(Exception): ...


class GreedyExplorativeInferManager(InferManager):
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
        `way`: either `set.union` or `set.intersection`. In the method `get_commom_users`, we use `way`
        for the users of each initial sexpr. `set.union` is a safer choice.
        """
        super().__init__(
            origin_sgraph, target_sgraphs, cut_groups, egg_runner, save_group
        )
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

    def get_mapped_targets(self, origin_name: str) -> list[list[str]]:
        if origin_name not in self.origin_to_targets:
            print(
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
    ) -> tuple[set[SExpr], set[SExpr], set[SExpr]]:
        """
        Returns results, begin_sexprs, and one_more_sexprs.

        This method seaerches at most `limit` layers from the initial `sexprs` and find the intersection
        or union of users of each sexpr in `sexprs` (dependends on `self.way`).
        During searching, all explored sexpr will be collected into `explored`, so that at the end, we
        can check if the found output sexprs are valid by checking if their args are all explored.

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

        one_more_sexprs = set()

        should_ignore = lambda s: s.op.constant or s.op == tgops.empty

        rich.print(f"{init_sexprs=}")

        while len(heads) > 0:
            new_heads = set()
            for sexpr in heads:
                if sexpr in init_sexprs:
                    min_steps_map[sexpr] = min_steps = 0
                    results.add(sexpr)
                else:
                    assert sexpr not in min_steps_map
                    arg_min_steps = [
                        min_steps_map[s] for s in sexpr.args if not should_ignore(s)
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
                    (min_steps > 0 and sexpr.name in self.barriers)
                ):
                    continue
                if min_steps > 1 and sexpr.op.skeleton:
                    if sexpr.op in (
                        tgops.matadd,
                        tgops.matsub,
                        tgops.matdiv,
                        tgops.ewmul,
                    ) and (
                        len(sexpr.params) >= 1 or any(s.shape == [] for s in sexpr.args)
                    ):
                        # We allow skeleton + - * / of at least one scalar.
                        pass
                    else:
                        # Save these skeleton sexprs for later check.
                        one_more_sexprs.add(sexpr)
                        continue
                for suc in sgraph.nx_graph.predecessors(sexpr.sexpr_id):
                    suc_sexpr: SExpr = sgraph.nx_graph.nodes[suc]["sexpr"]
                    if suc_sexpr in init_sexprs:
                        assert (
                            in_degrees[suc_sexpr] == 0
                        ), f"{suc_sexpr!r} in init_sexprs, but in_degrees={in_degrees[suc_sexpr]}"
                        continue
                    if suc_sexpr not in in_degrees:
                        # First time visiting.
                        # FIXME: This only considers the constant as direct input.
                        # Ideally, we should propogate a `constatnt` property along
                        # SExprs.
                        ignored_count = sum(should_ignore(s) for s in suc_sexpr.args)
                        in_degrees[suc_sexpr] = len(suc_sexpr.args) - ignored_count - 1
                    else:
                        in_degrees[suc_sexpr] -= 1
                    if should_ignore(sexpr):
                        # No need to descrease, because we exclude the constants
                        # when initializing. So add back.
                        in_degrees[suc_sexpr] += 1
                    if in_degrees[suc_sexpr] == 0:
                        new_heads.add(suc_sexpr)
            heads = new_heads

        # begin_sexprs should be the set of SExpr that any of its args is not visited during the exploration.
        begin_sexprs = set()
        explored_sexprs = set(min_steps_map.keys())
        for s in list(explored_sexprs):
            for a in s.args:
                if should_ignore(a):
                    explored_sexprs.add(a)
        for s in sexprs:
            if any(a not in explored_sexprs for a in s.args):
                begin_sexprs.add(s)
        return results, begin_sexprs, one_more_sexprs

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
            self.map_origin_to_targets([postcondition])

    def run(
        self,
        root_dirname: str,
        begin: int = None,
        explore_limit: int = 10,
        hint_ops: dict[tgops.Op, set[tgops.Op]] = None,
        max_workers: int = 1,
    ):
        all_begin_date = datetime.now()
        egg_runner = self.egg_runner
        topo = reversed(list(nx.topological_sort(self.origin_sgraph.nx_graph)))
        async_run = True

        if begin is None:
            begin = 0

        skipped_inputs = set()  # will be printed to help debug if failed.

        for topo_idx, origin_sexpr_id in enumerate(topo):
            if topo_idx < begin:
                continue
            begin_date = datetime.now()
            group_name = f"group{topo_idx}"
            dirname = osp.join(root_dirname, group_name)
            os.makedirs(dirname, exist_ok=True)

            print_ft(f"{BRI}{group_name}{RST}")

            origin_attr = self.origin_sgraph.nx_graph.nodes[origin_sexpr_id]
            origin_sexpr: SExpr = origin_attr["sexpr"]
            print(f"{BRI}Origin Sexpr{RST}: {origin_sexpr!r}")
            if self.through_sexpr_cb(origin_sexpr):
                print(f"{BYELLOW}Pass through for {origin_sexpr!r}{RST}")
                os.system(f"touch {dirname}/through.log")
            # elif origin_sexpr.name in self.origin_to_targets:
            #     for target_names in self.origin_to_targets[origin_sexpr.name]:
            #         target_sexprs = [self.name_to_sexpr[name] for name in target_names]
            #         cut_group = CutGroup(origin_sexpr, target_sexprs)
            #         self.run_one(
            #             dirname, cut_group, False, group_name, assume_provided=True
            #         )
            elif (
                origin_sexpr.name not in self.origin_to_targets
                and origin_sexpr.op == tgops.inpt
            ):
                # For input, we don't need to infer postconditions.
                skipped_inputs.add(origin_sexpr.name)
                print(f"{BRED}Skipped Inputs{RST}: Not provided.")
            elif origin_sexpr.op.constant:
                # For constants, we don't need to infer postconditions if not provided.
                print(f"{BRED}Skipped Constants{RST}: Not provided.")
            elif origin_sexpr.op.dist and origin_sexpr.op != tgops.dist_wait:
                print(f"{BYELLOW}Skipped{RST}: DistOp for origin.")
            elif origin_sexpr.name in self.manual_cut_mapping:
                print(f"{BGREEN}Found user-specified CutGroup.{RST}")
                target_sexprs = [
                    self.name_to_sexpr[name]
                    for name in self.manual_cut_mapping[origin_sexpr.name]
                ]
                cut_group = CutGroup(origin_sexpr, target_sexprs)
                origin_args = list(self.get_mapped_args(origin_sexpr))
                raw_econditions: set[ECondition] = set()
                for origin_arg in origin_args:
                    if origin_arg.name in self.origin_to_econditions:
                        raw_econditions.update(
                            self.origin_to_econditions[origin_arg.name]
                        )
                self.run_one(
                    dirname,
                    cut_group,
                    False,
                    group_name,
                    raw_econditions=raw_econditions,
                )
                self.post_run_process(cut_group, False, dirname)
            else:
                egg_runner.clean(dirname, keep_root=True)
                starting_from_inputs = origin_sexpr.name in self.origin_to_targets
                if starting_from_inputs:
                    origin_args = [origin_sexpr]
                else:
                    origin_args = list(self.get_mapped_args(origin_sexpr))
                raw_econditions: set[ECondition] = set()
                for origin_arg in origin_args:
                    if origin_arg.name in self.origin_to_econditions:
                        raw_econditions.update(
                            self.origin_to_econditions[origin_arg.name]
                        )
                print(f"{BRI}Origin Args{RST}:", end="")
                rich.print(origin_args)
                assert len(origin_args) > 0, (
                    f"No mapped args found, maybe this is input and you forget providing preconditions?\n"
                    f"{BRI}Inputs without conditions{RST}: {skipped_inputs}"
                )
                # Args of an origin_sexpr must have been conditioned due to topologic.
                # `potential_target_args_list_per_origin_arg` maps each origin_arg into a list of potential target args.
                potential_target_args_list_per_origin_arg: list[list[str]] = []
                for arg in origin_args:
                    target_args_list = self.get_mapped_targets(arg.name)
                    potential_target_args_list_per_origin_arg.append(
                        list(target_args_list)
                    )
                print(f"{BRI}Potential Target Args{RST}:", end="")
                rich.print(potential_target_args_list_per_origin_arg)
                print()

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
                print(f"{BRI}by_sg{RST}=", end="")
                rich.print(by_sg)
                if len(by_sg) == 0:
                    raise CannotFindPotentialTargetOutputs(
                        f"{BRED}len(by_sg) is 0, forget providing preconditions for some inputs?{RST}\n"
                        f"{BRI}Inputs without conditions{RST}: {skipped_inputs}"
                    )

                def generate_next_exploration(
                    by_sg: dict[SGraph, list[list[SExpr]]], hop_limit: int
                ) -> Generator[
                    tuple[int, bool, set[SExpr], set[SExpr], set[SExpr]], None, None
                ]:
                    begin_sexprs = set()
                    t_outs_per_sg: dict[SGraph, set[SExpr]] = {}
                    for sg, t_args_per_o_arg in by_sg.items():
                        t_outs_per_sg[sg] = set.union(
                            *[set(t) for t in t_args_per_o_arg]
                        )
                    for hop_idx in range(hop_limit):
                        # In each iteration, `t_outs_per_sg` will be updated
                        # with new `common_t_outs` (which also includes the
                        # set from previous iteration or initals).
                        one_more_sexprs = set()
                        for sg, t_args_all_o_arg in t_outs_per_sg.items():
                            (
                                common_t_outs,
                                begin_sexprs_this_sg,
                                one_more_sexprs_this_sg,
                            ) = self.get_commom_users(sg, t_args_all_o_arg, limit=limit)
                            t_outs_per_sg[sg] = set(common_t_outs)
                            begin_sexprs.update(begin_sexprs_this_sg)
                            one_more_sexprs.update(one_more_sexprs_this_sg)
                            print(f"For {sg=}, we found {t_outs_per_sg[sg]}")
                        t_outs: set[SExpr] = set.union(*t_outs_per_sg.values())

                        # NOTE: yield 1: First, try without one more skeleton node.
                        t_outs_without_one_more = t_outs - one_more_sexprs
                        yield hop_idx, False, begin_sexprs, t_outs_without_one_more, one_more_sexprs
                        # Allow one more layer of skeleton ops

                        # NOTE: yield 2: Second, try with one more skeleton node.
                        # (In bulky exploration, this helps assert even we explore
                        # one more, there won't be any new relations)
                        yield hop_idx, True, begin_sexprs, t_outs, one_more_sexprs

                # limit is a large enough number of steps for each explorations.
                limit = 10
                # hop_limit is the maximum number of hops we can explore, where
                # a hop means one exploration until skeletons.
                hop_limit = 2

                succeed = False
                if async_run:
                    executor = ThreadPoolExecutor(max_workers=max_workers)
                # Iteration over hops
                callback_and_infos: list[tuple[Future, str, CutGroup]] = []
                break_hop_loop = False
                for (
                    hop_idx,
                    with_one_more,
                    begin_sexprs,
                    t_outs,
                    one_more_sexprs,
                ) in generate_next_exploration(by_sg, hop_limit=hop_limit):
                    print(
                        f"------- {BRI}g{topo_idx}-{hop_idx=}{RST}, {len(t_outs)=}, {len(one_more_sexprs)=}"
                    )
                    print(f"{BRI}t_outs{RST}=", end="")
                    rich.print(sorted(t_outs, key=lambda s: s.name))

                    if with_one_more and len(one_more_sexprs) == 0:
                        # This can happend when we are trying one more skeleton
                        # nodes, but there is no such nodes since there can be
                        # some nodes that are not considered as args based on
                        # the single-device sexpr, and thus the potential next
                        # skeleton nodes are all not valid.
                        # In this case, we can just skip, because we are not
                        # going to add any new computations to saturate.
                        break_hop_loop = True
                        jobs = []
                    else:
                        if len(t_outs) == 0:
                            continue
                        # fmt: off
                        jobs = [(hop_idx, t_outs, all_target_args, begin_sexprs, with_one_more, one_more_sexprs)]
                        # fmt: on
                        print(
                            f"---------- {BRI}Found {len(jobs)} for {hop_idx=} {with_one_more=}{RST}"
                        )

                    # 1. Jobs prepared, run/start all jobs.
                    # Iteration over hops
                    for (
                        hop_idx,
                        target_sexprs,
                        target_args,
                        begin_sexprs,
                        with_one_more,
                        one_more_sexprs,
                    ) in jobs:
                        trial_id = (
                            f"{hop_idx}.onemore" if with_one_more else f"{hop_idx}"
                        )
                        trial_name = f"trial{trial_id}"
                        trial_dirname = osp.join(dirname, trial_id)
                        print(f"------- Begin {BRI}{trial_name} | {hop_idx=}{RST}")
                        print(f"{BRI}Trying target_sexprs:{RST}", end="")
                        rich.print(sorted(target_sexprs, key=lambda s: s.name))
                        print(f"{BRI}Target Args{RST}:", end="")
                        rich.print(sorted(target_args, key=lambda s: s.name))

                        target_sexprs = list(target_sexprs)
                        cut_group = CutGroup(origin_sexpr, target_sexprs)
                        is_only_input = cut_group.is_only_input()
                        begin_names = set(
                            [a.name for a in origin_args]
                            + [s.name for s in begin_sexprs]
                        )
                        args = (trial_dirname, cut_group, is_only_input, trial_dirname)
                        kwargs = {
                            "begin_names": begin_names,
                            "intermediate_name": True,
                            "raw_econditions": list(raw_econditions),
                            "additional_input_names": {
                                s.name for s in chain(target_args, target_sexprs)
                            },
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
                        callback_and_infos.append(
                            (
                                res,
                                trial_name,
                                trial_dirname,
                                cut_group,
                                with_one_more,
                                one_more_sexprs,
                            )
                        )
                        print(f"------- {BRI}g{topo_idx}-{trial_name}{RST}: {status}")

                    # We only continue to process results when it is `with_one_more`
                    if not with_one_more:
                        continue

                    # 2. Wait for processes at the end of `hop_idx` iteration.
                    collected_post_info: list[tuple[ECondition, CutGroup, str]] = []
                    print(f"----------------- {BRI}Callbacks for {hop_idx=}{RST}")
                    for (
                        run_one_res,
                        trial_name,
                        trial_dirname,
                        cut_group,
                        with_one_more,
                        one_more_sexprs,
                    ) in callback_and_infos:
                        try:
                            if async_run:
                                run_one_res.result()
                                assert run_one_res.done()
                                if (e := run_one_res.exception()) is not None:
                                    raise e
                            else:
                                if run_one_res is not None:
                                    raise run_one_res
                        except (CannotFindPostconditions, CannotMatchAllDistOps) as e:
                            status = f"{BRED}Failed: {type(e).__name__}{RST}"
                            continue
                        else:
                            status = f"{BGREEN}Success{RST}"
                        finally:
                            print(
                                f"------- {BRI}g{topo_idx}-{trial_name}{RST}: {status}"
                            )
                        postcondition_str = egg_runner.get_postcondition(trial_dirname)
                        postcondition = ECondition.from_str(postcondition_str)
                        collected_post_info.append(
                            (postcondition, cut_group, trial_dirname, one_more_sexprs)
                        )
                    callback_and_infos = []

                    # 3. Add postconditions and update conditioned names.
                    # According to our algorithm, we break out whenever no more postconditions found.
                    until_found_new = False
                    for (
                        postcondition,
                        cut_group,
                        trial_dirname,
                        one_more_sexprs,
                    ) in collected_post_info:
                        # If there is a forbidden name, this method raises error.
                        num_econds_before = len(self.econditions)
                        kept_econditions = self.add_econdition(
                            postcondition,
                            required_name=cut_group.origin_cut.name,
                            filter_info_dirname=trial_dirname,
                            forbidden_names={s.name for s in one_more_sexprs},
                        )
                        num_econds_after = len(self.econditions)
                        any_new = num_econds_after != num_econds_before
                        if any_new:
                            # This means we have found at least one new valid post-condition,
                            # meaning that 1). succeed in finding a relation for this
                            # single-device SExpr; 2). we found new relation.
                            succeed = True
                            until_found_new = True
                        # Map the conditions no matter what.
                        self.conditioned_names.update([c.name for c in cut_group.cuts])
                        if postcondition is not None:
                            self.map_origin_to_targets(kept_econditions)
                        print(
                            f"------------------------- Done {BRI}g{topo_idx}-{trial_name}{RST}: {succeed=}, {until_found_new=}, {len(kept_econditions)=}"
                        )
                        if not any_new:
                            until_found_new = False
                            break

                    if not until_found_new:
                        break  # If there is no more condition found, just break.

                    if break_hop_loop:
                        break

                if async_run:
                    executor.shutdown(wait=False, cancel_futures=True)

                if not succeed and not starting_from_inputs:
                    # NOTE: if `starting_from_inputs`, allow no postcondition.
                    # Log conditions for debugging.
                    if self.egg_runner.verbose:
                        print("======================================================")
                        for ec in self.econditions:
                            rich.print(ec)
                        print("------------------------------------------------------")
                        rich.print(self.origin_to_targets)
                    elapsed = datetime.now() - all_begin_date
                    print(f"{BRED}Stopped, time elapsed: {elapsed}{RST}")
                    # Save state.
                    self.through_sexpr_cb = None
                    self.save(osp.join(dirname, "checkpoint.pkl"))
                    raise CannotFindPostconditions(
                        f"{BRED}Failed, check the conditions above. {RST}\n"
                        f"{BRI}Inputs without conditions{RST}: {skipped_inputs}\n"
                        f"The last dirname: {dirname}"
                    )

            elapsed = datetime.now() - begin_date
            all_elapsed = datetime.now() - all_begin_date
            print_ft(f"Done with {BRI}{group_name}{RST} in {elapsed}/{all_elapsed}")
            print()

    def save(self, path):
        self.global_states = entangle.get_global_states()
        pickle.dump(self, open(path, "wb"))

    def resume(self, resumed: "GreedyExplorativeInferManager"):
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

        entangle.resume_global_states(resumed.global_states)
