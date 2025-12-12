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

import json
import os
import os.path as osp
import pickle
from datetime import datetime
from itertools import chain
from multiprocessing.dummy import Pool as DummyPool
from typing import Iterable, Optional

import rich

import threading
import entangle
import entangle.sym as sym
from entangle.sgraph.egraph import ECondition, SExprECondition
from entangle.sgraph.sexpr import SExpr, ShapeLike
from entangle.sgraph.sgraph import SGraph
from entangle.sgraph.sskeleton import CutGroup
from entangle.sgraph.transform import CannotMatchAllDistOps, SGraphTransformer
from entangle.sgraph.visualization import visualize_sgraph_to_infer
from entangle.tools.egg import EggRunner
from entangle.utils import ENODE_SPLIT
from entangle.utils.print_utils import BGREEN, BRED, BRI, BYELLOW, RST, print_ft


graph_merge_transform_lock = threading.Lock()


class FoundForbiddenNameError(Exception): ...


def replace_all(s: str, replacements: dict[str, str]) -> str:
    if replacements is None:
        return s
    for key, value in replacements.items():
        s = s.replace(key, value)
    return s


def save_shapes(sexprs: list[SExpr], path: str, mode: str = "w"):
    mappings: dict[str, list[str]] = json.load(open(path, "r")) if mode == "a" else {}
    for sexpr in sexprs:
        egg_str = sexpr.shape.to_egg_str()
        if mode == "a":
            if sexpr.name in mappings:
                assert mappings[sexpr.name] == egg_str
        mappings[f"{sexpr.name}@"] = egg_str
    with open(path, "w") as f:
        json.dump(mappings, f, indent=2)


def save_expected(
    dirname: str,
    expected: SExprECondition,
    replacements: dict[str, str] = None,
):
    """
    `replacements`: maps transformed output name to original output name.
    """
    checklist = []
    with open(osp.join(dirname, "ce.sexpr"), "w") as f:
        sexprs = []
        for eclass in expected.eclasses:
            sexprs.extend(eclass)
            to_check_equivalence = []
            for sexpr in eclass:
                f.write(replace_all(sexpr.to_egg_str(), replacements) + ENODE_SPLIT)
                assert sexpr.name is not None
                named_input = replace_all(sexpr.to_egg_str_as_inpt(), replacements)
                to_check_equivalence.append(named_input)
                f.write(replace_all(named_input, replacements) + "\n")
            checklist.append(to_check_equivalence)
        save_shapes(sexprs, osp.join(dirname, "shapes.json"), mode="a")
    with open(osp.join(dirname, "impl_checklist.sexpr"), "w") as f:
        for to_check_equivalence in checklist:
            f.write(ENODE_SPLIT.join(to_check_equivalence) + "\n")


def save_merged(
    sgraphs: list[SGraph],
    path: str,
    return_merged: bool = False,
    intermediate_name: bool = False,
) -> tuple[dict[str, str], SGraph]:
    """
    Merge sgraphs and save all at once into path.
    This function returns a dict that maps the new output names to the original
    output names if any.
    """
    sexpr_transformer = SGraphTransformer(sgraphs)
    merged_sgraph = (
        sexpr_transformer.merge_dist_ops()
        .merge_duplicated_dist_wait()
        .lower_dist_ops()
        .merge_clones()
        .rebuild_sepxrs()
        .to_sgraph()
    )

    merged_sgraph.save(path, intermediate_name)

    if return_merged:
        return sexpr_transformer.collapsed_replacements(), merged_sgraph
    else:
        return sexpr_transformer.collapsed_replacements()


class SubInferInfo:
    """
    This is a class that stores necessary and mimimum information to run a sub graphs inference.
    """

    def __init__(
        self,
        origin_sgraph,
        target_sgraphs,
        econditions: list[ECondition],
        scalar_econditions: list[ECondition],
    ):
        self.origin_sgraph = origin_sgraph
        self.target_sgraphs = target_sgraphs
        self.econditions = econditions
        self.scalar_econditions = scalar_econditions

    def save(self, path):
        with open(path, "wb") as f:
            self.global_states = entangle.get_global_states()
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
            entangle.resume_global_states(obj.global_states)


class InferManager:
    def __init__(
        self,
        origin_sgraph: SGraph,
        target_sgraphs: list[SGraph],
        cut_groups: list[CutGroup],
        egg_runner: EggRunner,
        save_group: bool = False,
        max_self_provable_check_worker: int = None,
    ):
        self.origin_sgraph: SGraph = origin_sgraph
        self.target_sgraphs: list[SGraph] = target_sgraphs
        self.sgraphs = [origin_sgraph] + target_sgraphs
        self.cut_groups: list[CutGroup] = cut_groups

        self.name_to_sexpr: dict[str, SExpr] = {}
        self.name_to_sgraph: dict[SExpr, SGraph] = {}
        for sgraph in self.sgraphs:
            num_before = len(self.name_to_sexpr)
            num_add = len(sgraph.name_to_sexpr)
            self.name_to_sexpr.update(sgraph.name_to_sexpr)
            for name, sexpr in sgraph.name_to_sexpr.items():
                assert (
                    name not in self.name_to_sgraph
                ), f"Name conflict: {name}: {sexpr!r}"
                self.name_to_sgraph[name] = sgraph
            assert len(self.name_to_sexpr) == num_before + num_add, "Name conflict."

        self.egg_runner = egg_runner
        self.save_group = save_group
        self.max_self_provable_check_worker = (
            max_self_provable_check_worker or os.cpu_count() * 2
        )

        self.econditions: list[ECondition] = []
        self.added_econdition_strs: set[str] = set()  # This is just for unique..
        self.scalar_econditions: list[ECondition] = []
        # `conditioned_names` stores names of sexprs that has been infered postconditions.
        self.conditioned_names = set()

        self.all_scalar_econd_str: str = None

        # `origin_to_targets` maps from origin_name to a set of "tuples containing representale target name",
        # Each tuple can be used to represent the origin.
        self.origin_to_targets: dict[str, set[tuple[str]]] = {}
        self.origin_to_econditions: dict[str, set[ECondition]] = {}

    def map_origin_to_targets(self, econditions: list[ECondition]):
        if not isinstance(econditions, Iterable):
            econditions = [econditions]
        origin_inpt = None
        target_inpts: list[str] = []
        pure_sym = True
        input_names = set.union(*[e.input_names_set for e in econditions])
        for inpt in input_names:
            if inpt.startswith("Sym"):
                continue
            elif inpt.startswith("Sn"):
                assert origin_inpt is None, "Only one origin input is allowed."
                origin_inpt = inpt
            elif inpt.startswith("(fill"):
                continue
            else:
                target_inpts.append(inpt)
            pure_sym = False
        if pure_sym:
            return
        if origin_inpt not in self.origin_to_targets:
            self.origin_to_targets[origin_inpt] = set()
        new = tuple(sorted([t for t in target_inpts]))
        rich.print(f"origin={origin_inpt} <---> targets={new}")
        self.origin_to_targets[origin_inpt].add(new)
        if origin_inpt not in self.origin_to_econditions:
            self.origin_to_econditions[origin_inpt] = set()
        self.origin_to_econditions[origin_inpt].update(econditions)

    def name_mapper(self, name):
        assert name in self.name_to_sexpr, f"{name} not found in sexpr set."
        sexpr = self.name_to_sexpr[name]
        assert sexpr.shape is not None, f"Shape not found for {sexpr=}."
        return f"(input {sexpr.name}@{sexpr.shape})"

    def get_sexpr(self, name):
        assert name in self.name_to_sexpr, f"{name} not found in sexpr set."
        return self.name_to_sexpr[name]

    def add_econdition(
        self,
        econdition: ECondition,
        check_self_provable: bool = True,
        required_name: str = None,
        filter_info_dirname: str = None,
        forbidden_names: set[str] = None,
        do_add: bool = True,
    ) -> list[ECondition]:
        """
        `check_self_provable`: if True, we may prune some econditions. So this
            method will return the kept econditions.
        `filter_info_dirname`: a dirname. If specified with "--verbose", we will
            dump the filter information.
        `do_add`: if True, we will add the econdition to `self.econditions`.

        Returns:
        `kept_econditions`: the econditions that are kept after filtering. We keep it even
            if it exists. Keeping here just means valid.
        """
        split_econditions = econdition.split()
        non_scalar_econds: list[ECondition] = []
        non_scalar_econd_strs: list[str] = []
        # Add scalar conditions
        for idx, econd in enumerate(split_econditions):
            econd_str = econd.to_egg_str()
            if len(econd.eclasses) == 0:
                # This can happen for user-specified conditions that just want to
                # specify some tensors should be considered together.
                continue
            if econd.all_scalar:
                self.scalar_econditions.append(econd)
            else:
                assert len(econd.eclasses) == 1
                econd.idx = idx
                non_scalar_econds.append(econd)
                non_scalar_econd_strs.append(econd_str)
        pruned_idx = []
        kept_econds = []
        kept_econd_strs = []
        pruned_econd_strs = []
        exisited_econd_strs = []
        if check_self_provable:
            # Check and add non-scalar conditions
            econd_strs = non_scalar_econd_strs
            econds = non_scalar_econds
            econd_name_shape_mappings = [
                {f"{n}@": self.name_to_sexpr[n].shape for n in e.input_names}
                for e in econds
            ]
            num_proc = max(
                min(len(econd_strs) // 8, self.max_self_provable_check_worker), 1
            )
            with DummyPool(num_proc) as p:
                provable_ids_list = p.map(
                    self.egg_runner.check_self_provable,
                    zip(econd_strs, econd_name_shape_mappings),
                )
        else:
            provable_ids_list = [None] * len(non_scalar_econd_strs)

        have_required_name = False
        for econd, econd_str, provable_ids in zip(
            non_scalar_econds, non_scalar_econd_strs, provable_ids_list
        ):
            enode_strs: list[str] = econd_str.split(ENODE_SPLIT)
            if provable_ids is not None:
                assert len(enode_strs) == len(
                    provable_ids
                ), f"{econd_strs=}, {provable_ids=}, {econd=}, {econd.eclasses=}"
                pruned_econd = econd.prune_self_provable(provable_ids)
                pruned_econd_str = pruned_econd.to_egg_str()
            else:
                pruned_econd = econd
                pruned_econd_str = econd_str
            if len(pruned_econd.eclasses[0]) <= 1:
                pruned_idx.append(econd.idx)
                pruned_econd_strs.append(econd_str)
                # Self-provable, skip.
                continue
            if required_name is not None:
                if pruned_econd_str.find(required_name) != -1:
                    have_required_name = True
            # Check if `pruned_econd` uses any forbidden names.
            if forbidden_names is not None:
                if any(
                    name in pruned_econd.input_names_set for name in forbidden_names
                ):
                    raise FoundForbiddenNameError(
                        f"Found forbidden name in econdition: {pruned_econd=}\n{forbidden_names=}"
                    )
            kept_econds.append(pruned_econd)
            kept_econd_strs.append(pruned_econd_str)
            if do_add:
                if pruned_econd_str in self.added_econdition_strs:
                    exisited_econd_strs.append(pruned_econd_str)
        if required_name is not None and not have_required_name:
            print(f"{BRED}Required name not satisfied: {required_name}{RST}")
            return []
        if do_add:
            for econd, econd_str in zip(kept_econds, kept_econd_strs):
                if econd_str not in self.added_econdition_strs:
                    self.econditions.append(econd)
                    self.added_econdition_strs.add(econd_str)

        # Below are all logging for debugging.
        print(f"{BRI}Required name{RST}: {required_name}")
        print(f"{BRI}Kept EConditions:{RST}")
        rich.print(kept_econd_strs)
        print(f"{BRI}Pruned EConditions:{RST} {pruned_idx=}")
        if self.egg_runner.verbose:
            rich.print(pruned_econd_strs)
        print(f"{BRI}Existed EConditions:{RST} {pruned_idx=}")
        if self.egg_runner.verbose:
            rich.print(exisited_econd_strs)
        if self.egg_runner.verbose and filter_info_dirname is not None:
            path = osp.join(filter_info_dirname, "pruned_econditions.sexpr")
            with open(path, "w") as f:
                f.write("\n".join(pruned_econd_strs))
            path = osp.join(filter_info_dirname, "kept_econditions.sexpr")
            with open(path, "w") as f:
                f.write("\n".join(kept_econd_strs))
            path = osp.join(filter_info_dirname, "existing_econditions.sexpr")
            with open(path, "w") as f:
                f.write("\n".join(exisited_econd_strs))
        return kept_econds

    def filter_econditions(
        self,
        raw_econditions: list[ECondition],
        input_sexpr_names: set[str],
        output_replacements: dict[str, str],
        split_scalar: bool = False,
    ) -> list[ECondition] | tuple[list[ECondition], list[ECondition]]:
        """
        This method returns econditions whose input names are subset of `input_sexpr_names`.
        """
        if type(input_sexpr_names) is not set:
            input_sexpr_names = set(input_sexpr_names)
        econditions = []
        scalar_econditions = []
        for econdition in raw_econditions:
            extracted = econdition.extract(input_sexpr_names, output_replacements)
            if self.egg_runner.verbose:
                print(f"{BRI}Try to extracted ECondition from :{RST}")
                rich.print(econdition)
                print(f"{BRI}Extracted ECondition:{RST}")
                rich.print(extracted)
            if extracted:
                if split_scalar and extracted.all_scalar:
                    scalar_econditions.append(extracted)
                else:
                    econditions.append(extracted)
        if split_scalar:
            return econditions, scalar_econditions
        else:
            return econditions

    def save_preconditions(self, econditions: list[ECondition], dirname: str):
        path = osp.symabspath(osp.join(dirname, "precondition.sexpr"))
        precondition_strs = {
            s for s in map(ECondition.to_egg_str, econditions) if s.find("true") == -1
        }
        precondition_str = "\n".join(precondition_strs)
        unique = "\n".join(set(precondition_str.split("\n")))
        if unique.count("\n") > 20:
            print(f"{BRI}Precondition:{RST}: too long to print, please check {path}")
        else:
            print(f"{BRI}Precondition:{RST}")
            rich.print(unique)
        with open(path, "a") as f:
            f.write(f"# Preconditions\n")
            f.write(unique)

    def save_constant_preconditions(self, constant_sexprs: list[SExpr], dirname: str):
        constant_conditions = [
            f"{s.get_placeholderized(keep_constant=False).to_egg_str()}{ENODE_SPLIT}{s.to_egg_str()}"
            for s in constant_sexprs
        ]
        with open(osp.join(dirname, "precondition.sexpr"), "a") as f:
            f.write("".join(["\n" + s for s in constant_conditions]))

    def save_scalar_conditions(
        self, scalar_econditions: list[ECondition], dirname: str, used_names: set[str]
    ):
        with open(osp.join(dirname, "precondition.scalar.sexpr"), "a") as f:
            scalar_econditions_strs = [
                s for s in [c.to_egg_str(eq_only=True) for c in scalar_econditions]
            ]
            if self.egg_runner.verbose:
                print("Scalar pre-conditions: ", end="")
                if len(scalar_econditions_strs) > 0:
                    strs = "\n".join(scalar_econditions_strs).split("\n")
                    rich.print(list(filter(lambda x: x != "", strs)))
                else:
                    rich.print([])
            f.write("".join([f"\n{s}" for s in scalar_econditions_strs]))
        with open(osp.join(dirname, "precondition.scalar.smtlib"), "a") as f:
            assertions = list(
                chain(*[ec.to_smtlib_strs() for ec in scalar_econditions])
            )
            if self.egg_runner.verbose:
                print("Scalar conditions (smtlib): ", end="")
                rich.print(assertions)
            f.write("\n".join([s for s in assertions]))

    def precompute_all_scalar_conditions(self):
        """
        This method should be called after initializing the infer mangaer.
        It pre-computes all the scalar preconditions for EggRunner to use
        such that the self-provable check can correctly work.
        """
        assertions = list(
            chain(*[ec.to_smtlib_strs() for ec in self.scalar_econditions])
        )
        scalar_econd_str = "\n".join([s for s in assertions])
        self.all_scalar_econd_str = scalar_econd_str
        self.egg_runner.all_scalar_econd_str = scalar_econd_str

    def replace_back_postcondition(
        self,
        sgraphs: list[SGraph],
        transformed_sgraphs: list[SGraph],
        output_replacements: dict[str, str],
        dirname: str,
    ):
        raw_postcondition_str = self.egg_runner.get_postcondition(dirname, raw=True)
        # This is for the original version of names.
        output_names = set(chain(*[[s.name for s in sg.outputs] for sg in sgraphs]))
        tsf_output_names = {s.name for sg in transformed_sgraphs for s in sg.outputs}
        # Convert output names back and save.
        output_back_replacements: dict[str, list[str]] = {}
        for old, new in output_replacements.items():
            if old not in output_names:
                # Skip if the output is not in the original outputs.
                continue
            if new not in tsf_output_names:
                # Skip if the output is not in the merged outputs.
                continue
            if new not in output_back_replacements:
                output_back_replacements[new] = []
            output_back_replacements[new].append(old)

        # Since we skipped unrelated names, v[0] must be original output name.
        postcondition_str = replace_all(
            raw_postcondition_str,
            {k: v[0] for k, v in output_back_replacements.items()},
        )
        with open(osp.join(dirname, "postcondition.sexpr"), "w") as f:
            # Except for using the `output_back_replacements`, we also need to add
            # the equivalence of the outputs that are replaced by the same name.

            # This is for the replaced version from Egg.
            used_back_replacements = [
                [n for n in v if n in output_names]
                for k, v in output_back_replacements.items()
                if k in tsf_output_names and len(v) > 1
            ]
            print("Back replacing:")
            print("output_names: ", end="")
            rich.print(sorted(output_names))
            print("tsf_output_names: ", end="")
            rich.print(sorted(tsf_output_names))
            print("output_back_replacements: ", end="")
            rich.print(output_back_replacements)
            print("used_back_replacements: ", end="")
            rich.print(used_back_replacements)
            if len(used_back_replacements) > 0:
                eclasses = [
                    [self.name_to_sexpr[n].get_placeholderized() for n in names]
                    for names in used_back_replacements
                ]
                econdition = ECondition.from_sexpr_econdition(
                    SExprECondition(inputs=list(chain(*eclasses)), eclasses=eclasses)
                )
                postcondition = ECondition.from_str(postcondition_str)
                # self.add_econdition(econdition)
                # # Merge into `postcondition` because we add it later.
                # postcondition = postcondition.merge(econdition)
                f.write(econdition.to_egg_str())
                f.write("\n")
                f.write(postcondition.to_egg_str())
            else:
                f.write(postcondition_str)

    def run_one(
        self,
        dirname: str,
        cut_group: CutGroup,
        is_pass_through: bool,
        group_name: str,
        *,
        begin_names: set[str] = None,
        intermediate_name: bool = False,
        assume_provided: bool = False,
        raw_econditions: list[ECondition] = None,
        additional_input_names: set[str] = None,
    ) -> dict[str, str]:
        """
        If `origin_sgraph` and `target_sgraph` are provided, we use them;
        otherwise, we extract the sgraphs from cut_group.

        `intermediate_name`: also put intermediate names into the preconditions.
        `raw_econditions`: By default, if it is None, we will use all collected self.econditions.
        """
        egg_runner = self.egg_runner
        cut_group: CutGroup
        is_only_input = cut_group.is_only_input()
        is_only_constant = cut_group.is_only_constant()
        egg_runner.clean(dirname)
        os.makedirs(dirname, exist_ok=True)
        # Only do inference from the begin index.
        # But we still need to add the econditions after this if statement.

        rich.print(cut_group)

        if begin_names is None:
            begin_names = self.conditioned_names
        with graph_merge_transform_lock:
            origin_sgraph, target_sgraphs = cut_group.extract_sgraphs(begin_names)

        if egg_runner.visualize:
            visualize_sgraph_to_infer(origin_sgraph, target_sgraphs, dirname)
        sgraphs = [origin_sgraph] + target_sgraphs
        if type(sgraphs) is SGraph:
            sgraphs = [sgraphs]

        # -------------------------------------------------------------------------------------------------
        # 1. Save precondition
        # 1.0. Collect the names of related sexprs.
        sexpr_names_set = set()
        if additional_input_names:
            sexpr_names_set.update(additional_input_names)
        scalar_sexprs = [
            s
            for s in chain(
                *[sg.get_scalar_sexprs() + sg.get_shape_scalars() for sg in sgraphs]
            )
            if s.name is not None
        ]
        scalar_names = set()
        for s in scalar_sexprs:
            scalar_names.add(s.name)
            symbols = s.sym_expr.free_symbols
            scalar_names.update([s.name for s in symbols])
        sexpr_names_set.update(scalar_names)
        input_sexprs = set(chain(*[sg.get_input_sexprs() for sg in sgraphs]))
        output_sexprs = set(chain(*[sg.outputs for sg in sgraphs]))
        constant_sexprs = set(chain(*[sg.get_constant_sexprs() for sg in sgraphs]))
        sexpr_names_set.update([s.name for s in input_sexprs])
        sexpr_names_set.update([s.name for s in output_sexprs])
        sexpr_names_set.update([s.name for s in constant_sexprs])
        print(f"{BRI}Econdition leaves names:{RST}")
        print("Input leaves:", end="")
        rich.print(sorted([s.name for s in input_sexprs]))
        print("Scalar leaves:", end="")
        rich.print(list(set([n for n in scalar_names])))
        print("Constant leaves:", end="")
        rich.print(sorted([s.name for s in constant_sexprs]))
        print("Output leaves:", end="")
        rich.print(sorted([s.name for s in output_sexprs]))
        # -------------------------------------------------------------------------------------------------
        # 1. Save cs and cd.
        # Note that it is possible the outputs were changed
        output_replacements: dict[str, str] = {}
        try:
            with graph_merge_transform_lock:
                # 1.1. Save cs
                out_re, tsf_origin_sgraph = save_merged(
                    [origin_sgraph], osp.join(dirname, "cs.sexpr"), return_merged=True
                )
                output_replacements.update(out_re)
                # 1.2. Save cd
                out_re, tsf_target_sgraph = save_merged(
                    target_sgraphs, osp.join(dirname, "cd.sexpr"), return_merged=True
                )
            output_replacements.update(out_re)
            # 1.3. Save shapes
            save_shapes(
                chain(
                    tsf_origin_sgraph.sexprs,
                    tsf_target_sgraph.sexprs,
                    [
                        self.name_to_sexpr[n]
                        for n in sexpr_names_set
                        if n not in scalar_names
                    ],
                ),
                osp.join(dirname, "shapes.json"),
            )
        except CannotMatchAllDistOps as e:
            if not egg_runner.visualize:
                visualize_sgraph_to_infer(origin_sgraph, target_sgraphs, dirname)
            raise e

        # -------------------------------------------------------------------------------------------------
        # 2.0. Save intermediate names
        if intermediate_name:
            path = osp.join(dirname, "precondition.sexpr")
            tsf_origin_sgraph.save(path, True)
            tsf_target_sgraph.save(path, True)
        # 2.1. Save preconditions
        used_econditions = (
            self.econditions if raw_econditions is None else raw_econditions
        )
        used_econditions = chain(used_econditions, self.scalar_econditions)
        econditions, scalar_econditions = self.filter_econditions(
            used_econditions,
            sexpr_names_set,
            output_replacements,
            split_scalar=True,
        )
        self.save_preconditions(econditions, dirname)
        # -------------------------------------------------------------------------------------------------
        # 2.2. Save named constants (Sometimes we want to specify constants' conditions)
        self.save_constant_preconditions(constant_sexprs, dirname)
        # -------------------------------------------------------------------------------------------------
        # 2.3. Save scalar conditions
        self.save_scalar_conditions(self.scalar_econditions, dirname, sexpr_names_set)
        print()

        # -------------------------------------------------------------------------------------------------
        # 3. Save replacements
        with open(osp.join(dirname, "replacements.py"), "w") as f:
            f.write(str(output_replacements))

        # -------------------------------------------------------------------------------------------------
        # 4. Save SubInferInfo if requested.
        if self.save_group:
            sub_infer_info = SubInferInfo(
                origin_sgraph, target_sgraphs, self.econditions, self.scalar_econditions
            )
            info_path = osp.join(dirname, "sub_infer_info.pkl")
            sub_infer_info.save(info_path)
            print(f"{BGREEN}Saved SubInferInfo into {info_path}{RST}")

        # -------------------------------------------------------------------------------------------------
        # 5. Skip if only input.
        if is_only_input or is_only_constant or assume_provided:
            rich.print("\n".join(str(cut_group).split("\n")[:3]))
            self.egg_runner.copy_pre_as_post(dirname)
            if len(econditions) == 0:
                print(f"{BRED}Skipped{RST}: Not provided.")
            else:
                print(
                    f"{BGREEN}Skipped{RST}: Found {len(econditions)}, assuming {BRI}Provieded{RST}."
                )
            return

        egg_runner.upload(dirname)
        # -------------------------------------------------------------------------------------------------
        if not is_pass_through:
            # 6.1. Invoke Egg to infer postcondition.
            run_begin = datetime.now()
            print(f"{BGREEN}Started at {run_begin}{RST}")
            egg_runner.run(dirname, tmux_window_id=f"{group_name}", mode="infer")

            print(f"Saturation and Extraction done in {datetime.now() - run_begin}")
            egg_runner.download(dirname)
            try:
                self.replace_back_postcondition(
                    sgraphs,
                    [tsf_origin_sgraph, tsf_target_sgraph],
                    output_replacements,
                    dirname,
                )
            except FileNotFoundError as e:
                print(
                    f"{BRED}Result file wasn't found. Please check the log {osp.join(dirname, 'output.log')}{RST}"
                )
                raise e
            # Also print the group here because if might be flooded by debug outputs.
            rich.print(cut_group)
        else:
            # 6.2. Do the pass-through, call egg runner if we need visualization.
            # Pass through the preconditions and computations as postconditions.
            egg_runner.copy_pre_as_post(dirname)
            print(f"{BYELLOW}Pass through for postcondition in {group_name}.{RST}")

        return output_replacements

    def post_run_process(
        self,
        cut_group: CutGroup,
        is_pass_through: bool,
        dirname: str,
    ) -> Optional[ECondition]:
        if (
            not cut_group.is_only_input()
            and not cut_group.is_only_constant()
            and not is_pass_through
        ):
            # 1. Precondition for inputs will be added before `run` using `add_econdition`
            # 2. Only-constant econditions will also be added before.
            # 3. Pass-through cut groups doesn't generate new conditions.
            # So we only need to handle non-input and non-pass-through cut groups here.
            postcondition_str = self.egg_runner.get_postcondition(dirname)
            postcondition = ECondition.from_str(postcondition_str)
            self.add_econdition(postcondition)
        else:
            postcondition = None

        if not is_pass_through:
            self.conditioned_names.update([c.name for c in cut_group.cuts])

        return postcondition

    def check_impl(self, root_dirname, expected_list: list[SExprECondition]):
        egg_runner = self.egg_runner
        check_impl_dirname = osp.join(root_dirname, "check_impl")

        begin_date = datetime.now()
        print_ft(f"{BRI}Final check_impl{RST}")

        for idx, expected in enumerate(expected_list):
            this_begin_date = datetime.now()
            group_name = f"expected{idx}"
            dirname = osp.join(check_impl_dirname, group_name)
            os.makedirs(dirname, exist_ok=True)

            origin_cut = None
            target_cuts = []
            for s in expected.inputs:
                if self.name_to_sgraph[s.name] == self.origin_sgraph:
                    assert origin_cut is None, "Multiple origin cuts found."
                    origin_cut = self.name_to_sexpr[s.name]
                else:
                    target_cuts.append(self.name_to_sexpr[s.name])
            cut_group = CutGroup(origin_cut, target_cuts)
            # `run_one` will add the preconditions because sexprs in `cur_group` are
            # already conditioned.
            additional_econditions = self.origin_to_econditions.get(
                origin_cut.name, set()
            )
            additional_input_names = set(
                chain(*[econd.input_names for econd in additional_econditions])
            )
            # print(additional_input_names)
            output_replacement = self.run_one(
                dirname,
                cut_group,
                True,
                group_name,
                additional_input_names=additional_input_names,
                raw_econditions=self.econditions + list(additional_econditions),
            )

            # Save the expected post condition.
            print(f"{BRI}Checking expected:{RST}")
            rich.print(expected)
            save_expected(dirname, expected, output_replacement)
            egg_runner.upload(dirname)
            egg_runner.run(dirname, group_name, mode="check_impl")
            egg_runner.download(dirname, postcondition=False)
            print(f"\n{BGREEN}Check Implication Succeed! {RST}")
            print_ft(
                f"Done with {BRI}Final check_impl{RST} in {datetime.now() - this_begin_date}"
            )
            print()
        print_ft(
            f"Done all {BRI}Final check_impl{RST} in {datetime.now() - begin_date}"
        )
