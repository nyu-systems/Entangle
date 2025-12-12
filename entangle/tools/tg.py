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
import importlib
import json
import logging
import os
import os.path as osp
import pickle
import random
import re
import shutil
from datetime import datetime
import tempfile
from typing import Callable, Sequence, Type

import numpy as np
import rich
from tqdm import tqdm

from entangle.convert.convert import pgraph_to_sgraph
from entangle.pgraph.pickleable import PickleableGraph, load_pgraph
from entangle.sgraph.sgraph import SGraph
from entangle.tools.egg import EggRunner
from entangle.tools.utils import dot_exists
from entangle.utils import visual_utils
from entangle.utils.print_utils import BGREEN, BRED, BRI, BYELLOW, RST

random.seed(123456)
np.random.seed(123456)


parser = argparse.ArgumentParser(
    description="entangle: a tool for graph dumping and verification of PyTorch models."
)
parser.add_argument(
    "--disable_rich", "--poor", action="store_true", help="Disable rich printing."
)
parser.add_argument("--log", help="Logging level", type=str, default="INFO")
parser.add_argument(
    "--log_path", help="Logging path.", type=str, default=osp.join(tempfile.gettempdir(), "entangle.log")
)

subparsers = parser.add_subparsers(
    help="Sub-commands for entangle", dest="subcommand"
)

###################################################################################################
# subparser for `test`
###################################################################################################
parser_test = subparsers.add_parser("test", help="test help")
parser_test.add_argument(
    "test_module",
    help=f"Abosulte test module path, e.g. `tests.test_transformer`",
    type=str,
)
parser_test.add_argument(
    "-m",
    "--method",
    help=f"method to dump graph: run|jit.trace|dynamo",
    type=str,
    default="dynamo",
)
parser_test.add_argument(
    "-o", "--output", help=f"Output dirname", type=str, default=tempfile.gettempdir()
)
parser_test.add_argument("-d", "--draw", help=f"Draw graph", action="store_true")
parser_test.add_argument("--dynamo_log", help=f"Dynamo logs", action="store_true")
parser_test.add_argument(
    "--dsp_size", help=f"Distributed sequence parallelism Size", type=int
)
parser_test.add_argument("--tp_size", help=f"Tensor parallelism size", type=int)

###################################################################################################
# subparser for `texify`
###################################################################################################
parser_textify = subparsers.add_parser("textify", help="Texify a gpickle file(s)")
parser_textify.add_argument(
    "input_path", help=f"Input file name or directory path", type=str
)
parser_textify.add_argument("-o", "--output", help=f"Output dirname", type=str)

###################################################################################################
# subparser for `picklize`
###################################################################################################
parser_picklize = subparsers.add_parser("picklize", help="Generate gpickle from text.")
parser_picklize.add_argument(
    "input_path", help=f"Input file name or directory path", type=str
)
parser_picklize.add_argument("-o", "--output", help=f"Output dirname", type=str)

###################################################################################################
# subparser for `visualize`
###################################################################################################
parser_textify = subparsers.add_parser("visualize", help="Visualize a gpickle file(s)")
parser_textify.add_argument(
    "input_path", help=f"Input file name or directory path", type=str
)
parser_textify.add_argument("-o", "--output", help=f"Output dirname", type=str)

###################################################################################################
# subparser for `infer`
###################################################################################################
parser_infer = subparsers.add_parser(
    "infer", help="Infer post-conditions using e-graph."
)
parser_infer.add_argument(
    "--cache",
    action="store_true",
    help=f"If set, load sgraph from cache.",
)
parser_infer.add_argument(
    "-g",
    "--graph_path",
    required=True,
    help=f"This can be either 1). the path to graphs, i.e., the root dirname of dumped gpickles, containing origin and target; or 2). the path to a SubInferInfo pickle file.",
    type=str,
)
parser_infer.add_argument(
    "--origin",
    help=f"Origin graphs sub-dirname under specified `graph_path`.",
    type=str,
    default="origin",
)
parser_infer.add_argument(
    "--target",
    help=f"Target graphs sub-dirname under specified `graph_path`.",
    type=str,
    default="target",
)
parser_infer.add_argument(
    "-i",
    "--config_module",
    required=True,
    help=f"Python file that contains a class inheriting from `entangle.tools.config.Config`.",
    type=str,
)
parser_infer.add_argument(
    "--graph_prefix",
    help='Graph file name prefix. E.g., for "bw.g0.r2.gpickle", the prefix is "bw.g0"',
    default="",
    type=str,
)
parser_infer.add_argument(
    "--focus",
    help=f"Focus on a specific group id. This will remove all other unrelated nodes.",
    nargs="+",
    type=str,
    default=[],
)
parser_infer.add_argument(
    "--infer_manager",
    help=f"Type of InferManager to use. For `sskeleton`, use `Config`; for the other two, use `ExplorativeConfig`.",
    choices=["sskeleton", "explorative", "greedy"],
    default="sskeleton",
)
parser_infer.add_argument(
    "--post_type",
    type=str,
    default="candidates",
    choices=["candidates", "representative"],
    help="Post-condition type, can be candidates|representative, yis is buggy now, use candidates.",
)
####################
# Explorative only
####################
# Constructor parameters
parser_infer.add_argument(
    "--resume",
    help=f"Resume infer manager (use group id)",
    type=int,
)
parser_infer.add_argument(
    "--way",
    help=f"When finding target outputs, use intersection or union for potential sets. Union is safer while intersection might be faster.",
    type=str,
    choices=["intersection", "union"],
    default="union",
)
# Run parameters
parser_infer.add_argument(
    "--explore_limit",
    help=f"Explore step limit.",
    type=int,
    default=4,
)
parser_infer.add_argument(
    "-j",
    "--max_worker",
    help=f"Max number of workers to explore.",
    type=int,
    default=4,
)
####################
# SSkeleton only
####################
parser_infer.add_argument(
    "--begin", help=f"Begin from which sgroup.", default=0, type=int
)
parser_infer.add_argument(
    "--end",
    help=f"Stop at which sgroup id (excluding this group).",
    default=-1,
    type=int,
)
parser_infer.add_argument(
    "--through",
    help=f"Allow directly pass the preconditions and computations through as postconditions. Should be a list of group id.",
    nargs="+",
    type=int,
    default=[],
)
parser_infer.add_argument("-o", "--output", help=f"Output dirname", type=str)
parser_infer.add_argument(
    "-t",
    "--egg_dirname",
    type=str,
    default="../egger",
    help=f"The directory to the root of egg.",
)
parser_infer.add_argument(
    "-d",
    "--egg_data_dirname",
    type=str,
    help=f"The temporary dirname for data like conditions and computations used by egg.",
)
parser_infer.add_argument(
    "--save_group",
    help="Enable saving a group (sub-graphs) for ease of debug.",
    action="store_true",
)
parser_infer.add_argument(
    "--inverse_lemma", help="Use inverse lemmas.", action="store_true"
)
parser_infer.add_argument("--debug", help="Use debug version Egg.", action="store_true")
parser_infer.add_argument("--tmux", help="Use tmux.", action="store_true")
parser_infer.add_argument("--verbose", help="Verbose.", action="store_true")
parser_infer.add_argument("--visualize", help="Visualize.", action="store_true")
parser_infer.add_argument("--stats", help="Statistics.", action="store_true")


###################################################################################################
# Test
###################################################################################################
def identity_or_get_first(obj):
    return obj if type(obj) not in (tuple, list) else obj[0]


def test(args: argparse.Namespace):
    import torch
    import torch.distributed as dist

    import entangle
    from entangle.graph.dynamo import GraphCollector

    test_pymodule = importlib.import_module(args.test_module)
    setup_test: callable = test_pymodule.setup_test
    model, input_args, input_kwargs, local_variables = setup_test(args)
    is_dist: bool = test_pymodule.IS_DIST
    if is_dist:
        rank = dist.get_rank()
    else:
        rank = 0
    os.environ["MY_RANK"] = str(rank)

    if rank == 0:
        rich.print(model)
    if args.method == "run":
        res = identity_or_get_first(model(*input_args, **input_kwargs)).mean()
        res.backward()
    elif args.method == "jit.trace":
        entangle.graph.trace.trace_and_dump(
            model, input_args, dirname=args.output, rank=rank
        )
    elif args.method == "dynamo":
        import logging

        # optimizer = get_megatron_optimizer([model], None, None, 1.0)
        import torch._dynamo as torchdynamo

        if rank == 0 and args.dynamo_log:
            torch._logging.set_logs(
                dynamo=logging.DEBUG, aot=logging.DEBUG, graph_code=True
            )
        collector = GraphCollector(
            formats=["code"], distributed=rank is not None, rank=rank, draw=False
        )
        # @torchdynamo.optimize(backend=collector.make_backend(GraphCollector.Direction.FW))
        # @torchdynamo.optimize("eager", dynamic=False)
        # def fn(model):
        #     res = model(*input_args, **input_kwargs)
        #     if type(res) is tuple:
        #         res = res[0].mean()
        #     else:
        #         res = res.mean()
        # res.backward()
        # args = get_args()
        # optimizer.reduce_model_grads(args, timers=None)
        # optimizer.step(args, timers=None)

        # fn(model)
        # exit(0)

        def fn(model):
            res = identity_or_get_first(model(*input_args, **input_kwargs)).mean()
            res.backward()

        collector, compiled_model, default_backend_model = (
            entangle.graph.dynamo.dynamo_and_dump(
                model,
                fn,
                dirname=args.output,
                formats=["code"],
                logs=args.dynamo_log,
                rank=rank,
                draw=args.draw,
            )
        )
        if rank == 0:
            for g in collector.pickleable_graphs():
                rich.print(f"------------")
                rich.print(g)

        # Test compiled.
        result = identity_or_get_first(model(*input_args, **input_kwargs))
        compiled_result = identity_or_get_first(
            default_backend_model(*input_args, **input_kwargs)
        )
        if not torch.allclose(result, compiled_result, rtol=0.0, atol=0.0):
            print(
                f"Error: results not close.\n{result=}\n{compiled_result=}", flush=True
            )
        else:
            print(f"[rank {rank}] Results are close", flush=True)
    else:
        raise NotImplementedError(
            f"Unknown mode: {args.method}, only supports jit.trace|dynamo"
        )
    if is_dist:
        dist.destroy_process_group()
    print(f"[rank={rank}] Exiting normally.")


###################################################################################################
# Textify
###################################################################################################
def textify_file(input_path: str, output_path: str):
    print(f"{input_path} ---> {output_path}")
    output_dir = osp.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    graph = load_pgraph(input_path, force_no_lift=True)
    with open(output_path, "w") as out_f:
        out_f.write(str(graph))
        out_f.write("\n\n\n------------------------\n\n\n")


def textify(args: argparse.Namespace):
    rich.print(f"Textifying {args.input_path}")
    if osp.isdir(args.input_path):
        input_dir = args.input_path
        output_dir = args.output if args.output is not None else input_dir
        rich.print(f"Outputing to {output_dir}")
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                if not filename.endswith(".gpickle"):
                    continue
                input_path = osp.join(dirpath, filename)
                output_path = osp.join(
                    output_dir, osp.relpath(input_path, start=input_dir)
                )
                output_path = output_path.removesuffix(".gpickle") + ".log"
                textify_file(input_path, output_path)
    else:
        output_dir = (
            args.output if args.output is not None else osp.dirname(args.input_path)
        )
        rich.print(f"Outputing to {output_dir}")
        output_path = osp.join(
            output_dir, osp.basename(args.input_path.removesuffix(".gpickle") + ".log")
        )
        textify_file(args.input_path, output_path)


###################################################################################################
# Picklize
###################################################################################################
def picklize_file(input_path: str, output_path: str):
    print(f"{input_path} ---> {output_path}")
    output_dir = osp.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(input_path, "r") as f:
        text_graph = f.read().split("\n\n\n------------------------\n\n\n")
        pgraph = PickleableGraph.from_text(text_graph)
    with open(output_path, "wb") as f:
        pickle.dump(pgraph, f, pickle.HIGHEST_PROTOCOL)


def picklize(args: argparse.Namespace):
    rich.print(f"Textifying {args.input_path}")
    if osp.isdir(args.input_path):
        input_dir = args.input_path
        output_dir = args.output if args.output is not None else input_dir
        rich.print(f"Outputing to {output_dir}")
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                if not filename.endswith(".log"):
                    continue
                input_path = osp.join(dirpath, filename)
                output_path = osp.join(
                    output_dir, osp.relpath(input_path, start=input_dir)
                )
                output_path = output_path.removesuffix(".log") + ".gpickle"
                picklize_file(input_path, output_path)
    else:
        output_dir = (
            args.output if args.output is not None else osp.dirname(args.input_path)
        )
        rich.print(f"Outputing to {output_dir}")
        output_path = osp.join(
            output_dir, osp.basename(args.input_path.removesuffix(".log") + ".gpickle")
        )
        picklize_file(args.input_path, output_path)


###################################################################################################
# Visualize
###################################################################################################
def visualize_file(input_path: str, output_path_prefix: str):
    output_dir = osp.dirname(output_path_prefix)
    os.makedirs(output_dir, exist_ok=True)
    pg: PickleableGraph = load_pgraph(input_path)
    import entangle.sgraph.visualization as sgraph_viz

    sgraph = pgraph_to_sgraph(pg)
    nx_graph = sgraph.nx_graph
    output_path = output_path_prefix + f".{pg.direction}.{pg.gid}.html"
    print(f"{input_path} ---> {output_path}")
    graph_to_draw = sgraph_viz.setup(nx_graph)
    visual_utils.draw_pyvis(graph_to_draw, output_path=output_path, reverse=True)


def visualize(args: argparse.Namespace):
    rich.print(f"Visualizing {args.input_path}")
    if osp.isdir(args.input_path):
        input_dir = args.input_path
        output_dir = args.output if args.output is not None else input_dir
        rich.print(f"Outputing to {output_dir}")
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for filename in filenames:
                if not filename.endswith(".gpickle"):
                    continue
                input_path = osp.join(dirpath, filename)
                output_path = osp.join(
                    output_dir, osp.relpath(input_path, start=input_dir)
                )
                output_path = output_path.removesuffix(".gpickle")
                visualize_file(input_path, output_path)
    else:
        output_dir = (
            args.output if args.output is not None else osp.dirname(args.input_path)
        )
        rich.print(f"Outputing to {output_dir}")
        output_path = osp.join(
            output_dir, osp.basename(args.input_path.removesuffix(".gpickle"))
        )
        visualize_file(args.input_path, output_path)


###################################################################################################
# Infer
###################################################################################################
def build_egg_runner(args: argparse.Namespace) -> EggRunner:
    return EggRunner(
        egg_dirname=args.egg_dirname,
        egg_data_dirname=args.egg_data_dirname,
        inverse_lemma=args.inverse_lemma,
        post_type=args.post_type,
        log_level=args.log,
        debug=args.debug,
        tmux=args.tmux,
        verbose=args.verbose,
        visualize=args.visualize,
        stats=args.stats,
        use_local_directory=True,
    )


def load_sgraph(
    path: str,
    name_prefix: str,
    get_lift_fresh_copy_constant_value: Callable[[str], int] = None,
    filter_output: Callable[[list], list] = None,
) -> SGraph:
    """
    `path`: a path to either pgraph or sgraph.
        for pgraph, the ext must be ".gpickle"
        for sgraph, the ext must be ".sgraph"
    """
    if path.endswith(".gpickle"):
        return pgraph_to_sgraph(
            load_pgraph(
                path,
                get_lift_fresh_copy_constant_value=get_lift_fresh_copy_constant_value,
            ),
            name_prefix,
            filter_output=filter_output,
        )
    elif path.endswith(".sgraph"):
        return pickle.load(open(path, "rb"))
    else:
        raise RuntimeError(f"Unknown file type: {path}")


def make_run_name(args: argparse.Namespace) -> str:
    return f"precondition.{osp.basename(args.graph_path)}.{args.graph_prefix}.{args.origin}-{args.target}.{args.infer_manager}"


def infer(args: argparse.Namespace, unknown_args: Sequence[str]):
    import entangle as tg
    from entangle.sgraph.egraph import ECondition, SExprECondition
    from entangle.sgraph.infer import (
        ExplorativeInferManager,
        GreedyExplorativeInferManager,
        SSkeletonInferManager,
        SubInferInfo,
    )
    from entangle.sgraph.sgraph import SGraph
    from entangle.sgraph.transform import SGraphTransformer
    from entangle.sgraph.visualization import visualize_sgraph_to_infer
    from entangle.sym.sym_manager import SymManager
    from entangle.tools.config import Config, ExplorativeConfig, load_config_class
    from entangle.utils.module_utils import load_module

    os.system("pkill dot")

    # Load the module, so that ops and mapping defined outside can be used.
    config_module = load_module(args.config_module, "config_module")
    ConfigClass: Type[Config] = load_config_class(config_module)
    config: Config = ConfigClass(unknown_args)
    SymManager.setup_sym_instantiation(config.get_symval_instantiation())
    SGraphTransformer.oracle_group_id_to_size = config.oracle_group_id_to_size

    graph_prefix = args.graph_prefix
    run_name = make_run_name(args)
    cache_dirname = osp.join("tgcache", run_name)

    if not args.cache:
        # 1. Load the computational graphs from; or load SubInferInfo
        if osp.isfile(args.graph_path):
            sub_infer_info = SubInferInfo.load(args.graph_path)
            assert (
                type(sub_infer_info) == SubInferInfo
            ), f"Expect SubInferInfo, got {sub_infer_info}"
            if args.infer_manager == "explorative":
                raise RuntimeError(
                    f"Cannot use ExplorativeInferManager (--infer_manager explorative) with SubInferInfo. "
                    f"Please use SSkeletonInferManager (--infer_manager sskeleton) instead."
                )
            origin_sgraph = sub_infer_info.origin_sgraph
            target_sgraphs = sub_infer_info.target_sgraphs
            if args.output is None:
                args.output = osp.join(osp.dirname(args.graph_path), "subinfer")
            use_sub_infer_info = True
        else:
            origin_dirname = osp.join(args.graph_path, args.origin)
            origin_filename = [
                f
                for f in os.listdir(origin_dirname)
                if f.startswith(graph_prefix)
                and (f.endswith(".gpickle") or f.endswith(".sgraph"))
            ]
            assert (
                len(origin_filename) == 1
            ), f"Expect exactly one origin, got {origin_filename}"
            origin_sgraph = load_sgraph(
                osp.join(origin_dirname, origin_filename[0]),
                name_prefix="S",
                get_lift_fresh_copy_constant_value=config.get_lift_fresh_copy_constant_value,
                filter_output=config.filter_output,
            )

            target_dirname = osp.join(args.graph_path, args.target)
            gpickle_filenames = sorted(
                [
                    f
                    for f in os.listdir(target_dirname)
                    if f.startswith(graph_prefix)
                    and (f.endswith(".gpickle") or f.endswith(".sgraph"))
                ]
            )
            target_sgraphs = [
                load_sgraph(
                    osp.join(target_dirname, gpickle_filename),
                    name_prefix="D",
                    get_lift_fresh_copy_constant_value=config.get_lift_fresh_copy_constant_value,
                    filter_output=config.filter_output,
                )
                for gpickle_filename in gpickle_filenames
            ]
            if args.output is None:
                args.output = run_name
            use_sub_infer_info = False
        os.makedirs(args.output, exist_ok=True)

        # 2. SGraphs transformations
        # 2.1. Focus on specific nodes if required.
        if args.focus:
            assert (
                len(args.focus) == len(target_sgraphs) + 1
            ), "Should provide the same number of focus node as the number of target graphs + 1 (for origin graph)."

            def focus(sgraph: SGraph, focus_name: str) -> SGraph:
                for output in sgraph.outputs:
                    for sexpr in output.post_order_dfs():
                        if sexpr.name == focus_name:
                            return SGraph([sexpr], sgraph.sexpr_order)
                raise RuntimeError(
                    f"Cannot find the focus node {focus_name} in the graph."
                )

            origin_sgraph = focus(origin_sgraph, args.focus[0])
            target_sgraphs = [
                focus(sg, focus_name)
                for sg, focus_name in zip(target_sgraphs, args.focus[1:])
            ]

        # 2.2. Simplify graph by removing clones and duplicated dist/wait.
        def simplify(sgraph: SGraph):
            return (
                SGraphTransformer(sgraph)
                .add_dist_wait_if_not_exists()
                .merge_duplicated_dist_wait()
                .merge_clones()
                .rebuild_sepxrs()
                .to_sgraph()
            )

        origin_sgraph = simplify(origin_sgraph)
        target_sgraphs = [simplify(sg) for sg in target_sgraphs]

        # 2.3. Force leaf
        def force_leaf(sgraph: SGraph):
            return (
                SGraphTransformer(sgraph)
                .force_leaf(config.get_force_leaf_set())
                .to_sgraph()
            )

        origin_sgraph = force_leaf(origin_sgraph)
        target_sgraphs = [force_leaf(sg) for sg in target_sgraphs]

        # 2.4. Mark the dist ops.
        SGraphTransformer(sgraphs=[origin_sgraph]).merge_dist_ops(just_mark=True)
        SGraphTransformer(sgraphs=target_sgraphs).merge_dist_ops(just_mark=True)

        # 2.5. Visualize the sgraphs
        visualize_sgraph_to_infer(
            origin_sgraph, target_sgraphs, args.output, graph_prefix=graph_prefix
        )

        # 2.6. Save the sgraphs
        shutil.rmtree(cache_dirname, ignore_errors=True)
        os.makedirs(cache_dirname, exist_ok=True)
        with open(osp.join(cache_dirname, "origin.sgraph"), "wb") as f:
            pickle.dump(origin_sgraph, f)
        for i, sgraph in enumerate(target_sgraphs):
            with open(osp.join(cache_dirname, f"target{i}.sgraph"), "wb") as f:
                pickle.dump(sgraph, f)
        tg.save_global_states(osp.join(cache_dirname, "global_states.pkl"))
        print(f"{BGREEN}Saved sgraphs to {cache_dirname}{RST}")
    else:
        print(f"{BGREEN}Loading sgraphs from {cache_dirname}{RST}")
        origin_path = osp.join(cache_dirname, "origin.sgraph")
        origin_sgraph = pickle.load(open(origin_path, "rb"))
        target_filenames = [
            f for f in os.listdir(cache_dirname) if re.match(r"target\d+.sgraph", f)
        ]
        target_sgraphs = [
            pickle.load(open(osp.join(cache_dirname, target_filename), "rb"))
            for target_filename in target_filenames
        ]
        if args.output is None:
            args.output = run_name
        tg.load_and_resume_global_states(osp.join(cache_dirname, "global_states.pkl"))

    name_to_sexpr = {}
    for sgraph in [origin_sgraph, *target_sgraphs]:
        name_to_sexpr.update(sgraph.name_to_sexpr)

    for sgraph in [origin_sgraph, *target_sgraphs]:
        name_to_sexpr.update(sgraph.name_to_sexpr)

    # Save graph stats information.
    path = osp.join(args.output, "graph_stats.json")
    stats = []
    for sgraph in [origin_sgraph, *target_sgraphs]:
        stats.append(sgraph.get_stats().to_dict())
    with open(path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"{BGREEN}Graph Stats dumped into {path}{RST}")
    ################### pre-processing done ###################

    ################# start running inference #################
    # 3. Create config for this run.
    config.set_get_sexpr(lambda x: name_to_sexpr[x])

    # 4. Create `EggRunner` to execute Egg.
    egg_runner = build_egg_runner(args)
    # We put begin here because building egg_runner can take a while, but it is actually a one-time thing.
    begin = datetime.now()

    # 5. Get cut group if any.
    try:
        cut_groups = config.get_cut_groups(origin_sgraph, target_sgraphs)
    except AssertionError as e:
        raise RuntimeError(
            f"Got error when trying to get_cut_groups. If you believe there "
            f"is a problem of the used cutting strategy, try to implement "
            f"your own `Config.get_cut_groups` method. The error is {e}"
        )

    if args.infer_manager == "sskeleton":
        print(f"{BRI}Using InferManager: SSkeletonInferManager{RST}")
        assert not issubclass(
            ConfigClass, ExplorativeConfig
        ), f"Cannot use ExplorativeConfig for SSkeletonInferManager."

        # 6. Create `InferManager` to infer post conditions.
        infer_manager = SSkeletonInferManager(
            origin_sgraph,
            target_sgraphs,
            cut_groups,
            egg_runner,
            save_group=args.save_group,
        )
        input_sexpr_econdition_list: list[SExprECondition] = (
            config.build_preconditions()
        )
        for sexpr_econdition in input_sexpr_econdition_list:
            econdition = ECondition.from_sexpr_econdition(sexpr_econdition)
            infer_manager.add_econdition(econdition, check_self_provable=False)

        if use_sub_infer_info:
            for cond in sub_infer_info.econditions:
                infer_manager.add_econdition(cond, check_self_provable=False)
            for cond in sub_infer_info.scalar_econditions:
                infer_manager.add_econdition(cond, check_self_provable=False)
        infer_manager.precompute_all_scalar_conditions()

        # 7. Actually run the inference group-by-group.
        begin_infer = datetime.now()
        try:
            infer_manager.run(
                root_dirname=args.output,
                begin=args.begin,
                end=args.end,
                through=set(args.through),
            )
        finally:
            elapsed = datetime.now() - begin_infer
            print(f"{BRI}Time Elapased for Inference: {elapsed}{RST}\n")
    elif args.infer_manager == "explorative":
        if not isinstance(config, ExplorativeConfig):
            raise RuntimeError(
                f"Must use ExplorativeConfig for explorative mode, "
                f"got {type(config)}, please check {args.config_module}"
            )
        way = set.intersection if args.way == "intersection" else set.union

        # 6. Create `InferManager` to infer post conditions.
        infer_manager = ExplorativeInferManager(
            origin_sgraph,
            target_sgraphs,
            cut_groups,
            egg_runner,
            args.save_group,
            through_sexpr_cb=config.get_through_sexpr_cb(),
            way=way,
        )
        if args.resume is not None:
            resume_path = f"{args.output}/group{args.resume}/checkpoint.pkl"
            with open(resume_path, "rb") as f:
                resume_infer_manager: ExplorativeInferManager = pickle.load(f)
            infer_manager.resume(resume_infer_manager)
        else:
            input_sexpr_econdition_list: list[SExprECondition] = (
                config.build_preconditions()
            )
            for sexpr_econdition in input_sexpr_econdition_list:
                econdition = ECondition.from_sexpr_econdition(sexpr_econdition)
                infer_manager.add_econdition(econdition, check_self_provable=False)
                infer_manager.map_origin_to_targets(econdition)
        infer_manager.precompute_all_scalar_conditions()

        # 7. Actually run the inference group-by-group.
        begin_infer = datetime.now()
        try:
            infer_manager.run(
                root_dirname=args.output,
                begin=args.resume,
                explore_limit=args.explore_limit,
                hint_ops=config.get_hint_ops(),
                max_workers=args.max_worker,
            )
        finally:
            elapsed = datetime.now() - begin_infer
            print(f"{BRI}Time Elapased for Inference: {elapsed}{RST}\n")
    else:
        assert args.infer_manager == "greedy"
        if not isinstance(config, ExplorativeConfig):
            raise RuntimeError(
                f"Must use ExplorativeConfig for explorative mode, "
                f"got {type(config)}, please check {args.config_module}"
            )
        way = set.intersection if args.way == "intersection" else set.union

        # 6. Create `InferManager` to infer post conditions.
        infer_manager = GreedyExplorativeInferManager(
            origin_sgraph,
            target_sgraphs,
            cut_groups,
            egg_runner,
            args.save_group,
            through_sexpr_cb=config.get_through_sexpr_cb(),
            way=way,
        )
        if args.resume is not None:
            resume_path = f"{args.output}/group{args.resume}/checkpoint.pkl"
            with open(resume_path, "rb") as f:
                resume_infer_manager: GreedyExplorativeInferManager = pickle.load(f)
            infer_manager.resume(resume_infer_manager)
        else:
            print(f"{BGREEN}Building Preconditions...{RST}")
            input_sexpr_econdition_list: list[SExprECondition] = (
                config.build_preconditions()
            )
            print(f"{BGREEN}Loading preconditions...{RST}")
            for sexpr_econdition in tqdm(input_sexpr_econdition_list, leave=False):
                econdition = ECondition.from_sexpr_econdition(sexpr_econdition)
                infer_manager.add_econdition(econdition, check_self_provable=False)
                infer_manager.map_origin_to_targets(econdition)
        infer_manager.precompute_all_scalar_conditions()

        # 7. Actually run the inference group-by-group.
        begin_infer = datetime.now()
        try:
            infer_manager.run(
                root_dirname=args.output,
                begin=args.resume,
                explore_limit=args.explore_limit,
                hint_ops=config.get_hint_ops(),
                max_workers=args.max_worker,
            )
        finally:
            elapsed = datetime.now() - begin_infer
            print(f"{BRI}Time Elapased for Inference: {elapsed}{RST}\n")

    # 8. Check final post condition if available.
    expected_list = config.build_expected()
    if len(expected_list) == 0:
        print(
            f"{BYELLOW}Warning: No user expectation provided. Just inferred post-conditions.{RST}"
        )
    infer_manager.check_impl(args.output, expected_list)

    now = datetime.now()
    print(f"Exiting normally. \nEverything done in {now - begin}.")


def main():
    args, unknown_args = parser.parse_known_args()

    args.log = logging._nameToLevel[args.log.upper()]
    if args.disable_rich:
        rich_print_backup = rich.print
        rich.print = print

    if args.subcommand == "test":
        test(args)
    elif args.subcommand == "textify":
        textify(args)
    elif args.subcommand == "picklize":
        picklize(args)
    elif args.subcommand == "visualize":
        visualize(args)
    elif args.subcommand == "infer":
        args.graph_path = osp.normpath(args.graph_path)
        if args.visualize:
            assert (
                dot_exists()
            ), "Please install graphviz and make sure `dot` is in your PATH to use `visualize`"
        assert (
            args.end == -1 or args.begin <= args.end
        ), f"Invalid range: {args.begin=}, {args.end=}"
        infer(args, unknown_args)
        # import cProfile
        # from pstats import SortKey

        # profiler = cProfile.Profile()
        # try:
        #     profiler.runcall(infer, args, unknown_args)
        # finally:
        #     profiler.print_stats(SortKey.CUMULATIVE)
        #     profiler.dump_stats(osp.join(tempfile.gettempdir(), "infer.prof"))
    else:
        assert args.subcommand is None
        parser.print_help()


if __name__ == "__main__":
    main()
