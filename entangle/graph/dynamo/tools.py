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
import os.path as osp
import pickle
import tempfile
from typing import Callable, Tuple

import rich
import torch
import torch._dynamo
import torch.distributed as dist
import torch.distributed._functional_collectives as fdist
from torch.fx.graph_module import GraphModule
from functorch.compile import aot_module_simplified, default_partition, make_boxed_func
from torch.fx.passes.graph_drawer import FxGraphDrawer

from entangle.pgraph.pickleable import *

DYNAMO_SUPPRESS_ERRORS = os.environ.get("DYNAMO_SUPPRESS_ERRORS", "0") != "0"
torch._dynamo.config.suppress_errors = DYNAMO_SUPPRESS_ERRORS
DYNAMO_LOG_LEVEL = logging._nameToLevel[os.environ.get("DYNAMO_LOG_LEVEL", "WARN")]
torch._logging.set_logs(
    dynamo=DYNAMO_LOG_LEVEL,
    aot=DYNAMO_LOG_LEVEL,
    graph_code=DYNAMO_LOG_LEVEL >= logging.DEBUG,
)

HACK_FOR_DYNAMO = os.environ.get("TG_HACK_FOR_DYNAMO", "0") == "1"
USING_DYNAMO = os.environ.get("TG_USING_DYNAMO", "0") == "1"
DYNAMO_TRACING = False

if torch.__version__ >= "2.7":
    torch._dynamo.config.ignore_logger_methods.add(logging.Logger.info)


torch._dynamo.reset()


def get_group_rank(rank: int, group_ranks: list[int]) -> int:
    """
    Get the rank in the group given the global rank and the group ranks.
    """
    if rank not in group_ranks:
        raise ValueError(f"Rank {rank} is not in the group ranks: {group_ranks}")
    return group_ranks.index(rank)


def default_backend(gm, sample_inputs):
    return gm


class GraphCollector:
    class Direction:
        FW = "fw"
        BW = "bw"
        BOTH = "both"

    def __init__(
        self,
        formats: list[str],
        distributed=False,
        rank=None,
        draw: bool = False,
        dirname: str = None,
        single_direction: bool = False,
    ):
        """
        dirname: if provided, we dump the graph as soon as collected.
        """
        self.formats = formats
        self.distributed = distributed
        self.rank = rank
        if self.distributed and self.rank is None:
            rank = dist.get_rank()
        self.graphs = []
        self.gid = 0
        self.draw = draw
        self.dirname = dirname
        self.single_direction = single_direction

    def add_graph(self, gm: GraphModule, gid: int, direction: Direction, save=False):
        readable_str = gm.print_readable(print_output=False)
        attr = {
            "gid": gid,
            "direction": direction,
            "readable_str": readable_str,
        }
        self.graphs.append((gm, attr))
        if self.dirname is not None:
            save_graph(self.dirname, self.rank, gm, attr)

    def print_graph(self, gm: torch.fx.GraphModule, direction: str, gid: int):
        if len(self.formats) > 0:
            print(f"----- {direction} {gid}")
        for f in self.formats:
            if f == "code":
                gm.print_readable()
            elif f == "tabular":
                gm.graph.print_tabular()
            else:
                raise ValueError(f"Unknown format: {format}")

    def make_backend(self, direction: str, increase_gid: bool = False):
        if direction in (GraphCollector.Direction.FW, GraphCollector.Direction.BW):

            def backend(
                gm: torch.fx.GraphModule,
                sample_inputs,
                rank=self.rank,
                formats=None,
                gid=self.gid,
                increase_gid=increase_gid,
            ):
                if self.draw:
                    name = f"{direction}_{gid}"
                    FxGraphDrawer(
                        gm,
                        name=name,
                    ).get_dot_graph().write_svg(osp.join(tempfile.gettempdir(), f"{name}.svg"))
                if formats is None:
                    formats = []
                self.add_graph(gm, gid, direction, save=self.single_direction)
                if rank is None or rank == 0:
                    self.print_graph(gm, direction, gid)
                if increase_gid and direction == "fw":
                    # If under single direction mode, `increase_gid` is True, we increase gid here.
                    # Otherwise, it is BOTH backend that should update gid.
                    self.gid += 1
                if self.single_direction:
                    return gm.forward
                else:
                    return make_boxed_func(gm.forward)

            return backend
        else:
            assert direction in (GraphCollector.Direction.BOTH,)

            def backend(gm: GraphModule, sample_inputs, gid=self.gid):
                fw = self.make_backend(direction="fw", increase_gid=False)
                bw = self.make_backend(direction="bw", increase_gid=False)

                def partition_fn(
                    joint_module: GraphModule,
                    _joint_inputs,
                    *,
                    num_fwd_outputs,
                    gid=self.gid,
                    **kwargs,  # To catch any new arguments
                ) -> Tuple[GraphModule, GraphModule]:
                    if self.draw:
                        name = f"both_{gid}"
                        FxGraphDrawer(
                            gm,
                            name=name,
                        ).get_dot_graph().write_svg(osp.join(tempfile.gettempdir(), f"{name}.svg"))
                    if self.rank is None or self.rank == 0:
                        self.print_graph(joint_module, direction="both", gid=0)
                    self.add_graph(
                        joint_module, gid, GraphCollector.Direction.BOTH, save=True
                    )
                    fwd_module, bwd_module = default_partition(
                        joint_module=joint_module,
                        _joint_inputs=_joint_inputs,
                        num_fwd_outputs=num_fwd_outputs,
                        **kwargs,
                    )
                    return fwd_module, bwd_module

                if increase_gid:
                    self.gid += 1

                gm_forward = aot_module_simplified(
                    gm,
                    sample_inputs,
                    fw_compiler=fw,
                    bw_compiler=bw,
                    partition_fn=partition_fn,
                )
                return gm_forward

            return backend

    def pickleable_graphs(self) -> list[PickleableGraph]:
        pgraphs = []
        for graph in self.graphs:
            gm, attr = graph
            pgraph = to_pickleable_graph(gm.graph, rank=self.rank, **attr)
            pgraphs.append(pgraph)
        return pgraphs


def save_graph(dirname, rank, gm: GraphModule, attr: dict, just_check: bool = False):
    direction = attr["direction"]
    gid = attr["gid"]
    # 1. Save in pt2
    pt2_path = osp.join(dirname, f"{direction}.g{gid}.r{rank}.pt2")
    if just_check and rank == 0:
        print(f"{pt2_path}: exists?={osp.exists(pt2_path)}")
    else:
        torch.save(gm, pt2_path)
        if rank == 0:
            print(f"graph saved to {pt2_path}")

    # 2. Save in gpickle
    pgraph = to_pickleable_graph(gm.graph, rank=rank, **attr)
    # Check if pickleable
    for node in pgraph.nodes:
        try:
            pickle.dumps(node)
        except Exception as e:
            rich.print(node)
            logging.error(
                f"Error when pickling node: {node}\nmeta keys: {list(node.meta.keys())}"
            )
            raise e
    gpickle_path = osp.join(dirname, f"{direction}.g{gid}.r{rank}.gpickle")
    if just_check and rank == 0:
        print(f"{gpickle_path}: exists?={osp.exists(gpickle_path)}")
    else:
        with open(gpickle_path, "wb") as f:
            pickle.dump(pgraph, f, pickle.HIGHEST_PROTOCOL)
        if rank == 0:
            print(f"graph saved to {gpickle_path}")
    # 3. Save readable
    readable_path = osp.join(dirname, f"{direction}.g{gid}.r{rank}.py")
    if just_check and rank == 0:
        print(f"{readable_path}: exists?={osp.exists(readable_path)}")
    else:
        with open(readable_path, "w") as f:
            # Write some prologue so that editors can recognize and parse this file in Python well.
            f.write(
                "import torch\n"
                "from torch import device\n"
                "\n"
                "b8: type\n"
                "i32: type\n"
                "i64: type\n"
                "bf16: type\n"
                "f32: type\n"
                "f64: type\n"
                "fx_pytree: type\n"
                "Sym: type\n\n\n"
            )
            f.write(
                f"# Graph[rank={pgraph.rank}]({pgraph.direction}, gid={pgraph.gid})\n"
            )
            f.write(pgraph.readable_str)
            f.write("\n\n")
        if rank == 0:
            print(f"readable saved to {readable_path}")


def save_graphs(dirname, rank, collector: GraphCollector, just_check: bool = False):
    os.makedirs(dirname, exist_ok=True)
    for gm, attr in collector.graphs:
        save_graph(dirname, rank, gm, attr, just_check)


def dynamo_and_dump(
    model,
    fn,
    dirname,
    formats: list[str],
    logs: bool = False,
    rank: int = None,
    draw: bool = False,
    compile_model_or_fn: str = "model",
    return_res: bool = False,
    forward_only: bool = False,
) -> tuple[GraphCollector, torch.nn.Module | Callable, torch.nn.Module | Callable]:
    if logs:
        torch._logging.set_logs(
            dynamo=logging.DEBUG, aot=logging.DEBUG, graph_code=True
        )
    collector = GraphCollector(
        formats=formats,
        distributed=rank is not None,
        rank=rank,
        draw=draw,
        dirname=dirname,
        single_direction=forward_only,
    )
    backend_direction = (
        GraphCollector.Direction.FW if forward_only else GraphCollector.Direction.BOTH
    )
    if compile_model_or_fn == "model":
        compiled_model = torch.compile(
            model,
            dynamic=False,
            backend=collector.make_backend(backend_direction, increase_gid=True),
        )
        res = fn(compiled_model)
        save_graphs(dirname, rank, collector, just_check=True)
        default_backend_model = torch.compile(
            model,
            dynamic=False,
            backend=default_backend,
        )
        if return_res:
            return collector, compiled_model, default_backend_model, res
        else:
            return collector, compiled_model, default_backend_model
    elif compile_model_or_fn == "fn":
        compiled_fn = torch.compile(
            fn,
            dynamic=False,
            backend=collector.make_backend(backend_direction, increase_gid=True),
        )
        res = compiled_fn(model)
        save_graphs(dirname, rank, collector, just_check=True)
        default_backend_fn = torch.compile(
            fn,
            dynamic=False,
            backend=default_backend,
        )
        if return_res:
            return collector, compiled_fn, default_backend_fn, res
        else:
            return collector, compiled_fn, default_backend_fn
    else:
        raise ValueError(f"Unknown compile_model_or_fn: {compile_model_or_fn}")
