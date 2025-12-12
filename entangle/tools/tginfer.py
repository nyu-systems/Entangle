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

"""
This is a shortcut to re-run one step of inference.

Example:
tginfer
"""

import argparse
import importlib
import importlib.util
import logging
import os
import os.path as osp
import pickle
import random
from datetime import datetime
import tempfile
from typing import Callable, Type

import numpy as np
import rich
from rich.logging import RichHandler

from entangle.convert.convert import pgraph_to_sgraph
from entangle.pgraph.pickleable import PickleableGraph, load_pgraph
from entangle.sgraph.sgraph import SGraph
from entangle.tools.config import Config, ExplorativeConfig, load_config_class
from entangle.tools.egg import EggRunner
from entangle.tools.utils import dot_exists
from entangle.utils import visual_utils
from entangle.utils.module_utils import load_module
from entangle.utils.print_utils import BGREEN, BRED, BRI, BYELLOW, RST

random.seed(123456)
np.random.seed(123456)


parser = argparse.ArgumentParser(
    description="entangle: a tool for graph dumping and verification of PyTorch models."
)
parser.add_argument("--log", help="Logging level", type=str, default="INFO")
parser.add_argument(
    "--log_path", help="Logging path.", type=str, default=osp.join(tempfile.gettempdir(), "entangle.log")
)
parser.add_argument(
    "--post_type",
    type=str,
    default="candidates",
    choices=["candidates", "representative"],
    help="Post-condition type, can be candidates|representative, yis is buggy now, use candidates.",
)
parser.add_argument(
    "-t",
    "--egg_dirname",
    type=str,
    default="../egger",
    help=f"The directory to the root of egg.",
)
parser.add_argument(
    "-d",
    "--dirname",
    type=str,
    required=True,
    help=f"The temporary dirname for data like conditions and computations used by egg.",
)
parser.add_argument(
    "-i",
    "--config_module",
    # required=True,
    help=f"Python file that contains a class inheriting from `entangle.tools.config.Config`.",
    type=str,
)
parser.add_argument("--inverse_lemma", help="Use inverse lemmas.", action="store_true")
parser.add_argument("--debug", help="Use debug version Egg.", action="store_true")
parser.add_argument("--verbose", help="Verbose.", action="store_true")
parser.add_argument("--visualize", help="Visualize.", action="store_true")


def build_egg_runner(args: argparse.Namespace) -> EggRunner:
    return EggRunner(
        egg_dirname=args.egg_dirname,
        egg_data_dirname=None,
        inverse_lemma=args.inverse_lemma,
        post_type=args.post_type,
        log_level=args.log,
        debug=args.debug,
        tmux=False,
        verbose=args.verbose,
        visualize=args.visualize,
        use_local_directory=True,
    )


def main():
    # Move things into python code so that we don't have to type in many things...
    os.system("killall redis-server")
    os.environ["MEGATRON_USE_TIMELINE_SDK"] = "0"

    args = parser.parse_args()

    if args.visualize:
        assert (
            dot_exists()
        ), "Please install graphviz and make sure `dot` is in your PATH to use `visualize`"

    args.log = logging._nameToLevel[args.log.upper()]

    egg_runner = build_egg_runner(args)

    name = osp.split(args.dirname)[-1]

    if args.config_module is not None:
        config_module = load_module(args.config_module, "config_module")
        ConfigClass: Type[Config] = load_config_class(config_module)

    files_to_remove = ["start.json", "saturated.json"]
    files_to_remove = [osp.join(args.dirname, f) for f in files_to_remove]
    os.system(
        f"echo 'cleaning...' && rm -f {args.dirname}/*.dot {args.dirname}/*.svg {' '.join(files_to_remove)}"
    )

    egg_runner.run(args.dirname, name, mode="infer")


if __name__ == "__main__":
    main()
