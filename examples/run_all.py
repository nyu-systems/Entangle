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
import os.path as osp
from subprocess import DEVNULL, PIPE, STDOUT, Popen

parser = argparse.ArgumentParser("Experiments Runner")
parser.add_argument(
    "model",
    type=str,
    choices=[
        "gpt",
        "qwen2",
        "aws_llama",
    ],
    help="Model name.",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Run all configurations. If specified, the `--world_size` argument will be ignored.",
)
parser.add_argument(
    "--tp",
    nargs="+",
    type=int,
    default=[2],
    help="The tp size, valid for `gpt`.",
)
parser.add_argument(
    "--num_layers",
    nargs="+",
    type=int,
    default=[1],
    help="The number of layers",
)
parser.add_argument(
    "--output_root",
    type=str,
    default=".",
    help="Output root directory, we will create a directory under this to store the output.",
)
parser.add_argument(
    "--graph_prefix",
    type=str,
    default="",
    help="Graph prefix.",
)
parser.add_argument(
    "--stdout",
    action="store_true",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
)
parser.add_argument(
    "--disable_rich", action="store_true", help="Disable rich printing."
)
parser.add_argument(
    "--verbose", action="store_true", help="Enable verbose output."
)
args = parser.parse_args()


def run_one(model, origin_name, target_name, cmd):
    output_dir = osp.join(
        args.output_root,
        f"precondition.{model}.{args.graph_prefix}.{origin_name}-{target_name}.greedy",
    )
    additional_args = ""
    if args.verbose:
        additional_args += " --verbose"
    cmd = f"export OUTDIR={output_dir} && mkdir -p $OUTDIR; {cmd} -o {output_dir} {additional_args} | tee $OUTDIR/output.log "
    print(cmd)
    if args.dry_run:
        return
    if args.stdout:
        p = Popen(cmd, shell=True)
    else:
        p = Popen(cmd, shell=True, stdout=DEVNULL)
    p.wait()
    if p.returncode != 0:
        print(f"Command failed: {cmd}")
        exit(p.returncode)


def main():
    model = args.model

    tg_args = ""
    if args.disable_rich:
        tg_args += " --disable_rich"

    if model == "gpt":
        if args.all:
            num_layers_list = [1, 2, 4, 8]
            tp = [2, 4, 6, 8]
        else:
            num_layers_list = args.num_layers
            tp = args.tp
        for num_layers in num_layers_list:
            for tp_size in tp:
                origin_name = f"paral1_layer{num_layers}"
                target_name = f"paral{tp_size}_layer{num_layers}"
                args.graph_prefix = "fw.g0"
                cmd = f"tg {tg_args} infer -g data/{model} --origin {origin_name} --target {target_name} --graph_prefix {args.graph_prefix} --config_module verify_{model}.py --tp={tp_size} --num_layers={num_layers} --infer_manager greedy --stats"
                run_one(model, origin_name, target_name, cmd)
    elif model == "qwen2":
        if args.all:
            num_layers_list = [1]
            tp = [2, 4]
        else:
            num_layers_list = args.num_layers
            tp = args.tp
        for num_layers in num_layers_list:
            for tp_size in tp:
                origin_name = f"paral1_layer{num_layers}"
                target_name = f"paral{tp_size}_layer{num_layers}"
                args.graph_prefix = "fw.g0"
                cmd = f"tg {tg_args} infer -g data/{model} --origin {origin_name} --target {target_name} --graph_prefix {args.graph_prefix} --config_module verify_{model}.py --tp={tp_size} --num_layers={num_layers} --infer_manager greedy --stats"
                run_one(model, origin_name, target_name, cmd)
    elif model == "aws_llama":
        if args.all:
            num_layers_list = [1, 2, 4, 8]
            tp = [2, 4, 8]
        else:
            num_layers_list = args.num_layers
            tp = args.tp
        for num_layers in num_layers_list:
            for tp_size in tp:
                origin_name = f"tp1_layer{num_layers}"
                target_name = f"tp{tp_size}_layer{num_layers}"
                args.graph_prefix = "fw.g0"
                cmd = f"tg {tg_args} infer -g data/{model} --origin {origin_name} --target {target_name} --graph_prefix {args.graph_prefix} --config_module verify_{model}.py --tp={tp_size} --num_layers={num_layers} --infer_manager greedy --stats"
                run_one(model, origin_name, target_name, cmd)
    else:
        raise ValueError(f"Unknown model: {model}")


if __name__ == "__main__":
    main()
