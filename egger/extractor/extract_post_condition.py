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
import random

import matplotlib.pyplot as plt
import networkx as nx
import rich
from entangle.sgraph.egraph import *

"""
# Examples
python extractor/extract_post_condition.py target/saturated.json
python3 extractor/extract_post_condition.py target/saturated.json
"""
parser = argparse.ArgumentParser(description="Post-condition Extractor")
parser.add_argument(
    "egraph_file",
    type=str,
    help="Path to the egraph json file.",
)
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose outputs.")
parser.add_argument(
    "-g",
    "--using_egraph",
    nargs="+",
    default=["candidates"],
    help="Can be candidates|representative|yis",
)


def main():
    args = parser.parse_args()
    precondition_dirname = osp.dirname(args.egraph_file)
    egraph = EGraph(args.egraph_file, verbose=args.verbose)
    egraph.compute_post_condition_eclasses()
    for egraph_type in args.using_egraph:
        if egraph_type == "candidates":
            used_egraph = egraph.extract_to_postcondition(all_candidates=True)
            filename = "candidates.dot"
        elif egraph_type == "representative":
            used_egraph = egraph.extract_to_postcondition(representative_only=True)
            filename = "postcondition_representative.dot"
        elif egraph_type == "yis":
            used_egraph = egraph.extract_to_postcondition(
                representative_only=True,
                including_yis=True,
            )
            filename = "postcondition_including_yis.dot"
        else:
            raise ValueError(f"Invalid using_egraph: {egraph_type}")
        used_egraph.to_dot(osp.join(precondition_dirname, filename))


if __name__ == "__main__":
    main()
