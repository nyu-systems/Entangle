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

import importlib
import importlib.util
import os.path as osp
import sys

from entangle.utils.print_utils import BRI, RST


def load_module(path: str, module_name: str, check_keys: list[str] = None):
    if check_keys is None:
        check_keys = []
    print(f"Loading `{BRI}{module_name}{RST}` from {BRI}{path}{RST} ...")
    if not osp.exists(path):
        raise FileNotFoundError(f"Provided module file {path} doesn't exist.")
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)

    for key in check_keys:
        if key not in module.__dict__:
            raise AttributeError(f"Expected {BRI}{key}{RST} in {path}.")

    return module
