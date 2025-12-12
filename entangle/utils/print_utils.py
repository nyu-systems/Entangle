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

from colorama import Fore, Style
from rich.logging import RichHandler

RST = Style.RESET_ALL
BRED = Style.BRIGHT + Fore.RED
BGREEN = Style.BRIGHT + Fore.GREEN
BYELLOW = Style.BRIGHT + Fore.YELLOW
BRI = Style.BRIGHT


def print_ft(s: str = None, c: str = None, leading=10, *args, **kwargs):
    if s is None or s == "":
        s = ""
    else:
        s = f" {s} "
    print(filling_terminal(s, c=c, leading=leading), *args, **kwargs)


def filling_terminal(s: str = None, c: str = None, leading=10) -> str:
    if s is None:
        s = ""
    if c is None:
        c = "-"
    try:
        columns = os.get_terminal_size().columns
    except OSError:
        columns = 256
    trailing = columns - len(s) - leading
    if trailing < 0:
        trailing = 0
    s = f"{BRI}" + c * leading + s + c * trailing + f"{RST}"
    return s


def get_logger(name, level=logging.INFO, mode: str = "w", path=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # if level == logging.DEBUG:
    if level == logging.DEBUG:
        logger.addHandler(RichHandler(level=logging.DEBUG))
    if path is not None:
        logger.addHandler(logging.FileHandler(path, mode))
    return logger
