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

from typing import Callable

from entangle.convert.convert import *
from entangle.sgraph import sexpr
from entangle.sgraph.egraph import SExprECondition
from entangle.sgraph.sskeleton import *
from entangle.tools.config import ExplorativeConfig
from entangle.utils.mesh import DeviceMesh


class MyConfig(ExplorativeConfig):
    def build_preconditions(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(
            self.get_sexpr
        )
        return [
            SExprECondition(
                inputs=[
                    s := placeholder("Sn__r0__arg0_1"),
                    d := placeholder("Dn__r0__arg0_1"),
                ],
                eclasses=[[s, d]],
            ),
            SExprECondition(
                inputs=[
                    s := placeholder("Sn__r0__arg3_1"),
                    *(
                        d := [
                            placeholder("Dn__r0__arg3_1"),
                            placeholder("Dn__r0__arg6_1"),
                        ]
                    ),
                ],
                eclasses=[[s, sexpr.concat([d[0], d[1]], dim=0)]],
            ),
            SExprECondition(
                inputs=[
                    s := placeholder("Sn__r0__arg1_1"),
                    d := placeholder("Dn__r0__arg1_1"),
                ],
                eclasses=[[s, d]],
            ),
            SExprECondition(
                inputs=[
                    s := placeholder("Sn__r0__arg2_1"),
                    *(
                        d := [
                            placeholder("Dn__r0__arg2_1"),
                            placeholder("Dn__r0__arg5_1"),
                        ]
                    ),
                ],
                eclasses=[[s, sexpr.concat([d[0], d[1]], dim=0)]],
            ),
            SExprECondition(
                inputs=[
                    s := placeholder("Sn__r0__arg4_1"),
                    d := placeholder("Dn__r0__arg4_1"),
                ],
                eclasses=[[s, d]],
            ),
        ]

    def build_expected(self) -> list[SExprECondition]:
        placeholder: Callable[[str], SExpr] = sexpr.get_placeholder_maker(
            self.get_sexpr
        )
        expected = SExprECondition(
            inputs=[
                s := placeholder("Sn__r0__add_1"),
                d := placeholder("Dn__r0__add_3"),
            ],
            eclasses=[[s, d]],
        )
        return [expected]
