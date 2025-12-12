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

from entangle.ops.op import Op

broadcast = Op("hlo_broadcast", skeleton=True)
dot = Op("hlo_dot", multi_in=True, skeleton=True)
gather = Op("hlo_gather", multi_in=True, skeleton=True)
logistic = Op("hlo_logistic", skeleton=True)
reduce_max = Op("hlo_max", skeleton=True)
rms_norm = Op("hlo_rms_norm", multi_in=True, skeleton=True)
select = Op("hlo_select", noncompute=True, multi_in=True, relation=True)
