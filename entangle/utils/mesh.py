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

from typing import Union

import numpy as np


class DeviceMesh:
    def __init__(
        self,
        mesh_shape: tuple[int],
        dim_names: list[str],
        group_ids: dict[str, dict[tuple[int], str]] = None,
    ):
        self.mesh = np.arange(np.prod(mesh_shape)).reshape(mesh_shape)
        self.num_dims = len(self.mesh.shape)
        assert self.num_dims == len(dim_names), f"{self.num_dims=} vs {len(dim_names)=}"
        self.dim_names = list(dim_names)
        self.name_to_dim = {name: i for i, name in enumerate(dim_names)}
        # NOTE: `group_ids` maps a group name to a dict,
        # where the dict has an `indices` as key and group id as value.
        # The `indices` is an int tuple that indexes the group in the mesh.
        # E.g., mesh_shape=(2,2), dim_names=['pdim0', 'pdim1']
        #                  pdim0
        #           ------------->
        #           |  r0    r2    g63
        #    pdim1  |
        #           |  r1    r3    g64
        #          \|
        #              g65   g67
        # Then, the indices for group 65 is [0, None];
        # the indices for group 64 is [None, 1].
        self.group_ids = group_ids
        self.group_id_to_name = {}
        for name, ids_dict in self.group_ids.items():
            for group_id in ids_dict.values():
                self.group_id_to_name[group_id] = name

    def get_dim(self, name):
        return self.name_to_dim[name]

    def get_group_names(self) -> list[str]:
        return self.dim_names

    def get_group_name(self, group_id):
        return self.group_id_to_name[group_id]

    def get_group_size(self, name):
        return self.mesh.shape[self.get_dim(name)]

    def get_groups(self, name):
        dim = self.get_dim(name)
        mesh = self.mesh.transpose(
            (*(i for i in range(self.num_dims) if i != dim), dim)
        )
        mesh = mesh.reshape((-1, mesh.shape[-1]))
        return mesh.tolist()

    def get_rank_indices(self, rank) -> list[int]:
        return np.argwhere(self.mesh == rank)[0].tolist()

    def get_group_indices(self, rank, name) -> tuple[Union[int, None]]:
        rank_indices = self.get_rank_indices(rank)
        rank_indices[self.get_dim(name)] = None
        return tuple(rank_indices)

    def get_local_rank(self, rank, name):
        return self.get_rank_indices(rank)[self.get_dim(name)]

    def get_group_ids(self) -> list[str]:
        return list(self.group_id_to_name.keys())

    def get_group_id(self, rank, name):
        return self.group_ids[name][self.get_group_indices(rank, name)]
