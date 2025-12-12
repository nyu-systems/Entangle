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

class Op:
    OPS = {}

    def __init__(
        self,
        name: str,
        dist: bool = False,
        multi_in: bool = False,
        multi_out: bool = False,
        skeleton: bool = False,
        noncompute: bool = False,
        leaf: bool = False,
        scalar: bool = False,
        boolean: bool = False,
        constant: bool = False,
        relation: bool = False,
        idx: int = None,
    ):
        """
        name: str for the name of the op
        dist: Whether the op is a distributed op. This is used to partition graph into chains.
        multi_in: Whether the op can have multiple tensor inputs (note that scalar inputs are not counted)
        multi_out: Whether the op can have multiple tensor outputs (note that scalar outputs are not counted)
        skeleton: Whether the op is a skeleton op. This is used to partition graph into chains.
        noncompute: For ops that just doing view operation like concat, slice, padding that doesn't change any value of the tensor.
        leaf: Whether the op is a leaf op that doesn't depend on other tensors. This is used to find input tensors.
        scalar: Whether the op is a scalar op.
        boolean: Whether the op is a boolean op.
        constant: Whether the op is a constant op, this is for zeros, ones, fill, etc.
        relation: Whether the op is allowed to be used to represent a relation.
        """
        if name in Op.OPS:
            raise ValueError(f"Op {name} already exists in OPS: {Op.OPS[name]}")
        self.name = name
        self.dist = dist
        self.multi_in = multi_in
        self.multi_out = multi_out
        assert not (
            skeleton and noncompute
        ), "skeleton and noncompute cannot be both True"
        self.skeleton = skeleton
        self.noncompute = noncompute
        self.leaf = leaf
        self.scalar = scalar
        self.boolean = boolean
        self.constant = constant
        self.relation = relation
        self.idx = idx

        Op.OPS[name] = self

    def __hash__(self):
        return hash((self.name, self.idx))

    def __eq__(self, other):
        """
        This is due to we have infernbn before, the loaded op cannot be directly equaled.
        We don't really need this now, but let's keep it because it doesn't hurt.
        """
        if self.name == other.name:
            assert self.idx == other.idx
            assert self.dist == other.dist
            assert self.multi_in == other.multi_in
            assert self.multi_out == other.multi_out
            assert self.skeleton == other.skeleton
            assert self.noncompute == other.noncompute
            assert self.scalar == other.scalar
            assert self.boolean == other.boolean
            assert self.constant == other.constant
            assert self.leaf == other.leaf
            assert self.relation == other.relation
            return True
        else:
            return False

    @staticmethod
    def get(name) -> "Op":
        return Op.OPS[name]

    @property
    def source_op(self):
        # If this op is an getitem version, return the source op.
        if self.idx is not None:
            return Op.OPS[self.name.removesuffix(f"_{self.idx}")]
        else:
            return self

    def getitem(self, idx: int):
        assert (
            self.multi_out
        ), f"Only multi-out ops can have getitem subops, got {str(self)}"
        name = f"{self.name}_{idx}"
        if name in Op.OPS:
            return Op.OPS[name]
        else:
            return Op(
                name,
                dist=self.dist,
                multi_in=self.multi_in,
                multi_out=False,
                skeleton=self.skeleton,
                noncompute=self.noncompute,
                leaf=self.leaf,
                scalar=self.scalar,
                boolean=self.boolean,
                constant=self.constant,
                relation=self.relation,
                idx=idx,
            )

    def __str__(self):
        return f"Op({self.name}, dist={self.dist}, multi_in={self.multi_in}, multi_out={self.multi_out}, skeleton={self.skeleton}, noncompute={self.noncompute}, leaf={self.leaf}, scalar={self.scalar}, boolean={self.boolean}, constant={self.constant})"

    def __repr__(self):
        return f"Op({self.name})"
