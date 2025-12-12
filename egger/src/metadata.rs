// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::symval::*;
use crate::utils::*;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DataKind {
    Name,
    Scalar,
    Tnsr,
    TnsrTuple,
}

impl DataKind {
    pub fn is_scalar_or_name(self) -> bool {
        self == DataKind::Scalar || self == DataKind::Name
    }
}

impl Default for DataKind {
    fn default() -> Self {
        DataKind::Name
    }
}

const MAX_DIM: usize = 8;

#[derive(Debug, Clone, Default, PartialEq, Hash)]
pub struct TensorMeta {
    /// Shape of the tensor. We deal with tensor up to MAX_DIM dimensions
    pub shape: [Option<SymVal>; MAX_DIM],
    /// Number of dimensions of this tensor
    pub n_dim: usize,
}

impl TensorMeta {
    pub fn new(shape: &[SymVal]) -> Self {
        let mut res = TensorMeta::default();
        assert!(shape.len() <= MAX_DIM);
        for i in 0..shape.len() {
            res.shape[i] = Some(shape[i].clone());
        }
        res.n_dim = shape.len();
        res
    }

    pub fn sliced_shape(self) -> ShapeLike {
        self.shape[..self.n_dim]
            .iter()
            .map(|s| s.clone().unwrap())
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct ValTnsr {
    /// The data type of this eclass, can be a name/scalar/tensor
    pub dtype: DataKind,
    /// The value of this eclass if it is a Scalar type
    pub val: Option<SymVal>,
    /// The name string of this eclass if it is a Name type
    pub name: String,
    /// The pointer to the tensor if it is a Tensor type
    pub meta: Option<TensorMeta>,
    /// The pointer to the second tensor if it is a TnsrTuple type (for split node)
    pub meta_2: Option<TensorMeta>,
}

#[allow(unconditional_recursion)]
impl Default for ValTnsr {
    fn default() -> Self {
        ValTnsr {
            dtype: DataKind::Name,
            val: Option::None,
            name: String::new(),
            meta: Option::None,
            meta_2: Option::None,
        }
    }
}

impl ValTnsr {
    pub fn new_val(val: SymVal) -> ValTnsr {
        ValTnsr {
            dtype: DataKind::Scalar,
            val: Some(val),
            ..Default::default()
        }
    }

    pub fn new_name(name: String) -> ValTnsr {
        ValTnsr {
            dtype: DataKind::Name,
            name: name,
            ..Default::default()
        }
    }

    pub fn new_tensor(shape: &ShapeLike) -> ValTnsr {
        let meta = TensorMeta::new(shape);
        ValTnsr {
            dtype: DataKind::Tnsr,
            meta: Some(meta),
            ..Default::default()
        }
    }

    pub fn parse_to_val_shape(&self) -> Vec<i64> {
        assert!(self.dtype == DataKind::Name, "dtype: {:?}", self.dtype);
        shapelike_name_to_val_vec(&self.name)
    }

    pub fn parse_to_shapelike(&self, manager: SymValManagerRef) -> ShapeLike {
        assert!(self.dtype == DataKind::Name);
        shapelike_name_to_vec(&self.name, manager.clone())
    }

    pub fn get_shape(&self) -> ShapeLike {
        // returns shape of tensor if dtype is Tnsr, otherwise vec![]
        if self.dtype == DataKind::Tnsr {
            self.meta.clone().unwrap().sliced_shape()
        } else if self.dtype == DataKind::Scalar || self.dtype == DataKind::Name {
            vec![]
        } else {
            panic!("get_shape() not implemented for {:?}", self.dtype);
        }
    }

    pub fn get_sym_val(&self) -> SymVal {
        if self.dtype == DataKind::Scalar {
            assert!(self.val.is_some());
            self.val.clone().unwrap()
        } else {
            panic!("get_sym_val() not implemented for {:?}", self.dtype);
        }
    }

    pub fn is_pure_val(&self) -> bool {
        if self.dtype == DataKind::Scalar {
            match self.val.as_ref().unwrap().symval_id {
                SymValId::Val(_) => true,
                _ => false,
            }
        } else {
            false
        }
    }

    pub fn get_val(&self) -> i64 {
        if self.dtype == DataKind::Scalar {
            assert!(self.val.is_some());
            match self.val.as_ref().unwrap().symval_id {
                SymValId::Val(val) => val,
                _ => panic!("Not a value, {:?}", self),
            }
        } else {
            panic!("get_val() not implemented for {:?}", self.dtype);
        }
    }
}

impl std::fmt::Display for ValTnsr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.dtype {
            DataKind::Name => write!(f, "{}", self.name),
            DataKind::Scalar => write!(f, "{}", self.val.as_ref().unwrap()),
            DataKind::Tnsr => write!(f, "{}", shape_to_underscore_name(&self.get_shape())),
            _ => {
                panic!("Display not implemented for {:?}", self.dtype);
            }
        }
    }
}
