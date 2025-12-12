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

use crate::metadata::*;
use crate::model::*;
use crate::symval::*;
use egg::*;
use itertools::Itertools;

pub const SYMVAL_PREFIX: &str = "Sym";
pub const SEXPR_SPLIT: &str = "===";

pub fn shapelike_to_val_shape(shape: &ShapeLike) -> Vec<i64> {
    shape
        .iter()
        .map(|x| {
            if let SymValId::Val(val) = &x.symval_id {
                *val
            } else {
                panic!("Not a value")
            }
        })
        .collect()
}

pub fn val_shape_to_shapelike(shape: &Vec<i64>, manager: SymValManagerRef) -> ShapeLike {
    shape
        .iter()
        .map(|x| SymVal::new_val(*x, manager.clone()))
        .collect()
}

pub fn get_dtype(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> DataKind {
    egraph[subst[s.parse().unwrap()]].data.dtype
}

pub fn get_sym_val(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> SymVal {
    egraph[subst[s.parse().unwrap()]].data.get_sym_val()
}

pub fn is_pure_val(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> bool {
    egraph[subst[s.parse().unwrap()]].data.is_pure_val()
}

pub fn get_val(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> i64 {
    egraph[subst[s.parse().unwrap()]].data.get_val()
}

pub fn get_usize_val(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> usize {
    egraph[subst[s.parse().unwrap()]].data.get_val() as usize
}

pub fn get_meta(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> TensorMeta {
    egraph[subst[s.parse().unwrap()]].data.meta.clone().unwrap()
}

pub fn get_n_dim(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> usize {
    egraph[subst[s.parse().unwrap()]]
        .data
        .meta
        .as_ref()
        .unwrap()
        .n_dim
}

pub fn get_shape(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> ShapeLike {
    egraph[subst[s.parse().unwrap()]]
        .data
        .meta
        .clone()
        .unwrap()
        .sliced_shape()
}

pub fn try_get_val_shape(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
    s: &str,
) -> Option<Vec<i64>> {
    let shape = egraph[subst[s.parse().unwrap()]]
        .data
        .meta
        .clone()
        .unwrap()
        .sliced_shape();
    if shape.iter().any(|x| !x.symval_id.is_val()) {
        None
    } else {
        Some(shapelike_to_val_shape(&shape))
    }
}

pub fn get_shape_from_name(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
    name: &str,
    manager: SymValManagerRef,
) -> ShapeLike {
    shapelike_name_to_vec(&get_name(egraph, subst, name), manager.clone())
}

pub fn get_val_shape_from_name<T>(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
    name: &str,
) -> Vec<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    shapelike_name_to_val_vec(&get_name(egraph, subst, name))
}

pub fn try_get_val_shape_from_name(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    subst: &Subst,
    name: &str,
    manager: SymValManagerRef,
) -> Option<Vec<i64>> {
    let shape = get_shape_from_name(egraph, subst, name, manager);
    if shape.iter().any(|x| !x.symval_id.is_val()) {
        None
    } else {
        Some(shapelike_to_val_shape(&shape))
    }
}

pub fn get_meta_by_id(egraph: &mut EGraph<Mdl, TensorAnalysis>, id: Id) -> TensorMeta {
    egraph[id].data.meta.clone().unwrap()
}

pub fn get_name(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, s: &str) -> String {
    egraph[subst[s.parse().unwrap()]].data.name.clone()
}

pub fn get_id(subst: &Subst, s: &str) -> Id {
    subst[s.parse().unwrap()]
}

pub fn is_val_shape(shape: &ShapeLike) -> bool {
    shape.iter().all(|x| x.is_val())
}

macro_rules! get_ids {
    ($subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_id($subst, s))
            .collect::<Vec<Id>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_dtypes {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_dtype($egraph, $subst, s))
            .collect::<Vec<DataKind>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_shapes {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_shape($egraph, $subst, s))
            .collect::<Vec<ShapeLike>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_shapes_from_names {
    ($egraph:expr, $subst:expr, $strs:expr, $manager:expr) => {
        $strs
            .iter()
            .map(|s| get_shape_from_name($egraph, $subst, s, $manager.clone()))
            .collect::<Vec<ShapeLike>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_sym_vals {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_sym_val($egraph, $subst, s))
            .collect::<Vec<SymVal>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_usize_vals {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_val($egraph, $subst, s) as usize)
            .collect::<Vec<usize>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_vals {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_val($egraph, $subst, s))
            .collect::<Vec<i64>>()
            .try_into()
            .unwrap()
    };
}

#[allow(unused)]
macro_rules! get_val_shapes {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| try_get_val_shape($egraph, $subst, s).unwrap())
            .collect::<Vec<Vec<i64>>>()
            .try_into()
            .unwrap()
    };
}

macro_rules! get_val_shapes_from_names {
    ($egraph:expr, $subst:expr, $strs:expr) => {
        $strs
            .iter()
            .map(|s| get_val_shape_from_name::<i64>($egraph, $subst, s))
            .collect::<Vec<Vec<i64>>>()
            .try_into()
            .unwrap()
    };
}

pub(crate) use get_dtypes;
pub(crate) use get_ids;
pub(crate) use get_shapes;
pub(crate) use get_shapes_from_names;
pub(crate) use get_sym_vals;
pub(crate) use get_usize_vals;
#[allow(unused)]
pub(crate) use get_val_shapes;
pub(crate) use get_val_shapes_from_names;
pub(crate) use get_vals;

pub fn shapelike_name_to_vec(name: &String, manager: SymValManagerRef) -> ShapeLike {
    assert!(
        name.starts_with("[") && name.ends_with("]"),
        "Got name: {}",
        name
    );
    name[1..name.len() - 1]
        .split(",")
        .filter(|s| !s.is_empty())
        .map(|s| SymVal::from_str(s.trim(), manager.clone()))
        .collect_vec()
}

pub fn shapelike_name_to_val_vec<T>(name: &String) -> Vec<T>
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    assert!(
        name.starts_with("[") && name.ends_with("]"),
        "Got name: {}",
        name
    );
    name[1..name.len() - 1]
        .split(",")
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect_vec()
}

pub fn val_shape_to_underscore_name(shape: &Vec<i64>) -> String {
    format!("\"[{}]\"", &shape.iter().map(|x| x.to_string()).join(","))
}

pub fn shape_to_underscore_name(shape: &ShapeLike) -> String {
    format!("\"[{}]\"", &shape.iter().map(|x| x.to_string()).join(","))
}

pub fn shape_eq_except(shape1: &ShapeLike, shape2: &ShapeLike, dim: usize) -> bool {
    if shape1.len() != shape2.len() {
        return false;
    }
    for i in 0..shape1.len() {
        if i == dim {
            continue;
        }
        if shape1[i] != shape2[i] {
            return false;
        }
    }
    return true;
}

pub fn broadcast<'a>(shape1: &ShapeLike, shape2: &ShapeLike) -> Option<ShapeLike> {
    if shape1.len() > shape2.len() {
        let mut shape = shape1[..shape1.len() - shape2.len()].to_vec();
        for (x, y) in shape1[shape1.len() - shape2.len()..]
            .iter()
            .zip(shape2.iter())
        {
            if x == y {
                shape.push(x.clone());
            } else if *x == 1 {
                shape.push(y.clone());
            } else if *y == 1 {
                shape.push(x.clone());
            } else {
                return None;
            }
        }
        return Some(shape);
    } else if shape1.len() < shape2.len() {
        let mut shape = shape2[..shape2.len() - shape1.len()].to_vec();
        for (x, y) in shape1
            .iter()
            .zip(shape2[shape2.len() - shape1.len()..].iter())
        {
            if x == y {
                shape.push(x.clone());
            } else if *x == 1 {
                shape.push(y.clone());
            } else if *y == 1 {
                shape.push(x.clone());
            } else {
                return None;
            }
        }
        return Some(shape);
    } else {
        let mut shape = vec![];
        for (x, y) in shape1.iter().zip(shape2.iter()) {
            if x == y {
                shape.push(x.clone());
            } else if *x == 1 {
                shape.push(y.clone());
            } else if *y == 1 {
                shape.push(x.clone());
            } else {
                return None;
            }
        }
        return Some(shape);
    }
}

pub fn mul_reduce(shape: &[SymVal]) -> SymVal {
    shape.iter().cloned().reduce(|a, b| a * b).unwrap()
}

// HLO helper
pub fn get_left_dim(
    shape_len: usize,
    lhs_batch_dims: &[i64],
    lhs_contracting_dims: &[i64],
) -> usize {
    let mut tmp = None;
    let mut dims = vec![true; shape_len];
    for i in lhs_batch_dims {
        dims[*i as usize] = false;
    }
    for i in lhs_contracting_dims {
        dims[*i as usize] = false;
    }
    for i in 0..dims.len() {
        if dims[i] {
            tmp = Some(i);
            break;
        }
    }
    tmp.unwrap()
}

pub fn str2ast<L>(s: &str) -> PatternAst<L>
where
    L: egg::Language + std::fmt::Debug + FromOp,
{
    s.parse::<Pattern<L>>().unwrap().ast
}
