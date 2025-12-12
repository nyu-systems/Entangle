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

use crate::model::*;
use crate::utils::SEXPR_SPLIT;
use egg::*;
use itertools::Itertools;

pub fn load_eclasses(path: &std::path::PathBuf) -> Vec<Vec<RecExpr<Mdl>>> {
    let content =
        std::fs::read_to_string(path).expect("Something went wrong reading the model file");
    content
        .split("\n")
        .filter(|s| !s.is_empty() && !s.starts_with("#"))
        .map(|line| {
            line.split(SEXPR_SPLIT)
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .collect()
}

pub fn load_computations(path: &std::path::PathBuf) -> (Vec<RecExpr<Mdl>>, Vec<RecExpr<Mdl>>) {
    let mut cs: Vec<RecExpr<Mdl>> = Vec::new();
    let mut ys: Vec<RecExpr<Mdl>> = Vec::new();

    let content =
        std::fs::read_to_string(path).expect("Something went wrong reading the model file");
    let lines = content.lines();
    for line in lines
        .into_iter()
        .filter(|s| !s.is_empty() && !s.starts_with("#"))
    {
        let c_and_y = line.split(SEXPR_SPLIT).collect_vec();
        cs.push(c_and_y[0].parse().unwrap());
        ys.push(c_and_y[1].parse().unwrap());
    }
    (cs, ys)
}

pub fn zip_computation_to_eclass(
    cs: Vec<RecExpr<Mdl>>,
    ys: Vec<RecExpr<Mdl>>,
) -> Vec<Vec<RecExpr<Mdl>>> {
    cs.into_iter()
        .zip(ys.into_iter())
        .map(|(c, y)| vec![c, y])
        .collect()
}
