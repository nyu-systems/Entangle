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

use crate::rewrite::lambda_rewrite::*;
use crate::symval::{ShapeLike, SymValManagerRef};
use crate::utils::*;
use std::convert::TryInto;

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let results = make_rewrites!(manager, verbose,
        ///////////////////////////////////////////////////////////////////////////////////////////////
        // Inverse
        // Try to use inverse lemmas as less as possible. It introduces cycles and more operators. Thus,
        // you must make sure you have other lemmas to eliminate the redundant operators and avoid cycle
        // issues.
        ///////////////////////////////////////////////////////////////////////////////////////////////
        "<inverse-concat>" =>
            "(concat ?t1 ?t2 ?dim)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                let unioned1 = la.union(egraph, subst, "?t1", &format!("(slice {} {dim} 0 {} 1)",la.src_pat, t1_shape[dim]));
                let unioned2 = la.union(egraph, subst, "?t2", &format!("(slice {} {dim} {} {} 1)",la.src_pat, t1_shape[dim], t1_shape[dim].clone() + t2_shape[dim].clone()));
                return unioned1 as usize + unioned2 as usize;
            };
    );

    results
}
