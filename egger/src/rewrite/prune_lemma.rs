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
use crate::symval::SymValManagerRef;
use crate::symval::*;
use crate::utils::*;
use std::convert::TryInto;

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let results = make_rewrites!(manager, verbose,
        "<slice-concat-swap>" =>
            "(slice (concat ?t1 ?t2 ?c_dim) ?s_dim ?begin ?end 1)" => |la, egraph, matched_id, subst| {
                let [c_dim, s_dim]: [i64;2] = get_vals!(egraph, subst, ["?c_dim", "?s_dim"]);
                let [begin, end] = get_sym_vals!(egraph, subst, ["?begin", "?end"]);
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, ["?t1", "?t2"]);
                let (result_str, case) = if c_dim == s_dim {
                    let dim = c_dim as usize;
                    if SymVal::zero(la.manager.clone()) <= begin && end <= t1_shape[dim] {
                        // 0 <= begin <= end <= t1_shape[dim]
                        let result_str = if begin == 0 && end == t1_shape[dim] {
                            format!("?t1")
                        } else {
                            format!("(slice ?t1 ?s_dim {begin} {end} 1)")
                        };
                        (Some(result_str), "1")
                    } else if t1_shape[dim] <= begin && end <= t1_shape[dim].clone() + t2_shape[dim].clone() {
                        // t1_shape[dim] <= begin <= end <= t1_shape[dim] + t2_shape[dim]
                        let result_str = if begin == t1_shape[dim] && end == t1_shape[dim].clone() + t2_shape[dim].clone() {
                            format!("?t2")
                        } else {
                            let slice_begin = begin.clone() - t1_shape[dim].clone();
                            let slice_end = end.clone() - t1_shape[dim].clone();
                            format!("(slice ?t2 ?s_dim {slice_begin} {slice_end} 1)")
                        };
                        (Some(result_str), "2")
                    } else if SymVal::zero(la.manager.clone()) <= begin && begin <= t1_shape[dim] && t1_shape[dim] <= end && end <= t1_shape[dim].clone() + t2_shape[dim].clone() {
                        // 0 <= begin <= t1_shape[dim] <= end <= t1_shape[dim] + t2_shape[dim]
                        let begin1 = begin.clone();
                        let end1 = t1_shape[dim].clone();
                        let begin2 = 0;
                        let end2 = end.clone() - t1_shape[dim].clone();
                        (Some(format!("(concat (slice ?t1 ?s_dim {begin1} {end1} 1) (slice ?t2 ?s_dim {begin2} {end2} 1) ?c_dim)")), "3")
                    } else {
                        (None, "")
                    }
                } else {
                    (Some(format!("(concat (slice ?t1 ?s_dim {begin} {end} 1) (slice ?t2 ?s_dim {begin} {end} 1) ?c_dim)")), "4")
                };
                if let Some(result_str)=result_str {
                    let unioned = la.union_src(egraph, subst, &result_str);
                    return unioned as usize;
                }
                return 0;
            };
        "<concat-slice-swap>" =>
            "(concat ?t1 (slice ?t2 ?dim 0 ?end 1) ?dim)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let end = get_sym_val(egraph, subst, "?end");
                let t1_shape = get_shape(egraph, subst, "?t1");
                let new_end = end.clone() + t1_shape[dim].clone();
                let result_str = format!("(slice (concat ?t1 ?t2 ?dim) ?dim 0 {new_end} 1)");
                let unioned = la.union_src(egraph, subst, &result_str);
                return unioned as usize;
            };
        "<slice-slice-swap>" =>
            "(slice (slice ?t ?dim2 ?begin2 ?end2 ?step2) ?dim1 ?begin1 ?end1 ?step1)" => |la, egraph, matched_id, subst| {
                let [dim1, dim2] = get_usize_vals!(egraph, subst, ["?dim1", "?dim2"]);
                let [begin1, end1, step1, begin2, end2, step2] = get_sym_vals!(egraph, subst, ["?begin1", "?end1", "?step1", "?begin2", "?end2", "?step2"]);
                if dim1 == dim2 && step1 == step2 {
                    let begin = begin1.clone() + begin2.clone();
                    let end = begin.clone() + end1.clone() - begin1.clone();
                    assert!(end <= end2);
                    let unioned = la.union_src(egraph, subst, &format!("(slice ?t ?dim1 {begin} {end} 1)"));
                    return unioned as usize;
                } else if dim1 != dim2 {
                    let unioned = la.union_src(egraph, subst, &format!("(slice (slice ?t ?dim1 {begin1} {end1} {step1}) ?dim2 {begin2} {end2} {step2})"));
                    return unioned as usize;
                }
                return 0;
            };
        // "<concat-zeros-rewrite>" =>
        //     "(concat (fill ?shape1 0) (fill ?shape2 0) ?dim)" => |la, egraph, matched_id, subst| {
        //         let dim = get_val(egraph, subst, "?dim") as usize;
        //         let shape1 = get_shape_from_name(egraph, subst, "?shape1", la.manager.clone());
        //         let shape2 = get_shape_from_name(egraph, subst, "?shape2", la.manager.clone());
        //         let new_shape = {
        //             let mut tmp = shape1.clone();
        //             tmp[dim] = shape1[dim].clone() + shape2[dim].clone();
        //             tmp
        //         };
        //         let new_shape_str = shape_to_underscore_name(&new_shape);
        //         let (_, unioned) = egraph.union_instantiations(
        //             &la.src_pat.parse::<Pattern<Mdl>>().unwrap().ast,
        //             &format!("(fill {new_shape_str} 0)").parse::<Pattern<Mdl>>().unwrap().ast,
        //             subst,
        //             la.name.as_str(),
        //         );
        //         if la.verbose && unioned {
        //             println!("Rewrite for concat-zeros-rewrite: (concat (fill ?shape1{shape1:?} 0) (fill ?shape2{shape2:?} 0) {dim}) ---> (fill {new_shape_str} 0)");
        //         }
        //         return unioned as usize;
        //     };
        // "<slice-zeros-rewrite>" =>
        //     "(slice (fill ?shape 0) ?dim ?begin ?end 1)" => |la, egraph, matched_id, subst| {
        //         let dim = get_val(egraph, subst, "?dim") as usize;
        //         let begin = get_sym_val(egraph, subst, "?begin");
        //         let end = get_sym_val(egraph, subst, "?end");
        //         let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
        //         let new_shape_str = {
        //             let mut tmp = shape.clone();
        //             tmp[dim] = end.clone() - begin.clone();
        //             shape_to_underscore_name(&tmp)
        //         };
        //         let (_, unioned) = egraph.union_instantiations(
        //             &la.src_pat.parse::<Pattern<Mdl>>().unwrap().ast,
        //             &format!("(fill {new_shape_str} 0)").parse::<Pattern<Mdl>>().unwrap().ast,
        //             subst,
        //             la.name.as_str(),
        //         );
        //         if la.verbose && unioned {
        //             println!("Rewrite for slice-zeros-rewrite: (slice (fill ?shape{shape:?} 0) {dim} {begin} {end} 1) ---> (fill {new_shape_str} 0)");
        //         }
        //         return unioned as usize;
        //     };
    );

    results
    // vec![]
}
