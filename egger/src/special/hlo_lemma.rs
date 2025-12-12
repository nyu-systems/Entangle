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
use crate::rewrite::norm_activation_lemma::activation_template;
use crate::symval::{ShapeLike, SymValManagerRef};
use crate::utils::*;
use std::convert::TryInto;

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    let mut results = vec![];
    results.extend(activation_template!(manager, verbose, "hlo_logistic"));
    #[allow(unused_variables)]
    results.extend(make_rewrites!(manager, verbose,
        "<hlo_broadcast-concat-swap>" =>
            "(hlo_broadcast (concat ?t1 ?t2 ?dim) ?dims ?shape)" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, ["?t1", "?t2"]);
                let dim = get_val(egraph, subst, "?dim") as usize;
                let dims = get_val_shape_from_name::<i64>(egraph, subst, "?dims");
                let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
                let new_dim = dims[dim] as usize;
                let new_shape1_str = {
                    let mut new_shape = shape.clone();
                    new_shape[new_dim] = t1_shape[dim].clone();
                    shape_to_underscore_name(&new_shape)
                };
                let new_shape2_str = {
                    let mut new_shape = shape.clone();
                    new_shape[new_dim] = t2_shape[dim].clone();
                    shape_to_underscore_name(&new_shape)
                };
                let unioned = la.union_src(egraph, subst, &format!("(concat (hlo_broadcast ?t1 ?dims {new_shape1_str}) (hlo_broadcast ?t2 ?dims {new_shape2_str}) {new_dim})"));
                return unioned as usize;
            };
        "<hlo_broadcast-maybe-concat>" =>
            "?s=(hlo_broadcast ?t ?dims ?shape), ?sub=(hlo_broadcast ?t ?dims ?sub_shape)" => |la, egraph, matched_id, subst| {
                let [shape, sub_shape] = get_val_shapes_from_names!(egraph, subst, ["?shape", "?sub_shape"]);
                if shape.len() != sub_shape.len() { return 0; }
                // Try to find the only different dim.
                let different_dim = {
                    let mut d = None;
                    for i in 0..shape.len() {
                        if shape[i] != sub_shape[i] {
                            if d.is_none() {
                                d = Some(i);
                            } else { return 0; }
                        }
                    }
                    if d.is_none() { return 0; }
                    d.unwrap()
                };

                if shape[different_dim] % sub_shape[different_dim] == 0 {
                    let count = shape[different_dim] / sub_shape[different_dim];
                    let mut concat_str = "?sub".to_string();
                    for _ in 1..count {
                        concat_str = format!("(concat {concat_str} ?sub {different_dim})");
                    }
                    let unioned = la.union(egraph, subst, "?s", &concat_str);
                    return unioned as usize;
                }
                return 0;
            };
        "<hlo_dot-concat-lhs-swap>" =>
            "(hlo_dot (concat ?lhs1 ?lhs2 ?dim) ?rhs ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch)" => |la, egraph, matched_id, subst| {
                let [lhs1_shape, lhs2_shape, rhs_shape] = get_shapes!(egraph, subst, ["?lhs1", "?lhs2", "?rhs"]);
                let [lhs_con, rhs_con, lhs_batch, rhs_batch] = get_val_shapes_from_names!(egraph, subst, ["?lhs_con", "?rhs_con", "?lhs_batch", "?rhs_batch"]);
                let dim = get_val(egraph, subst, "?dim");
                let lhs_left_dim = get_left_dim(lhs1_shape.len(), &lhs_con, &lhs_batch) as i64;
                let rhs_left_dim = get_left_dim(rhs_shape.len(), &rhs_con, &rhs_batch) as i64;

                let dim_pos_in_lhs_batch = lhs_batch.iter().position(|&d| d == dim);
                let dim_pos_in_rhs_batch = rhs_batch.iter().position(|&d| d == dim);
                if dim == lhs_left_dim {
                    // Concat at rhs_left_dim, this means a simple partition on rhs.
                    let new_n_dim = lhs_con.len() + lhs_batch.len() + 1;
                    let new_dim = new_n_dim - 2;
                    let unioned = la.union_src(egraph, subst, &format!("(concat (hlo_dot ?lhs1 ?rhs ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) (hlo_dot ?lhs2 ?rhs ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) {new_dim})"));
                    return unioned as usize;
                }
                return 0;
            };
        "<hlo_dot-concat-rhs-swap>" =>
            "(hlo_dot ?lhs (concat ?rhs1 ?rhs2 ?dim) ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch)" => |la, egraph, matched_id, subst| {
                let [lhs_shape, rhs1_shape, rhs2_shape] = get_shapes!(egraph, subst, ["?lhs", "?rhs1", "?rhs2"]);
                let [lhs_con, rhs_con, lhs_batch, rhs_batch] = get_val_shapes_from_names!(egraph, subst, ["?lhs_con", "?rhs_con", "?lhs_batch", "?rhs_batch"]);
                let dim = get_val(egraph, subst, "?dim");
                let lhs_left_dim = get_left_dim(lhs_shape.len(), &lhs_con, &lhs_batch) as i64;
                let rhs_left_dim = get_left_dim(rhs1_shape.len(), &rhs_con, &rhs_batch) as i64;

                let dim_pos_in_lhs_batch = lhs_batch.iter().position(|&d| d == dim);
                let dim_pos_in_rhs_batch = rhs_batch.iter().position(|&d| d == dim);
                if dim == rhs_left_dim {
                    // Concat at rhs_left_dim, this means a simple partition on rhs.
                    let new_n_dim = lhs_con.len() + lhs_batch.len() + 1;
                    let new_dim = new_n_dim - 1;
                    let unioned = la.union_src(egraph, subst, &format!("(concat (hlo_dot ?lhs ?rhs1 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) (hlo_dot ?lhs ?rhs2 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) {new_dim})"));
                    return unioned as usize;
                }
                return 0;
            };
        "<hlo_dot-dual-concat-swap>" =>
            "(hlo_dot (concat ?lhs1 ?lhs2 ?lhs_dim) (concat ?rhs1 ?rhs2 ?rhs_dim) ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch)" => |la, egraph, matched_id, subst| {
                let [lhs1_shape, lhs2_shape, rhs1_shape, rhs2_shape] = get_shapes!(egraph, subst, ["?lhs1", "?lhs2", "?rhs1", "?rhs2"]);
                let [lhs_con, rhs_con, lhs_batch, rhs_batch] = get_val_shapes_from_names!(egraph, subst, ["?lhs_con", "?rhs_con", "?lhs_batch", "?rhs_batch"]);
                let [lhs_dim, rhs_dim] = get_usize_vals!(egraph, subst, ["?lhs_dim", "?rhs_dim"]);
                let lhs_left_dim = get_left_dim(lhs1_shape.len(), &lhs_con, &lhs_batch) as i64;
                let rhs_left_dim = get_left_dim(rhs1_shape.len(), &rhs_con, &rhs_batch) as i64;

                let dim_pos_in_lhs_batch = lhs_batch.iter().position(|&d| d == lhs_dim as i64);
                let dim_pos_in_rhs_batch = rhs_batch.iter().position(|&d| d == rhs_dim as i64);
                if dim_pos_in_lhs_batch.is_some() && dim_pos_in_lhs_batch == dim_pos_in_rhs_batch {
                    // Concat at batch dims, then the result would simply be the concat again.
                    let new_dim = dim_pos_in_lhs_batch.unwrap();
                    let unioned = la.union_src(egraph, subst, &format!("(concat (hlo_dot ?lhs1 ?rhs1 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) (hlo_dot ?lhs2 ?rhs2 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) {new_dim})"));
                    return unioned as usize;
                } else if lhs_con.len() == 1 && rhs_con.len() == 1 && lhs_dim as i64 == lhs_con[0] && rhs_dim as i64 == rhs_con[0] {
                    let unioned = la.union_src(egraph, subst, &format!("(reduce_add (hlo_dot ?lhs1 ?rhs1 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) (hlo_dot ?lhs2 ?rhs2 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch))"));
                    return unioned as usize;
                }
                return 0;
            };

        "<hlo_dot-slice-swap>" =>
            "(hlo_dot ?t1 (slice ?t2 ?dim ?begin ?end 1) ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch)" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, ["?t1", "?t2"]);
                let [lhs_con, rhs_con, lhs_batch, rhs_batch] = get_val_shapes_from_names!(egraph, subst, ["?lhs_con", "?rhs_con", "?lhs_batch", "?rhs_batch"]);
                let dim = get_val(egraph, subst, "?dim");
                let lhs_left_dim = get_left_dim(t1_shape.len(), &lhs_con, &lhs_batch) as i64;
                let rhs_left_dim = get_left_dim(t2_shape.len(), &rhs_con, &rhs_batch) as i64;
                if lhs_batch.contains(&dim) || rhs_batch.contains(&dim) { return 0; }
                if dim == rhs_left_dim {
                    let new_n_dim = lhs_con.len() + lhs_batch.len() + 1;
                    let new_dim = new_n_dim - 1;
                    let unioned = la.union_src(egraph, subst, &format!("(slice (hlo_dot ?t1 ?t2 ?lhs_con ?rhs_con ?lhs_batch ?rhs_batch) {new_dim} ?begin ?end 1)"));
                    return unioned as usize;
                }
                return 0;
            };

        "<hlo_max-concat-swap>" =>
            "(hlo_max (concat ?t1 ?t2 ?dim1) ?dim2)" => |la, egraph, matched_id, subst| {
                let [dim1, dim2] = get_usize_vals!(egraph, subst, ["?dim1", "?dim2"]);
                if dim1 != dim2 {
                    let unioned = la.union_src(egraph, subst, &format!("(concat (hlo_max ?t1 ?dim2) (hlo_max ?t2 ?dim2) ?dim1)"));
                    return unioned as usize;
                }
                return 0;
            };

        "<hlo-reshape-concat-swap>" =>
            "?rc=(reshape (concat ?t1 ?t2 ?dim) ?shape),\
             ?rc1=(reshape ?t1 ?shape1), \
             ?rc2=(reshape ?t2 ?shape2)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let shape = if let Some(v)=try_get_val_shape_from_name(egraph, subst, "?shape", la.manager.clone()) { v } else { return 0; };
                let t1_shape = if let Some(val) = try_get_val_shape(egraph, subst, "?t1") { val } else { return 0; };
                let t2_shape = if let Some(val) = try_get_val_shape(egraph, subst, "?t2") { val } else { return 0; };
                let concat_shape = {
                    let mut tmp = t1_shape.clone();
                    tmp[dim as usize] += t2_shape[dim as usize];
                    tmp
                };
                let pattern_src = concat_shape.clone();
                let pattern_dst = shape.clone();

                let shape1 = if let Some(v)=try_get_val_shape_from_name(egraph, subst, "?shape1", la.manager.clone()) { v } else { return 0; };
                let shape2 = if let Some(v)=try_get_val_shape_from_name(egraph, subst, "?shape2", la.manager.clone()) { v } else { return 0; };

                let mut potential_new_info: Vec<(usize, &str)> = vec![];

                if dim == 1 && pattern_src.len() == 2 && pattern_dst.len() == 4 && shape1.len() == 4 && shape2.len() == 4
                    && shape[..2] == shape1[..2] && shape[..2] == shape2[..2] && shape[3] == shape1[3] && shape[3] == shape2[3]
                    && shape[2] == shape1[2] + shape2[2]
                {
                    // After partitioning fused qkv, there is a reshape [b*s, nh*h] -> [s, b, nh, h]
                    // When t1 is [b*s, nh1*h] and t2 is [b*s, nh2*h]. The concat can be swapped.
                    potential_new_info.push((2, "1"));
                } else if dim == 2 && pattern_src.len() == 4 && pattern_dst.len() == 2 && shape1.len() == 2 && shape2.len() == 2
                    && shape[0] == t1_shape[0] * t1_shape[1] && shape[0] == t2_shape[0] * t2_shape[1]
                    && shape[1] == t1_shape[2] * t1_shape[3] + t1_shape[2] * t1_shape[3]
                    && shape[0] == shape1[0] && shape[0] == shape2[0] && shape[1] == shape1[1] + shape2[1]
                {
                    // After attention, there is a reshape back [s, b, nh, h] -> [b*s, nh*h]
                    // When t1 is [b,s,nh1,h] and t2 is [b,s,nh2,h]. The concat can be swapped.
                    potential_new_info.push((1, "2"));
                }

                let mut unioned_count = 0;
                for (new_dim, case) in potential_new_info {
                    let unioned = la.union(egraph, subst, "?rc", &format!("(concat ?rc1 ?rc2 {new_dim})"));
                    unioned_count += unioned as usize;
                }
                return unioned_count;
            };
        "<hlo_select-all-concat-swap>" =>
            "(hlo_select (concat ?mask1 ?mask2 ?dim) (concat ?on_true1 ?on_true2 ?dim) (concat ?on_false1 ?on_false2 ?dim))" => "(concat (hlo_select ?mask1 ?on_true1 ?on_false1) (hlo_select ?mask2 ?on_true2 ?on_false2) ?dim)"; 
    ));

    results
}
