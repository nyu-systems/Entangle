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

macro_rules! full_consecutive_slices_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        assert!($n >= 2);
        let name = format!("<consecutive-slices-back({})>", $n);
        let pat = (0..$n)
            .map(|i| format!("?s{i}=(slice ?t ?dim ?offset{i} ?offset{} 1)", i+1))
            .collect::<Vec<_>>()
            .join(", ");
        make_rewrites!($manager, $verbose,
            name => pat => |la, egraph, _matched_id, subst| {
                let [first_offset, last_offset] = get_sym_vals!(egraph, subst, ["?offset0", &format!("?offset{}", $n)]);
                if first_offset != 0 { return 0; }
                let t_shape = get_shape(egraph, subst, "?t");
                let dim = get_val(egraph, subst, "?dim") as usize;
                if t_shape[dim] != last_offset { return 0; }
                let concat_pat = (1..$n).fold("?s0".to_string(), |acc, i| format!("(concat {acc} ?s{i} ?dim)"));
                let unioned = la.union(egraph, subst, "?t", &concat_pat);
                return unioned as usize;
            }
        )
    }};
}

macro_rules! full_consecutive_slice_scatters_to_zeros_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let name = format!("<full-slice_scatter-to-zeros-concat-equals-reduce_add-lower({})>", $n);
        let concat_src = (1..$n).fold("?src0".to_string(), |acc, i| format!("(concat {acc} ?src{i} ?dim)", i=i));
        let s_pat = format!("?s=(slice_scatter (fill ?zero_shape 0) {concat_src} ?dim ?offset0 ?offset{})", $n);
        let ss_pat = (0..$n).map(|i| format!("?s{i}=(slice_scatter (fill ?zero_shape 0) ?src{i} ?dim ?offset{i} ?offset{})", i+1)).collect::<Vec<_>>().join(", ");
        make_rewrite!(($manager, $verbose, name) => format!("{s_pat},{ss_pat}") => |la, egraph, _matched_id, subst| {
            let dim = get_val(egraph, subst, "?dim") as usize;
            let [first_offset, last_offset] = get_sym_vals!(egraph, subst, ["?offset0", &format!("?offset{}", $n)]);
            let zero_shape = get_shape_from_name(egraph, subst, "?zero_shape", la.manager.clone());

            if first_offset == 0 && last_offset == zero_shape[dim] {
                let sum = (1..$n).fold("?s0".to_string(), |acc, i| format!("(reduce_add {acc} ?s{i})", i=i));
                let unioned = la.union(egraph, subst, "?s", &sum);
               return unioned as usize;
            }
            return 0;
        })
    }};
}

macro_rules! full_consecutive_slice_scatters_to_empty_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let name = format!("<full-slice_scatter-to-empty-concat-swap({})>", $n);
        let concat_src = (1..$n).fold("?src0".to_string(), |acc, i| format!("(concat {acc} ?src{i} ?cdim)", i=i));
        let s_pat = format!("?s=(slice_scatter (empty ?shape) {concat_src} ?sdim ?begin ?end)");
        let ss_pat = (0..$n).map(|i| format!("?s{i}=(slice_scatter (empty ?sub_shape) ?src{i} ?sdim ?begin ?end)")).collect::<Vec<_>>().join(", ");
        make_rewrite!(($manager, $verbose, name) => format!("{s_pat},{ss_pat}") => |la, egraph, _matched_id, subst| {
            let [cdim] = get_usize_vals!(egraph, subst, ["?cdim"]);
            let [shape, sub_shape] = get_shapes_from_names!(egraph, subst, ["?shape", "?sub_shape"], la.manager.clone());
            if !shape_eq_except(&shape, &sub_shape, cdim) { return 0; }
            if shape[cdim] != sub_shape[cdim].clone() * SymVal::new_val($n, la.manager.clone()) { return 0; }
            let res = (1..$n).fold("?s0".to_string(), |acc, i| format!("(concat {acc} ?s{i} {cdim})", i=i));
            let unioned = la.union(egraph, subst, "?s", &res);
            return unioned as usize;
        })
    }};
}

macro_rules! index_matadd_swap_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let name = format!("<index-matadd-swap({})>", $n);
        let sum_pat = (1..$n).fold(
            "(slice_scatter (fill ?zero_shape 0) ?t0 ?dim ?offset0 ?offset1)".to_string(),
            |acc, i| format!("(matadd {acc} (slice_scatter (fill ?zero_shape 0) ?t{i} ?dim ?offset{} ?offset{}))", i, i+1),
        );
        let pat = format!("(index {sum_pat} ?indices)");
        make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
            let [first_offset, last_offset] = get_sym_vals!(egraph, subst, ["?offset0", &format!("?offset{}", $n)]);
            if first_offset != 0 { return 0; }
            let zero_shape = get_shape_from_name(egraph, subst, "?zero_shape", la.manager.clone());
            let dim = get_val(egraph, subst, "?dim") as usize;
            if last_offset != zero_shape[dim] { return 0; }
            let sum = (1..$n).fold(
                "(index (slice_scatter (fill ?zero_shape 0) ?t0 ?dim ?offset0 ?offset1) ?indices)".to_string(),
                |acc, i| format!("(matadd {acc} (index (slice_scatter (fill ?zero_shape 0) ?t{i} ?dim ?offset{} ?offset{}) ?indices))", i, i+1),
            );
            let unioned = la.union_src(egraph, subst, &sum);
            return unioned as usize;
        })
    }}
}

macro_rules! topk_concat_swap_template {
    ($manager:expr, $verbose:expr, $return_idx:expr) => {{
        make_rewrite!(($manager, $verbose, format!("<topk_{}-concat-swap>", $return_idx)) => 
            format!("(topk_{} (concat ?inpt1 ?inpt2 ?dim) ?k ?topk_dim ?largest ?sorted)", $return_idx) => |la, egraph, _matched_id, subst| 
        {
            let [dim, topk_dim] = get_usize_vals!(egraph, subst, ["?dim", "?topk_dim"]);
            if dim != topk_dim {
                let unioned = la.union_src(egraph, subst, &format!("(concat (topk_{ri} ?inpt1 ?k ?topk_dim ?largest ?sorted) (topk_{ri} ?inpt2 ?k ?topk_dim ?largest ?sorted) ?dim)", ri=$return_idx));
                return unioned as usize;
            }
            return 0;
        })  
    }};
}

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let mut results = vec![];
    results.extend(full_consecutive_slices_template!(manager, verbose, 2));
    results.extend(full_consecutive_slices_template!(manager, verbose, 4));
    results.extend(full_consecutive_slices_template!(manager, verbose, 6));
    results.extend(full_consecutive_slices_template!(manager, verbose, 8));
    #[allow(unused_variables)]
    results.extend(make_rewrites!(manager, verbose,
        "<full-slice-back>" =>
            "(slice ?t ?dim 0 ?end 1)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let end = get_sym_val(egraph, subst, "?end");
                let t_shape = get_shape(egraph, subst, "?t");
                if end == t_shape[dim] {
                    let unioned = la.union_src(egraph, subst, "?t");
                    return unioned as usize;
                }
                return 0;
            };
        "<full-index_put-back>" =>
            "(index_put ?inpt1 ?indices ?value)" => |la, egraph, matched_id, subst| {
                let [inpt1_shape, value_shape] = get_shapes!(egraph, subst, ["?inpt1", "?value"]);
                // FIXME: This is a hacky way to describe the lemma. This is true, but it is a very narrow case.
                if inpt1_shape == value_shape {
                    let unioned = la.union_src(egraph, subst, "?value");
                    return unioned as usize;
                }
                return 0;
            };
        "<reshape-to-same-shape>" =>
            "(reshape ?t ?shape)" => |la, egraph, matched_id, subst| {
                let t_shape = get_shape(egraph, subst, "?t");
                let target_shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
                if t_shape == target_shape {
                    let unioned = la.union_src(egraph, subst, "?t");
                    return unioned as usize;
                }
                return 0;
            };
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
                    // TODO: CASE
                    let unioned = la.union(egraph, subst, &la.src_pat, &result_str);
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
                let unioned = la.union(egraph, subst, &la.src_pat, &result_str);
                return unioned as usize;
            };
        "<concat-concat-2-2-swap>" =>
            "(concat (concat ?t1 ?t2 ?dim2) (concat ?t3 ?t4 ?dim2) ?dim1)" => |la, egraph, matched_id, subst| {
                let [dim1, dim2] = get_usize_vals!(egraph, subst, ["?dim1", "?dim2"]);
                let [t1_shape, t2_shape, t3_shape, t4_shape] = get_shapes!(egraph, subst, ["?t1", "?t2", "?t3", "?t4"]);
                if dim1 != dim2 && shape_eq_except(&t1_shape, &t3_shape, dim1) && shape_eq_except(&t2_shape, &t4_shape, dim1) {
                    let unioned = la.union_src(egraph, subst, "(concat (concat ?t1 ?t3 ?dim1) (concat ?t2 ?t4 ?dim1) ?dim2)");
                    return unioned as usize;
                }
                return 0;
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
        "<transpose-concat-swap>" =>
            "(transpose (concat ?t1 ?t2 ?dim) ?perm)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim");
                let perm = if let Some(v)=try_get_val_shape_from_name(egraph, subst, "?perm", la.manager.clone()) { v } else { return 0; };
                let new_concat_dim = perm.iter().position(|&x| x == dim).unwrap();

                let unioned = la.union_src(egraph, subst, &format!("(concat (transpose ?t1 ?perm) (transpose ?t2 ?perm) {new_concat_dim})"));
                return unioned as usize;
            };
        "<transpose-slice-swap>" =>
            "(transpose (slice ?t ?dim ?begin ?end ?step) ?perm)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim");
                let perm = if let Some(v)=try_get_val_shape_from_name(egraph, subst, "?perm", la.manager.clone()) { v } else { return 0; };
                let new_dim = perm.iter().position(|&x| x == dim).unwrap();
                let [begin, end, step] = get_sym_vals!(egraph, subst, ["?begin", "?end", "?step"]);
                let unioned = la.union_src(egraph, subst, &format!("(slice (transpose ?t ?perm) {new_dim} {begin} {end} {step})"));
                return unioned as usize;
            };
        "<scatter_src-concat-swap>" =>
            "(scatter_src (concat ?inpt1 ?inpt2 ?concat_dim) (concat ?index1 ?index2 ?concat_dim) (concat ?src1 ?src2 ?concat_dim) ?dim)" => |la, egraph, matched_id, subst| {
                let [concat_dim, dim] = get_usize_vals!(egraph, subst, ["?concat_dim", "?dim"]);
                let [index1_shape, index2_shape, src1_shape, src2_shape] = get_shapes!(egraph, subst, ["?index1", "?index2", "?src1", "?src2"]);
                if concat_dim != dim && index1_shape == src1_shape && index2_shape == src2_shape {
                    let unioned = la.union_src(egraph, subst, "(concat (scatter_src ?inpt1 ?index1 ?src1 ?dim) (scatter_src ?inpt2 ?index2 ?src2 ?dim) ?concat_dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<slice_scatter-concat-rewrite>" =>
            "(slice_scatter (concat ?inpt1 (concat ?inpt2 ?inpt3 ?dim) ?dim) ?src ?dim ?begin ?end)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [begin, end] = get_sym_vals!(egraph, subst, ["?begin", "?end"]);
                let [inpt1_shape, inpt2_shape, inpt3_shape] = get_shapes!(egraph, subst, ["?inpt1", "?inpt2", "?inpt3"]);
                println!("Pre slice_scatter-concat-rewrite: inpt1_shape={inpt1_shape:?}, inpt2_shape={inpt2_shape:?}, inpt3_shape={inpt3_shape:?}, begin={begin}, end={end}");
                if begin == inpt1_shape[dim].clone() && end - begin == inpt2_shape[dim] {
                    let unioned = la.union_src(egraph, subst, "(concat (concat ?inpt1 ?src ?dim) ?inpt3 ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<slice_scatter-concat-rewrite>" =>
            "(slice_scatter ?inpt ?src ?dim ?begin ?end)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [begin, end] = get_sym_vals!(egraph, subst, ["?begin", "?end"]);
                let [inpt_shape, src_shape] = get_shapes!(egraph, subst, ["?inpt", "?src"]);
                if begin == 0 && end == inpt_shape[dim] {
                    let unioned = la.union_src(egraph, subst, "?src");
                    return unioned as usize;
                } else if begin == 0 && end < inpt_shape[dim] {
                    let unioned = la.union_src(egraph, subst, &format!("(concat ?src (slice ?inpt {dim} {end} {} 1) ?dim)", inpt_shape[dim]));
                    return unioned as usize;
                } else if begin > 0 && end == inpt_shape[dim] {
                    let unioned = la.union_src(egraph, subst, &format!("(concat (slice ?inpt {dim} 0 {begin} 1) ?src ?dim)"));
                    return unioned as usize;
                } else {
                    // let unioned = la.union_src(egraph, subst, &format!("(concat (concat (slice ?inpt {dim} 0 {begin} 1) ?src ?dim) (slice ?inpt {dim} {end} {} 1) {dim})", inpt_shape[dim]));
                    // return unioned as usize;
                }
                return 0;
            };
        "<full-slice_backward-back>" =>
            "(slice_backward ?grad_output ?input_shape ?dim 0 ?end 1)" => |la, egraph, matched_id, subst| {
                let grad_output_shape = get_shape(egraph, subst, "?grad_output");
                let dim = get_val(egraph, subst, "?dim") as usize;
                let end = get_sym_val(egraph, subst, "?end");

                if grad_output_shape[dim] == end {
                    let unioned = la.union_src(egraph, subst, "?grad_output");
                    return unioned as usize;
                }
                return 0;
            };
        "<slice_scatter-consecutive>" =>
            "(slice_scatter (slice_scatter ?inpt1 ?src1 ?dim ?begin ?mid) ?src2 ?dim ?mid ?end)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [src1_shape, src2_shape] = get_shapes!(egraph, subst, ["?src1", "?src2"]);
                if shape_eq_except(&src1_shape, &src2_shape, dim) {
                    let src_n_dim = src1_shape.len();
                    let unioned = la.union_src(egraph, subst, "(slice_scatter ?inpt1 (concat ?src1 ?src2 ?dim) ?dim ?begin ?end)");
                    return unioned as usize;
                }
                return 0;
            };
        
        "<slice-slice_scatter-back>" => "(slice (slice_scatter ?inpt ?src ?dim ?start ?end) ?dim ?start ?end 1)" => "?src";

        "<full-slice_scatter-back>" =>
            "(slice_scatter ?inpt ?src ?dim 0 ?end)" => |la, egraph, matched_id, subst| {
                let [inpt_shape, src_shape] = get_shapes!(egraph, subst, ["?inpt", "?src"]);
                let dim = get_val(egraph, subst, "?dim") as usize;
                let end = get_sym_val(egraph, subst, "?end");
                if inpt_shape == src_shape && inpt_shape[dim] == end {
                    let unioned = la.union_src(egraph, subst, "?src");
                    return unioned as usize;
                }
                return 0;
            };
            
        "<padding-back>" =>
            "(pad ?t 0 0 0 ?i 0 0)" => |la, egraph, matched_id, subst| {
                // XXX: BACK_LEMMA
                let t_shape = get_shape(egraph, subst, "?t");
                let dim = t_shape.len() - 2;
                let end = &t_shape[dim];
                let unioned = la.union(egraph, subst, &format!("(slice {} {dim} 0 {end} 1)", la.src_pat), "?t");
                return unioned as usize;
            };
            
    ));

    // results.push(make_simple_rewrite!((manager, verbose, "<concat-associate>") => "(concat ?t (concat ?t1 ?t2 ?dim) ?dim)" => "(concat (concat ?t ?t1 ?dim) ?t2 ?dim)"));
    results.push(full_consecutive_slice_scatters_to_zeros_template!(manager, verbose, 2));
    results.push(full_consecutive_slice_scatters_to_zeros_template!(manager, verbose, 4));
    results.push(full_consecutive_slice_scatters_to_zeros_template!(manager, verbose, 8));
    results.push(full_consecutive_slice_scatters_to_empty_template!(manager, verbose, 2));
    results.push(full_consecutive_slice_scatters_to_empty_template!(manager, verbose, 4));
    results.push(full_consecutive_slice_scatters_to_empty_template!(manager, verbose, 8));
    results.push(index_matadd_swap_template!(manager, verbose, 2));
    results.push(index_matadd_swap_template!(manager, verbose, 4));
    results.push(index_matadd_swap_template!(manager, verbose, 8));
    results.push(topk_concat_swap_template!(manager, verbose, 0));
    results.push(topk_concat_swap_template!(manager, verbose, 1));

    results
}
