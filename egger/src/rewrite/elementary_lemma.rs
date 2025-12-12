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
use crate::rewrite::lambda_rewrite::*;
use crate::symval::SymValManagerRef;
use crate::symval::*;
use crate::utils::*;
use egg::*;
use std::convert::TryInto;

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let results = make_rewrites!(manager, verbose,
        "<add_scalar-concat-swap>" =>
            "(add_scalar (concat ?t1 ?t2 ?dim) ?scalar)" => |la, egraph, matched_id, subst| {
                let unioned = la.union_src(egraph, subst, "(concat (add_scalar ?t1 ?scalar) (add_scalar ?t2 ?scalar) ?dim)");
                return unioned as usize;
            };
        "<addcmul-concat-swap>" =>
            "(addcmul (concat ?inpt1 ?inpt2 ?dim) (concat ?tensor11 ?tensor12 ?dim) (concat ?tensor21 ?tensor22 ?dim) ?value)" => |la, egraph, matched_id, subst| {
                let [inpt1_shape, tensor11_shape, tensor21_shape] = get_shapes!(egraph, subst, vec!["?inpt1", "?tensor11", "?tensor21"]);
                if inpt1_shape == tensor11_shape && tensor11_shape == tensor21_shape {
                    let unioned = la.union_src(egraph, subst, "(concat (addcmul ?inpt1 ?tensor11 ?tensor21 ?value) (addcmul ?inpt2 ?tensor12 ?tensor22 ?value) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-concat-swap>" =>
            "(matadd (concat ?t1 ?t2 ?dim) ?t)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t_shape, t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t", "?t1", "?t2"]);
                if !(dim == t1_shape.len() - 1 && t_shape.len() == 1) && t_shape.len() == 1 || t_shape.len() == 0 {
                    let unioned = la.union_src(egraph, subst, "(concat (matadd ?t1 ?t) (matadd ?t2 ?t) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-dual_concat-swap>" =>
            "(matadd (concat ?t1 ?t2 ?dim1) (concat ?t3 ?t4 ?dim2))" => |la, egraph, matched_id, subst| {
                let [shape1, shape2, shape3, shape4] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3", "?t4"]);
                let [dim1, dim2] = get_usize_vals!(egraph, subst, vec!["?dim1", "?dim2"]);
                if shape1 == shape3 && shape2 == shape4 && dim1 == dim2
                    || (dim2 == 0 && shape3.len() == 1 && shape4.len() == 1 && shape1[shape1.len()-1] == shape3[0] && shape2[shape2.len()-1] == shape4[0])
                {
                    let unioned = la.union_src(egraph, subst, "(concat (matadd ?t1 ?t3) (matadd ?t2 ?t4) ?dim1)");
                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-slice-swap>" =>
            "(matadd (slice ?t1 ?dim ?begin ?end ?step) ?t2)" => |la, egraph, matched_id, subst| {
                if get_dtype(egraph, subst, "?t2") != DataKind::Tnsr { return 0; }
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [begin, end, step] = get_sym_vals!(egraph, subst, vec!["?begin", "?end", "?step"]);
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);

                if dim != t1_shape.len() - 1 && t2_shape.len() == 1 {
                    assert!(t2_shape[0] == t1_shape[t1_shape.len() - 1]);
                    // Only when `dim1` and `dim2` are the last dims and t2.shape.len() == 1.
                    let slice_matadd_str = format!("(slice (matadd ?t1 ?t2) ?dim ?begin ?end ?step)");
                    let unioned = la.union_src(egraph, subst, &slice_matadd_str);
                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-dual-slice-swap>" =>
            "(matadd (slice ?t1 ?dim1 ?begin ?end ?step) (slice ?t2 ?dim2 ?begin ?end ?step))" => |la, egraph, matched_id, subst| {
                let [dim1, dim2] = get_usize_vals!(egraph, subst, vec!["?dim1", "?dim2"]);
                let [begin, end, step] = get_sym_vals!(egraph, subst, vec!["?begin", "?end", "?step"]);
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                let n_t1 = t1_shape.len();
                let n_t2 = t2_shape.len();

                if (dim1 == t1_shape.len() - 1 && t2_shape.len() == 1 && dim2 == 0 && t1_shape[dim1] == t2_shape[dim2]) // 1. When `dim1` and `dim2` are the last dims and t2.shape.len() == 1.
                    || (t1_shape == t2_shape && dim1 == dim2) // 2. When same shape and same dim.
                {
                    let slice_matadd_str = format!("(slice (matadd ?t1 ?t2) ?dim1 ?begin ?end ?step)");
                    let unioned = la.union_src(egraph, subst, &slice_matadd_str);
                    return unioned as usize;
                }
                return 0;
            };
        "<tranpose-matadd-swap>" =>
            "(transpose (matadd ?t1 ?t2) ?trans)" => |la, egraph, matched_id, subst| {
                let dtypes: Vec<DataKind> = get_dtypes!(egraph, subst, vec!["?t1", "?t2"]);
                if dtypes.iter().any(|&x| x != DataKind::Tnsr) { return 0; }
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                if t1_shape == t2_shape {
                    let unioned = la.union_src(egraph, subst, "(matadd (transpose ?t1 ?trans) (transpose ?t2 ?trans))");
                    return unioned as usize;
                }
                return 0;
            };
        "<matsub-concat-swap>" =>
            "(matsub (concat ?t1 ?t2 ?dim) ?t3)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t1_shape, t2_shape, t3_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3"]);
                if dim == t3_shape.len() - 1 && broadcast(&t1_shape, &t3_shape).is_some() && broadcast(&t2_shape, &t3_shape).is_some() {
                    let unioned = la.union_src(egraph, subst, "(concat (matsub ?t1 ?t3) (matsub ?t2 ?t3) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<matsub-dual_concat-swap>" =>
            "(matsub (concat ?t1 ?t2 ?dim) (concat ?t3 ?t4 ?dim))" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t1_shape, t2_shape, t3_shape, t4_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3", "?t4"]);
                if broadcast(&t1_shape, &t3_shape).is_some() && broadcast(&t2_shape, &t4_shape).is_some() {
                    let unioned = la.union_src(egraph, subst, "(concat (matsub ?t1 ?t3) (matsub ?t2 ?t4) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<ewmul-concat-swap>" =>
            "(ewmul (concat ?t1 ?t2 ?dim) ?t)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                let concat_shape = {
                    let mut concat_shape = t1_shape.clone();
                    concat_shape[dim] += t2_shape[dim].clone();
                    concat_shape
                };
                let t_dtype = get_dtype(egraph, subst, "?t");
                if t_dtype.is_scalar_or_name()
                    || t_dtype == DataKind::Tnsr && {
                        let t_shape = get_shape(egraph, subst, "?t");
                        t_shape.len() == 0 || (t1_shape.len() == t2_shape.len() && t2_shape.len() == t_shape.len() && t_shape[dim] == 1)
                    }
                {
                    let unioned = la.union_src(egraph, subst, "(concat (ewmul ?t1 ?t) (ewmul ?t2 ?t) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<ewmul-dual-concat-swap>" =>
            "(ewmul (concat ?t1 ?t2 ?dim) (concat ?t3 ?t4 ?dim))" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape, t3_shape, t4_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3", "?t4"]);
                if t1_shape == t3_shape && t2_shape == t4_shape
                    || t3_shape[t3_shape.len()-1] == 1 && t4_shape[t4_shape.len()-1] == 1 && t1_shape[..t1_shape.len()-1] == t3_shape[..t3_shape.len()-1] && t2_shape[..t2_shape.len()-1] == t4_shape[..t4_shape.len()-1]
                {
                    let unioned = la.union_src(egraph, subst, "(concat (ewmul ?t1 ?t3) (ewmul ?t2 ?t4) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<ewmul-reduce_add-swap>" =>
            "(ewmul (reduce_add ?t1 ?t2) ?t3)" => |la, egraph, matched_id, subst| {
                let okay = if get_dtype(egraph, subst, "?t3").is_scalar_or_name() {
                    true
                } else {
                    let [t1_shape, t2_shape, t3_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3"]);
                    t1_shape == t2_shape && t2_shape == t3_shape
                };
                if okay {
                    let unioned = la.union_src(egraph, subst, "(reduce_add (ewmul ?t1 ?t3) (ewmul ?t2 ?t3))");
                    return unioned as usize;
                }
                return 0;
            };

        "<transpose-reduce_add-swap>" =>
            "(transpose (reduce_add ?t1 ?t2) ?trans)" => "(reduce_add (transpose ?t1 ?trans) (transpose ?t2 ?trans))";

        "<matdiv-concat-swap>" =>
            "(matdiv (concat ?t1 ?t2 ?dim) ?t)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                let t_dtype = get_dtype(egraph, subst, "?t");
                if t_dtype == DataKind::Tnsr {
                    let t_shape = get_shape(egraph, subst, "?t");
                    // FIXME: This is only a special case.
                    if t1_shape.len() == t2_shape.len()
                        && t2_shape.len() == t_shape.len()
                        && t_shape[dim] == 1
                    {
                        let unioned = la.union_src(egraph, subst, "(concat (matdiv ?t1 ?t) (matdiv ?t2 ?t) ?dim)");
                        return unioned as usize;
                    }
                } else if t_dtype.is_scalar_or_name() {
                    let unioned = la.union_src(egraph, subst, "(concat (matdiv ?t1 ?t) (matdiv ?t2 ?t) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<matdiv-dual_concat-swap>" =>
            "(matdiv (concat ?t1 ?t2 ?dim) (concat ?t3 ?t4 ?dim))" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [t1_shape, t2_shape, t3_shape, t4_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3", "?t4"]);
                if broadcast(&t1_shape, &t3_shape).is_some() && broadcast(&t2_shape, &t4_shape).is_some()
                {
                    let unioned = la.union_src(egraph, subst, "(concat (matdiv ?t1 ?t3) (matdiv ?t2 ?t4) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };

        "<matdiv-matadd-swap>" =>
            "(matdiv (matadd ?t1 ?t2) ?t)" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                let t_dtype = get_dtype(egraph, subst, "?t");
                if broadcast(&t1_shape, &t2_shape).is_some() && t_dtype == DataKind::Scalar {
                    let unioned = la.union_src(egraph, subst, "(matadd (matdiv ?t1 ?t) (matdiv ?t2 ?t))");
                    return unioned as usize;
                }
                return 0;
            };

        ///////////////////////////////////////////////////////////////////////////////////////////////
        // Hacky lemmas
        ///////////////////////////////////////////////////////////////////////////////////////////////
        "<ewmul-expand-and-concat>" =>
            "(ewmul (expand ?t1 ?shape) (concat ?t2 ?t3 ?dim))" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape, t3_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3"]);
                assert!(t1_shape.len() == 0, "Unconsidered circumstance.");
                let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
                let dim = get_val(egraph, subst, "?dim") as usize;
                let concat_shape = {
                    let mut concat_shape = t2_shape.clone();
                    concat_shape[dim] += t3_shape[dim].clone();
                    concat_shape
                };

                if shape == concat_shape && dim == t2_shape.len() - 1 && t1_shape.len() == 0 {
                    let shape2_name = shape_to_underscore_name(&t2_shape);
                    let shape3_name = shape_to_underscore_name(&t3_shape);
                    let unioned = la.union_src(egraph, subst, &format!("(concat (ewmul (expand ?t1 {shape2_name}) ?t2) (ewmul (expand ?t1 {shape3_name}) ?t3) ?dim)"));
                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-concat-and-expand>" =>
            "(matadd (concat ?t2 ?t3 ?dim) (expand ?t1 ?shape))" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape, t3_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3"]);
                assert!(t1_shape.len() == 1, "Unconsidered circumstance.");
                let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
                let dim = get_val(egraph, subst, "?dim") as usize;
                let concat_shape = {
                    let mut concat_shape = t2_shape.clone();
                    concat_shape[dim] += t3_shape[dim].clone();
                    concat_shape
                };

                if shape == concat_shape && dim != t2_shape.len() - 1 && t1_shape.len() == 1 && t1_shape[0] == t2_shape[t2_shape.len() - 1] {
                    let shape2_name = shape_to_underscore_name(&t2_shape);
                    let shape3_name = shape_to_underscore_name(&t3_shape);
                    let unioned = la.union_src(egraph, subst, &format!("(concat (matadd (expand ?t1 {shape2_name}) ?t2) (matadd (expand ?t1 {shape3_name}) ?t3) ?dim)"));

                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-slice-and-expand>" =>
            "(matadd (slice ?t2 ?dim ?begin ?end ?step) (expand ?t1 ?shape))" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                assert!(t1_shape.len() == 1, "Unconsidered circumstance.");
                let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [begin, end] = get_sym_vals!(egraph, subst, vec!["?begin", "?end"]);
                let slice_shape = {
                    let mut slice_shape = t2_shape.clone();
                    slice_shape[dim] = end.clone() - begin.clone();
                    slice_shape
                };
                if shape == slice_shape
                    && dim != t2_shape.len() - 1
                    && t1_shape.len() == 1
                    && t1_shape[0] == t2_shape[t2_shape.len() - 1]
                {
                    let shape2_name = shape_to_underscore_name(&t2_shape);
                    let unioned = la.union_src(egraph, subst, &format!("(slice (matadd (expand ?t1 {shape2_name}) ?t2) ?dim ?begin ?end ?step)"));
                    return unioned as usize;
                }
                return 0;
            };
        "<unsqueeze-matadd-swap>" =>
            "(unsqueeze (matadd ?t1 ?t2) ?dim)" => |la, egraph, matched_id, subst| {
                let dtypes: Vec<DataKind> = get_dtypes!(egraph, subst, vec!["?t1", "?t2"]);
                if dtypes.iter().any(|&x| x != DataKind::Tnsr) { return 0; }
                let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                let dim = get_val(egraph, subst, "?dim") as usize;
                if t1_shape == t2_shape {
                    let unioned = la.union_src(egraph, subst, "(matadd (unsqueeze ?t1 ?dim) (unsqueeze ?t2 ?dim))");
                    return unioned as usize;
                }
                return 0;
            };
        "<matadd-associate-swap>" =>
            "(matadd ?t1 (matadd ?t2 ?t3))" => |la, egraph, matched_id, subst| {
                let t_dtypes: Vec<DataKind> = get_dtypes!(egraph, subst, vec!["?t1", "?t2", "?t3"]);
                if t_dtypes.iter().any(|&x| x != DataKind::Tnsr) { return 0; }
                let [t1_id, t2_id, t3_id] = get_ids!(subst, vec!["?t1", "?t2", "?t3"]);
                if t1_id == t2_id || t1_id == t3_id || t2_id == t3_id { return 0; }
                let [t1_shape, t2_shape, t3_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2", "?t3"]);

                if t1_shape == t2_shape && t2_shape == t3_shape {
                    let unioned = la.union_src(egraph, subst, "(matadd (matadd ?t1 ?t2) ?t3)");
                    return unioned as usize;
                }
                return 0;
            };
        "<matdiv-sum-back>" =>
            "?s1=(matdiv ?x 1), ?s2=(matdiv ?x ?n)" => |la, egraph, matched_id, subst| {
                // XXX: BACK_LEMMA
                let x_shape = get_shape(egraph, subst, "?x");
                if x_shape.len() == 0 && is_pure_val(egraph, subst, "?n") {
                    let n = get_val(egraph, subst, "?n") as usize;
                    let mut sum = "?s2".to_string();
                    if n > 1 {
                        for i in 1..n {
                            sum = format!("(reduce_add {} ?s2)", sum);
                        }
                    }
                    let unioned = la.union(egraph, subst, "?s1", &sum);
                    return unioned as usize;
                }
                return 0;
            };
    );

    results
}
