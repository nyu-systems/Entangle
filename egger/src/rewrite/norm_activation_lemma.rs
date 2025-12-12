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

macro_rules! activation_template {
    ($manager:expr, $verbose:expr, $name:expr) => {{
        make_rewrites!($manager, $verbose,
            format!("<{}-slice-swap>", $name) => format!("({op} (slice ?t ?dim ?begin ?end ?step))", op=$name) => |la, egraph, _matched_id, subst| {
                let unioned = la.union_src(egraph, subst, &format!("(slice ({op} ?t) ?dim ?begin ?end ?step)", op=$name));
                return unioned as usize;
            };
            format!("<{}-concat-swap>", $name) => format!("({} (concat ?t1 ?t2 ?dim))", $name) => |la, egraph, _matched_id, subst| {
                let unioned = la.union_src(egraph, subst, &format!("(concat ({op} ?t1) ({op} ?t2) ?dim)", op=$name));
                return unioned as usize;
            };
        )
    }};
}

macro_rules! activation_backward_template {
    ($manager:expr, $verbose:expr, $name:expr) => {{
        make_rewrites!($manager, $verbose,
            format!("<{}-concat-swap>", $name) => format!("({} (concat ?grad_t1 ?grad_t2 ?dim) (concat ?t1 ?t2 ?dim))", $name) => |la, egraph, _matched_id, subst| {
                let unioned = la.union_src(egraph, subst, &format!("(concat ({op} ?grad_t1 ?t1) ({op} ?grad_t2 ?t2) ?dim)", op=$name));
                return unioned as usize;
            };
        )
    }};
}

macro_rules! rms_affine_concat_swap_template {
    ($manager:expr, $verbose:expr, $idx:expr) => {{
        let mut results = vec![];

        results.push(make_rewrite!(($manager, $verbose, format!("<rms_forward_affine_{}-concat-swap>", $idx)) => format!("(rms_forward_affine_{} (concat ?t1 ?t2 ?dim) ?weight ?eps)", $idx) => |la, egraph, _matched_id, subst| {
            let dim = get_val(egraph, subst, "?dim") as usize;
            let t1_shape = get_shape(egraph, subst, "?t1");
            if dim != t1_shape.len() - 1 {
                let new_dim = if $idx == 0 { dim } else { 0 };
                let unioned = la.union_src(egraph, subst, &format!("(concat (rms_forward_affine_{idx} ?t1 ?weight ?eps) (rms_forward_affine_{idx} ?t2 ?weight ?eps) {dim})", idx=$idx, dim=new_dim));
                return unioned as usize;
            }
            return 0;
        }));

        results.push(make_rewrite!(($manager, $verbose, format!("<rms_backward_affine_{}-all-concat-swap>", $idx)) =>
            format!("(rms_backward_affine_{} (concat ?t1 ?t2 ?dim) ?weight (concat ?invvar1 ?invvar2 ?invvar_dim) (concat ?input_or_output1 ?input_or_output2 ?input_or_output_dim) ?eps)", $idx) => |la, egraph, _matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let t1_n_dim = get_n_dim(egraph, subst, "?t1");
                let [t1_shape, t2_shape, invvar1_shape, invvar2_shape, input_or_output1_shape, input_or_output2_shape] = get_shapes!(egraph, subst, ["?t1", "?t2", "?invvar1", "?invvar2", "?input_or_output1", "?input_or_output2"]);
                if dim != t1_n_dim - 1
                    && mul_reduce(&t1_shape[..t1_shape.len()-1]) == invvar1_shape[0] && invvar1_shape[0] == mul_reduce(&input_or_output1_shape[..input_or_output1_shape.len()-1])
                    && mul_reduce(&t2_shape[..t2_shape.len()-1]) == invvar2_shape[0] && invvar2_shape[0] == mul_reduce(&input_or_output2_shape[..input_or_output2_shape.len()-1])
                {
                    let result = if $idx == 0 {
                        format!("(concat (rms_backward_affine_0 ?t1 ?weight ?invvar1 ?input_or_output1 ?eps) (rms_backward_affine_0 ?t2 ?weight ?invvar2 ?input_or_output2 ?eps) ?dim)")
                    } else {
                        assert!($idx == 1);
                        format!("(reduce_add (rms_backward_affine_1 ?t1 ?weight ?invvar1 ?input_or_output1 ?eps) (rms_backward_affine_1 ?t2 ?weight ?invvar2 ?input_or_output2 ?eps))")
                    };
                    let unioned = la.union_src(egraph, subst, &result);
                    return unioned as usize;
                }
                return 0;
            }
        ));

        results
    }};
}

pub(crate) use activation_template;

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    let mut results = vec![];
    ////////////////////////////////////////////////////////////////////////////////
    // Activation ops: they are elementwise computing that don't involve other elements.
    ////////////////////////////////////////////////////////////////////////////////
    results.extend(activation_template!(manager, verbose, "exp"));
    results.extend(activation_template!(manager, verbose, "gelu"));
    results.extend(activation_template!(manager, verbose, "log"));
    results.extend(activation_template!(manager, verbose, "silu"));
    results.extend(activation_template!(manager, verbose, "sigmoid"));
    results.extend(activation_backward_template!(manager, verbose, "gelu_backward"));
    results.extend(activation_backward_template!(manager, verbose, "sigmoid_backward"));
    results.extend(vec![
        make_simple_rewrite!((manager, verbose, "<pow_tensor_scalar-concat-swap>") => "(pow_tensor_scalar (concat ?t1 ?t2 ?dim) ?s)" => "(concat (pow_tensor_scalar ?t1 ?s) (pow_tensor_scalar ?t2 ?s) ?dim)"),
    ]);

    #[allow(unused_variables)]
    results.extend(make_rewrites!(manager, verbose,
        ////////////////////////////////////////////////////////////////////////////////
        // Norm ops: they only use the last dimension and they are batch-independent.
        ////////////////////////////////////////////////////////////////////////////////
        "<layernorm-repeat-swap>" =>
            "(fused_layernorm_affine_0 (repeat ?t ?repeats) ?weight ?bias ?last_dim_size ?eps)" => |la, egraph, matched_id, subst| {
                let [last_dim_size, repeats] = get_shapes_from_names!(egraph, subst, ["?last_dim_size", "?repeats"], la.manager.clone());
                if repeats[repeats.len() - 1] == 1 {
                    let result_str = format!("(repeat (fused_layernorm_affine_0 ?t ?weight ?bias ?last_dim_size ?eps) ?repeats)");
                    let unioned = la.union_src(egraph, subst, &result_str);
                    return unioned as usize;
                }
                return 0;
            };
        "<layernorm-concat-swap>" =>
            "(fused_layernorm_affine_0 (concat ?t1 ?t2 ?dim) ?weight ?bias ?last_dim_size ?eps)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim");
                let t1_n_dim = get_n_dim(egraph, subst, "?t1");
                if dim as usize == t1_n_dim - 1 { return 0; }
                let unioned = la.union_src(egraph, subst, &format!("(concat (fused_layernorm_affine_0 ?t1 ?weight ?bias ?last_dim_size ?eps) (fused_layernorm_affine_0 ?t2 ?weight ?bias ?last_dim_size ?eps) ?dim)"));
                return unioned as usize;
            };
        "<layernorm-slice-swap>" =>
            "(fused_layernorm_affine_0 (slice ?t ?dim ?begin ?end ?step) ?weight ?bias ?last_dim_size ?eps)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim");
                let t_n_dim = get_n_dim(egraph, subst, "?t");
                if dim as usize == t_n_dim - 1 { return 0; }
                let unioned = la.union_src(egraph, subst, "(slice (fused_layernorm_affine_0 ?t ?weight ?bias ?last_dim_size ?eps) ?dim ?begin ?end ?step)");
                return unioned as usize;
            };
        "<fused_layernorm_affine_bwd_1-concat_input-swap>" =>
            "(fused_layernorm_affine_bwd_1 (concat ?grad_output1 ?grad_output2 ?concat_dim) (concat ?mean1 ?mean2 ?concat_dim) (concat ?invvar1 ?invvar2 ?concat_dim) (concat ?input_or_output1 ?input_or_output2 ?concat_dim) ?weight ?bias ?eps)" => |la, egraph, matched_id, subst| {
                let [concat_dim] = get_usize_vals!(egraph, subst, ["?concat_dim"]);
                let [grad_output1_shape, grad_output2_shape, mean1_shape, mean2_shape, invvar1_shape, invvar2_shape, input_or_output1_shape, input_or_output2_shape] =
                    get_shapes!(egraph, subst, ["?grad_output1", "?grad_output2", "?mean1", "?mean2", "?invvar1", "?invvar2", "?input_or_output1", "?input_or_output2"]);
                if grad_output1_shape == input_or_output1_shape && grad_output2_shape == input_or_output2_shape
                    && mul_reduce(&grad_output1_shape[..grad_output1_shape.len()-1]) == mean1_shape[0] && mean1_shape[0] == invvar1_shape[0]
                    && mul_reduce(&grad_output2_shape[..grad_output2_shape.len()-1]) == mean2_shape[0] && mean2_shape[0] == invvar2_shape[0]
                    && concat_dim != grad_output1_shape.len() - 1
                {
                    let unioned = la.union_src(egraph, subst, &format!("(reduce_add (fused_layernorm_affine_bwd_1 ?grad_output1 ?mean1 ?invvar1 ?input_or_output1 ?weight ?bias ?eps) (fused_layernorm_affine_bwd_1 ?grad_output2 ?mean2 ?invvar2 ?input_or_output2 ?weight ?bias ?eps))"));
                    return unioned as usize;
                }
                return 0;
            };
        "<softmax-concat-swap>" =>
            "(softmax (concat ?inpt1 ?inpt2 ?concat_dim) ?dim)" => |la, egraph, matched_id, subst| {
                let [concat_dim, dim] = get_usize_vals!(egraph, subst, ["?concat_dim", "?dim"]);
                let inpt1_shape = get_shape(egraph, subst, "?inpt1");
                let n_dim = inpt1_shape.len();
                if concat_dim < dim {
                    let unioned = la.union_src(egraph, subst, "(concat (softmax ?inpt1 ?dim) (softmax ?inpt2 ?dim) ?concat_dim)");
                    return unioned as usize;
                }
                return 0;
            };
        "<softmax_backward_data-concat-swap>" =>
            "(softmax_backward_data (concat ?grad_output1 ?grad_output2 ?concat_dim) (concat ?output1 ?output2 ?concat_dim) ?dim)" => |la, egraph, matched_id, subst| {
                let [concat_dim, dim] = get_usize_vals!(egraph, subst, ["?concat_dim", "?dim"]);
                if concat_dim != dim {
                    let unioned = la.union_src(egraph, subst, "(concat (softmax_backward_data ?grad_output1 ?output1 ?dim) (softmax_backward_data ?grad_output2 ?output2 ?dim) ?concat_dim)");
                    return unioned as usize;
                }
                return 0;
            };
    ));

    results.extend(rms_affine_concat_swap_template!(manager, verbose, 0));
    results.extend(rms_affine_concat_swap_template!(manager, verbose, 1));

    results
}
