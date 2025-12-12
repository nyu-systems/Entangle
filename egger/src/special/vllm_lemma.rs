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

macro_rules! vllm_rotary_embedding_concat_swap_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let mut results = vec![];
        for idx in 0..2 {
            let pat = if idx == 0 {
                let concat_q = (1..$n).fold("?q0".to_string(), |acc, i| format!("(concat {} ?q{} 1)", acc, i));
                let s_pat = format!("?s=(vllm_rotary_embedding_{} ?positions {concat_q} ?k ?cos_sin_cache ?head_size)", idx, concat_q=concat_q);
                let si_pats = (0..$n).map(|i| format!("?s{i}=(vllm_rotary_embedding_{} ?positions ?q{i} ?k{i} ?cos_sin_cache ?head_size)", idx)).collect::<Vec<String>>().join(",");
                s_pat + "," + &si_pats
            } else {
                let concat_k = (1..$n).fold("?k0".to_string(), |acc, i| format!("(concat {} ?k{} 1)", acc, i));
                let s_pat = format!("?s=(vllm_rotary_embedding_{} ?positions ?q {concat_k} ?cos_sin_cache ?head_size)", idx, concat_k=concat_k);
                let si_pats = (0..$n).map(|i| format!("?s{i}=(vllm_rotary_embedding_{} ?positions ?q{i} ?k{i} ?cos_sin_cache ?head_size)", idx)).collect::<Vec<String>>().join(",");
                s_pat + "," + &si_pats
            };
            results.push(
                make_rewrite!(($manager, $verbose, format!("<vllm_rotary_embedding-concat-swap({},{})>", idx, $n)) => pat => |la, egraph, _matched_id, subst| {
                        let res = (1..$n).fold("?s0".to_string(), |acc, i| format!("(concat {} ?s{} 1)", acc, i));
                        let unioned = la.union(egraph, subst, "?s", &res);
                        return unioned as usize;
                    }
                )
            );
        }
        results
    }}
}

macro_rules! vllm_attention_concat_swap_template {
    ($manager:expr, $verbose:expr, $q_nh:expr, $kv_nh:expr, $world_size:expr) => {{
        assert!($world_size % $kv_nh == 0);
        // Multiple ranks need to uses same kv.
        // E.g., 12 query heads (q_nh), 2 key-value heads (kv_nh), 4 world size
        let num_ranks_per_kv = $world_size / $kv_nh;
        let mut pats = vec![];
        for i in 0..$world_size {
            let pat = format!("?s{i}=(vllm_unified_attention_with_output ?q{i} ?k{kv_i} ?v{kv_i})", i=i, kv_i=i/num_ranks_per_kv);
            pats.push(pat);
        }
        let concat_n_q = (1..$world_size).fold("?q0".to_string(), |acc, i| format!("(concat {} ?q{} 1)", acc, i));
        let concat_n_k = (1..$kv_nh).fold("?k0".to_string(), |acc, i| format!("(concat {} ?k{} 1)", acc, i));
        let concat_n_v = (1..$kv_nh).fold("?v0".to_string(), |acc, i| format!("(concat {} ?v{} 1)", acc, i));
        let pat = format!("?s=(vllm_unified_attention_with_output {concat_n_q} {concat_n_k} {concat_n_v}),") + &pats.join(",");
        make_rewrite!(($manager, $verbose, format!("<vllm_attention-concat-swap({}-{}-{})>", $q_nh, $kv_nh, $world_size)) => pat => |la, egraph, _matched_id, subst| {
            let concat_attn = (1..$world_size).fold("?s0".to_string(), |acc, i| format!("(concat {} ?s{} 1)", acc, i));
            let unioned = la.union(egraph, subst, "?s", &concat_attn);
            return unioned as usize;
        })
    }};
}

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let mut results = make_rewrites!(manager, verbose,
        // "<vllm_attention-concat-swap>" =>
        //     "(vllm_unified_attention_with_output (concat ?q1 ?q2 1) (concat ?k1 ?k2 1) (concat ?v1 ?v2 1))" => |la, egraph, _matched_id, subst| {
        //         let shapes: Vec<ShapeLike> = get_shapes!(egraph, subst, vec!["?q1", "?q2", "?k1", "?k2", "?v1", "?v2"]);
        //         if shapes.iter().any(|x| !is_val_shape(&x) || x.len() != 3) { return 0; }
        //         let [q1_shape, q2_shape, k1_shape, k2_shape, v1_shape, v2_shape] = shapes.iter().map(|x| shapelike_to_val_shape(&x)).collect::<Vec<Vec<i64>>>().try_into().unwrap();
        //         let [q1_nh, q2_nh, k1_nh, k2_nh, v1_nh, v2_nh] = [q1_shape[1], q2_shape[1], k1_shape[1], k2_shape[1], v1_shape[1], v2_shape[1]];
        //         if k1_nh != v1_nh || k2_nh != v2_nh { return 0; }
        //         let nh_q_per_kv = (q1_nh+q2_nh) / (k1_nh+k2_nh);

        //         if q2_nh == nh_q_per_kv * k2_nh {
        //             let (_unioned_id, unioned) = egraph.union_instantiations(
        //                 &la.src_pat.parse::<Pattern<Mdl>>().unwrap().ast,
        //                 &format!("(concat (vllm_unified_attention_with_output ?q1 ?k1 ?v1) (vllm_unified_attention_with_output ?q2 ?k2 ?v2) 1)").parse::<Pattern<Mdl>>().unwrap().ast,
        //                 subst,
        //                 la.name.as_str(),
        //             );
        //             if la.verbose && unioned {
        //                 println!("Rewrite for {} case1 succeeded", la.name);
        //             }
        //             return unioned as usize;
        //         } else if k2_nh == 1 && q2_nh < nh_q_per_kv {
        //             let new_q1_end = q1_nh - (nh_q_per_kv - q2_nh);
        //             let sliced_q1 = format!("(slice ?q1 1 0 {new_q1_end} 1)");
        //             let concat_q2 = format!("(concat (slice ?q1 1 {new_q1_end} {q1_nh} 1) ?q2 1)");
        //             let (_unioned_id, unioned) = egraph.union_instantiations(
        //                 &la.src_pat.parse::<Pattern<Mdl>>().unwrap().ast,
        //                 &format!("(concat (vllm_unified_attention_with_output {sliced_q1} ?k1 ?v1) (vllm_unified_attention_with_output {concat_q2} ?k2 ?v2) 1)").parse::<Pattern<Mdl>>().unwrap().ast,
        //                 subst,
        //                 la.name.as_str(),
        //             );
        //             if la.verbose && unioned {
        //                 println!("Rewrite for {} case2 succeeded", la.name);
        //             }
        //         }
        //         return 0;
        //     };
        // "<vllm_attention-kv1head-concat-swap>" =>
        //     "?s=(vllm_unified_attention_with_output (concat ?q1 ?q2 1) ?k ?v),?s1=(vllm_unified_attention_with_output ?q1 ?k ?v),?s2=(vllm_unified_attention_with_output ?q2 ?k ?v)" => |la, egraph, _matched_id, subst| {
        //         let shapes: Vec<ShapeLike> = get_shapes!(egraph, subst, vec!["?q1", "?q2", "?k", "?v"]);
        //         if shapes.iter().any(|x| !is_val_shape(&x) || x.len() != 3) { return 0; }
        //         let [q1_shape, q2_shape, k_shape, v_shape] = shapes.iter().map(|x| shapelike_to_val_shape(&x)).collect::<Vec<Vec<i64>>>().try_into().unwrap();
        //         let [q1_nh, q2_nh, k_nh, v_nh] = [q1_shape[1], q2_shape[1], k_shape[1], v_shape[1]];

        //         if k_nh == 1 && v_nh == 1 {
        //             let (_unioned_id, unioned) = egraph.union_instantiations(
        //                 &"?s".parse::<Pattern<Mdl>>().unwrap().ast,
        //                 &format!("(concat (vllm_unified_attention_with_output ?q1 ?k ?v) (vllm_unified_attention_with_output ?q2 ?k ?v) 1)").parse::<Pattern<Mdl>>().unwrap().ast,
        //                 subst,
        //                 la.name.as_str(),
        //             );
        //             if la.verbose && unioned {
        //                 println!("Rewrite for {} succeeded", la.name);
        //             }
        //             return unioned as usize;
        //         }
        //         return 0;
        //     };
        "<vllm_silu_and_mul-rewrite>" =>
            "(vllm_silu_and_mul (concat (concat ?inpt11 ?inpt12 ?dim) (concat ?inpt21 ?inpt22 ?dim) ?dim))" => |la, egraph, _matched_id, subst| {
                let [inpt11, inpt12, inpt21, inpt22] = get_shapes!(egraph, subst, vec!["?inpt11", "?inpt12", "?inpt21", "?inpt22"]);
                let n = inpt11.len();
                let dim = get_usize_val(egraph, subst, "?dim");
                if n == dim + 1 {
                    let unioned = la.union_src(egraph, subst, &format!("(concat (vllm_silu_and_mul (concat ?inpt11 ?inpt21 ?dim)) (vllm_silu_and_mul (concat ?inpt12 ?inpt22 ?dim)) ?dim)"));
                    return unioned as usize;
                }
                return 0;
            };

        "<vllm_rotary_embedding_ignore_k>" =>
            "?s0=(vllm_rotary_embedding_0 ?positions ?q ?k0 ?cos_sin_cache ?head_size),?s1=(vllm_rotary_embedding_0 ?positions ?q ?k1 ?cos_sin_cache ?head_size)" => |la, egraph, _matched_id, subst| {
                let unioned = la.union(egraph, subst, "?s0", "?s1");
                return unioned as usize;
            };

        "<vllm_rotary_embedding_ignore_q>" =>
            "?s0=(vllm_rotary_embedding_1 ?positions ?q0 ?k ?cos_sin_cache ?head_size),?s1=(vllm_rotary_embedding_1 ?positions ?q1 ?k ?cos_sin_cache ?head_size)" => |la, egraph, _matched_id, subst| {
                let unioned = la.union(egraph, subst, "?s0", "?s1");
                return unioned as usize;
            };

    );

    results.extend(vllm_rotary_embedding_concat_swap_template!(
        manager, verbose, 2
    ));
    results.extend(vllm_rotary_embedding_concat_swap_template!(
        manager, verbose, 4
    ));
    results.extend(vllm_rotary_embedding_concat_swap_template!(
        manager, verbose, 6
    ));
    results.push(vllm_attention_concat_swap_template!(
        manager, verbose, 12, 2, 2
    ));
    results.push(vllm_attention_concat_swap_template!(
        manager, verbose, 12, 2, 4
    ));
    results.push(vllm_attention_concat_swap_template!(
        manager, verbose, 12, 2, 6
    ));

    results
}
