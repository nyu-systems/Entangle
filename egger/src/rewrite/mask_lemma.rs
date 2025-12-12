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
use crate::symval::{ShapeLike, SymVal, SymValManagerRef};
use crate::utils::*;
use std::convert::TryInto;

macro_rules! vocabulary_size {
    () => {
        std::collections::HashSet::from([98304, 151936])
    };
}

fn outer_range_str(a: &str, b: &str) -> String {
    format!(
        "(bitwise_or (lt_scalar ?indices {}) (ge_scalar ?indices {}))",
        a, b
    )
}

fn outer_range_offset_str(a: usize, b: usize) -> String {
    format!(
        "(bitwise_or (lt_scalar ?indices ?offset{}) (ge_scalar ?indices ?offset{}))",
        a, b
    )
}

fn inner_range_str(a: &str, b: &str) -> String {
    format!(
        "(bitwise_and (ge_scalar ?indices {}) (lt_scalar ?indices {}))",
        a, b
    )
}

macro_rules! index_put_merge_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let mut results = vec![];
        assert!($n >= 2);
        results.push({
            let name = format!("<index_put-mask_select-swap({})>", $n);
            let pat = (0..$n)
                .map(|i| format!("?s{i}=(index_put (matsub ?indices ?offset{i}) (bitwise_or (lt_scalar ?indices ?offset{i}) (ge_scalar ?indices ?offset{})) (fill \"[]\" 0) )", i+1))
                .collect::<Vec<String>>().join(",");

            make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
                let [first_offset, last_offset] = get_vals!(egraph, subst, vec!["?offset0", &format!("?offset{}", $n)]);

                if first_offset == 0 && vocabulary_size!().contains(&last_offset) {
                    let mut masked_select = format!("(masked_select (matadd ?s0 ?offset0) (matadd ?s1 ?offset1) {} {})", outer_range_str("?offset0", "?offset1"), outer_range_str("?offset1", "?offset2"));
                    for i in 2..$n {
                        masked_select = format!(
                            "(masked_select {} (matadd ?s{i} ?offset{i}) {} {})", masked_select,
                            outer_range_str("?offset0", &format!("?offset{}", i)), outer_range_str(&format!("?offset{}", i), &format!("?offset{}", i + 1)));
                    }
                    let unioned = la.union(egraph, subst, &masked_select, &format!("(index_put (matsub ?indices 0) {} (fill \"[]\" 0))", outer_range_str("0", &format!("?offset{}", $n))));
                    return unioned as usize;
                }
                return 0;
            })
        });
        results.push({
            let name = format!("<index_put-mask_select-to_sum-swap({})>", $n);
            let pat = (0..$n)
                .map(|i| format!("?s{i}=(index_put ?embedding{i} (bitwise_or (lt_scalar ?indices ?offset{i}) (ge_scalar ?indices ?offset{})) (fill \"[]\" 0) )", i+1))
                .collect::<Vec<String>>().join(",");

            make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
                let embedding_shapes: Vec<ShapeLike> = (0..$n).map(|i| get_shape(egraph, subst, &format!("?embedding{}", i))).collect();
                if embedding_shapes.iter().any(|shape| shape != &embedding_shapes[0]) { return 0; }
                let [first_offset, last_offset] = get_vals!(egraph, subst, vec!["?offset0", &format!("?offset{}", $n)]);

                if first_offset == 0 && vocabulary_size!().contains(&last_offset) {
                    let mut masked_select = format!("(masked_select ?embedding0 ?embedding1 {} {})", outer_range_str("?offset0", "?offset1"), outer_range_str("?offset1", "?offset2"));
                    let mut sum = "(reduce_add ?s0 ?s1)".to_string();
                    for i in 2..$n {
                        masked_select = format!(
                            "(masked_select {} ?embedding{i} {} {})", masked_select,
                            outer_range_str("?offset0", &format!("?offset{}", i)), outer_range_str(&format!("?offset{}", i), &format!("?offset{}", i + 1)));
                            sum = format!("(reduce_add {} ?s{})", sum, i);
                    }
                    let unioned1 = la.union(egraph, subst, &masked_select, &sum);
                    let unioned2 = la.union(egraph, subst, &sum, &format!("(index_put {masked_select} {} (fill \"[]\" 0) )", outer_range_str("?offset0", &format!("?offset{}", $n))));
                    return unioned1 as usize + unioned2 as usize;
                }
                return 0;
            })
        });
        results
    }};
}

macro_rules! index_merge_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        assert!($n >= 2);
        let name = format!("<index-merge({})>", $n);
        let pat = (0..$n)
            .map(|i| format!("?s{i}=(index ?weight{i} (index_put (matsub ?indices ?offset{i}) (bitwise_or (lt_scalar ?indices ?offset{i}) (ge_scalar ?indices ?offset{})) (fill \"[]\" 0) ) )", i+1))
            .collect::<Vec<String>>().join(",");

        make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
            let [first_offset, last_offset] = get_vals!(egraph, subst, vec!["?offset0", &format!("?offset{}", $n)]);

            if first_offset == 0 && vocabulary_size!().contains(&last_offset) {
                let mut masked_select = format!("(masked_select ?s0 ?s1 {} {})", outer_range_str("?offset0", "?offset1"), outer_range_str("?offset1", "?offset2"));
                let mut concat = "(concat ?weight0 ?weight1 0)".to_string();
                for i in 2..$n {
                    masked_select = format!(
                        "(masked_select {} ?s{i} {} {})", masked_select,
                        outer_range_str("?offset0", &format!("?offset{}", i)), outer_range_str(&format!("?offset{}", i), &format!("?offset{}", i + 1)));
                    concat = format!("(concat {} ?weight{} 0)", concat, i);
                }
                let unioned = la.union(egraph, subst, &masked_select, &format!("(index {concat} ?indices)"));
                return unioned as usize;
            }
            return 0;
        })
    }};
}

macro_rules! matsub_masked_offset_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let mut results = vec![];
        let name = format!("<matsub-masked_indices-and-offset({})>", $n);
        let pat = (0..$n).map(|i| format!("?s{i}=(matsub ?indices (ewmul (bitwise_and (ge_scalar ?indices ?offset{i}) (lt_scalar ?indices ?offset{})) ?scalar{i}))", i+1)).collect::<Vec<String>>().join(",");
        results.push(make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
            let [first_offset, last_offset] = get_vals!(egraph, subst, vec!["?offset0", &format!("?offset{}", $n)]);
            if first_offset == 0 && vocabulary_size!().contains(&last_offset) {
                let mut mask_fill = format!("(reduce_add (mask_fill_scalar (matadd ?s0 ?scalar0) (bitwise_not {}) 0) (mask_fill_scalar (matadd ?s1 ?scalar1) (bitwise_not {}) 0))", inner_range_str("?offset0", "?offset1"), inner_range_str("?offset1", "?offset2"));
                for i in 2..$n {
                    mask_fill = format!("(reduce_add {mask_fill} (mask_fill_scalar (matadd ?s{i} ?scalar{i}) (bitwise_not {}) 0))", inner_range_str(&format!("?offset{}", i), &format!("?offset{}", i+1)));
                }
                let unioned = la.union(egraph, subst, &mask_fill, "?indices");
                return unioned as usize;
            }
            return 0;
        }));

        let name = format!("<ewmul-offsetted-indices-and-mask({})>", $n);
        let mut mask_fill = format!("(reduce_add (mask_fill_scalar (matadd ?s0 ?scalar0) (bitwise_not {}) 0) (mask_fill_scalar (matadd ?s1 ?scalar1) (bitwise_not {}) 0))", inner_range_str("?offset0", "?offset1"), inner_range_str("?offset1", "?offset2"));
        for i in 2..$n {
            mask_fill = format!("(reduce_add {mask_fill} (mask_fill_scalar (matadd ?s{i} ?scalar{i}) (bitwise_not {}) 0))", inner_range_str(&format!("?offset{}", i), &format!("?offset{}", i+1)));
        }
        let pat = format!("?sum={mask_fill},") + &(0..$n).map(|i| format!("?m{i}=(ewmul {} ?s{i})", inner_range_str(&format!("?offset{}", i), &format!("?offset{}", i+1))) ).collect::<Vec<String>>().join(",");
        results.push(make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
            let mut mask_fill = format!("(reduce_add (mask_fill_scalar (matadd ?m0 ?scalar0) (bitwise_not {}) 0) (mask_fill_scalar (matadd ?m1 ?scalar1) (bitwise_not {}) 0))", inner_range_str("?offset0", "?offset1"), inner_range_str("?offset1", "?offset2"));
            for i in 2..$n {
                mask_fill = format!("(reduce_add {mask_fill} (mask_fill_scalar (matadd ?m{i} ?scalar{i}) (bitwise_not {}) 0))", inner_range_str(&format!("?offset{}", i), &format!("?offset{}", i+1)));
            }
            let [first_offset, last_offset] = get_vals!(egraph, subst, vec!["?offset0", &format!("?offset{}", $n)]);
            if first_offset == 0 && vocabulary_size!().contains(&last_offset) {
                let unioned = la.union(egraph, subst, &mask_fill, "?indices");
                return unioned as usize;
            }
            return 0;
        }));

        let name = format!("<embedding-masked-offsetted-indices({})>", $n);
        let mut mask_fill = format!("(reduce_add (mask_fill_scalar (matadd ?m0 ?scalar0) (bitwise_not {}) 0) (mask_fill_scalar (matadd ?m1 ?scalar1) (bitwise_not {}) 0))", inner_range_str("?offset0", "?offset1"), inner_range_str("?offset1", "?offset2"));
        for i in 2..$n {
            mask_fill = format!("(reduce_add {mask_fill} (mask_fill_scalar (matadd ?m{i} ?scalar{i}) (bitwise_not {}) 0))", inner_range_str(&format!("?offset{}", i), &format!("?offset{}", i+1)));
        }
        let emb_pat = format!("?e=(embedding {} ?indices),", (1..$n).fold("?table0".to_string(), |acc, i| format!("(concat {} ?table{} 0)", acc, i)));
        let pat = format!("?sum={mask_fill},") + &emb_pat + &(0..$n).map(|i| format!("?e{i}=(embedding ?table{i} ?m{i})")).collect::<Vec<String>>().join(",");
        results.push(make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
            let mut mask_fill = format!("(reduce_add (mask_fill_scalar ?e0 (unsqueeze (bitwise_not {}) 1) 0) (mask_fill_scalar ?e1 (unsqueeze (bitwise_not {}) 1) 0))", inner_range_str("?offset0", "?offset1"), inner_range_str("?offset1", "?offset2"));
            for i in 2..$n {
                mask_fill = format!("(reduce_add {mask_fill} (mask_fill_scalar ?e{i} (unsqueeze (bitwise_not {}) 1) 0))", inner_range_str(&format!("?offset{}", i), &format!("?offset{}", i+1)));
            }
            let [first_offset, last_offset] = get_vals!(egraph, subst, vec!["?offset0", &format!("?offset{}", $n)]);
            let indices_shape = get_shape(egraph, subst, "?indices");
            if first_offset == 0 && vocabulary_size!().contains(&last_offset) && indices_shape.len() == 1 {
                let unioned = la.union(egraph, subst, &mask_fill, "?e");
                return unioned as usize;
            }
            return 0;
        }));

        results
    }};
}

macro_rules! embedding_concat_weight_with_masked_select_indices_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let mut results = Vec::new();
        let name = format!("<embedding-concat-weight-with-masked-select-indices({})>", $n);
        let concat_weight = (1..$n).fold("?table0".to_string(), |acc, i| format!("(concat {acc} ?table{i} 0)"));
        let masked_select_indices = (1..$n).fold("(matadd ?indices0 ?offset0)".to_string(), |acc, i| {
            format!("(masked_select {acc} (matadd ?indices{i} ?offset{i}) {} {})", outer_range_offset_str(0, i), outer_range_offset_str(i, i + 1))
        });
        let pat = format!("?e=(embedding {concat_weight} {masked_select_indices})");
        let sub_pats = (0..$n).map(|i|format!("?e{i}=(embedding ?table{i} ?indices{i})")).collect::<Vec<_>>().join(",");
        results.push(make_rewrite!(($manager, $verbose, name) => format!("{pat},{sub_pats}") => |la, egraph, _matched_id, subst| {
            let end_offsets: Vec<i64> = (1..$n).map(|i| get_val(egraph, subst, &format!("?offset{}", i))).collect();
            let table_lengths: Vec<SymVal> = (0..$n).map(|i| get_shape(egraph, subst, &format!("?table{}", i))[0].clone()).collect();
            for (end_offset, table_length) in end_offsets.into_iter().zip(table_lengths) {
                if table_length != end_offset { return 0; }
            }
            let masked_select_embedding = (1..$n).fold("?e0".to_string(), |acc, i| format!("(masked_select {acc} ?e{i} {} {})", outer_range_offset_str(0, i), outer_range_offset_str(i, i + 1)));
            let unioned = la.union(egraph, subst, &masked_select_embedding, "?e");
            return unioned as usize;
        }));

        results
    }};
}

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let mut results = make_rewrites!(manager, verbose,
        "<inverse-matsub>" =>
            "(matsub ?t ?scalar)" => |la, egraph, matched_id, subst| {
                // XXX BACK_LEMMA
                let scalar_dtype = get_dtype(egraph, subst, "?scalar");
                if scalar_dtype.is_scalar_or_name() {
                    let unioned = la.union(egraph, subst, "?t", "(matadd (matsub ?t ?scalar) ?scalar)");
                    return unioned as usize;
                }
                return 0;
            };
        // This lemma usually works with <index_put-mask_select-swap>, but also removing index_put for full.
        "<full-index_put>" =>
            "(index_put (masked_select \
                    (matadd ?indices1 0) (matadd ?indices2 ?b) \
                    (bitwise_or (lt_scalar ?indices 0) (ge_scalar ?indices ?b)) \
                    (bitwise_or (lt_scalar ?indices ?b) (ge_scalar ?indices ?c))) \
                (bitwise_or (lt_scalar ?indices 0) (ge_scalar ?indices ?c)) (fill \"[]\" 0))" => |la, egraph, matched_id, subst| {
                let c = get_val(egraph, subst, "?c") as i64;
                if vocabulary_size!().contains(&c) {
                    let mask1 = "(bitwise_or (lt_scalar ?indices 0) (ge_scalar ?indices ?b))";
                    let mask2 = "(bitwise_or (lt_scalar ?indices ?b) (ge_scalar ?indices ?c))";
                    let unioned = la.union_src(egraph, subst, &format!("(masked_select (matadd ?indices1 0) (matadd ?indices2 ?b) {mask1} {mask2})"));
                    return unioned as usize;
                }
                return 0;
             };
        "<index-masked_select-swap>" =>
            "(index (concat ?weight1 ?weight2 0) (masked_select (matadd ?indices1 ?a) (matadd ?indices2 ?b) (bitwise_or (lt_scalar ?indices ?a) (ge_scalar ?indices ?b)) (bitwise_or (lt_scalar ?indices ?b) (ge_scalar ?indices ?c))) )" => |la, egraph, matched_id, subst| {
                let [weight1_shape, weight2_shape] = get_shapes!(egraph, subst, vec!["?weight1", "?weight2"]);
                let [a, b, c] = get_vals!(egraph, subst, vec!["?a", "?b", "?c"]);
                if weight1_shape.len() == 2 && weight2_shape.len() == 2 && weight1_shape[0] == b - a && weight2_shape[0] == c - b {
                    let mask1 = "(bitwise_or (lt_scalar ?indices ?a) (ge_scalar ?indices ?b))";
                    let mask2 = "(bitwise_or (lt_scalar ?indices ?b) (ge_scalar ?indices ?c))";
                    let index_1 = "(index ?weight1 ?indices1)";
                    let index_2 = "(index ?weight2 ?indices2)";
                    let unioned = la.union_src(egraph, subst, &format!("(masked_select {index_1} {index_2} {mask1} {mask2})"));
                    return unioned as usize;
                }
                return 0;
            };

        "<matadd-0-back>" => "(matadd ?t 0)" => "?t";

        "<demerge-index>" =>
            "(index \
                (concat ?weight1 ?weight2 1) (arange ?begin ?end) \
                (masked_select \
                    ?indices1 (matadd ?indices2 ?b) \
                    (reshape (bitwise_or (lt_scalar ?indices ?a) (ge_scalar ?indices ?b)) ?shape) \
                    (reshape (bitwise_or (lt_scalar ?indices ?b) (ge_scalar ?indices ?c)) ?shape)) )" => |la, egraph, matched_id, subst| {
                let [weight1_shape, weight2_shape] = get_shapes!(egraph, subst, vec!["?weight1", "?weight2"]);
                let [a, b, c] = get_vals!(egraph, subst, vec!["?a", "?b", "?c"]);
                if weight1_shape.len() == 2 || weight2_shape.len() == 2 {
                    let weight_n_dim = weight1_shape.len();
                    let mask1 = "(bitwise_or (lt_scalar ?indices ?a) (ge_scalar ?indices ?b))";
                    let mask2 = "(bitwise_or (lt_scalar ?indices ?b) (ge_scalar ?indices ?c))";
                    let index_1 = "(index ?weight1 (arange ?begin ?end) ?indices1)";
                    let index_2 = "(index ?weight2 (arange ?begin ?end) ?indices2)";
                    let unioned = la.union_src(egraph, subst, &format!("(masked_select {index_1} {index_2} (reshape {mask1} ?shape) (reshape {mask2} ?shape) )"));
                    return unioned as usize;
                }
                return 0;
            };

        // Below are required by vLLM
        "<ewmul-zeros-is-zeros>" => "(ewmul (fill ?shape 0) ?scalar)" => "(fill ?shape 0)";
        "<matadd-rhs-zeros-back>" => "(matadd ?t (fill ?shape 0))" => "?t";
        "<matadd-lhs-zeros-back>" => "(matadd (fill ?shape 0) ?t)" => "?t";
        "<or-zeros-is-self>" => "(bitwise_or ?a (fill ?shape 0))" => "?a";

        "<ge-lt-and-empty-range-is-zeros>" =>
            "(bitwise_and (ge_scalar ?indices ?a) (lt_scalar ?indices ?b))" => |la, egraph, matched_id, subst| {
                let a = get_val(egraph, subst, "?a");
                let b = get_val(egraph, subst, "?b");
                if a >= b {
                    let shape = shape_to_underscore_name(&get_shape(egraph, subst, "?indices"));
                    let unioned = la.union_src(egraph, subst, &format!("(fill {shape} 0)"));
                    return unioned as usize;
                }
                return 0;
            };
    );

    results.extend(index_put_merge_template!(manager, verbose, 2));
    results.extend(index_put_merge_template!(manager, verbose, 4));
    results.extend(index_put_merge_template!(manager, verbose, 6));
    results.extend(index_put_merge_template!(manager, verbose, 8));
    results.push(index_merge_template!(manager, verbose, 2));
    results.push(index_merge_template!(manager, verbose, 4));
    results.push(index_merge_template!(manager, verbose, 6));
    results.push(index_merge_template!(manager, verbose, 8));
    results.extend(matsub_masked_offset_template!(manager, verbose, 2));
    results.extend(matsub_masked_offset_template!(manager, verbose, 4));
    results.extend(matsub_masked_offset_template!(manager, verbose, 8));
    results
        .extend(embedding_concat_weight_with_masked_select_indices_template!(manager, verbose, 2));
    results
        .extend(embedding_concat_weight_with_masked_select_indices_template!(manager, verbose, 4));
    results
        .extend(embedding_concat_weight_with_masked_select_indices_template!(manager, verbose, 8));

    results
}
