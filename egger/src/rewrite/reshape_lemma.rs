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

use crate::metadata::DataKind;
use crate::model::*;
use crate::rewrite::lambda_rewrite::*;
use crate::symval::ShapeLike;
use crate::symval::SymValManagerRef;
use crate::utils::*;
use egg::*;
use std::cmp::min;
use std::convert::TryInto;

macro_rules! reshape_concat_swap_with_reshape_back_template {
    // XXX: BACK_LEMMA
    ($manager:expr, $verbose:expr, $n:expr) => {{
        assert!($n >= 2);
        let name = format!("<reshape-concat-swap(other_reshape_exists-{})>", $n);
        let s_pat = {
            let mut tmp = "(concat ?t0 ?t1 ?dim)".to_string();
            for i in 2..$n {
                tmp = format!("(concat {} ?t{} ?dim)", tmp, i);
            }
            format!("?s=(reshape {} ?shape)", tmp)
        };
        let si_pat = (0..$n).map(|i| format!("?s{i}=(reshape ?t{i} ?shape{i})")).collect::<Vec<String>>().join(",");
        let pat = format!("{},{}", s_pat, si_pat);
        make_rewrites!($manager, $verbose,
            name => pat => |la, egraph, _matched_id, subst| {
                let t_shapes: Vec<ShapeLike> = (0..$n).map(|i| get_shape(egraph, subst, format!("?t{i}").as_str())).collect();
                let t_shape_names: Vec<String> = t_shapes.iter().map(|s| shape_to_underscore_name(s)).collect();
                let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());
                let shapes: Vec<ShapeLike> = (0..$n).map(|i| get_shape_from_name(egraph, subst, format!("?shape{i}").as_str(), la.manager.clone())).collect();
                let dim = get_val(egraph, subst, "?dim") as usize;
                if (t_shapes.iter().all(|s| s.len() == 4) && shape.len() == 3 && shapes.iter().all(|s| s.len() ==3) && dim == 2
                    && shape[1] == t_shapes[0][1].clone() * (t_shapes.iter().map(|s| s[2].clone()).reduce(|acc, e| acc + e).unwrap()) && t_shapes.iter().all(|s| shape[0] == s[0] && shape[2] == s[3])
                    && shapes.iter().zip(t_shapes.iter()).all(|(shape, t_shape)| shape[1] == t_shape[1].clone() * t_shape[2].clone() && shape[0] == t_shape[0] && shape[2] == t_shape[3]))

                    || (t_shapes.iter().all(|s| s.len() == 4) && shape.len() == 3 && shapes.iter().all(|s| s.len() ==3) && dim == 1
                    && shape[0] == t_shapes[0][0].clone() * (t_shapes.iter().map(|s| s[1].clone()).reduce(|acc, e| acc + e).unwrap()) && t_shapes.iter().all(|s| shape[1] == s[2] && shape[2] == s[3])
                    && shapes.iter().zip(t_shapes.iter()).all(|(shape, t_shape)| shape[0] == t_shape[0].clone() * t_shape[1].clone() && shape[1] == t_shape[2] && shape[2] == t_shape[3]))
                {
                    // s: [a,b,c1,d], [a,b,c2,d] -> [a,b,c1+c2,d] -> [a, b*(c1+c2), d]
                    // s0: [a,b,c1,d] -> [a,b*c1,d]
                    // s1: [a,b,c2,d] -> [a,b*c2,d]
                    let mut unioned = false;
                    for i in 0..$n {
                        let cur_unioned = la.union(egraph, subst, &format!("?t{i}"), &format!("(reshape ?s{} {})", i, t_shape_names[i]));
                        unioned |= cur_unioned;
                    }
                    return unioned as usize;
                }
                return 0;
            };
        )
    }};
}

macro_rules! reshape_concat_swap_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        assert!($n >= 2);
        let name = format!("<reshape-concat-swap({})>", $n);
        let s_pat = {
            let concat_pat = (1..$n).fold("?t0".to_string(), |acc, i| format!("(concat {} ?t{} ?dim)", acc, i));
            format!("?s=(reshape {} ?shape)", concat_pat)
        };
        let si_pat = (0..$n).map(|i| format!("?s{i}=(reshape ?t{i} ?shape{i})")).collect::<Vec<String>>().join(",");
        let pat = format!("{},{}", s_pat, si_pat);
        make_rewrite!(($manager, $verbose, name) => pat => |la, egraph, _matched_id, subst| {
            let t_shapes: Vec<Option<Vec<i64>>> = (0..$n).map(|i| try_get_val_shape(egraph, subst, &format!("?t{}", i))).collect();
            if t_shapes.iter().any(|s| s.is_none()) { return 0; }
            let t_shapes = t_shapes.iter().cloned().map(|s| s.unwrap()).collect::<Vec<Vec<i64>>>();
            let si_shapes: Vec<Option<Vec<i64>>> = (0..$n).map(|i| try_get_val_shape(egraph, subst, &format!("?s{}", i))).collect();
            if si_shapes.iter().any(|s| s.is_none()) { return 0; }
            let si_shapes = si_shapes.iter().cloned().map(|s| s.unwrap()).collect::<Vec<Vec<i64>>>();

            let s_shape = if let Some(v)=try_get_val_shape_from_name(egraph, subst, "?shape", la.manager.clone()) { v } else { return 0; };
            let dim = get_val(egraph, subst, "?dim") as usize;
            let c_shape = {
                let mut tmp = t_shapes[0].clone();
                for i in 1..$n {
                    tmp[dim] += t_shapes[i][dim];
                }
                tmp
            };

            if dim == 1 && c_shape.len() == 2 && s_shape.len() == 4
                && c_shape[0] == s_shape[0] * s_shape[1] && c_shape[1] == s_shape[2] * s_shape[3]
                && si_shapes.iter().all(|s| s[0] == s_shape[0] && s[1] == s_shape[1] && s[3] == s_shape[3])
            {
                // ti_shape=[a*b, ci*d] -> si_shape=[a,b,ci,d]
                // c_shape=[a*b, sum(ci)*d] -> s_shape=[a,b,sum(ci),d]
                let concat_si = (1..$n).fold("?s0".to_string(), |acc, i| format!("(concat {} ?s{} 2)", acc, i));
                let unioned = la.union(egraph, subst, "?s", &concat_si);
                return unioned as usize;
            } else if dim == 2 && c_shape.len() == 4 && s_shape.len() == 2
                && c_shape[0] * c_shape[1] == s_shape[0] && c_shape[2] * c_shape[3] == s_shape[1]
                && si_shapes.iter().all(|s| s[0] == s_shape[0])
            {
                // ti_shape=[a,b,ci,d] -> si_shape=[a*b, ci*d]
                // c_shape=[a,b,sum(ci),d] -> s_shape=[a*b, sum(ci)*d]
                let concat_si = (1..$n).fold("?s0".to_string(), |acc, i| format!("(concat {} ?s{} 1)", acc, i));
                let unioned = la.union(egraph, subst, "?s", &concat_si);
                return unioned as usize;
            }
            return 0;
        })
    }};
}

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let mut results = make_rewrites!(manager, verbose,
        //////////////////// reshape - concat/slice ////////////////////
        "<reshape-concat-swap>" =>
            "(reshape (concat ?t1 ?t2 ?dim) ?shape)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let shape = if let Some(v) = try_get_val_shape_from_name(egraph, subst, "?shape", la.manager.clone()) { v } else { return 0; };
                let t1_shape = if let Some(val) = try_get_val_shape(egraph, subst, "?t1") { val } else { return 0; };
                let t2_shape = if let Some(val) = try_get_val_shape(egraph, subst, "?t2") { val } else { return 0; };
                let concat_shape = {
                    let mut tmp = t1_shape.clone();
                    tmp[dim as usize] += t2_shape[dim as usize];
                    tmp
                };
                let pattern_src = concat_shape.clone();
                let pattern_dst = shape.clone();

                if shape == concat_shape { return 0; }

                assert!(t1_shape.len() == pattern_src.len());
                let min_n_dim = min(pattern_src.len(), pattern_dst.len());
                let mut first_affected = min_n_dim;
                for i in 0..min_n_dim {
                    if pattern_src[i] != pattern_dst[i] {
                        first_affected = i;
                        break;
                    }
                }

                // (new_shape1, new_shape2, new_dim, case)
                let mut potential_new_info: Vec<(Vec<i64>, Vec<i64>, usize, &str)> = vec![];
                if dim < first_affected {
                    // Reshape doesn't affect the slice dim, we can rewrite anyways.
                    if first_affected > t1_shape.len() {
                        assert!(first_affected == t1_shape.len() + 1);
                    }
                    let new_reshape_shape1 = {
                        let mut tmp = t1_shape[..min(t1_shape.len(), first_affected)].to_vec();
                        tmp.extend(&pattern_dst[first_affected..]);
                        tmp
                    };
                    if first_affected > t2_shape.len() {
                        assert!(first_affected == t2_shape.len() + 1);
                    }
                    let new_reshape_shape2 = {
                        let mut tmp = t2_shape[..min(t2_shape.len(), first_affected)].to_vec();
                        tmp.extend(&pattern_dst[first_affected..]);
                        tmp
                    };
                    potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim, "1"));
                } else if dim > first_affected {
                    // TODO: This may be a valid case when the merge/divide is squeeze/unsqueeze.
                    if pattern_src.len() - 1 == pattern_dst.len() {
                        // Case 4: Merge reshape when dim > first_affected
                        if first_affected >= 1
                            && pattern_src[first_affected] == 1
                            && t1_shape[first_affected] == 1
                            && t2_shape[first_affected] == 1
                        {
                            // Case 4.1: merged to previous dim.
                            // Note that the affected must be 1, otherwise, the previous dim should be affected.
                            let new_reshape_shape1 = {
                                let mut tmp = pattern_dst[..first_affected].to_vec();
                                tmp.extend(&t1_shape[first_affected + 1..]);
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = pattern_dst[..first_affected].to_vec();
                                tmp.extend(&t2_shape[first_affected + 1..]);
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim - 1, "4.1"));
                        }
                        if first_affected + 1 < dim {
                            // Case 4.2: merged to next dim.
                            let new_reshape_shape1 = {
                                let mut tmp = pattern_dst[..first_affected + 1].to_vec();
                                tmp.extend(&t1_shape[first_affected + 2..]);
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = pattern_dst[..first_affected + 1].to_vec();
                                tmp.extend(&t2_shape[first_affected + 2..]);
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim - 1, "4.2"));
                        }
                    }
                } else if dim == first_affected {
                    // Case 8: Merge reshape when dim == first_affected
                    // [a, b, c, d] -> [a, b, c*d] and dim == first_affected == c
                    if first_affected + 1 < pattern_src.len()
                        && pattern_src[first_affected] * pattern_src[first_affected + 1]
                            == pattern_dst[first_affected]
                    {
                        let new_reshape_shape1 = {
                            let mut tmp = pattern_dst.clone();
                            tmp[dim] = t1_shape[dim] * t1_shape[dim + 1];
                            tmp
                        };
                        let new_reshape_shape2 = {
                            let mut tmp = pattern_dst.clone();
                            tmp[dim] = t2_shape[dim] * t2_shape[dim + 1];
                            tmp
                        };
                        potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim, "8"));
                    }
                    if pattern_src.len() - 1 == pattern_dst.len() {
                        // Case 2: Merge reshape
                        if dim >= 1 {
                            // Case 2.1: merged to previous dim.
                            // This case is not meaningful for reshape-concat-swap.
                        }
                        {
                            // Case 2.2: merged to next dim.
                            let new_reshape_shape1 = {
                                let mut tmp = t1_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim] = t1_shape[dim] * t1_shape[dim + 1];
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = t2_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim] = t2_shape[dim] * t2_shape[dim + 1];
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim, "2.2"));
                        }
                    } else if pattern_src.len() + 1 == pattern_dst.len() {
                        // Case 3: Divide reshape
                        if pattern_dst[dim] == 1 {
                            // Case 3.1: divide the dim by pattern_dst[dim], pattern_dst[dim] == 1
                            let new_reshape_shape1 = {
                                let mut tmp = t1_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim + 1] = t1_shape[dim] / pattern_dst[dim];
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = t2_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim + 1] = t2_shape[dim] / pattern_dst[dim];
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim + 1, "3.1"));
                        }
                        // [16, 24], [16, 6], [16, 1, 6]
                        if t1_shape[dim] % pattern_dst[dim + 1] == 0
                            && t2_shape[dim] % pattern_dst[dim + 1] == 0
                        {
                            // Case 3.2: divide the dim by pattern_dst[dim+1]
                            let second_size = pattern_dst[dim + 1];
                            let new_reshape_shape1 = {
                                let mut tmp = t1_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim] = t1_shape[dim] / second_size;
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = t2_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim] = t2_shape[dim] / second_size;
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim, "3.2"));
                        }
                    }
                }
                if pattern_src.len() + 1 == pattern_dst.len() && dim > 0 {
                    // Case 5: Divide reshape when dim > divide_dim (i.e., 0 for this)
                    let divide_dim = 0; // Generalize this when I have time.
                    if divide_dim + 1 < pattern_dst.len()
                        && pattern_src[divide_dim]
                            == pattern_dst[divide_dim] * pattern_dst[divide_dim + 1]
                        && pattern_src[..divide_dim] == pattern_dst[..divide_dim]
                        && pattern_src[divide_dim + 1..] == pattern_dst[divide_dim + 2..]
                        && t1_shape[divide_dim] == pattern_src[divide_dim]
                        && t2_shape[divide_dim] == pattern_src[divide_dim]
                    {
                        // Case 5.1: [a*b, c, d] -> [a, b, c, d] and the divided dim has
                        // the same size for t1 and t2.
                        let new_reshape_shape1 = {
                            let mut tmp = pattern_dst[..dim + 1].to_vec();
                            tmp.extend(&t1_shape[dim..]);
                            tmp
                        };
                        let new_reshape_shape2 = {
                            let mut tmp = pattern_dst[..dim + 1].to_vec();
                            tmp.extend(&t2_shape[dim..]);
                            tmp
                        };
                        potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim + 1, "5.1"));
                    }
                } else if pattern_src.len() == pattern_dst.len() {
                    {
                        // Case 6: Transpose reshape, only consider 0 and 1 for now.
                        // TODO: Generalize this lemma.
                        let mut swap_dim1 = 0;
                        let mut swap_dim2 = 1;
                        if swap_dim1 > swap_dim2 {
                            (swap_dim1, swap_dim2) = (swap_dim2, swap_dim1);
                        }
                        let mut valid = pattern_src[swap_dim1] == pattern_dst[swap_dim2]
                            && pattern_src[swap_dim2] == pattern_dst[swap_dim1];
                        if valid {
                            for i in 0..pattern_src.len() {
                                if i == swap_dim1 || i == swap_dim2 {
                                    continue;
                                }
                                if pattern_src[i] != pattern_dst[i] {
                                    valid = false;
                                    break;
                                }
                            }
                        }
                        if valid && swap_dim1 < dim && swap_dim2 < dim {
                            // Case 6.1: The concat dim is later than any of the swap dims.
                            let new_reshape_shape1 = {
                                let mut tmp = t1_shape.clone();
                                tmp[swap_dim1] = t1_shape[swap_dim2];
                                tmp[swap_dim2] = t1_shape[swap_dim1];
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = t2_shape.clone();
                                tmp[swap_dim1] = t2_shape[swap_dim2];
                                tmp[swap_dim2] = t2_shape[swap_dim1];
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, dim, "6.1"));
                        }
                        if valid && swap_dim2 == dim && swap_dim1 == 0 {
                            // Case 6.2: [1, a, b, c] -> [a, 1, b, c]
                            let new_reshape_shape1 = {
                                let mut tmp = t1_shape.clone();
                                tmp[swap_dim1] = t1_shape[swap_dim2];
                                tmp[swap_dim2] = t1_shape[swap_dim1];
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = t2_shape.clone();
                                tmp[swap_dim1] = t2_shape[swap_dim2];
                                tmp[swap_dim2] = t2_shape[swap_dim1];
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, swap_dim1, "6.2"));
                        }
                    }
                    if pattern_src.len() > 1 {
                        // Case 7: merge into a dim but keep one on left.
                        // [a, b, c, d] -> [1, a*b, c, d]
                        let merge_dim1 = 0;
                        let merge_dim2 = merge_dim1 + 1;
                        assert!(merge_dim2 == merge_dim1 + 1);
                        if dim == merge_dim1
                            && pattern_dst[merge_dim1] == 1
                            && pattern_src[merge_dim1] * pattern_src[merge_dim2] == pattern_dst[merge_dim2]
                        {
                            let new_reshape_shape1 = {
                                let mut tmp = t1_shape.clone();
                                tmp[merge_dim1] = 1;
                                tmp[merge_dim2] = t1_shape[merge_dim1] * t1_shape[merge_dim2];
                                tmp
                            };
                            let new_reshape_shape2 = {
                                let mut tmp = t2_shape.clone();
                                tmp[merge_dim1] = 1;
                                tmp[merge_dim2] = t2_shape[merge_dim1] * t2_shape[merge_dim2];
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape1, new_reshape_shape2, merge_dim2, "7"));
                        }
                    }
                }
                let mut unioned_count = 0;
                for (new_reshape_shape1, new_reshape_shape2, new_dim, case) in potential_new_info {
                    let new_slice_pat = format!(
                        "(concat (reshape ?t1 {}) (reshape ?t2 {}) {new_dim})",
                        val_shape_to_underscore_name(&new_reshape_shape1),
                        val_shape_to_underscore_name(&new_reshape_shape2),
                    ).parse::<Pattern<Mdl>>().unwrap();
                    if la.verbose {
                        println!("Pre-Rewrite reshape-concat-swap(case {case}) t1_shape{t1_shape:?}, t2_shape{t2_shape:?}, dim={dim}, pattern_src{pattern_src:?}, pattern_dst{pattern_dst:?}, new_shape1={new_reshape_shape1:?}, new_shape2={new_reshape_shape2:?}, new_dim={new_dim}");
                    }
                    let unioned = la.union_src(egraph, subst, &new_slice_pat.to_string());
                    // egraph.rebuild();
                    unioned_count += unioned as usize;
                }
                return unioned_count;
            };

        "<reshape-slice-swap>" =>
            "(reshape (slice ?t ?dim ?begin ?end ?step) ?shape)" => |la, egraph, matched_id, subst| {
                // FIXME: I don't handle SymVal for reshape lemmas now, because this might introduce too many complicated computation
                // and would require user to provide much more relations among those SymVals. If you really need this lemma, please
                // implement `get_symval_instantiation` in your tg Config file.
                if !is_pure_val(egraph, subst, "?begin") || !is_pure_val(egraph, subst, "?end") || !is_pure_val(egraph, subst, "?step") {
                    return 0;
                }
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [begin, end, step] = get_vals!(egraph, subst, ["?begin", "?end", "?step"]);
                let shape = if let Some(v) = try_get_val_shape_from_name(egraph, subst, "?shape", la.manager.clone()) { v } else { return 0; };
                let t_shape = if let Some(val) = try_get_val_shape(egraph, subst, "?t") { val } else { return 0; };
                let pattern_src = {
                    let mut tmp = t_shape.clone();
                    tmp[dim as usize] = end - begin;
                    tmp
                };
                let pattern_dst = shape.clone();

                assert!(t_shape.len() == pattern_src.len());
                let min_n_dim = min(pattern_src.len(), pattern_dst.len());
                let mut first_affected = min_n_dim;
                for i in 0..min_n_dim {
                    if pattern_src[i] != pattern_dst[i] {
                        first_affected = i;
                        break;
                    }
                }

                // (new_shape, new_dim, new_begin, new_end, case)
                let mut potential_new_info: Vec<(Vec<i64>, usize, i64, i64, &str)> = vec![];
                if dim < first_affected {
                    // Reshape doesn't affect the slice dim, we can rewrite anyways.
                    if first_affected > t_shape.len() {
                        assert!(first_affected == t_shape.len() + 1);
                    }
                    let new_reshape_shape = {
                        let mut tmp = t_shape[..min(t_shape.len(), first_affected)].to_vec();
                        tmp.extend(&pattern_dst[first_affected..]);
                        tmp
                    };
                    potential_new_info.push((new_reshape_shape, dim, begin, end, "1"));
                } else if dim == first_affected && pattern_src.len() - 1 == pattern_dst.len() {
                    // Case 2: Merge reshape
                    if dim >= 1 && begin == 0 && end == 1 && t_shape[dim] == 1{
                        // Case 2.1: merged `1` to previous dim.
                        assert!(pattern_src[first_affected] == 1);
                        assert!(pattern_src[dim + 1..] == pattern_dst[dim..]);
                        let new_reshape_shape = {
                            let mut tmp = t_shape[..dim].to_vec();
                            tmp.extend(&pattern_dst[dim..]);
                            tmp
                        };
                        let new_end = new_reshape_shape[dim-1];
                        potential_new_info.push((new_reshape_shape, dim - 1, begin, new_end, "2.1"));
                    }
                    {
                        // Case 2.2: merged to next dim.
                        let new_reshape_shape = {
                            let mut tmp = t_shape[..dim].to_vec();
                            tmp.extend(&pattern_dst[dim..]);
                            tmp[dim] = t_shape[dim] * t_shape[dim + 1];
                            tmp
                        };
                        potential_new_info.push((new_reshape_shape, dim, begin * t_shape[dim + 1], end * t_shape[dim + 1], "2.2"));
                    }
                } else if dim == first_affected && pattern_src.len() + 1 == pattern_dst.len() {
                    // Case 3: Divide reshape
                    // [16, 24], [16, 6], [16, 1, 6]
                    if t_shape[dim] % pattern_dst[dim + 1] == 0
                        && begin % pattern_dst[dim + 1] == 0
                        && end % pattern_dst[dim + 1] == 0
                        && pattern_src[..dim] == pattern_dst[..dim]
                    {
                        let second_size = pattern_dst[dim + 1];
                        if begin % second_size == 0 && end % second_size == 0 {
                            // Case 3.2: divide the dim by pattern_dst[dim+1]
                            let new_reshape_shape = {
                                let mut tmp = t_shape[..dim].to_vec();
                                tmp.extend(&pattern_dst[dim..]);
                                tmp[dim] = t_shape[dim] / second_size;
                                tmp
                            };
                            potential_new_info.push((new_reshape_shape, dim, begin / second_size, end / second_size, "3.2"));
                        }
                    }
                    if pattern_dst[dim] == 1 && dim >= 1 {
                        // Case 3.3: divide from previous dim, but just unsqueeze.
                        // E.g., (16, 24) -[:, 12:18]-> (16, 6) -> (16, 1, 6)
                        let new_reshape_shape = {
                            let mut tmp = t_shape.to_vec();
                            tmp.insert(dim, 1);
                            tmp
                        };
                        potential_new_info.push((new_reshape_shape, dim + 1, begin, end, "3.3"));
                    }
                }
                let mut unioned_count = 0;
                for (new_reshape_shape, new_dim, new_begin, new_end, case) in potential_new_info
                {
                    assert!(new_begin <= new_end && new_end <= new_reshape_shape[new_dim], "new_begin={}, new_end={}, new_shape={:?}, new_dim={}", new_begin, new_end, new_reshape_shape, new_dim);
                    let new_slice_pat = format!("(slice (reshape ?t {}) {new_dim} {new_begin} {new_end} {step})", val_shape_to_underscore_name(&new_reshape_shape)).parse::<Pattern<Mdl>>().unwrap();
                    if la.verbose {
                        println!("Pre-Rewrite reshape-slice-swap(case {case}) {t_shape:?}, ({dim}, {begin}, {end}), ({new_dim}, {new_begin}, {new_end}), {pattern_src:?}, {pattern_dst:?}, new_shape: {new_reshape_shape:?}, t_id={}", get_id(subst, "?t"));
                    }
                    let unioned = la.union_src(egraph, subst, &new_slice_pat.to_string());
                    // egraph.rebuild();
                    unioned_count += unioned as usize;
                }
                return unioned_count;
            };

        "<transpose-reshape-swap>" =>
            "(transpose (reshape ?t ?re_shape) ?trans)" => |la, egraph, matched_id, subst| {
                let re_shape = shapelike_name_to_val_vec::<i64>(&get_name(egraph, subst, "?re_shape"));
                let trans = shapelike_name_to_val_vec::<i64>(&get_name(egraph, subst, "?trans"));
                let t_shape = if let Some(v) = try_get_val_shape(egraph, subst, "?t") { v } else { return 0; };
                let pattern_src = t_shape.clone();
                let pattern_dst = re_shape.clone();
                let result_shape_name = {
                    let mut tmp = vec![];
                    for i in &trans {
                        tmp.push( pattern_dst[*i as usize]);
                    }
                    val_shape_to_underscore_name(&tmp)
                };

                let (new_trans, pattern) = if pattern_src.len() == 4 && pattern_dst.len() == 3 && trans == vec![1, 0, 2]
                    && pattern_src[1] * pattern_src[2] == pattern_dst[1] && pattern_src[0] == pattern_dst[0] && pattern_src[3] == pattern_dst[2]
                {
                    (vec![1,2,0,3], "[a,b,c,d] -> [a, b*c, d]")
                } else if pattern_src.len() == 4 && pattern_dst.len() == 3 && trans == vec![0, 2, 1]
                    && pattern_src[0] * pattern_src[1] == pattern_dst[0] && pattern_src[2] == pattern_dst[1] && pattern_src[3] == pattern_dst[2]
                {
                    (vec![0,1,3,2], "[a,b,c,d] -> [a*b, c, d]")
                } else if pattern_src.len() == 3 && pattern_dst.len() == 4 && trans == vec![1, 2, 0, 3]
                    && pattern_src[1] == pattern_dst[1] * pattern_dst[2] && pattern_src[0] == pattern_dst[0] && pattern_src[2] == pattern_dst[3]
                {
                    (vec![1,0,2], "[a, b*c, d] -> [a,b,c,d] -> [b,c,a,d]")
                } else if pattern_src.len() == 3 && pattern_dst.len() == 4 && trans == vec![0, 1, 3, 2]
                    && pattern_src[0] == pattern_dst[0] * pattern_dst[1] && pattern_src[1] == pattern_dst[2] && pattern_src[2] == pattern_dst[3]
                {
                    (vec![0,2,1], "[a*b, c, d] -> [a,b,c,d] -> [a,b,d,c]")
                } else {
                    return 0;
                };
                let new_trans_str = val_shape_to_underscore_name(&new_trans);
                let unioned = la.union_src(egraph, subst, &format!("(reshape (transpose ?t {new_trans_str}) {result_shape_name})"));
                return unioned as usize;
            };

        "<reshape-matadd-swap>" =>
            "(reshape (matadd ?t1 ?t2) ?shape)" => |la, egraph, matched_id, subst| {
                let [t1_dtype, t2_dtype] = get_dtypes!(egraph, subst, vec!["?t1", "?t2"]);
                let reshape_t1 = if t1_dtype == DataKind::Tnsr { "(reshape ?t1 ?shape)" } else { "?t1" };
                let reshape_t2 = if t2_dtype == DataKind::Tnsr { "(reshape ?t2 ?shape)" } else { "?t2" };
                if t1_dtype == DataKind::Tnsr && t2_dtype == DataKind::Tnsr {
                    let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                    if t1_shape != t2_shape { return 0; }
                }
                let unioned = la.union_src(egraph, subst, &format!("(matadd {} {})", reshape_t1, reshape_t2));
                return unioned as usize;
            };
        "<reshape-reduce_add-swap>" =>
            "(reshape (reduce_add ?t1 ?t2) ?shape)" => |la, egraph, matched_id, subst| {
                let [t1_dtype, t2_dtype] = get_dtypes!(egraph, subst, vec!["?t1", "?t2"]);
                let reshape_t1 = if t1_dtype == DataKind::Tnsr { "(reshape ?t1 ?shape)" } else { "?t1" };
                let reshape_t2 = if t2_dtype == DataKind::Tnsr { "(reshape ?t2 ?shape)" } else { "?t2" };
                if t1_dtype == DataKind::Tnsr && t2_dtype == DataKind::Tnsr {
                    let [t1_shape, t2_shape] = get_shapes!(egraph, subst, vec!["?t1", "?t2"]);
                    if t1_shape != t2_shape { return 0; }
                }
                let unioned = la.union_src(egraph, subst, &format!("(reduce_add {} {})", reshape_t1, reshape_t2));
                return unioned as usize;
            };
        "<reshape-masked_select-swap>" =>
            "(reshape (masked_select ?t1 ?t2 ?mask1 ?mask2) ?shape)" => |la, egraph, matched_id, subst| {
                let [t1_shape, t2_shape, mask1_shape, mask2_shape] = get_shapes!(egraph, subst, ["?t1", "?t2", "?mask1", "?mask2"]);
                let shape = get_shape_from_name(egraph, subst, "?shape", la.manager.clone());

                if t1_shape == t2_shape && mask1_shape == mask2_shape{
                    if t1_shape == mask1_shape {
                        let unioned = la.union_src(egraph, subst, "(masked_select (reshape ?t1 ?shape) (reshape ?t2 ?shape) (reshape ?mask1 ?shape) (reshape ?mask2 ?shape))");
                        return unioned as usize;
                    } else if t1_shape.len() > mask1_shape.len() && t1_shape[..mask1_shape.len()] == mask1_shape && t1_shape == shape {
                        let unioned = la.union_src(egraph, subst, "(masked_select (reshape ?t1 ?shape) (reshape ?t2 ?shape) ?mask1 ?mask2)");
                        return unioned as usize;
                    }
                }
                return 0;
            };

        "<reshape-reshape-collapse>" =>
            "(reshape (reshape ?t ?shape1) ?shape2)" => |la, egraph, matched_id, subst| {
                let t_shape = get_shape(egraph, subst, "?t");
                let shape2 = get_shape_from_name(egraph, subst, "?shape2", la.manager.clone());

                if t_shape == shape2 {
                    let unioned = la.union_src(egraph, subst, "?t");
                    return unioned as usize;
                }
                return 0;
            };
    );
    results.extend(reshape_concat_swap_with_reshape_back_template!(
        manager, verbose, 2
    ));
    results.extend(reshape_concat_swap_with_reshape_back_template!(
        manager, verbose, 4
    ));
    results.extend(reshape_concat_swap_with_reshape_back_template!(
        manager, verbose, 6
    ));
    results.extend(reshape_concat_swap_with_reshape_back_template!(
        manager, verbose, 8
    ));
    results.push(reshape_concat_swap_template!(manager, verbose, 2));
    results.push(reshape_concat_swap_template!(manager, verbose, 4));
    results.push(reshape_concat_swap_template!(manager, verbose, 8));
    results
}
