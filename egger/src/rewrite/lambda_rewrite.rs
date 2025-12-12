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
use crate::symval::{ShapeLike, SymValManagerRef, SymVal};
use crate::utils::*;
use crate::metadata::DataKind;
use egg::{rewrite as rw, *};
use std::collections::HashSet;
use std::convert::TryInto;
use regex::Regex;

pub struct SendSyncFn {
    // We expect this `func` returns
    // 1. The node id list of src patterns;
    // 2. The result node id list.
    pub func: Box<dyn Fn(&LambdaRewrite, &mut EGraph<Mdl, TensorAnalysis>, Id, &Subst) -> usize>,
}

impl SendSyncFn {
    pub fn new<F>(f: F) -> Self
    where
        F: Fn(&LambdaRewrite, &mut EGraph<Mdl, TensorAnalysis>, Id, &Subst) -> usize + 'static,
    {
        SendSyncFn { func: Box::new(f) }
    }
}

unsafe impl Send for SendSyncFn {}

unsafe impl Sync for SendSyncFn {}


pub fn substitute_pattern_str(egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, pat: &str) -> String {
    let mut result = String::new();

    let re = Regex::new(r"\?[\w\d]+").unwrap();
    let mut last_end = 0;
    for matched in re.find_iter(pat) {
        let begin = matched.start();
        let end = matched.end();
        let key = matched.as_str();
        let meta = &egraph[subst[key.parse().unwrap()]].data;
        let val = format!("{}", meta);
        result.push_str(&pat[last_end..begin]);
        if meta.dtype == DataKind::Tnsr {
            result.push_str(key);
            result.push_str(&format!("({})", val));
        } else {
            result.push_str(&val);
        }
        last_end = end;
    }
    result.push_str(&pat[last_end..]);
    result
}

pub struct LambdaRewrite {
    pub name: String,
    pub src_pat: String,
    pub multi: bool,
    pub dst: SendSyncFn,
    pub manager: SymValManagerRef,
    pub verbose: bool,
}

impl LambdaRewrite {
    pub fn union_src(&self, egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, pat: &str) -> bool {
        self.union(egraph, subst, &self.src_pat, pat)
    }

    pub fn union(
        &self, egraph: &mut EGraph<Mdl, TensorAnalysis>, subst: &Subst, pat1: &str, pat2: &str,
    ) -> bool
    {
        let (_, unioned) = egraph.union_instantiations(
            &str2ast(pat1), 
            &str2ast(pat2), 
            subst, 
            &self.name
        );
        if self.verbose && unioned {
            println!(
                "Rewrite for {}: {} -> {}, following {pat1} -> {pat2}",
                self.name,
                substitute_pattern_str(egraph, subst, pat1),
                substitute_pattern_str(egraph, subst, pat2)
            );
        }
        unioned
    }
}

pub fn collect_lambda_rewrites_to_egg_rewrites(
    lambda_rewrites: Vec<LambdaRewrite>,
    filter: Option<HashSet<&str>>,
) -> Vec<Rewrite<Mdl, TensorAnalysis>> {
    if filter.is_some() {
        // Use filter if provided.
        lambda_rewrites
            .into_iter()
            .filter(|la| filter.as_ref().unwrap().contains(la.name.as_str()))
            .map(|la| {
                let name = la.name.clone();
                let src_pat = la.src_pat.clone();
                if !la.multi {
                    rw!(name;  {src_pat.clone().parse::<Pattern<Mdl>>().unwrap()}  => {la} )
                } else {
                    rw!(name;  {src_pat.clone().parse::<MultiPattern<Mdl>>().unwrap()}  => {la} )
                }
            })
            .collect()
    } else {
        lambda_rewrites
            .into_iter()
            .map(|la| {
                let name = la.name.clone();
                let src_pat = la.src_pat.clone();
                if !la.multi {
                    rw!(name;  {src_pat.clone().parse::<Pattern<Mdl>>().unwrap()}  => {la} )
                } else {
                    rw!(name;  {src_pat.clone().parse::<MultiPattern<Mdl>>().unwrap()}  => {la} )
                }
            })
            .collect()
    }
}

impl Applier<Mdl, TensorAnalysis> for LambdaRewrite {
    /// Apply the pattern once. Check the new nodes are valid before actually
    /// apply. See Applier trait in egg for more information.
    fn apply_one(
        &self,
        egraph: &mut EGraph<Mdl, TensorAnalysis>,
        matched_id: Id,
        subst: &Subst,
        _searcher_ast: Option<&PatternAst<Mdl>>,
        _rule_name: Symbol,
    ) -> Vec<Id> {
        if self.verbose {
            println!("Prerewrite for {}: {}", self.name, substitute_pattern_str(egraph, subst, self.src_pat.as_str()));
        }
        let count = (self.dst.func)(self, egraph, matched_id, subst);
        if egraph.analysis.enable_stats && count > 0 {
            *egraph
                .analysis
                .lemma_applied_count
                .entry(self.name.clone())
                .or_insert(0) += count;
        }
        return vec![];
    }

    fn vars(&self) -> Vec<Var> {
        vec![]
    }
}

macro_rules! make_rewrite {
    (($manager:expr, $verbose:expr, $name:expr) => $src_pat:expr => $dst_pat:literal) => {
        make_rewrite!(($manager, $verbose, $name) => $src_pat => |la, egraph, _matched_id, subst| {
            let unioned = la.union_src(egraph, subst, $dst_pat);
            return unioned as usize;
        })
    };

    (($manager:expr, $verbose:expr, $name:expr) => $src_pat:expr => $lambda:expr) => {
        if $src_pat.to_string().contains(",") {
            LambdaRewrite {
                name: $name.to_string(),
                src_pat: $src_pat.to_string(),
                multi: true,
                manager: $manager.clone(),
                verbose: $verbose,
                dst: SendSyncFn::new($lambda),
            }
        } else {
            LambdaRewrite {
                name: $name.to_string(),
                src_pat: $src_pat.to_string(),
                multi: false,
                manager: $manager.clone(),
                verbose: $verbose,
                dst: SendSyncFn::new($lambda),
            }
        }
    };
}

macro_rules! make_rewrites {
    ($manager:expr, $verbose:expr, $($name:expr => $src_pat:expr => $lambda:expr);+ $(;)?) => {
        vec![
            $(make_rewrites!($name, $manager, $verbose, $src_pat => $lambda),)+
        ]
    };

    ($name:expr, $manager:expr, $verbose:expr, $src_pat:expr => $lambda:expr) => {
        make_rewrite!(($manager, $verbose, $name) => $src_pat => $lambda)
    };
}

macro_rules! make_simple_rewrite {
    (($manager:expr, $verbose:expr, $name:expr) => $src_pat:expr => $dst_pat:expr) => {
        make_rewrite!(($manager, $verbose, $name) => $src_pat => |la, egraph, _matched_id, subst| {
            let unioned = la.union_src(egraph, subst, $dst_pat);
            return unioned as usize;
        })
    };
}

pub(crate) use make_rewrite;
pub(crate) use make_rewrites;
pub(crate) use make_simple_rewrite;

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    #[allow(unused_variables)]
    let mut results = vec![];
    #[allow(unused_variables)]
    results.extend(make_rewrites!(manager, verbose,
        "<sum-concat-swap>" =>
            "(sum_dim_int_list (concat ?inpt1 ?inpt2 ?concat_dim) ?dims)" => |la, egraph, matched_id, subst| {
                let concat_dim = get_val(egraph, subst, "?concat_dim") as usize;
                let concat_n_dim = get_n_dim(egraph, subst, "?inpt1");
                let dims = get_val_shape_from_name(egraph, subst, "?dims");
                let new_concat_dim = {
                    let mut count = 0;
                    for d in dims.iter() {
                        if d < &concat_dim {
                            count += 1;
                        }
                    }
                    concat_dim - count
                };
                assert!(concat_n_dim >= dims.len());

                if !dims.contains(&concat_dim) {
                    let unioned = la.union_src(egraph, subst, &format!("(concat (sum_dim_int_list ?inpt1 ?dims) (sum_dim_int_list ?inpt2 ?dims) {new_concat_dim})"));
                    return unioned as usize;
                } else if dims.len() == 1 && dims[0] == concat_dim {
                    let unioned = la.union_src(egraph, subst, "(matadd (sum_dim_int_list ?inpt1 ?dims) (sum_dim_int_list ?inpt2 ?dims))");
                    return unioned as usize;
                }
                return 0;
            };
        "<sum_keep-concat-swap>" =>
            "(sum_dim_int_list_keep (concat ?inpt1 ?inpt2 ?concat_dim) ?dims)" => |la, egraph, matched_id, subst| {
                let concat_dim = get_val(egraph, subst, "?concat_dim") as usize;
                let dims = get_val_shape_from_name(egraph, subst, "?dims");

                if !dims.contains(&concat_dim) {
                    let unioned = la.union_src(egraph, subst, "(concat (sum_dim_int_list_keep ?inpt1 ?dims) (sum_dim_int_list_keep ?inpt2 ?dims) ?concat_dim)");
                    return unioned as usize;
                }
                return 0;
            };

        "<baddbmm-concat-swap>" =>
            "(baddbmm (concat ?inpt11 ?inpt12 0) (concat ?inpt21 ?inpt22 0) (concat ?inpt31 ?inpt32 0) ?beta ?alpha)" => |la, egraph, matched_id, subst| {
                let [inpt11_shape, inpt12_shape, inpt21_shape, inpt22_shape, inpt31_shape, inpt32_shape] = get_shapes!(egraph, subst, ["?inpt11", "?inpt12", "?inpt21", "?inpt22", "?inpt31", "?inpt32"]);
                if inpt11_shape[0].clone() == inpt21_shape[0].clone() && inpt21_shape[0].clone() == inpt31_shape[0].clone() {
                    let unioned = la.union_src(egraph, subst, &format!("(concat (baddbmm ?inpt11 ?inpt21 ?inpt31 ?beta ?alpha) (baddbmm ?inpt12 ?inpt22 ?inpt32 ?beta ?alpha) 0)"));
                    return unioned as usize;
                }
                return 0;
            };
        
        "<addmm-concat-swap>" =>
            "(addmm (concat ?inpt1 ?inpt2 0) ?mat1 (concat ?mat21 ?mat22 1) ?beta ?alpha)" => |la, egraph, matched_id, subst| {
                let [inpt1_shape, inpt2_shape, mat1_shape, mat21_shape, mat22_shape] = get_shapes!(egraph, subst, ["?inpt1", "?inpt2", "?mat1", "?mat21", "?mat22"]);
                if inpt1_shape.len() == 1 && mat21_shape.len() == 2 && inpt1_shape[0] == mat21_shape[1] {
                    let unioned = la.union_src(egraph, subst, "(concat (addmm ?inpt1 ?mat1 ?mat21 ?beta ?alpha) (addmm ?inpt2 ?mat1 ?mat22 ?beta ?alpha) 1)");
                    return unioned as usize;
                }
                return 0;
            };
        
        "<addmm-slice-swap>" =>
            "(addmm (slice ?inpt 0 ?begin ?end ?step) ?mat1 (slice ?mat2 1 ?begin ?end ?step) ?beta ?alpha)" => |la, egraph, matched_id, subst| {
                let [inpt_shape, mat1_shape, mat2_shape] = get_shapes!(egraph, subst, ["?inpt", "?mat1", "?mat2"]);
                let [begin, end] = get_sym_vals!(egraph, subst, ["?begin", "?end"]);
                if inpt_shape.len() == 1 && mat2_shape.len() == 2 && inpt_shape[0] == mat2_shape[1] {
                    let unioned = la.union_src(egraph, subst, "(slice (addmm ?inpt ?mat1 ?mat2 ?beta ?alpha) 1 ?begin ?end ?step)");
                    return unioned as usize;
                }
                return 0;
            };

        "<baddbmm-to-bmm_alpha>" => "(baddbmm ?inpt1 ?inpt2 ?inpt3 ?beta ?alpha)" => "(bmm_alpha ?inpt2 ?inpt3 ?alpha)";
        "<bmm-to-bmm_alpha>" => "(bmm ?inpt1 ?inpt2)" => "(bmm_alpha ?inpt1 ?inpt2 1)";

        "<bmm_alpha-reshape-swap>" =>
            "(bmm_alpha (reshape ?inpt1 ?shape1) (reshape ?inpt2 ?shape2) ?alpha)" => |la, egraph, matched_id, subst| {
                let [shape1, shape2] = get_shapes_from_names!(egraph, subst, ["?shape1", "?shape2"], la.manager.clone());
                let [inpt1_shape, inpt2_shape] = get_shapes!(egraph, subst, ["?inpt1", "?inpt2"]);
                let inpt_n_dim = inpt1_shape.len();
                let reshaped_n_dim = shape1.len();

                if inpt1_shape.len() == inpt2_shape.len() && inpt1_shape[..inpt_n_dim-2] == inpt2_shape[..inpt_n_dim-2]
                    && inpt1_shape[inpt_n_dim-2..] == shape1[reshaped_n_dim-2..] && inpt2_shape[inpt_n_dim-2..] == shape2[reshaped_n_dim-2..]
                {
                    let result_shape = {
                        let mut shape = shape1.clone();
                        shape[reshaped_n_dim-1] = shape2[reshaped_n_dim-1].clone();
                        shape
                    };
                    let unioned = la.union_src(egraph, subst, &format!("(reshape (bmm_alpha ?inpt1 ?inpt2 ?alpha) {})", shape_to_underscore_name(&result_shape)));
                    return unioned as usize;
                }
                return 0;
            };

        "<bmm_alpha-concat-swap>" =>
            "(bmm_alpha (concat ?inpt11 ?inpt12 ?dim) (concat ?inpt21 ?inpt22 ?dim) ?alpha)" => |la, egraph, matched_id, subst| {
                let [inpt11_shape, inpt12_shape, inpt21_shape, inpt22_shape] = get_shapes!(egraph, subst, ["?inpt11", "?inpt12", "?inpt21", "?inpt22"]);
                let [n11, n12, n21, n22] = [inpt11_shape.len(), inpt12_shape.len(), inpt21_shape.len(), inpt22_shape.len()];
                let dim = get_val(egraph, subst, "?dim") as usize;

                if dim < n11 - 2 && inpt11_shape[..n11-2] == inpt21_shape[..n21-2] && inpt12_shape[..n12-2] == inpt22_shape[..n22-2]
                    && inpt11_shape[n11-2..] == inpt12_shape[n12-2..] && inpt21_shape[n21-2..] == inpt22_shape[n22-2..]
                {
                    let unioned = la.union_src(egraph, subst, "(concat (bmm_alpha ?inpt11 ?inpt21 ?alpha) (bmm_alpha ?inpt12 ?inpt22 ?alpha) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };

        "<mask_fill_scalar-concat-swap>" =>
            "(mask_fill_scalar (concat ?inpt1 ?inpt2 ?dim) ?mask ?value)" => |la, egraph, matched_id, subst| {
                let dim = get_val(egraph, subst, "?dim") as usize;
                let mask_shape = get_shape(egraph, subst, "?mask");

                if mask_shape[dim] == 1 {
                    // FIXME: This is just a special case.
                    let unioned = la.union_src(egraph, subst, "(concat (mask_fill_scalar ?inpt1 ?mask ?value) (mask_fill_scalar ?inpt2 ?mask ?value) ?dim)");
                    return unioned as usize;
                }
                return 0;
            };

        "<max_dim_0-concat-swap>" =>
            "(max_dim_0 (concat ?t1 ?t2 ?concat_dim) ?max_dim)" => |la, egraph, matched_id, subst| {
                let [concat_dim, max_dim] = get_usize_vals!(egraph, subst, ["?concat_dim", "?max_dim"]);
                if concat_dim == max_dim {
                    let unioned = la.union_src(egraph, subst, "(maximum (max_dim_0 ?t1 ?max_dim) (max_dim_0 ?t2 ?max_dim) )");
                    return unioned as usize;
                }
                return 0;
            };
        
        "<matmul-first-concat-swap>" => "(matmul (concat ?x0 ?x1 0) ?A)" => "(concat (matmul ?x0 ?A) (matmul ?x1 ?A) 0)";
        "<matmul-second-concat-swap>" => "(matmul ?x0 (concat ?x1 ?x2 1))" => "(concat (matmul ?x0 ?x1) (matmul ?x0 ?x2) 1)";
        "<matmul-both-concat-swap>" => "(matmul (concat ?x0 ?x1 1) (concat ?A0 ?A1 0))" => "(reduce_add (matmul ?x0 ?A0) (matmul ?x1 ?A1))";
        "<matmul-first-slice-swap>" => "(matmul (slice ?x0 0 ?begin ?end 1) ?x1)" => "(slice (matmul ?x0 ?x1) 0 ?begin ?end 1)";
        "<matmul-second-slice-swap>" => "(matmul ?t1 (slice ?t2 1 ?begin ?end 1))" => "(slice (matmul ?t1 ?t2) 1 ?begin ?end 1)";
        "<sum_default-concat-swap>" => "(sum_default (concat ?t1 ?t2 ?dim))" => "(reduce_add (sum_default ?t1) (sum_default ?t2))";
        "<sum_default-reduce_add-swap>" => "(sum_default (reduce_add ?t1 ?t2))" => "(reduce_add (sum_default ?t1) (sum_default ?t2))";
        "<reduce_add-to-matadd-rewrite>" => "(reduce_add ?t1 ?t2)" => "(matadd ?t1 ?t2)";

        "<index_put_acc_2-concat-swap(full-arange)>" => 
            "?s=(index_put_acc_2 (concat ?inpt1 ?inpt2 ?dim) (arange 0 ?end) (concat ?indices1 ?indices2 ?dim) (concat ?value1 ?value2 ?dim)),
             ?s1=(index_put_acc_2 ?inpt1 (arange 0 ?end0) ?indices1 ?value1),
             ?s2=(index_put_acc_2 ?inpt2 (arange 0 ?end1) ?indices2 ?value2)" => |la, egraph, matched_id, subst| {
                let [end, end0, end1] = get_usize_vals!(egraph, subst, ["?end", "?end0", "?end1"]);
                let dim = get_val(egraph, subst, "?dim") as usize;
                let [inpt1_shape, inpt2_shape, indices1_shape, indices2_shape, value1_shape, value2_shape] = get_shapes!(egraph, subst, ["?inpt1", "?inpt2", "?indices1", "?indices2", "?value1", "?value2"]);
                if dim == 0 && end == end0 + end1 && inpt1_shape[0] == end0 && inpt2_shape[0] == end1 && indices1_shape[0] == value1_shape[0] && indices2_shape[0] == value2_shape[0] {
                    let unioned = la.union(egraph, subst, "?s", "(concat ?s1 ?s2 ?dim)");
                    return unioned as usize;
                }
                return 0;
             };
        
        "<squeeze-concat-swap>" =>
            "(squeeze (concat ?inpt1 ?inpt2 ?cdim) ?sdim)" => |la, egraph, matched_id, subst| {
                let [cdim, sdim] = get_vals!(egraph, subst, ["?cdim", "?sdim"]);
                if cdim != sdim {
                    let unioned = la.union_src(egraph, subst, "(concat (squeeze ?inpt1 ?sdim) (squeeze ?inpt2 ?sdim) ?cdim)");
                    return unioned as usize;
                }
                return 0;
            }

    ));

    results
}
