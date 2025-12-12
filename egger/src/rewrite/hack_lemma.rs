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

macro_rules! vocab_paral_cross_entropy_backward_template {
    ($manager:expr, $verbose:expr, $n:expr) => {{
        let mut results = vec![];

        let concat_of_softmax = (1..$n).fold("?softmax0".to_string(), |acc, i| format!("(concat {} ?softmax{} 2)", acc, i));
        let targets_pat = (0..$n).map(|i| format!("?target{i}=(vocab_parallel_cross_entropy_backward ?grad_output{i} ?softmax{i} ?target_mask{i} ?masked_target_1d{i} ?label_smoothing{i} ?vocab_size{i})")).collect::<Vec<_>>().join(", ");
        let pat = format!("?origin=(vocab_parallel_cross_entropy_backward ?grad_output {concat_of_softmax} ?target_mask ?masked_target_1d ?label_smoothing ?vocab_size), {}", targets_pat);
        results.push(make_rewrite!(($manager, $verbose, format!("<vocab_paral_cross_entropy_backward-swap({})>", $n)) => pat => |la, egraph, _matched_id, subst| {
            let unioned = la.union(egraph, subst,
                &format!("?origin"),
                &(1..$n).fold("?target0".to_string(), |acc, i| format!("(concat {} ?target{} 2)", acc, i)),
            );
            return unioned as usize;
        }));

        let concat_of_grad_output = (1..$n).fold("?grad_output0".to_string(), |acc, i| format!("(concat {} ?grad_output{} 0)", acc, i));
        let concat_of_softmax = (1..$n).fold("?softmax0".to_string(), |acc, i| format!("(concat {} ?softmax{} 2)", acc, i));
        let concat_of_target_mask = (1..$n).fold("?target_mask0".to_string(), |acc, i| format!("(concat {} ?target_mask{} 0)", acc, i));
        let concat_of_masked_target_1d = (1..$n).fold("?masked_target_1d0".to_string(), |acc, i| format!("(concat {} ?masked_target_1d{} 0)", acc, i));
        let pat = format!("(vocab_parallel_cross_entropy_backward {concat_of_grad_output} {concat_of_softmax} {concat_of_target_mask} {concat_of_masked_target_1d} ?label_smoothing ?vocab_size)");
        results.push(make_rewrite!(($manager, $verbose, format!("<vocab_paral_cross_entropy_backward-sp-concat-swap({})>", $n)) => pat => |la, egraph, _matched_id, subst| {
            let targets_pats: Vec<String> = (0..$n).map(|i| format!("(vocab_parallel_cross_entropy_backward ?grad_output{i} ?softmax{i} ?target_mask{i} ?masked_target_1d{i} ?label_smoothing ?vocab_size)")).collect();
            let concat_targets = (1..$n).fold(targets_pats[0].clone(), |acc, i| format!("(concat {} {} 0)", acc, targets_pats[i]));
            let unioned = la.union_src(egraph, subst, &concat_targets);
            return unioned as usize;
        }));

        results
    }};
}

pub fn get_rules(manager: SymValManagerRef, verbose: bool) -> Vec<LambdaRewrite> {
    let mut results = vec![];
    results.extend(vocab_paral_cross_entropy_backward_template!(
        manager, verbose, 2
    ));
    results.extend(vocab_paral_cross_entropy_backward_template!(
        manager, verbose, 4
    ));
    results.extend(vocab_paral_cross_entropy_backward_template!(
        manager, verbose, 8
    ));
    results
}
