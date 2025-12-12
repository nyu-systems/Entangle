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

use std::convert::TryInto;

use crate::metadata::*;
use crate::model::{Mdl, TensorAnalysis};
use crate::symval::{ShapeLike, SymVal};
use crate::utils::{is_val_shape, shapelike_to_val_shape};
use egg::*;

macro_rules! vllm_rotary_embedding_template {
    ($args:expr, $x:expr, $idx:expr) => {{
        assert!($args.len() == 5);
        let [positions, query, key, cos_sin_cache, head_size]: [Id; 5] =
            $args[..5].try_into().unwrap();
        assert!($x(&positions).dtype == DataKind::Tnsr);
        assert!($x(&query).dtype == DataKind::Tnsr);
        assert!($x(&key).dtype == DataKind::Tnsr);
        assert!($x(&cos_sin_cache).dtype == DataKind::Tnsr);
        assert!($x(&head_size).dtype.is_scalar_or_name());
        let positions_shape = $x(&positions).get_shape();
        let query_shape = $x(&query).get_shape();
        let key_shape = $x(&key).get_shape();
        let cos_sin_cache_shape = $x(&cos_sin_cache).get_shape();
        assert!(positions_shape.len() == 1);
        assert!(query_shape.len() == 2);
        assert!(key_shape.len() == 2);
        assert!(cos_sin_cache_shape.len() == 2);
        assert!(positions_shape[0] == query_shape[0]);
        assert!(positions_shape[0] == key_shape[0]);
        assert!(positions_shape[0] == cos_sin_cache_shape[0]);

        let shape = match $idx {
            0 => &query_shape,
            1 => &key_shape,
            _ => panic!("Invalid index"),
        };
        Some(ValTnsr::new_tensor(&shape))
    }};
}

pub fn make(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    name: &Symbol,
    args: &Vec<Id>,
) -> Option<ValTnsr> {
    let manager = &egraph.analysis.manager.clone().unwrap();
    let x = |i: &Id| &egraph[*i].data;
    match name.as_str() {
        "vllm_rotary_embedding_0" => vllm_rotary_embedding_template!(args, x, 0),
        "vllm_rotary_embedding_1" => vllm_rotary_embedding_template!(args, x, 1),
        "vllm_unified_attention_with_output" => {
            assert!(args.len() == 3);
            let [q, k, v]: [Id; 3] = args[..3].try_into().unwrap();
            assert!(x(&q).dtype == DataKind::Tnsr);
            assert!(x(&k).dtype == DataKind::Tnsr);
            assert!(x(&v).dtype == DataKind::Tnsr);

            let q_sym_shape = x(&q).get_shape();
            let k_sym_shape = x(&k).get_shape();
            let v_sym_shape = x(&v).get_shape();
            assert!(is_val_shape(&q_sym_shape));
            assert!(is_val_shape(&k_sym_shape));
            assert!(is_val_shape(&v_sym_shape));
            let q_shape = shapelike_to_val_shape(&q_sym_shape);
            let k_shape = shapelike_to_val_shape(&k_sym_shape);
            let v_shape = shapelike_to_val_shape(&v_sym_shape);

            assert!(q_shape.len() == 3);
            assert!(k_shape.len() == 3);
            assert!(v_shape.len() == 3);
            assert!(q_shape[2] == k_shape[2]);
            assert!(k_shape[1] == v_shape[1]);
            assert!(k_shape[2] == v_shape[2]);
            assert!(q_shape[0] == k_shape[0]);
            assert!(q_shape[0] == v_shape[0]);

            // The q is evenly split for each k and v.
            assert!(q_shape[1] % k_shape[1] == 0);
            assert!(k_shape[1] == v_shape[1]);
            let shape = vec![
                q_sym_shape[0].clone(),
                q_sym_shape[1].clone(),
                v_sym_shape[2].clone(),
            ];
            Some(ValTnsr::new_tensor(&shape))
        }
        "vllm_fused_add_rms_norm" => {
            assert!(args.len() == 4);
            let [inpt, residual, weight, eps] = args[..4].try_into().unwrap();
            assert!(x(&inpt).dtype == DataKind::Tnsr);
            assert!(x(&residual).dtype == DataKind::Tnsr);
            assert!(x(&weight).dtype == DataKind::Tnsr);
            assert!(x(&eps).dtype.is_scalar_or_name());
            let inpt_shape = x(&inpt).get_shape();
            let residual_shape = x(&residual).get_shape();
            let weight_shape = x(&weight).get_shape();

            assert!(inpt_shape == residual_shape);
            assert!(weight_shape.len() == 1);
            assert!(inpt_shape[inpt_shape.len() - 1] == weight_shape[0]);
            Some(ValTnsr::new_tensor(&inpt_shape))
        }
        "vllm_silu_and_mul" => {
            assert!(args.len() == 1);
            let inpt = args[0];
            assert!(x(&inpt).dtype == DataKind::Tnsr);
            let inpt_sym_shape = x(&inpt).get_shape();
            let inpt_shape = shapelike_to_val_shape(&inpt_sym_shape);

            assert!(inpt_shape[inpt_shape.len() - 1] % 2 == 0);
            let mut shape = inpt_shape.clone();
            shape[inpt_shape.len() - 1] /= 2;
            let shape: ShapeLike = shape
                .iter()
                .map(|&x| SymVal::new_val(x, manager.clone()))
                .collect();
            Some(ValTnsr::new_tensor(&shape))
        }
        _ => None,
    }
}
