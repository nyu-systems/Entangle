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
use crate::model::{Mdl, TensorAnalysis};
use crate::utils::get_left_dim;
use egg::*;
use std::convert::TryInto;

pub fn make(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    name: &Symbol,
    args: &Vec<Id>,
) -> Option<ValTnsr> {
    let x = |i: &Id| &egraph[*i].data;
    let manager = egraph.analysis.manager.clone().unwrap();
    match name.as_str() {
        "hlo_broadcast" => {
            assert!(
                args.len() == 3,
                "{:?}",
                args.iter()
                    .map(|id| egraph.id_to_expr(*id).to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let [inpt, dims, shape] = args[..3].try_into().unwrap();
            assert!(x(&inpt).dtype == DataKind::Tnsr);
            assert!(x(&dims).dtype.is_scalar_or_name());
            assert!(x(&shape).dtype.is_scalar_or_name());

            let inpt_shape = x(&inpt).get_shape();
            let dims = x(&dims).parse_to_val_shape();
            let shape = x(&shape).parse_to_shapelike(manager);
            for i in 0..dims.len() {
                assert!(
                    shape[dims[i] as usize] == inpt_shape[i],
                    "Got inpt_shape: {:?}, dims {:?} and shape {:?} that do not match",
                    inpt_shape,
                    dims,
                    shape
                );
            }
            Some(ValTnsr::new_tensor(&shape))
        }
        "hlo_dot" => {
            assert!(args.len() == 6);
            let [lhs, rhs, lhs_contracting_dims, rhs_contracting_dims, lhs_batch_dims, rhs_batch_dims] =
                args[..6].try_into().unwrap();
            assert!(x(&lhs).dtype == DataKind::Tnsr);
            assert!(x(&rhs).dtype == DataKind::Tnsr);
            assert!(x(&lhs_contracting_dims).dtype.is_scalar_or_name());
            assert!(x(&rhs_contracting_dims).dtype.is_scalar_or_name());
            assert!(x(&lhs_batch_dims).dtype.is_scalar_or_name());
            assert!(x(&rhs_batch_dims).dtype.is_scalar_or_name());

            let lhs_shape = x(&lhs).get_shape();
            let rhs_shape = x(&rhs).get_shape();
            let lhs_contracting_dims = x(&lhs_contracting_dims).parse_to_val_shape();
            let rhs_contracting_dims = x(&rhs_contracting_dims).parse_to_val_shape();
            let lhs_batch_dims = x(&lhs_batch_dims).parse_to_val_shape();
            let rhs_batch_dims = x(&rhs_batch_dims).parse_to_val_shape();

            assert!(
                lhs_contracting_dims.len() == rhs_contracting_dims.len()
                    && lhs_contracting_dims.len() == 1,
                "Got contracting dims lhs={:?} rhs={:?}",
                lhs_contracting_dims,
                rhs_contracting_dims
            );
            assert!(
                lhs_batch_dims.len() == rhs_batch_dims.len(),
                "Got batch dims lhs={:?} rhs={:?}",
                lhs_batch_dims,
                rhs_batch_dims
            );
            assert!(lhs_batch_dims.len() + lhs_contracting_dims.len() + 1 == lhs_shape.len());
            assert!(rhs_batch_dims.len() + rhs_contracting_dims.len() + 1 == rhs_shape.len());
            let mut lhs_batch_sizes = vec![];
            for i in 0..lhs_batch_dims.len() {
                lhs_batch_sizes.push(lhs_shape[lhs_batch_dims[i] as usize].clone());
            }
            let mut rhs_batch_sizes = vec![];
            for i in 0..rhs_batch_dims.len() {
                rhs_batch_sizes.push(rhs_shape[rhs_batch_dims[i] as usize].clone());
            }
            assert!(
                lhs_batch_sizes == rhs_batch_sizes,
                "Batch sizes do not match: lhs={:?} rhs={:?}",
                lhs_batch_sizes,
                rhs_batch_sizes
            );
            let lhs_dim = get_left_dim(lhs_shape.len(), &lhs_batch_dims, &lhs_contracting_dims);
            let rhs_dim = get_left_dim(rhs_shape.len(), &rhs_batch_dims, &rhs_contracting_dims);
            let shape = {
                let mut tmp = lhs_batch_sizes.clone();
                tmp.push(lhs_shape[lhs_dim].clone());
                tmp.push(rhs_shape[rhs_dim].clone());
                tmp
            };
            Some(ValTnsr::new_tensor(&shape))
        }
        "hlo_gather" => {
            assert!(args.len() == 7);
            let [operand, start_indices, offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim, slice_sizes] =
                args[..7].try_into().unwrap();
            assert!(x(&operand).dtype == DataKind::Tnsr);
            assert!(x(&start_indices).dtype == DataKind::Tnsr);
            assert!(x(&offset_dims).dtype.is_scalar_or_name());
            assert!(x(&collapsed_slice_dims).dtype.is_scalar_or_name());
            assert!(x(&start_index_map).dtype.is_scalar_or_name());
            assert!(x(&index_vector_dim).dtype.is_scalar_or_name());
            assert!(x(&slice_sizes).dtype.is_scalar_or_name());

            let start_indices_shape = x(&start_indices).get_shape();
            let collapsed_slice_dims = x(&collapsed_slice_dims).parse_to_val_shape();
            let index_vector_dim = x(&index_vector_dim).get_val();
            let slice_sizes = x(&slice_sizes).parse_to_shapelike(manager.clone());

            let mut shape = vec![];
            for i in 0..start_indices_shape.len() {
                if index_vector_dim != i as i64 {
                    shape.push(start_indices_shape[i].clone());
                }
            }
            for i in 0..slice_sizes.len() {
                if !collapsed_slice_dims.contains(&(i as i64)) {
                    shape.push(slice_sizes[i].clone());
                }
            }
            Some(ValTnsr::new_tensor(&shape))
        }
        "hlo_logistic" => {
            assert!(args.len() == 1);
            let [inpt] = args[..1].try_into().unwrap();
            assert!(x(&inpt).dtype == DataKind::Tnsr);
            let inpt_shape = x(&inpt).get_shape();
            Some(ValTnsr::new_tensor(&inpt_shape))
        }
        "hlo_max" => {
            assert!(args.len() == 2);
            let [t, dim] = args[..2].try_into().unwrap();
            assert!(x(&t).dtype == DataKind::Tnsr);
            assert!(x(&dim).dtype.is_scalar_or_name());

            let t_shape = x(&t).get_shape();
            let dim = x(&dim).get_val();
            assert!(dim < t_shape.len() as i64);

            let mut shape = vec![];
            for i in 0..t_shape.len() {
                if i != dim as usize {
                    shape.push(t_shape[i].clone());
                }
            }
            Some(ValTnsr::new_tensor(&shape))
        }
        "hlo_rms_norm" => {
            assert!(args.len() == 3);
            let [inpt, weight, eps] = args[..3].try_into().unwrap();
            assert!(x(&inpt).dtype == DataKind::Tnsr);
            assert!(x(&weight).dtype == DataKind::Tnsr);
            assert!(x(&eps).dtype.is_scalar_or_name());

            let inpt_shape = x(&inpt).get_shape();
            let weight_shape = x(&weight).get_shape();
            assert!(weight_shape.len() == 1 && weight_shape[0] == inpt_shape[0]);
            Some(ValTnsr::new_tensor(&inpt_shape))
        }
        "hlo_select" => {
            assert!(args.len() == 3);
            let [mask, on_true, on_false] = args[..3].try_into().unwrap();
            assert!(x(&mask).dtype == DataKind::Tnsr);
            assert!(x(&on_true).dtype == DataKind::Tnsr);
            assert!(x(&on_false).dtype == DataKind::Tnsr);

            let mask_shape = x(&mask).get_shape();
            let on_true_shape = x(&on_true).get_shape();
            let on_false_shape = x(&on_false).get_shape();
            assert!(mask_shape == on_true_shape && mask_shape == on_false_shape);
            Some(ValTnsr::new_tensor(&mask_shape))
        }
        _ => None,
    }
}
