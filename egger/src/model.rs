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

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::metadata::*;
use crate::special::hlo_model;
use crate::special::special_model;
use crate::special::vllm_model;
use crate::symval::*;
use crate::utils::*;
use core::panic;
use itertools::zip;
use std::collections::{HashMap, HashSet};
use std::usize;
use std::vec;

use egg::*;

define_language! {
    pub enum Mdl {
        Num(i64),
        Var(Symbol),
        "input"     = Input([Id; 1]), // takes a Var, format: name@dim1_dim2...
        "empty"     = Empty([Id; 1]), // shape
        "noop"      = Noop([Id; 2]), // No op, use to combine the outputs of a graph in case there are multiple, since egg works with single root graph

        // From open-sourced libs
        // aten
        "add_scalar" = AddScalar([Id; 2]), // input, scalar
        "addcmul" = Addcmul([Id; 4]), // input, tensor1, tensor2, value
        "addmm"     = Addmm([Id; 5]), // input, mat1, mat2, beta, alpha
        "arange"    = Arange([Id; 2]), // start, end
        "argsort" = Argsort([Id; 1]), // input
        "baddbmm"   = Baddbmm([Id; 5]), // input1, input2, input3, beta, alpha
        "bmm" = Bmm([Id; 2]), // input1, input2
        "bmm_alpha" = BmmAlpha([Id; 3]), // input2, input3, alpha
        "bitwise_and" = BitwiseAnd([Id; 2]), // input1, input2
        "bitwise_not" = BitwiseNot([Id; 1]), // input
        "bitwise_or" = BitwiseOr([Id; 2]), // input1, input2
        "clamp" = Clamp([Id; 2]), // input, min
        "concat"    = Concat([Id; 3]), // input1, input2, axis
        "cos"       = Cos([Id; 1]), // input
        "embedding" = Embedding([Id; 2]), // weight, indices
        "ewadd"     = Ewadd([Id; 2]),
        "ewmul"     = Ewmul([Id; 2]),
        "eq_scalar" = EqScalar([Id; 2]), // input, scalar
        "exp"       = Exp([Id; 1]), // input
        "expand" = Expand([Id; 2]), //input, shape_name
        "fill" = Fill([Id;2]), // shape, value
        "ge_scalar" = GEScalar([Id; 2]), // input, scalar
        "ge_tensor" = GETensor([Id; 2]), // input1, input2
        "gelu" = Gelu([Id; 1]), // input
        "gelu_backward" = GeluBackward([Id; 2]), // grad_output, input
        "index_put" = IndexPut([Id; 3]), // input, indices, value
        "index_put_acc_2" = IndexPutAcc2([Id; 4]), // input indices0, indices1, value
        "index_fill_scalar" = IndexFillScalar([Id; 4]), // input, index, dim, value
        "log"       = Log([Id; 1]), // input
        "le_tensor" = LETensor([Id; 2]), // input1, input2
        "lt_scalar" = LTScalar([Id; 2]), // input, scalar
        "mask_fill_scalar" = MaskFillScalar([Id; 3]), // input, mask, scalar
        "matadd"    = Matadd([Id; 2]), // input1, input2
        "matdiv"    = Matdiv([Id; 2]), // input1, input2
        "matmul"    = Matmul([Id; 2]), // input1, input2
        "matsub"    = Matsub([Id; 2]), // input1, input2
        "maximum" = Maximum([Id; 2]), // input1, input2
        "max_dim_0" = MaxDim0([Id; 2]), // input, dim. This returns max values
        "max_dim_1" = MaxDim1([Id; 2]), // input, dim. This returns max indices
        "mean_default" = MeanDefault([Id; 1]),
        "pad"       = Pad([Id; 7]), // inpt, pad_before0, pad_after_0, pad_before_1, pad_after_1, pad_before_2, pad_after_2
        "pow_tensor_scalar" = PowTensorScalar([Id; 2]), // input, exp
        "relu"      = Relu(Id),
        "repeat" = Repeat([Id; 2]), // input, repeat_factor
        "reshape"   = Reshape([Id; 2]), // input, shape_name (format: dim1_dim2...)
        "scatter_src" = ScatterSrc([Id; 4]), // inpt, index, src, dim
        "masked_select" = MaskedSelect([Id; 4]), // inpt1, inpt2, mask1, mask2
        "sigmoid" = Sigmoid([Id; 1]), // inpt
        "sigmoid_backward" = SigmoidBackward([Id; 2]), // grad_output, input
        "silu" = Silu([Id; 1]), // input
        "sin" = Sin([Id; 1]), // input
        "sinkhorn" = Sinkhorn([Id; 1]), // input
        "slice"     = Slice([Id; 5]), // input, axis, start, end, step
        "slice_backward" = SliceBackward([Id; 6]), // grad_output, input_shape, axis, start, end, step
        "slice_scatter" = SliceScatter([Id; 5]), // inpt, src, dim, start, end
        "softmax" = Softmax([Id; 2]), // input, dim
        "softmax_backward_data" = SoftmaxBackwardData([Id; 3]), // grad_output, output, dim
        "sort_0" =  Sort0([Id; 1]), // input, returns values
        "sort_1" = Sort1([Id; 1]), // input, returns indices
        "squeeze" = Squeeze([Id; 2]), // input, dim
        "sum_dim_int_list" = SumDimIntList([Id; 2]), // input, dims
        "sum_dim_int_list_keep" = SumDimIntListKeep([Id; 2]), // input, dims
        "sum_default" = SumDefault([Id; 1]),
        "topk_0" = Topk0([Id; 5]), // input, k, dim, largest, sorted, returns values
        "topk_1" = Topk1([Id; 5]), // input, k, dim, largest, sorted, returns indices
        "transpose" = Transpose([Id; 2]), // input, perm_name (format: dim1_dim2...)
        "unsqueeze" = Unsqueeze([Id; 2]), // input, dim

        // Dist
        "reduce_add" = ReduceAdd([Id; 2]), // input1, input2

        // Megatron-LM
        "vocab_parallel_cross_entropy_backward" = VocabParallelCrossEntropyBackward([Id; 6]), // grad_output, softmax, target_mask, masked_target_1d, label_smoothing, vocab_size

        // apex
        "fused_layernorm_affine_0" = FusedLayernormAffine0([Id; 5]), // input, weight, bias, shape, eps
        "fused_layernorm_affine_bwd_1" = FusedLayernormAffineBwd1([Id; 7]), // grad_output, mean, invvar, input_or_output, weight, bias, eps
        "rms_backward_affine_0" = RmsBackwardAffine0([Id; 5]), // grad_output, weight, invvar, input_or_output, (removed shape), eps, (removed memory_efficient)
        "rms_backward_affine_1" = RmsBackwardAffine1([Id; 5]), // grad_output, weight, invvar, input_or_output, (removed shape), eps, (removed memory_efficient)
        "rms_forward_affine_0" = RmsForwardAffine0([Id; 3]), // output: input, weight, eps
        "rms_forward_affine_1" = RmsForwardAffine1([Id; 3]), // invvar: input, weight, eps

        // Special
        // SymInt
        "+" = SymAdd([Id; 2]),
        "-" = SymSub([Id; 2]),
        "*" = SymMul([Id; 2]),

        // Predicates
        "all_ones" = AllOnes([Id; 1]), // input
        "all_zeros" = AllZeros([Id; 1]), // input

        // Fused
        Other(Symbol, Vec<Id>),
    }
}

///////////////////////////////////////////////////////////////////////////////
/// TensorAnalysis
///////////////////////////////////////////////////////////////////////////////
/// Struct for metadata analysis
///
/// This analysis mainly maintains the shape of tensors.
pub struct TensorAnalysis {
    /// Record blacklisted nodes for filtering cycles
    pub blacklist_nodes: HashSet<Mdl>,
    /// Newly added nodes by order
    pub newly_added: Vec<Mdl>,
    /// SymValManagerRef
    pub manager: Option<SymValManagerRef>,
    /// Name to shapes
    pub name_to_shapes: HashMap<String, ShapeLike>,

    /// Enable stats
    pub enable_stats: bool,
    /// Lemma Used Count
    pub lemma_applied_count: HashMap<String, usize>,
}

macro_rules! a2a_template {
    ($args:expr, $x:expr, $return_idx:expr) => {{
        let len = $args.len();
        let inpts = &$args[..len - 0];
        // FIXME: Let's assume evenly partition now.
        let inpt_shapes = inpts
            .iter()
            .map(|id| $x(id).get_shape())
            .collect::<Vec<_>>();
        // FIXME: Let's assume shapes of the inputs are all the same.
        for inpt_shape in inpt_shapes.iter() {
            assert!(inpt_shapes[0].clone() == inpt_shape.clone());
        }
        Some(ValTnsr::new_tensor(&inpt_shapes[$return_idx]))
    }};
}

fn variable_op_make(
    egraph: &mut EGraph<Mdl, TensorAnalysis>,
    name: &Symbol,
    args: &Vec<Id>,
) -> Option<ValTnsr> {
    let _manager = &egraph.analysis.manager.clone().unwrap();
    let x = |i: &Id| &egraph[*i].data;
    match name.as_str() {
        "index" => {
            assert!(args.len() >= 2);
            let inpt = &args[0];
            let index_ids = &args[1..];

            assert!(x(inpt).dtype == DataKind::Tnsr);
            let inpt_shape = x(inpt).get_shape();

            if index_ids.len() == 1 {
                let indices = &index_ids[0];
                // Check types
                assert!(x(indices).dtype == DataKind::Tnsr);

                // Get arguments
                let indices_shape = x(indices).get_shape();
                assert!(indices.len() == 1, "Haven't seen other case yet.");

                let mut shape = indices_shape.clone();
                shape.extend_from_slice(&inpt_shape[1..]);
                Some(ValTnsr::new_tensor(&shape))
            } else if index_ids.len() == inpt_shape.len() {
                let indices_shapes = index_ids
                    .iter()
                    .map(|id| x(id).get_shape())
                    .collect::<Vec<_>>();
                assert!(
                    indices_shapes.iter().all(|s| s.len() == 1),
                    "Haven't seen other case yet: {:?}",
                    indices_shapes
                );
                let first_len = &indices_shapes[0][0];
                let result_shape = vec![first_len.clone()];
                assert!(indices_shapes.iter().all(|s| s[0] == *first_len));
                Some(ValTnsr::new_tensor(&result_shape))
            } else {
                None
            }
        }
        "concat_n" => {
            assert!(args.len() >= 3);
            let inpts = &args[0..args.len() - 1];
            let dim = &args[args.len() - 1];
            let inpt_shapes = inpts.iter().map(|id| x(id).get_shape()).collect::<Vec<_>>();
            let mut shape = inpt_shapes[0].clone();
            for i in 1..inpt_shapes.len() {
                assert!(inpt_shapes[i].len() == shape.len());
                for j in 0..shape.len() {
                    if j == x(dim).get_val() as usize {
                        shape[j] += inpt_shapes[i][j].clone();
                    } else {
                        assert!(shape[j] == inpt_shapes[i][j]);
                    }
                }
            }
            Some(ValTnsr::new_tensor(&shape))
        }
        "a2a_0" => a2a_template!(args, x, 0),
        "a2a_1" => a2a_template!(args, x, 1),
        "a2a_2" => a2a_template!(args, x, 2),
        "a2a_3" => a2a_template!(args, x, 3),
        "a2a_4" => a2a_template!(args, x, 4),
        "a2a_5" => a2a_template!(args, x, 5),
        "a2a_6" => a2a_template!(args, x, 6),
        "a2a_7" => a2a_template!(args, x, 7),
        _ => None,
    }
}

impl Default for TensorAnalysis {
    fn default() -> Self {
        TensorAnalysis {
            blacklist_nodes: HashSet::<Mdl>::new(),
            newly_added: Vec::<Mdl>::new(),
            manager: None,
            name_to_shapes: HashMap::<String, ShapeLike>::new(),
            enable_stats: false,
            lemma_applied_count: HashMap::<String, usize>::new(),
        }
    }
}

impl Analysis<Mdl> for TensorAnalysis {
    type Data = ValTnsr;

    fn pre_union(
        egraph: &EGraph<Mdl, Self>,
        id1: Id,
        id2: Id,
        _justification: &Option<Justification>,
    ) {
        // Check if the two nodes are blacklisted
        let node1 = &egraph[id1].data;
        let node2 = &egraph[id2].data;
        if node1.dtype == DataKind::Tnsr
            && node2.dtype == DataKind::Tnsr
            && node1.meta.is_some()
            && node2.meta.is_some()
        {
            let meta1 = node1.meta.as_ref().unwrap();
            let meta2 = node2.meta.as_ref().unwrap();
            let shape1 = meta1.clone().sliced_shape();
            let shape2 = meta2.clone().sliced_shape();

            if shape1 != shape2 {
                let expr1 = egraph.id_to_expr(id1);
                let expr2 = egraph.id_to_expr(id2);
                let n_dim1 = meta1.n_dim;
                let n_dim2 = meta2.n_dim;
                panic!(
                    "{}",
                    &format!(
                        "Should union on same shape, \
                        got id{id1}.shape({n_dim1})={shape1:?} and id{id2}.shape({n_dim2})={shape2:?}\n\
                        expr1: {}\nexpr2: {}\n",
                        expr1.to_string(),
                        expr2.to_string()
                    )
                )
            }
        }
    }

    /// Merges two metadata when two eclasses are merged.
    fn merge(&mut self, _to: &mut Self::Data, _from: Self::Data) -> DidMerge {
        DidMerge(false, false)
    }

    // Constructs metadata for a new enode.
    fn make(egraph: &mut EGraph<Mdl, Self>, enode: &Mdl) -> Self::Data {
        assert!(egraph.analysis.manager.is_some());
        let manager = &egraph.analysis.manager.clone().unwrap();
        let analysis = &egraph.analysis;
        let x = |i: &Id| &egraph[*i].data;
        let _dims_from_name = |name: &Id| {
            let name_vec: Vec<&str> = x(name).name.split("@").collect();
            assert!(name_vec.len() == 2);
            let dims: ShapeLike = shapelike_name_to_vec(&name_vec[1].to_string(), manager.clone());
            dims
        };

        match enode {
            Mdl::Num(_n) => Self::Data::new_val(SymVal::new_val(*_n, manager.clone())),

            Mdl::Var(s) => {
                let res = s.to_string().parse::<i64>();
                if res.is_ok() {
                    // Representing i64
                    Self::Data::new_val(SymVal::new_val(res.unwrap() as i64, manager.clone()))
                } else if s.to_string().starts_with(SYMVAL_PREFIX) || s.to_string().starts_with("(")
                {
                    // Representing SymVal
                    Self::Data::new_val(SymVal::new_name(s.to_string(), manager.clone()))
                } else {
                    // Normal name
                    Self::Data::new_name(s.to_string())
                }
            }

            Mdl::Input([name]) => {
                // Check types
                assert!(x(name).dtype == DataKind::Name);

                // Get arguments
                // let mut dims = dims_from_name(name);
                // let _ndim = dims.len();
                // dims.shrink_to_fit();
                // assert!(dims.len() == dims.capacity());
                assert!(
                    analysis.name_to_shapes.contains_key(&x(name).name),
                    "name {} not found",
                    x(name).name
                );
                let dims = analysis.name_to_shapes[&x(name).name].clone();
                Self::Data::new_tensor(&dims)
            }

            Mdl::Empty([name]) => {
                assert!(x(name).dtype == DataKind::Name);
                let shape = x(name).parse_to_shapelike(manager.clone());
                Self::Data::new_tensor(&shape)
            }

            Mdl::Noop([_a, _b]) => Self::Data {
                dtype: DataKind::Tnsr,
                ..Default::default()
            },

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // aten
            ////////////////////////////////////////////////////////////////////////////////////////////////
            Mdl::Addcmul([inpt, tensor1, tenso2, value]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(tensor1).dtype == DataKind::Tnsr);
                assert!(x(tenso2).dtype == DataKind::Tnsr);
                assert!(x(value).dtype.is_scalar_or_name());
                let inpt_shape = x(inpt).get_shape();
                let tensor1_shape = x(tensor1).get_shape();
                let tensor2_shape = x(tenso2).get_shape();
                assert!(inpt_shape == tensor1_shape);
                assert!(tensor1_shape == tensor2_shape);
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Addmm([inpt, mat1, mat2, beta, alpha]) => {
                // Check types
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(mat1).dtype == DataKind::Tnsr);
                assert!(x(mat2).dtype == DataKind::Tnsr);
                assert!(x(beta).dtype.is_scalar_or_name());
                assert!(x(alpha).dtype.is_scalar_or_name());

                let inpt_shape: Vec<SymVal> = x(inpt).get_shape();
                let mat1_shape = x(mat1).get_shape();
                let mat2_shape = x(mat2).get_shape();
                let inpt_n = inpt_shape.len();
                let mat1_n = mat1_shape.len();
                let mat2_n = mat2_shape.len();
                assert!(inpt_n == 1);
                assert!(mat1_n == 2);
                assert!(mat2_n == 2);
                // out = β input + α(mat1@mat2)
                assert!(mat1_shape[1] == mat2_shape[0]);
                let shape = vec![mat1_shape[0].clone(), mat2_shape[1].clone()];
                assert!(broadcast(&inpt_shape, &shape).is_some());
                Self::Data::new_tensor(&shape)
            }

            Mdl::Arange([begin, end]) => {
                assert!(x(begin).dtype == DataKind::Scalar);
                assert!(x(end).dtype == DataKind::Scalar);
                let begin_val = x(begin).get_val();
                let end_val = x(end).get_val();
                let shape = vec![SymVal::new_val(end_val - begin_val, manager.clone())];
                Self::Data::new_tensor(&shape)
            }

            Mdl::Argsort([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let inpt_shape = x(inpt).get_shape();
                assert!(inpt_shape.len() == 1, "got {:?}", inpt_shape);
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Baddbmm([input1, input2, input3, beta, alpha]) => {
                assert!(x(input1).dtype == DataKind::Tnsr);
                assert!(x(input2).dtype == DataKind::Tnsr);
                assert!(x(input3).dtype == DataKind::Tnsr);
                assert!(x(beta).dtype.is_scalar_or_name());
                assert!(x(alpha).dtype.is_scalar_or_name());
                let t1_shape = x(input1).get_shape();
                let t2_shape = x(input2).get_shape();
                let t3_shape = x(input3).get_shape();
                let n1 = t1_shape.len();
                let n2 = t2_shape.len();
                let n3 = t3_shape.len();
                assert!(n1 == n2);
                assert!(n1 == n3);
                assert!(t1_shape[..n1 - 2] == t2_shape[..n2 - 2]);
                assert!(t1_shape[..n1 - 2] == t3_shape[..n3 - 2]);

                // (*b, n, p), (*b, n, m), (*b, m, p)
                assert!(t1_shape[n1 - 2] == t2_shape[n2 - 2]);
                assert!(t1_shape[n1 - 1] == t3_shape[n3 - 1]);
                assert!(t2_shape[n2 - 1] == t3_shape[n3 - 2]);

                let mut shape = vec![];
                shape.extend_from_slice(&t2_shape[..n2 - 2]);
                shape.push(t2_shape[n2 - 2].clone());
                shape.push(t3_shape[n3 - 1].clone());
                Self::Data::new_tensor(&shape)
            }

            Mdl::Bmm([input2, input3]) => {
                assert!(x(input2).dtype == DataKind::Tnsr);
                assert!(x(input3).dtype == DataKind::Tnsr);
                let t2_shape = x(input2).get_shape();
                let t3_shape = x(input3).get_shape();
                let n2 = t2_shape.len();
                let n3 = t3_shape.len();
                assert!(n2 == n3);
                assert!(t2_shape[..n2 - 2] == t3_shape[..n3 - 2]);

                // (*b, n, m), (*b, m, p)
                assert!(t2_shape[n2 - 1] == t3_shape[n3 - 2]);

                let mut shape = vec![];
                shape.extend_from_slice(&t2_shape[..n2 - 2]);
                shape.push(t2_shape[n2 - 2].clone());
                shape.push(t3_shape[n3 - 1].clone());
                Self::Data::new_tensor(&shape)
            }

            Mdl::BmmAlpha([input2, input3, alpha]) => {
                assert!(x(input2).dtype == DataKind::Tnsr);
                assert!(x(input3).dtype == DataKind::Tnsr);
                assert!(x(alpha).dtype.is_scalar_or_name());
                let t2_shape = x(input2).get_shape();
                let t3_shape = x(input3).get_shape();
                let n2 = t2_shape.len();
                let n3 = t3_shape.len();
                assert!(n2 == n3);
                assert!(t2_shape[..n2 - 2] == t3_shape[..n3 - 2]);

                // (*b, n, m), (*b, m, p)
                assert!(t2_shape[n2 - 1] == t3_shape[n3 - 2]);
                let mut shape = vec![];
                shape.extend_from_slice(&t2_shape[..n2 - 2]);
                shape.push(t2_shape[n2 - 2].clone());
                shape.push(t3_shape[n3 - 1].clone());
                Self::Data::new_tensor(&shape)
            }

            Mdl::BitwiseAnd([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                let a_meta = x(a).meta.as_ref().unwrap();
                let b_meta = x(b).meta.as_ref().unwrap();
                let a_shape = a_meta.clone().sliced_shape();
                let b_shape = b_meta.clone().sliced_shape();
                assert!(a_shape == b_shape);
                Self::Data::new_tensor(&a_shape)
            }

            Mdl::BitwiseNot([a]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                let a_shape = x(a).get_shape();
                Self::Data::new_tensor(&a_shape)
            }

            Mdl::BitwiseOr([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                let a_meta = x(a).meta.as_ref().unwrap();
                let b_meta = x(b).meta.as_ref().unwrap();
                let a_shape = a_meta.clone().sliced_shape();
                let b_shape = b_meta.clone().sliced_shape();
                assert!(a_shape == b_shape);
                Self::Data::new_tensor(&a_shape)
            }

            Mdl::Clamp([inpt, min]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(min).dtype == DataKind::Scalar);
                let inpt_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Concat([a, b, axis]) => {
                assert!(x(axis).dtype == DataKind::Scalar);
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                let t_a = x(a).meta.as_ref().unwrap();
                let t_b = x(b).meta.as_ref().unwrap();
                let axis_val = x(axis).get_val();
                let dim = axis_val as usize;

                let t_a_shape = t_a.clone().sliced_shape();
                let t_b_shape = t_b.clone().sliced_shape();

                assert!(
                    t_a_shape[..dim] == t_b_shape[..dim],
                    "dim={}, t_a_shape={:?}, t_b_shape={:?}",
                    dim,
                    t_a_shape,
                    t_b_shape
                );
                assert!(
                    dim == t_a_shape.len() - 1 || t_a_shape[dim + 1..] == t_b_shape[dim + 1..],
                    "dim={}, t_a_shape={:?}, t_b_shape={:?}",
                    dim,
                    t_a_shape,
                    t_b_shape
                );
                let mut shape = t_a.clone().sliced_shape();
                shape[axis_val as usize] += t_b.shape[axis_val as usize].clone().unwrap();
                Self::Data::new_tensor(&shape)
            }

            Mdl::Cos([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let inpt_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Embedding([weight, indices]) => {
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(indices).dtype == DataKind::Tnsr);
                let weight_shape = x(weight).get_shape();
                let indices_shape = x(indices).get_shape();
                let mut shape = indices_shape.clone();
                shape.push(weight_shape[1].clone());
                Self::Data::new_tensor(&shape)
            }

            Mdl::Ewadd([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                Self::Data::new_tensor(&x(a).get_shape())
            }

            Mdl::Ewmul([a, b]) => {
                let a_dtype = x(a).dtype;
                let b_dtype = x(b).dtype;
                if a_dtype != DataKind::Tnsr && b_dtype != DataKind::Tnsr {
                    panic!(
                        "ewmul should take at least one tensor, got a_dtype={:?}, b_dtype={:?}",
                        a_dtype, b_dtype
                    );
                }
                let a_shape = x(a).get_shape();
                let b_shape = x(b).get_shape();
                let shape = broadcast(&a_shape, &b_shape);
                assert!(
                    shape.is_some(),
                    "ewmul should be broadcastable, got a_shape={:?}, b_shape={:?}",
                    a_shape,
                    b_shape
                );
                Self::Data::new_tensor(&shape.unwrap())
            }

            Mdl::EqScalar([inpt, scalar]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(scalar).dtype == DataKind::Scalar);
                let inpt_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Exp([input]) => {
                assert!(x(input).dtype == DataKind::Tnsr);
                let t_shape = x(input).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::Expand([input, shape_name]) => {
                let shape = x(shape_name).parse_to_shapelike(manager.clone());
                assert!(x(input).dtype == DataKind::Tnsr);
                let t_meta = x(input).meta.as_ref().unwrap();
                let t_shape = t_meta.clone().sliced_shape();

                assert!(
                    t_shape.len() == 0
                        || t_shape.len() == 1
                            && t_shape[t_shape.len() - 1] == shape[shape.len() - 1],
                    "Incompatible shapes for expand: t_shape={:?}, shape={:?}",
                    t_shape,
                    shape
                );
                Self::Data::new_tensor(&shape)
            }

            Mdl::GEScalar([inpt, scalar]) => {
                assert!(
                    x(inpt).dtype == DataKind::Tnsr,
                    "{:?}",
                    egraph.id_to_expr(*inpt)
                );
                assert!(
                    x(scalar).dtype == DataKind::Scalar,
                    "{:?}",
                    egraph.id_to_expr(*scalar)
                );
                Self::Data::new_tensor(&x(inpt).get_shape())
            }

            Mdl::GETensor([inpt1, inpt2]) => {
                assert!(x(inpt1).dtype == DataKind::Tnsr);
                assert!(x(inpt2).dtype == DataKind::Tnsr);
                let inpt1_shape = x(inpt1).get_shape();
                let inpt2_shape = x(inpt2).get_shape();
                assert!(inpt1_shape == inpt2_shape);
                Self::Data::new_tensor(&inpt1_shape)
            }

            Mdl::Gelu([input]) => {
                assert!(x(input).dtype == DataKind::Tnsr);
                let t_shape = x(input).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::GeluBackward([grad_output, input]) => {
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(input).dtype == DataKind::Tnsr);
                let grad_shape = x(grad_output).get_shape();
                let input_shape = x(input).get_shape();
                assert!(grad_shape == input_shape);
                Self::Data::new_tensor(&grad_shape)
            }

            Mdl::IndexPut([inpt, indices, value]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(indices).dtype == DataKind::Tnsr);
                assert!(x(value).dtype == DataKind::Tnsr);
                let inpt_shape = x(inpt).get_shape();
                let indices_shape = x(indices).get_shape();
                let value_shape = x(value).get_shape();
                assert!(indices.len() == 1, "Haven't seen other case yet.");

                assert!(value_shape.len() == 0 || indices_shape[0] == value_shape[0]);
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::IndexPutAcc2([inpt, indices0, indices1, value]) => {
                // Torch equivalence: inpt[indices0, indices1] = value
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(indices0).dtype == DataKind::Tnsr);
                assert!(x(indices1).dtype == DataKind::Tnsr);
                assert!(x(value).dtype == DataKind::Tnsr);

                let inpt_shape = x(inpt).get_shape();
                let indices0_shape = x(indices0).get_shape();
                let indices1_shape = x(indices1).get_shape();
                let value_shape = x(value).get_shape();
                assert!(indices0_shape.len() == 1, "Haven't seen other case yet.");
                assert!(indices1_shape.len() == 1, "Haven't seen other case yet.");
                assert!(value_shape.len() == 0 || indices0_shape[0] == value_shape[0]);
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Pad([inpt, before0, after0, before1, after1, before2, after2]) => {
                assert!(x(before0).dtype == DataKind::Scalar);
                assert!(x(after0).dtype == DataKind::Scalar);
                assert!(x(before1).dtype == DataKind::Scalar);
                assert!(x(after1).dtype == DataKind::Scalar);
                assert!(x(before2).dtype == DataKind::Scalar);
                assert!(x(after2).dtype == DataKind::Scalar);
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_inpt = x(inpt).meta.as_ref().unwrap();
                let before0_val: i64 = x(before0).get_val();
                let after0_val: i64 = x(after0).get_val();
                let before1_val: i64 = x(before1).get_val();
                let after1_val: i64 = x(after1).get_val();
                let before2_val: i64 = x(before2).get_val();
                let after2_val: i64 = x(after2).get_val();

                let mut shape = t_inpt.clone().sliced_shape();
                shape[t_inpt.n_dim - 1] +=
                    SymVal::new_val(before0_val + after0_val, manager.clone());
                if t_inpt.n_dim >= 2 {
                    shape[t_inpt.n_dim - 2] +=
                        SymVal::new_val(before1_val + after1_val, manager.clone());
                }
                if t_inpt.n_dim >= 3 {
                    shape[t_inpt.n_dim - 3] +=
                        SymVal::new_val(before2_val + after2_val, manager.clone());
                }
                Self::Data::new_tensor(&shape)
            }

            Mdl::IndexFillScalar([inpt, indices, dim, value]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(indices).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                assert!(x(value).dtype.is_scalar_or_name());
                let inpt_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Log([input]) => {
                assert!(x(input).dtype == DataKind::Tnsr);
                let t_shape = x(input).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::LETensor([inpt1, inpt2]) => {
                assert!(x(inpt1).dtype == DataKind::Tnsr);
                assert!(x(inpt2).dtype == DataKind::Tnsr);
                let inpt1_shape = x(inpt1).get_shape();
                let inpt2_shape = x(inpt2).get_shape();
                assert!(inpt1_shape == inpt2_shape);
                Self::Data::new_tensor(&inpt1_shape)
            }

            Mdl::LTScalar([inpt, scalar]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(scalar).dtype == DataKind::Scalar);
                Self::Data::new_tensor(&x(inpt).get_shape())
            }

            Mdl::MaskFillScalar([inpt, mask, scalar]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(mask).dtype == DataKind::Tnsr);
                assert!(x(scalar).dtype.is_scalar_or_name());
                let inpt_shape = x(inpt).get_shape();
                let mask_shape = x(mask).get_shape();
                assert!(broadcast(&inpt_shape, &mask_shape).is_some());
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Matadd([a, b]) => {
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype.is_scalar_or_name());
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype.is_scalar_or_name());
                let a_shape = x(a).get_shape();
                let b_shape = x(b).get_shape();
                let broadcast_shape = broadcast(&a_shape, &b_shape);
                assert!(
                    broadcast_shape.is_some(),
                    "matadd should be broadcastable, got a_shape={:?}, b_shape={:?}: {:?} and {:?}",
                    a_shape,
                    b_shape,
                    egraph.id_to_expr(*a).pretty(usize::MAX),
                    egraph.id_to_expr(*b).pretty(usize::MAX)
                );
                Self::Data::new_tensor(&broadcast_shape.unwrap())
            }

            Mdl::Matdiv([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype.is_scalar_or_name());
                let a_shape = x(a).get_shape();
                let b_shape = x(b).get_shape();
                let shape = broadcast(&a_shape, &b_shape);
                assert!(
                    shape.is_some(),
                    "matdiv should be broadcastable, got a_shape={:?}, b_shape={:?}",
                    a_shape,
                    b_shape
                );
                Self::Data::new_tensor(&shape.unwrap())
            }

            Mdl::Matmul([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr);
                let t_a = x(a).meta.as_ref().unwrap();
                let t_b = x(b).meta.as_ref().unwrap();
                assert!(t_a.n_dim == 2);
                assert!(t_b.n_dim == 2);
                Self::Data::new_tensor(&vec![
                    t_a.shape[0].clone().unwrap(),
                    t_b.shape[1].clone().unwrap(),
                ])
            }

            Mdl::Matsub([a, b]) => {
                assert!(x(a).dtype == DataKind::Tnsr);
                assert!(x(b).dtype == DataKind::Tnsr || x(b).dtype == DataKind::Scalar);
                let res_shape = if x(b).dtype == DataKind::Tnsr {
                    let a_meta = x(a).meta.as_ref().unwrap();
                    let b_meta = x(b).meta.as_ref().unwrap();
                    let a_shape = a_meta.clone().sliced_shape();
                    let b_shape = b_meta.clone().sliced_shape();
                    let broadcast_shape = broadcast(&a_shape, &b_shape);
                    assert!(
                        broadcast_shape.is_some(),
                        "matadd should be broadcastable, got a_shape={:?}, b_shape={:?}, a={:?}, b={:?}",
                        a_shape,
                        b_shape,
                        egraph.id_to_expr(*a).pretty(usize::MAX),
                        egraph.id_to_expr(*b).pretty(usize::MAX),
                    );
                    broadcast_shape.unwrap()
                } else {
                    x(a).get_shape()
                };
                Self::Data::new_tensor(&res_shape)
            }

            Mdl::Maximum([inpt1, inpt2]) => {
                assert!(x(inpt1).dtype == DataKind::Tnsr);
                assert!(x(inpt2).dtype == DataKind::Tnsr);
                let t1_shape = x(inpt1).get_shape();
                let t2_shape = x(inpt2).get_shape();
                assert!(t1_shape == t2_shape);
                Self::Data::new_tensor(&t1_shape)
            }

            Mdl::MaxDim0([inpt, dim]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let t_shape = x(inpt).get_shape();
                let dim = x(dim).get_val() as usize;
                assert!(dim < t_shape.len());
                let mut new_shape = t_shape[..dim].to_vec();
                new_shape.extend(t_shape[dim + 1..].iter().cloned());
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::MaxDim1([inpt, dim]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let t_shape = x(inpt).get_shape();
                let dim = x(dim).get_val() as usize;
                assert!(dim < t_shape.len());
                let mut new_shape = t_shape[..dim].to_vec();
                new_shape.extend(t_shape[dim + 1..].iter().cloned());
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::MeanDefault([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                Self::Data::new_tensor(&vec![])
            }

            Mdl::PowTensorScalar([inpt, exp]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(exp).dtype == DataKind::Scalar);
                let inpt_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Repeat([input, repeats]) => {
                assert!(x(input).dtype == DataKind::Tnsr);
                assert!(x(repeats).dtype == DataKind::Name);
                let t_shape = x(input).get_shape();
                let repeats = shapelike_name_to_vec(&x(repeats).name, manager.clone());
                let new_shape = zip(t_shape.iter(), repeats.iter())
                    .map(|(a, b)| a.clone() * b.clone())
                    .collect::<ShapeLike>();
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::Reshape([inpt, shape_name]) => {
                let shape = x(shape_name).parse_to_shapelike(manager.clone());
                assert!(x(inpt).dtype == DataKind::Tnsr);

                let inpt_shape = x(inpt).get_shape();
                println!("shape={:?}, inpt_shape={:?}", shape, inpt_shape);
                let inpt_num_elem = shape
                    .iter()
                    .cloned()
                    .fold(SymVal::new_val(1, manager.clone()), |a, b| a * b);
                let shape_num_elem = inpt_shape
                    .iter()
                    .cloned()
                    .fold(SymVal::new_val(1, manager.clone()), |a, b| a * b);
                assert!(
                    inpt_num_elem == shape_num_elem,
                    "inpt_shape={:?}, shape={:?}, inpt={:?}",
                    inpt_shape,
                    shape,
                    egraph.id_to_expr(*inpt).pretty(usize::MAX)
                );
                Self::Data::new_tensor(&shape)
            }

            Mdl::ScatterSrc([inpt, index, src, dim]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(index).dtype == DataKind::Tnsr);
                assert!(x(src).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let inpt_shape = x(inpt).get_shape();
                let index_shape = x(index).get_shape();
                let src_shape = x(src).get_shape();
                let dim = x(dim).get_val() as usize;

                assert!(
                    index_shape == src_shape,
                    "index_shape={:?}, src_shape={:?}, inpt={:?}, index={:?}, src={:?}",
                    index_shape,
                    src_shape,
                    egraph.id_to_expr(*inpt).pretty(usize::MAX),
                    egraph.id_to_expr(*index).pretty(usize::MAX),
                    egraph.id_to_expr(*src).pretty(usize::MAX)
                );
                assert!(dim < inpt_shape.len());
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::MaskedSelect([inpt1, inpt2, mask1, mask2]) => {
                assert!(x(inpt1).dtype == DataKind::Tnsr);
                assert!(x(inpt2).dtype == DataKind::Tnsr);
                assert!(x(mask1).dtype == DataKind::Tnsr);
                assert!(x(mask2).dtype == DataKind::Tnsr);
                let t1_shape = x(inpt1).get_shape();
                let t2_shape = x(inpt2).get_shape();
                let mask1_shape = x(mask1).get_shape();
                let mask2_shape = x(mask2).get_shape();
                assert!(
                    t1_shape == t2_shape,
                    "t1_shape={:?}, t2_shape={:?}",
                    t1_shape,
                    t2_shape
                );
                assert!(mask1_shape == t1_shape[..mask1_shape.len()]);
                assert!(mask2_shape == t2_shape[..mask2_shape.len()]);
                Self::Data::new_tensor(&t1_shape)
            }

            Mdl::Sigmoid([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::SigmoidBackward([grad_output, input]) => {
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(input).dtype == DataKind::Tnsr);
                let grad_shape = x(grad_output).get_shape();
                let input_shape = x(input).get_shape();
                assert!(grad_shape == input_shape);
                Self::Data::new_tensor(&grad_shape)
            }

            Mdl::Silu([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::Sin([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let inpt_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Sinkhorn([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::Slice([inpt, axis, begin, end, step]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(axis).dtype == DataKind::Scalar);
                let axis_val = x(axis).get_val() as usize;
                let begin_val = x(begin).get_sym_val();
                let end_val = x(end).get_sym_val();
                let step_val = x(step).get_sym_val();
                let t_inpt = x(inpt).meta.as_ref().unwrap();

                let mut shape = t_inpt.clone().sliced_shape();
                let dim = axis_val as usize;
                assert!(
                    end_val <= shape[dim],
                    "begin={}, end={}, dim={}, but shape={:?}, t_id={}",
                    begin_val,
                    end_val,
                    axis_val,
                    shape,
                    inpt
                );
                shape[axis_val as usize] = (end_val - begin_val) / step_val;
                Self::Data::new_tensor(&shape)
            }

            Mdl::SliceBackward([grad_output, input_shape, dim, start, end, step]) => {
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(input_shape).dtype == DataKind::Name);
                assert!(x(dim).dtype == DataKind::Scalar);
                assert!(x(start).dtype == DataKind::Scalar);
                assert!(x(end).dtype == DataKind::Scalar);
                assert!(x(step).dtype == DataKind::Scalar);

                let input_shape_vec = shapelike_name_to_vec(&x(input_shape).name, manager.clone());
                let dim = x(dim).get_val() as usize;
                let start = x(start).get_sym_val();
                let end = x(end).get_sym_val();

                assert!(dim < input_shape_vec.len());
                assert!(start < end);
                assert!(end <= input_shape_vec[dim]);
                Self::Data::new_tensor(&input_shape_vec)
            }

            Mdl::SliceScatter([inpt, src, dim, start, end]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(src).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                assert!(x(start).dtype == DataKind::Scalar);
                assert!(x(end).dtype == DataKind::Scalar);
                let inpt_shape = x(inpt).meta.clone().unwrap().sliced_shape();
                let src_shape = x(src).meta.clone().unwrap().sliced_shape();
                let dim = x(dim).get_val() as usize;
                let start = x(start).get_sym_val();
                let end = x(end).get_sym_val();

                assert!(SymVal::zero(manager.clone()) <= start, "start={:?}", start);
                assert!(start <= end, "start={}, end={}", start, end);
                assert!(end <= inpt_shape[dim], "end={}, inpt_shape[dim]={}", end, inpt_shape[dim]);
                assert!(
                    end.clone() - start.clone() == src_shape[dim],
                    "shape={:?}, src_shape={:?}, start={:?}, end={:?}",
                    inpt_shape,
                    src_shape,
                    start,
                    end
                );
                Self::Data::new_tensor(&inpt_shape)
            }

            Mdl::Softmax([inpt, dim]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::SoftmaxBackwardData([grad_output, output, dim]) => {
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(output).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let t_shape = x(&output).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::Sort0([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                Self::Data::new_tensor(&x(inpt).get_shape())
            }

            Mdl::Sort1([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                Self::Data::new_tensor(&x(inpt).get_shape())
            }

            Mdl::Squeeze([inpt, dim]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let t_shape = x(inpt).get_shape();
                let dim = x(dim).get_val() as usize;
                assert!(t_shape[dim] == 1, "t_shape={:?}, dim={}", t_shape, dim);
                let mut new_shape = t_shape.clone();
                new_shape.remove(dim);
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::SumDimIntList([inpt, dims]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                let dims = shapelike_name_to_val_vec(&x(dims).name);

                let mut new_shape = vec![];
                for idx in 0..t_shape.len() {
                    if !dims.contains(&idx) {
                        new_shape.push(t_shape[idx].clone());
                    }
                }
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::SumDimIntListKeep([inpt, dims]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                let dims = shapelike_name_to_val_vec(&x(dims).name);

                let mut new_shape = vec![];
                for idx in 0..t_shape.len() {
                    if !dims.contains(&idx) {
                        new_shape.push(t_shape[idx].clone());
                    } else {
                        new_shape.push(SymVal::new_val(1, manager.clone()));
                    }
                }
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::SumDefault([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                Self::Data::new_tensor(&vec![])
            }

            Mdl::Topk0([inpt, k, dim, largest, sorted]) => {
                // Check types
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(k).dtype == DataKind::Scalar);
                assert!(x(dim).dtype == DataKind::Scalar);
                assert!(x(largest).dtype == DataKind::Name);
                assert!(x(sorted).dtype == DataKind::Name);

                // Get arguments
                let t_shape = x(inpt).get_shape();
                let k_val = x(k).get_val();
                let dim_val = x(dim).get_val() as usize;

                let shape = {
                    let mut tmp = t_shape.clone();
                    tmp[dim_val] = SymVal::new_val(k_val, manager.clone());
                    tmp
                };
                Self::Data::new_tensor(&shape)
            }

            Mdl::Topk1([inpt, k, dim, largest, sorted]) => {
                // Check types
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(k).dtype == DataKind::Scalar);
                assert!(x(dim).dtype == DataKind::Scalar);
                assert!(x(largest).dtype == DataKind::Name);
                assert!(x(sorted).dtype == DataKind::Name);

                // Get arguments
                let t_shape = x(inpt).get_shape();
                let k_val = x(k).get_val();
                let dim_val = x(dim).get_val() as usize;

                let shape = {
                    let mut tmp = t_shape.clone();
                    tmp[dim_val] = SymVal::new_val(k_val, manager.clone());
                    tmp
                };
                Self::Data::new_tensor(&shape)
            }

            Mdl::Transpose([inpt, perm_name]) => {
                // Check types
                assert!(x(perm_name).dtype == DataKind::Name);
                assert!(x(inpt).dtype == DataKind::Tnsr);

                // Get arguments
                let perms: ShapeLike = shapelike_name_to_vec(&x(perm_name).name, manager.clone());
                let t_inpt = x(inpt).meta.as_ref().unwrap();

                let mut new_shape = t_inpt.clone().sliced_shape();
                for (idx, perm) in perms.into_iter().enumerate() {
                    assert!(matches!(&perm.symval_id, SymValId::Val { .. }));
                    let val = perm.val();
                    new_shape[idx] = t_inpt.shape[val as usize].clone().unwrap();
                }
                Self::Data::new_tensor(&new_shape)
            }

            Mdl::Unsqueeze([inpt, dim]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(dim).dtype == DataKind::Scalar);
                let t_inpt = x(inpt).meta.clone().unwrap();
                let dim_val: i64 = x(dim).get_val();

                let mut shape = t_inpt.sliced_shape();
                shape.insert(dim_val as usize, SymVal::new_val(1, manager.clone()));
                Self::Data::new_tensor(&shape)
            }

            // aten
            Mdl::AddScalar([inpt, scalar]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(scalar).dtype == DataKind::Scalar);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::Fill([shape, value]) => {
                assert!(x(shape).dtype == DataKind::Name);
                assert!(x(value).dtype == DataKind::Scalar);
                let shape = x(shape).parse_to_shapelike(manager.clone());
                Self::Data::new_tensor(&shape)
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // Dist
            ////////////////////////////////////////////////////////////////////////////////////////////////
            Mdl::ReduceAdd([inpt1, inpt2]) => {
                assert!(x(inpt1).dtype == DataKind::Tnsr);
                assert!(x(inpt2).dtype == DataKind::Tnsr);
                let t1_shape = x(inpt1).get_shape();
                let t2_shape = x(inpt2).get_shape();
                assert!(t1_shape == t2_shape);
                Self::Data::new_tensor(&t1_shape)
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // Megatron-LM
            ////////////////////////////////////////////////////////////////////////////////////////////////
            Mdl::VocabParallelCrossEntropyBackward(
                [grad_output, softmax, target_mask, masked_target_1d, label_smoothing, vocab_size],
            ) => {
                // Check types
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(softmax).dtype == DataKind::Tnsr);
                assert!(x(target_mask).dtype == DataKind::Tnsr);
                assert!(x(masked_target_1d).dtype == DataKind::Tnsr);
                assert!(x(label_smoothing).dtype == DataKind::Name);
                assert!(x(vocab_size).dtype == DataKind::Scalar);

                let t_softmax = x(softmax).get_shape();
                Self::Data::new_tensor(&t_softmax)
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // apex
            ////////////////////////////////////////////////////////////////////////////////////////////////
            Mdl::FusedLayernormAffine0([inpt, weight, bias, last_dim_size, eps]) => {
                // Check types
                assert!(x(inpt).dtype == DataKind::Tnsr);
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(bias).dtype == DataKind::Tnsr);
                let last_dim_size = x(last_dim_size).parse_to_shapelike(manager.clone());
                assert!(
                    last_dim_size.len() == 1,
                    "got last_dim_size={:?}",
                    last_dim_size
                );
                assert!(x(eps).dtype == DataKind::Name);
                let t_shape = x(inpt).get_shape();
                assert!(t_shape[t_shape.len() - 1] == last_dim_size[0]);
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::FusedLayernormAffineBwd1(
                [grad_output, mean, invvar, input_or_output, weight, bias, eps],
            ) => {
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(mean).dtype == DataKind::Tnsr);
                assert!(x(invvar).dtype == DataKind::Tnsr);
                assert!(x(input_or_output).dtype == DataKind::Tnsr);
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(bias).dtype == DataKind::Tnsr);
                assert!(x(eps).dtype == DataKind::Name);
                let grad_output_shape = x(grad_output).get_shape();
                let mean_shape = x(mean).get_shape();
                let invvar_shape = x(invvar).get_shape();
                let input_or_output_shape = x(input_or_output).get_shape();
                let weight_shape = x(weight).get_shape();
                let bias_shape = x(bias).get_shape();
                assert!(grad_output_shape == input_or_output_shape);
                let non_norm_numel = grad_output_shape[..grad_output_shape.len() - 1]
                    .iter()
                    .cloned()
                    .fold(SymVal::new_val(1, manager.clone()), |a, b| a * b);
                assert!(mean_shape.len() == 1 && mean_shape[0] == non_norm_numel);
                assert!(invvar_shape.len() == 1 && invvar_shape[0] == non_norm_numel);
                assert!(
                    weight_shape.len() == 1
                        && weight_shape[0] == grad_output_shape[grad_output_shape.len() - 1]
                );
                assert!(
                    bias_shape.len() == 1
                        && bias_shape[0] == grad_output_shape[grad_output_shape.len() - 1]
                );
                Self::Data::new_tensor(&weight_shape)
            }

            Mdl::RmsBackwardAffine0([grad_output, weight, invvar, input_or_output, eps]) => {
                // Check types
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(invvar).dtype == DataKind::Tnsr);
                assert!(x(input_or_output).dtype == DataKind::Tnsr);
                assert!(x(eps).dtype == DataKind::Name);

                let grad_output_shape = x(grad_output).get_shape();
                Self::Data::new_tensor(&grad_output_shape)
            }

            Mdl::RmsBackwardAffine1([grad_output, weight, invvar, input_or_output, eps]) => {
                // Check types
                assert!(x(grad_output).dtype == DataKind::Tnsr);
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(invvar).dtype == DataKind::Tnsr);
                assert!(x(input_or_output).dtype == DataKind::Tnsr);
                assert!(x(eps).dtype == DataKind::Name);

                let weight_shape = x(weight).get_shape();
                Self::Data::new_tensor(&weight_shape)
            }

            Mdl::RmsForwardAffine0([input, weight, eps]) => {
                // Check types
                assert!(x(input).dtype == DataKind::Tnsr);
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(eps).dtype.is_scalar_or_name());

                let input_shape = x(input).get_shape();
                let weight_shape = x(weight).get_shape();
                assert!(input_shape.len() >= 1, "input_shape={:?}", input_shape);
                assert!(weight_shape.len() == 1, "weight_shape={:?}", weight_shape);
                assert!(
                    input_shape[input_shape.len() - 1].clone() == weight_shape[0],
                    "input_shape={:?}, weight_shape={:?}",
                    input_shape,
                    weight_shape
                );
                Self::Data::new_tensor(&input_shape)
            }

            Mdl::RmsForwardAffine1([input, weight, eps]) => {
                assert!(x(input).dtype == DataKind::Tnsr);
                assert!(x(weight).dtype == DataKind::Tnsr);
                assert!(x(eps).dtype.is_scalar_or_name());

                let input_shape = x(input).get_shape();
                let weight_shape = x(weight).get_shape();
                assert!(weight_shape.len() == 1, "weight_shape={:?}", weight_shape);
                assert!(
                    input_shape[input_shape.len() - 1].clone() == weight_shape[0],
                    "input_shape={:?}, weight_shape={:?}",
                    input_shape,
                    weight_shape
                );
                let shape = {
                    let idiff = input_shape.len() - weight_shape.len();
                    let mut s = SymVal::new_val(1, manager.clone());
                    for i in 0..idiff {
                        s = s * input_shape[i].clone();
                    }
                    vec![s]
                };
                Self::Data::new_tensor(&shape)
            }

            ////////////////////////////////////////////////////////////////////////////////////////////////
            // Special
            // SymInt
            ////////////////////////////////////////////////////////////////////////////////////////////////
            Mdl::SymAdd([a, b]) => {
                // Check types
                assert!(x(a).dtype.is_scalar_or_name());
                assert!(x(b).dtype.is_scalar_or_name());
                let a = x(a).get_sym_val();
                let b = x(b).get_sym_val();
                let res = a + b;
                Self::Data::new_val(res)
            }

            Mdl::SymSub([a, b]) => {
                // Check types
                assert!(x(a).dtype.is_scalar_or_name());
                assert!(x(b).dtype.is_scalar_or_name());
                let a = x(a).get_sym_val();
                let b = x(b).get_sym_val();
                let res = a - b;
                Self::Data::new_val(res)
            }

            Mdl::SymMul([a, b]) => {
                assert!(x(a).dtype.is_scalar_or_name());
                assert!(x(b).dtype.is_scalar_or_name());
                let a = x(a).get_sym_val();
                let b = x(b).get_sym_val();
                let res = a * b;
                Self::Data::new_val(res)
            }

            // Predicates
            Mdl::AllOnes([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            Mdl::AllZeros([inpt]) => {
                assert!(x(inpt).dtype == DataKind::Tnsr);
                let t_shape = x(inpt).get_shape();
                Self::Data::new_tensor(&t_shape)
            }

            // Fused
            Mdl::Other(name, args) => {
                if let Some(res) = variable_op_make(egraph, name, args) {
                    res
                } else if let Some(res) = special_model::make(egraph, name, args) {
                    res
                } else if let Some(res) = hlo_model::make(egraph, name, args) {
                    res
                } else if let Some(res) = vllm_model::make(egraph, name, args) {
                    res
                } else {
                    panic!("Unknown op {}: {:?}", name, enode);
                }
            }

            other => {
                println!("{:?}", other);
                todo!()
            }
        }
    }

    // Not needed to modify anything
    fn modify(_egraph: &mut EGraph<Mdl, Self>, _id: Id) {}
}
