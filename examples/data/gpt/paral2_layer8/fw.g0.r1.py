# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import device

b8: type
i32: type
i64: type
bf16: type
f32: type
f64: type
fx_pytree: type
Sym: type


# Graph[rank=1](fw, gid=0)
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[49152, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[1152, 768]", primals_5: "f32[1152]", primals_6: "f32[768, 384]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[1536, 768]", primals_11: "f32[1536]", primals_12: "f32[768, 1536]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[1152, 768]", primals_17: "f32[1152]", primals_18: "f32[768, 384]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[1536, 768]", primals_23: "f32[1536]", primals_24: "f32[768, 1536]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[1152, 768]", primals_29: "f32[1152]", primals_30: "f32[768, 384]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[1536, 768]", primals_35: "f32[1536]", primals_36: "f32[768, 1536]", primals_37: "f32[768]", primals_38: "f32[768]", primals_39: "f32[768]", primals_40: "f32[1152, 768]", primals_41: "f32[1152]", primals_42: "f32[768, 384]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[1536, 768]", primals_47: "f32[1536]", primals_48: "f32[768, 1536]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[1152, 768]", primals_53: "f32[1152]", primals_54: "f32[768, 384]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[768]", primals_58: "f32[1536, 768]", primals_59: "f32[1536]", primals_60: "f32[768, 1536]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[1152, 768]", primals_65: "f32[1152]", primals_66: "f32[768, 384]", primals_67: "f32[768]", primals_68: "f32[768]", primals_69: "f32[768]", primals_70: "f32[1536, 768]", primals_71: "f32[1536]", primals_72: "f32[768, 1536]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[1152, 768]", primals_77: "f32[1152]", primals_78: "f32[768, 384]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[1536, 768]", primals_83: "f32[1536]", primals_84: "f32[768, 1536]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[768]", primals_88: "f32[1152, 768]", primals_89: "f32[1152]", primals_90: "f32[768, 384]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[1536, 768]", primals_95: "f32[1536]", primals_96: "f32[768, 1536]", primals_97: "f32[768]", primals_98: "f32[768]", primals_99: "f32[768]", primals_100: "f32[49152, 768]", primals_101: "f32[1536, 768]", primals_102: "i64[8, 1536]", primals_103: "b8[8, 1, 1536, 1536]", primals_104: "i64[8, 1536]", primals_105: "i64[8, 1536]", primals_106: "f32[8, 1536]"):
        # File: /opt/tiger/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:456 in forward_backward_no_pipelining, code: total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        zeros: "i32[]" = torch.ops.aten.zeros.default([], dtype = torch.int32, device = device(type='cuda'), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:106 in forward_step_func, code: tokens = data['tokens'].to(device)
        _to_copy: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_102, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=1));  primals_102 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:107 in forward_step_func, code: attention_mask = data['attention_mask'].to(device)
        _to_copy_1: "b8[8, 1, 1536, 1536]" = torch.ops.aten._to_copy.default(primals_103, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=1));  primals_103 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:108 in forward_step_func, code: position_ids = data['position_ids'].to(device)
        _to_copy_2: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_104, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=1));  primals_104 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:109 in forward_step_func, code: labels = data['labels'].to(device)
        _to_copy_3: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_105, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=1));  primals_105 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:110 in forward_step_func, code: loss_mask = data['loss_mask'].to(device)
        _to_copy_4: "f32[8, 1536]" = torch.ops.aten._to_copy.default(primals_106, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1));  primals_106 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:252 in forward, code: input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        lt: "b8[8, 1536]" = torch.ops.aten.lt.Scalar(_to_copy, 49152)
        ge: "b8[8, 1536]" = torch.ops.aten.ge.Scalar(_to_copy, 98304)
        bitwise_or: "b8[8, 1536]" = torch.ops.aten.bitwise_or.Tensor(lt, ge);  lt = ge = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:254 in forward, code: masked_input = input_.clone() - self.vocab_start_index
        clone: "i64[8, 1536]" = torch.ops.aten.clone.default(_to_copy);  _to_copy = None
        sub: "i64[8, 1536]" = torch.ops.aten.sub.Tensor(clone, 49152);  clone = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:255 in forward, code: masked_input[input_mask] = 0
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        index_put: "i64[8, 1536]" = torch.ops.aten.index_put.default(sub, [bitwise_or], lift_fresh_copy);  sub = lift_fresh_copy = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:260 in forward, code: output_parallel = self.weight[masked_input]
        index: "f32[8, 1536, 768]" = torch.ops.aten.index.Tensor(primals_1, [index_put]);  primals_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:266 in forward, code: output_parallel[input_mask, :] = 0.0
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        slice_1: "f32[8, 1536, 768]" = torch.ops.aten.slice.Tensor(index, 1, 0, 9223372036854775807)
        index_put_1: "f32[8, 1536, 768]" = torch.ops.aten.index_put.default(slice_1, [bitwise_or], lift_fresh_copy_1);  slice_1 = lift_fresh_copy_1 = None
        slice_scatter: "f32[8, 1536, 768]" = torch.ops.aten.slice_scatter.default(index, index_put_1, 1, 0, 9223372036854775807);  index = index_put_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:484 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        all_reduce: "f32[8, 1536, 768]" = torch.ops._c10d_functional.all_reduce.default(slice_scatter, 'sum', '12')
        wait_tensor: "f32[8, 1536, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        copy: "f32[8, 1536, 768]" = torch.ops.aten.copy.default(slice_scatter, wait_tensor);  slice_scatter = wait_tensor = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding: "f32[8, 1536, 768]" = torch.ops.aten.embedding.default(primals_101, _to_copy_2);  primals_101 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:112 in forward, code: embeddings = word_embeddings + position_embeddings
        view_1: "f32[8, 1536, 768]" = torch.ops.aten.view.default(copy, [8, 1536, 768]);  copy = None
        add: "f32[8, 1536, 768]" = torch.ops.aten.add.Tensor(view_1, embedding);  view_1 = embedding = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:118 in forward, code: embeddings = embeddings.transpose(0, 1).contiguous()
        transpose: "f32[1536, 8, 768]" = torch.ops.aten.transpose.int(add, 0, 1);  add = None
        clone_1: "f32[1536, 8, 768]" = torch.ops.aten.clone.default(transpose, memory_format = torch.contiguous_format);  transpose = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:499 in scatter_to_sequence_parallel_region, code: return _ScatterToSequenceParallelRegion.apply(input_)
        slice_4: "f32[768, 8, 768]" = torch.ops.aten.slice.Tensor(clone_1, 0, 768, 1536);  clone_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:140 in forward, code: embeddings = embeddings.clone()
        clone_2: "f32[768, 8, 768]" = torch.ops.aten.clone.default(slice_4);  slice_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:142 in forward, code: embeddings = self.embedding_dropout(embeddings)
        native_dropout = torch.ops.aten.native_dropout.default(clone_2, 0.1, True);  clone_2 = None
        getitem: "f32[768, 8, 768]" = native_dropout[0]
        getitem_1: "b8[768, 8, 768]" = native_dropout[1];  native_dropout = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd = torch.ops.apex.fused_layer_norm_affine_fwd.default(getitem, primals_2, primals_3, [768], 1e-05)
        getitem_2: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd[0]
        getitem_3: "f32[6144]" = fused_layer_norm_affine_fwd[1]
        getitem_4: "f32[6144]" = fused_layer_norm_affine_fwd[2];  fused_layer_norm_affine_fwd = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_2, 2, '12')
        wait_tensor_1: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        t: "f32[768, 1152]" = torch.ops.aten.t.default(primals_4);  primals_4 = None
        view_2: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_1, [12288, 768]);  wait_tensor_1 = None
        mm: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_2, t);  view_2 = t = None
        view_3: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm, [1536, 8, 1152]);  mm = None
        add_1: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_3, primals_5);  view_3 = primals_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_4: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_1, [1536, 8, 12, 96]);  add_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_4, [32, 32, 32], 3);  view_4 = None
        getitem_5: "f32[1536, 8, 12, 32]" = split_with_sizes[0]
        getitem_6: "f32[1536, 8, 12, 32]" = split_with_sizes[1]
        getitem_7: "f32[1536, 8, 12, 32]" = split_with_sizes[2];  split_with_sizes = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_5: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_5, [1536, 8, 12, 32]);  getitem_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_6: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_5, [1536, 96, 32]);  view_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_7: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_6, [1536, 96, -1]);  getitem_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_1: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_6, 0, 1);  view_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_2: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_7, 0, 1);  view_7 = None
        transpose_3: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_2, 1, 2);  transpose_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty, transpose_1, transpose_3, beta = 0.0, alpha = 0.17677669529663687);  empty = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_8: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm, [8, 12, 1536, 1536]);  baddbmm = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_8, _to_copy_1, -10000.0);  view_8 = None
        view_9: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill, [96, 1536, 1536]);  masked_fill = None
        view_10: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_9, [8, 12, 1536, 1536]);  view_9 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_10, -1, False);  view_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_1 = torch.ops.aten.native_dropout.default(_softmax, 0.1, True)
        getitem_8: "f32[8, 12, 1536, 1536]" = native_dropout_1[0]
        getitem_9: "b8[8, 12, 1536, 1536]" = native_dropout_1[1];  native_dropout_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_12: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_7, [1536, 96, -1]);  getitem_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_13: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_8, [96, 1536, -1]);  getitem_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_4: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_12, 0, 1);  view_12 = None
        bmm: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_13, transpose_4)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_14: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm, [8, 12, 1536, 32]);  bmm = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_14, [2, 0, 1, 3]);  view_14 = None
        clone_3: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_15: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_3, [1536, 8, 384]);  clone_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_1: "f32[384, 768]" = torch.ops.aten.t.default(primals_6)
        view_16: "f32[12288, 384]" = torch.ops.aten.view.default(view_15, [12288, 384])
        mm_1: "f32[12288, 768]" = torch.ops.aten.mm.default(view_16, t_1);  view_16 = t_1 = None
        view_17: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_1, [1536, 8, 768]);  mm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_1: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_17, 'sum', 2, '12');  view_17 = None
        wait_tensor_2: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        copy_1: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_1, wait_tensor_2);  empty_1 = wait_tensor_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_2: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_1, primals_7);  copy_1 = primals_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_2 = torch.ops.aten.native_dropout.default(add_2, 0.1, True);  add_2 = None
        getitem_10: "f32[768, 8, 768]" = native_dropout_2[0]
        getitem_11: "b8[768, 8, 768]" = native_dropout_2[1];  native_dropout_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_3: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(getitem, getitem_10);  getitem_10 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_1 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_3, primals_8, primals_9, [768], 1e-05)
        getitem_12: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_1[0]
        getitem_13: "f32[6144]" = fused_layer_norm_affine_fwd_1[1]
        getitem_14: "f32[6144]" = fused_layer_norm_affine_fwd_1[2];  fused_layer_norm_affine_fwd_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_1: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_12, 2, '12')
        wait_tensor_3: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        t_2: "f32[768, 1536]" = torch.ops.aten.t.default(primals_10);  primals_10 = None
        view_18: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_3, [12288, 768]);  wait_tensor_3 = None
        mm_2: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_18, t_2);  view_18 = t_2 = None
        view_19: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_2, [1536, 8, 1536]);  mm_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_4: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_19, primals_11);  view_19 = primals_11 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_4)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_3: "f32[1536, 768]" = torch.ops.aten.t.default(primals_12)
        view_20: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu, [12288, 1536])
        mm_3: "f32[12288, 768]" = torch.ops.aten.mm.default(view_20, t_3);  view_20 = t_3 = None
        view_21: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_3, [1536, 8, 768]);  mm_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_2: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_1: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_21, 'sum', 2, '12');  view_21 = None
        wait_tensor_4: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        copy_2: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_2, wait_tensor_4);  empty_2 = wait_tensor_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_5: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_2, primals_13);  copy_2 = primals_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_3 = torch.ops.aten.native_dropout.default(add_5, 0.1, True);  add_5 = None
        getitem_15: "f32[768, 8, 768]" = native_dropout_3[0]
        getitem_16: "b8[768, 8, 768]" = native_dropout_3[1];  native_dropout_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_6: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_15);  getitem_15 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_2 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_6, primals_14, primals_15, [768], 1e-05)
        getitem_17: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_2[0]
        getitem_18: "f32[6144]" = fused_layer_norm_affine_fwd_2[1]
        getitem_19: "f32[6144]" = fused_layer_norm_affine_fwd_2[2];  fused_layer_norm_affine_fwd_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_2: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_17, 2, '12')
        wait_tensor_5: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        t_4: "f32[768, 1152]" = torch.ops.aten.t.default(primals_16);  primals_16 = None
        view_22: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_5, [12288, 768]);  wait_tensor_5 = None
        mm_4: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_22, t_4);  view_22 = t_4 = None
        view_23: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_4, [1536, 8, 1152]);  mm_4 = None
        add_7: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_23, primals_17);  view_23 = primals_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_24: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_7, [1536, 8, 12, 96]);  add_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_24, [32, 32, 32], 3);  view_24 = None
        getitem_20: "f32[1536, 8, 12, 32]" = split_with_sizes_1[0]
        getitem_21: "f32[1536, 8, 12, 32]" = split_with_sizes_1[1]
        getitem_22: "f32[1536, 8, 12, 32]" = split_with_sizes_1[2];  split_with_sizes_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_25: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_20, [1536, 8, 12, 32]);  getitem_20 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_26: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_25, [1536, 96, 32]);  view_25 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_27: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_21, [1536, 96, -1]);  getitem_21 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_3: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_5: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_26, 0, 1);  view_26 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_6: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_27, 0, 1);  view_27 = None
        transpose_7: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_6, 1, 2);  transpose_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_1: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_3, transpose_5, transpose_7, beta = 0.0, alpha = 0.17677669529663687);  empty_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_28: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_1, [8, 12, 1536, 1536]);  baddbmm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_1: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_28, _to_copy_1, -10000.0);  view_28 = None
        view_29: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_1, [96, 1536, 1536]);  masked_fill_1 = None
        view_30: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_29, [8, 12, 1536, 1536]);  view_29 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_1: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_30, -1, False);  view_30 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_4 = torch.ops.aten.native_dropout.default(_softmax_1, 0.1, True)
        getitem_23: "f32[8, 12, 1536, 1536]" = native_dropout_4[0]
        getitem_24: "b8[8, 12, 1536, 1536]" = native_dropout_4[1];  native_dropout_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_32: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_22, [1536, 96, -1]);  getitem_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_33: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_23, [96, 1536, -1]);  getitem_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_8: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_32, 0, 1);  view_32 = None
        bmm_1: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_33, transpose_8)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_34: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_1, [8, 12, 1536, 32]);  bmm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_1: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_34, [2, 0, 1, 3]);  view_34 = None
        clone_4: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_35: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_4, [1536, 8, 384]);  clone_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_5: "f32[384, 768]" = torch.ops.aten.t.default(primals_18)
        view_36: "f32[12288, 384]" = torch.ops.aten.view.default(view_35, [12288, 384])
        mm_5: "f32[12288, 768]" = torch.ops.aten.mm.default(view_36, t_5);  view_36 = t_5 = None
        view_37: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_5, [1536, 8, 768]);  mm_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_4: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_2: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_37, 'sum', 2, '12');  view_37 = None
        wait_tensor_6: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        copy_3: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_4, wait_tensor_6);  empty_4 = wait_tensor_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_8: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_3, primals_19);  copy_3 = primals_19 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_5 = torch.ops.aten.native_dropout.default(add_8, 0.1, True);  add_8 = None
        getitem_25: "f32[768, 8, 768]" = native_dropout_5[0]
        getitem_26: "b8[768, 8, 768]" = native_dropout_5[1];  native_dropout_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_9: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_6, getitem_25);  getitem_25 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_3 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_9, primals_20, primals_21, [768], 1e-05)
        getitem_27: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_3[0]
        getitem_28: "f32[6144]" = fused_layer_norm_affine_fwd_3[1]
        getitem_29: "f32[6144]" = fused_layer_norm_affine_fwd_3[2];  fused_layer_norm_affine_fwd_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_3: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_27, 2, '12')
        wait_tensor_7: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        t_6: "f32[768, 1536]" = torch.ops.aten.t.default(primals_22);  primals_22 = None
        view_38: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_7, [12288, 768]);  wait_tensor_7 = None
        mm_6: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_38, t_6);  view_38 = t_6 = None
        view_39: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_6, [1536, 8, 1536]);  mm_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_10: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_39, primals_23);  view_39 = primals_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_1: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_10)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_7: "f32[1536, 768]" = torch.ops.aten.t.default(primals_24)
        view_40: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_1, [12288, 1536])
        mm_7: "f32[12288, 768]" = torch.ops.aten.mm.default(view_40, t_7);  view_40 = t_7 = None
        view_41: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_7, [1536, 8, 768]);  mm_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_5: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_3: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_41, 'sum', 2, '12');  view_41 = None
        wait_tensor_8: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_3);  reduce_scatter_tensor_3 = None
        copy_4: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_5, wait_tensor_8);  empty_5 = wait_tensor_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_11: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_4, primals_25);  copy_4 = primals_25 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_6 = torch.ops.aten.native_dropout.default(add_11, 0.1, True);  add_11 = None
        getitem_30: "f32[768, 8, 768]" = native_dropout_6[0]
        getitem_31: "b8[768, 8, 768]" = native_dropout_6[1];  native_dropout_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_12: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_9, getitem_30);  getitem_30 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_4 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_12, primals_26, primals_27, [768], 1e-05)
        getitem_32: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_4[0]
        getitem_33: "f32[6144]" = fused_layer_norm_affine_fwd_4[1]
        getitem_34: "f32[6144]" = fused_layer_norm_affine_fwd_4[2];  fused_layer_norm_affine_fwd_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_4: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_32, 2, '12')
        wait_tensor_9: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        t_8: "f32[768, 1152]" = torch.ops.aten.t.default(primals_28);  primals_28 = None
        view_42: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_9, [12288, 768]);  wait_tensor_9 = None
        mm_8: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_42, t_8);  view_42 = t_8 = None
        view_43: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_8, [1536, 8, 1152]);  mm_8 = None
        add_13: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_43, primals_29);  view_43 = primals_29 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_44: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_13, [1536, 8, 12, 96]);  add_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(view_44, [32, 32, 32], 3);  view_44 = None
        getitem_35: "f32[1536, 8, 12, 32]" = split_with_sizes_2[0]
        getitem_36: "f32[1536, 8, 12, 32]" = split_with_sizes_2[1]
        getitem_37: "f32[1536, 8, 12, 32]" = split_with_sizes_2[2];  split_with_sizes_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_45: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_35, [1536, 8, 12, 32]);  getitem_35 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_46: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_45, [1536, 96, 32]);  view_45 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_47: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_36, [1536, 96, -1]);  getitem_36 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_6: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_9: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_46, 0, 1);  view_46 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_10: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_47, 0, 1);  view_47 = None
        transpose_11: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_10, 1, 2);  transpose_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_2: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_6, transpose_9, transpose_11, beta = 0.0, alpha = 0.17677669529663687);  empty_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_48: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_2, [8, 12, 1536, 1536]);  baddbmm_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_2: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_48, _to_copy_1, -10000.0);  view_48 = None
        view_49: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_2, [96, 1536, 1536]);  masked_fill_2 = None
        view_50: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_49, [8, 12, 1536, 1536]);  view_49 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_2: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_50, -1, False);  view_50 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_7 = torch.ops.aten.native_dropout.default(_softmax_2, 0.1, True)
        getitem_38: "f32[8, 12, 1536, 1536]" = native_dropout_7[0]
        getitem_39: "b8[8, 12, 1536, 1536]" = native_dropout_7[1];  native_dropout_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_52: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_37, [1536, 96, -1]);  getitem_37 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_53: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_38, [96, 1536, -1]);  getitem_38 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_12: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_52, 0, 1);  view_52 = None
        bmm_2: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_53, transpose_12)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_54: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_2, [8, 12, 1536, 32]);  bmm_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_2: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_54, [2, 0, 1, 3]);  view_54 = None
        clone_5: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_55: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_5, [1536, 8, 384]);  clone_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_9: "f32[384, 768]" = torch.ops.aten.t.default(primals_30)
        view_56: "f32[12288, 384]" = torch.ops.aten.view.default(view_55, [12288, 384])
        mm_9: "f32[12288, 768]" = torch.ops.aten.mm.default(view_56, t_9);  view_56 = t_9 = None
        view_57: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_9, [1536, 8, 768]);  mm_9 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_7: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_4: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_57, 'sum', 2, '12');  view_57 = None
        wait_tensor_10: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_4);  reduce_scatter_tensor_4 = None
        copy_5: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_7, wait_tensor_10);  empty_7 = wait_tensor_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_14: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_5, primals_31);  copy_5 = primals_31 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_8 = torch.ops.aten.native_dropout.default(add_14, 0.1, True);  add_14 = None
        getitem_40: "f32[768, 8, 768]" = native_dropout_8[0]
        getitem_41: "b8[768, 8, 768]" = native_dropout_8[1];  native_dropout_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_15: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_12, getitem_40);  getitem_40 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_5 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_15, primals_32, primals_33, [768], 1e-05)
        getitem_42: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_5[0]
        getitem_43: "f32[6144]" = fused_layer_norm_affine_fwd_5[1]
        getitem_44: "f32[6144]" = fused_layer_norm_affine_fwd_5[2];  fused_layer_norm_affine_fwd_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_5: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_42, 2, '12')
        wait_tensor_11: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        t_10: "f32[768, 1536]" = torch.ops.aten.t.default(primals_34);  primals_34 = None
        view_58: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_11, [12288, 768]);  wait_tensor_11 = None
        mm_10: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_58, t_10);  view_58 = t_10 = None
        view_59: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_10, [1536, 8, 1536]);  mm_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_16: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_59, primals_35);  view_59 = primals_35 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_2: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_16)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_11: "f32[1536, 768]" = torch.ops.aten.t.default(primals_36)
        view_60: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_2, [12288, 1536])
        mm_11: "f32[12288, 768]" = torch.ops.aten.mm.default(view_60, t_11);  view_60 = t_11 = None
        view_61: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_11, [1536, 8, 768]);  mm_11 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_8: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_5: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_61, 'sum', 2, '12');  view_61 = None
        wait_tensor_12: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_5);  reduce_scatter_tensor_5 = None
        copy_6: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_8, wait_tensor_12);  empty_8 = wait_tensor_12 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_17: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_6, primals_37);  copy_6 = primals_37 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_9 = torch.ops.aten.native_dropout.default(add_17, 0.1, True);  add_17 = None
        getitem_45: "f32[768, 8, 768]" = native_dropout_9[0]
        getitem_46: "b8[768, 8, 768]" = native_dropout_9[1];  native_dropout_9 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_18: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_15, getitem_45);  getitem_45 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_6 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_18, primals_38, primals_39, [768], 1e-05)
        getitem_47: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_6[0]
        getitem_48: "f32[6144]" = fused_layer_norm_affine_fwd_6[1]
        getitem_49: "f32[6144]" = fused_layer_norm_affine_fwd_6[2];  fused_layer_norm_affine_fwd_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_6: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_47, 2, '12')
        wait_tensor_13: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        t_12: "f32[768, 1152]" = torch.ops.aten.t.default(primals_40);  primals_40 = None
        view_62: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_13, [12288, 768]);  wait_tensor_13 = None
        mm_12: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_62, t_12);  view_62 = t_12 = None
        view_63: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_12, [1536, 8, 1152]);  mm_12 = None
        add_19: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_63, primals_41);  view_63 = primals_41 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_64: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_19, [1536, 8, 12, 96]);  add_19 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(view_64, [32, 32, 32], 3);  view_64 = None
        getitem_50: "f32[1536, 8, 12, 32]" = split_with_sizes_3[0]
        getitem_51: "f32[1536, 8, 12, 32]" = split_with_sizes_3[1]
        getitem_52: "f32[1536, 8, 12, 32]" = split_with_sizes_3[2];  split_with_sizes_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_65: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_50, [1536, 8, 12, 32]);  getitem_50 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_66: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_65, [1536, 96, 32]);  view_65 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_67: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_51, [1536, 96, -1]);  getitem_51 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_9: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_13: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_66, 0, 1);  view_66 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_14: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_67, 0, 1);  view_67 = None
        transpose_15: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_14, 1, 2);  transpose_14 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_3: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_9, transpose_13, transpose_15, beta = 0.0, alpha = 0.17677669529663687);  empty_9 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_68: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_3, [8, 12, 1536, 1536]);  baddbmm_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_3: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_68, _to_copy_1, -10000.0);  view_68 = None
        view_69: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_3, [96, 1536, 1536]);  masked_fill_3 = None
        view_70: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_69, [8, 12, 1536, 1536]);  view_69 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_3: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_70, -1, False);  view_70 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_10 = torch.ops.aten.native_dropout.default(_softmax_3, 0.1, True)
        getitem_53: "f32[8, 12, 1536, 1536]" = native_dropout_10[0]
        getitem_54: "b8[8, 12, 1536, 1536]" = native_dropout_10[1];  native_dropout_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_72: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_52, [1536, 96, -1]);  getitem_52 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_73: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_53, [96, 1536, -1]);  getitem_53 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_16: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_72, 0, 1);  view_72 = None
        bmm_3: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_73, transpose_16)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_74: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_3, [8, 12, 1536, 32]);  bmm_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_3: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_74, [2, 0, 1, 3]);  view_74 = None
        clone_6: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_75: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_6, [1536, 8, 384]);  clone_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_13: "f32[384, 768]" = torch.ops.aten.t.default(primals_42)
        view_76: "f32[12288, 384]" = torch.ops.aten.view.default(view_75, [12288, 384])
        mm_13: "f32[12288, 768]" = torch.ops.aten.mm.default(view_76, t_13);  view_76 = t_13 = None
        view_77: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_13, [1536, 8, 768]);  mm_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_10: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_6: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_77, 'sum', 2, '12');  view_77 = None
        wait_tensor_14: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_6);  reduce_scatter_tensor_6 = None
        copy_7: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_10, wait_tensor_14);  empty_10 = wait_tensor_14 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_20: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_7, primals_43);  copy_7 = primals_43 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_11 = torch.ops.aten.native_dropout.default(add_20, 0.1, True);  add_20 = None
        getitem_55: "f32[768, 8, 768]" = native_dropout_11[0]
        getitem_56: "b8[768, 8, 768]" = native_dropout_11[1];  native_dropout_11 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_21: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_18, getitem_55);  getitem_55 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_7 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_21, primals_44, primals_45, [768], 1e-05)
        getitem_57: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_7[0]
        getitem_58: "f32[6144]" = fused_layer_norm_affine_fwd_7[1]
        getitem_59: "f32[6144]" = fused_layer_norm_affine_fwd_7[2];  fused_layer_norm_affine_fwd_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_7: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_57, 2, '12')
        wait_tensor_15: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        t_14: "f32[768, 1536]" = torch.ops.aten.t.default(primals_46);  primals_46 = None
        view_78: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_15, [12288, 768]);  wait_tensor_15 = None
        mm_14: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_78, t_14);  view_78 = t_14 = None
        view_79: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_14, [1536, 8, 1536]);  mm_14 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_22: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_79, primals_47);  view_79 = primals_47 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_3: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_22)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_15: "f32[1536, 768]" = torch.ops.aten.t.default(primals_48)
        view_80: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_3, [12288, 1536])
        mm_15: "f32[12288, 768]" = torch.ops.aten.mm.default(view_80, t_15);  view_80 = t_15 = None
        view_81: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_15, [1536, 8, 768]);  mm_15 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_11: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_7: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_81, 'sum', 2, '12');  view_81 = None
        wait_tensor_16: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_7);  reduce_scatter_tensor_7 = None
        copy_8: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_11, wait_tensor_16);  empty_11 = wait_tensor_16 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_23: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_8, primals_49);  copy_8 = primals_49 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_12 = torch.ops.aten.native_dropout.default(add_23, 0.1, True);  add_23 = None
        getitem_60: "f32[768, 8, 768]" = native_dropout_12[0]
        getitem_61: "b8[768, 8, 768]" = native_dropout_12[1];  native_dropout_12 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_24: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_21, getitem_60);  getitem_60 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_8 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_24, primals_50, primals_51, [768], 1e-05)
        getitem_62: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_8[0]
        getitem_63: "f32[6144]" = fused_layer_norm_affine_fwd_8[1]
        getitem_64: "f32[6144]" = fused_layer_norm_affine_fwd_8[2];  fused_layer_norm_affine_fwd_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_8: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_62, 2, '12')
        wait_tensor_17: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_8);  all_gather_into_tensor_8 = None
        t_16: "f32[768, 1152]" = torch.ops.aten.t.default(primals_52);  primals_52 = None
        view_82: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_17, [12288, 768]);  wait_tensor_17 = None
        mm_16: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_82, t_16);  view_82 = t_16 = None
        view_83: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_16, [1536, 8, 1152]);  mm_16 = None
        add_25: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_83, primals_53);  view_83 = primals_53 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_84: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_25, [1536, 8, 12, 96]);  add_25 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(view_84, [32, 32, 32], 3);  view_84 = None
        getitem_65: "f32[1536, 8, 12, 32]" = split_with_sizes_4[0]
        getitem_66: "f32[1536, 8, 12, 32]" = split_with_sizes_4[1]
        getitem_67: "f32[1536, 8, 12, 32]" = split_with_sizes_4[2];  split_with_sizes_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_85: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_65, [1536, 8, 12, 32]);  getitem_65 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_86: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_85, [1536, 96, 32]);  view_85 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_87: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_66, [1536, 96, -1]);  getitem_66 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_12: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_17: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_86, 0, 1);  view_86 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_18: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_87, 0, 1);  view_87 = None
        transpose_19: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_18, 1, 2);  transpose_18 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_4: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_12, transpose_17, transpose_19, beta = 0.0, alpha = 0.17677669529663687);  empty_12 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_88: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_4, [8, 12, 1536, 1536]);  baddbmm_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_4: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_88, _to_copy_1, -10000.0);  view_88 = None
        view_89: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_4, [96, 1536, 1536]);  masked_fill_4 = None
        view_90: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_89, [8, 12, 1536, 1536]);  view_89 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_4: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_90, -1, False);  view_90 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_13 = torch.ops.aten.native_dropout.default(_softmax_4, 0.1, True)
        getitem_68: "f32[8, 12, 1536, 1536]" = native_dropout_13[0]
        getitem_69: "b8[8, 12, 1536, 1536]" = native_dropout_13[1];  native_dropout_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_92: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_67, [1536, 96, -1]);  getitem_67 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_93: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_68, [96, 1536, -1]);  getitem_68 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_20: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_92, 0, 1);  view_92 = None
        bmm_4: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_93, transpose_20)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_94: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_4, [8, 12, 1536, 32]);  bmm_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_4: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_94, [2, 0, 1, 3]);  view_94 = None
        clone_7: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_95: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_7, [1536, 8, 384]);  clone_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_17: "f32[384, 768]" = torch.ops.aten.t.default(primals_54)
        view_96: "f32[12288, 384]" = torch.ops.aten.view.default(view_95, [12288, 384])
        mm_17: "f32[12288, 768]" = torch.ops.aten.mm.default(view_96, t_17);  view_96 = t_17 = None
        view_97: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_17, [1536, 8, 768]);  mm_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_13: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_8: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_97, 'sum', 2, '12');  view_97 = None
        wait_tensor_18: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_8);  reduce_scatter_tensor_8 = None
        copy_9: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_13, wait_tensor_18);  empty_13 = wait_tensor_18 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_26: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_9, primals_55);  copy_9 = primals_55 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_14 = torch.ops.aten.native_dropout.default(add_26, 0.1, True);  add_26 = None
        getitem_70: "f32[768, 8, 768]" = native_dropout_14[0]
        getitem_71: "b8[768, 8, 768]" = native_dropout_14[1];  native_dropout_14 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_27: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_24, getitem_70);  getitem_70 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_9 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_27, primals_56, primals_57, [768], 1e-05)
        getitem_72: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_9[0]
        getitem_73: "f32[6144]" = fused_layer_norm_affine_fwd_9[1]
        getitem_74: "f32[6144]" = fused_layer_norm_affine_fwd_9[2];  fused_layer_norm_affine_fwd_9 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_9: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_72, 2, '12')
        wait_tensor_19: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_9);  all_gather_into_tensor_9 = None
        t_18: "f32[768, 1536]" = torch.ops.aten.t.default(primals_58);  primals_58 = None
        view_98: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_19, [12288, 768]);  wait_tensor_19 = None
        mm_18: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_98, t_18);  view_98 = t_18 = None
        view_99: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_18, [1536, 8, 1536]);  mm_18 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_28: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_99, primals_59);  view_99 = primals_59 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_4: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_28)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_19: "f32[1536, 768]" = torch.ops.aten.t.default(primals_60)
        view_100: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_4, [12288, 1536])
        mm_19: "f32[12288, 768]" = torch.ops.aten.mm.default(view_100, t_19);  view_100 = t_19 = None
        view_101: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_19, [1536, 8, 768]);  mm_19 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_14: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_9: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_101, 'sum', 2, '12');  view_101 = None
        wait_tensor_20: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_9);  reduce_scatter_tensor_9 = None
        copy_10: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_14, wait_tensor_20);  empty_14 = wait_tensor_20 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_29: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_10, primals_61);  copy_10 = primals_61 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_15 = torch.ops.aten.native_dropout.default(add_29, 0.1, True);  add_29 = None
        getitem_75: "f32[768, 8, 768]" = native_dropout_15[0]
        getitem_76: "b8[768, 8, 768]" = native_dropout_15[1];  native_dropout_15 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_30: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_27, getitem_75);  getitem_75 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_10 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_30, primals_62, primals_63, [768], 1e-05)
        getitem_77: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_10[0]
        getitem_78: "f32[6144]" = fused_layer_norm_affine_fwd_10[1]
        getitem_79: "f32[6144]" = fused_layer_norm_affine_fwd_10[2];  fused_layer_norm_affine_fwd_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_10: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_77, 2, '12')
        wait_tensor_21: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_10);  all_gather_into_tensor_10 = None
        t_20: "f32[768, 1152]" = torch.ops.aten.t.default(primals_64);  primals_64 = None
        view_102: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_21, [12288, 768]);  wait_tensor_21 = None
        mm_20: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_102, t_20);  view_102 = t_20 = None
        view_103: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_20, [1536, 8, 1152]);  mm_20 = None
        add_31: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_103, primals_65);  view_103 = primals_65 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_104: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_31, [1536, 8, 12, 96]);  add_31 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(view_104, [32, 32, 32], 3);  view_104 = None
        getitem_80: "f32[1536, 8, 12, 32]" = split_with_sizes_5[0]
        getitem_81: "f32[1536, 8, 12, 32]" = split_with_sizes_5[1]
        getitem_82: "f32[1536, 8, 12, 32]" = split_with_sizes_5[2];  split_with_sizes_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_105: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_80, [1536, 8, 12, 32]);  getitem_80 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_106: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_105, [1536, 96, 32]);  view_105 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_107: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_81, [1536, 96, -1]);  getitem_81 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_15: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_21: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_106, 0, 1);  view_106 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_22: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_107, 0, 1);  view_107 = None
        transpose_23: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_22, 1, 2);  transpose_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_5: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_15, transpose_21, transpose_23, beta = 0.0, alpha = 0.17677669529663687);  empty_15 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_108: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_5, [8, 12, 1536, 1536]);  baddbmm_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_5: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_108, _to_copy_1, -10000.0);  view_108 = None
        view_109: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_5, [96, 1536, 1536]);  masked_fill_5 = None
        view_110: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_109, [8, 12, 1536, 1536]);  view_109 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_5: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_110, -1, False);  view_110 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_16 = torch.ops.aten.native_dropout.default(_softmax_5, 0.1, True)
        getitem_83: "f32[8, 12, 1536, 1536]" = native_dropout_16[0]
        getitem_84: "b8[8, 12, 1536, 1536]" = native_dropout_16[1];  native_dropout_16 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_112: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_82, [1536, 96, -1]);  getitem_82 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_113: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_83, [96, 1536, -1]);  getitem_83 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_24: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_112, 0, 1);  view_112 = None
        bmm_5: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_113, transpose_24)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_114: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_5, [8, 12, 1536, 32]);  bmm_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_5: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_114, [2, 0, 1, 3]);  view_114 = None
        clone_8: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_115: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_8, [1536, 8, 384]);  clone_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_21: "f32[384, 768]" = torch.ops.aten.t.default(primals_66)
        view_116: "f32[12288, 384]" = torch.ops.aten.view.default(view_115, [12288, 384])
        mm_21: "f32[12288, 768]" = torch.ops.aten.mm.default(view_116, t_21);  view_116 = t_21 = None
        view_117: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_21, [1536, 8, 768]);  mm_21 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_16: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_10: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_117, 'sum', 2, '12');  view_117 = None
        wait_tensor_22: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_10);  reduce_scatter_tensor_10 = None
        copy_11: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_16, wait_tensor_22);  empty_16 = wait_tensor_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_32: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_11, primals_67);  copy_11 = primals_67 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_17 = torch.ops.aten.native_dropout.default(add_32, 0.1, True);  add_32 = None
        getitem_85: "f32[768, 8, 768]" = native_dropout_17[0]
        getitem_86: "b8[768, 8, 768]" = native_dropout_17[1];  native_dropout_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_33: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_30, getitem_85);  getitem_85 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_11 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_33, primals_68, primals_69, [768], 1e-05)
        getitem_87: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_11[0]
        getitem_88: "f32[6144]" = fused_layer_norm_affine_fwd_11[1]
        getitem_89: "f32[6144]" = fused_layer_norm_affine_fwd_11[2];  fused_layer_norm_affine_fwd_11 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_11: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_87, 2, '12')
        wait_tensor_23: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_11);  all_gather_into_tensor_11 = None
        t_22: "f32[768, 1536]" = torch.ops.aten.t.default(primals_70);  primals_70 = None
        view_118: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_23, [12288, 768]);  wait_tensor_23 = None
        mm_22: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_118, t_22);  view_118 = t_22 = None
        view_119: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_22, [1536, 8, 1536]);  mm_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_34: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_119, primals_71);  view_119 = primals_71 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_5: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_34)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_23: "f32[1536, 768]" = torch.ops.aten.t.default(primals_72)
        view_120: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_5, [12288, 1536])
        mm_23: "f32[12288, 768]" = torch.ops.aten.mm.default(view_120, t_23);  view_120 = t_23 = None
        view_121: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_23, [1536, 8, 768]);  mm_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_17: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_11: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_121, 'sum', 2, '12');  view_121 = None
        wait_tensor_24: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_11);  reduce_scatter_tensor_11 = None
        copy_12: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_17, wait_tensor_24);  empty_17 = wait_tensor_24 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_35: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_12, primals_73);  copy_12 = primals_73 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_18 = torch.ops.aten.native_dropout.default(add_35, 0.1, True);  add_35 = None
        getitem_90: "f32[768, 8, 768]" = native_dropout_18[0]
        getitem_91: "b8[768, 8, 768]" = native_dropout_18[1];  native_dropout_18 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_36: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_33, getitem_90);  getitem_90 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_12 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_36, primals_74, primals_75, [768], 1e-05)
        getitem_92: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_12[0]
        getitem_93: "f32[6144]" = fused_layer_norm_affine_fwd_12[1]
        getitem_94: "f32[6144]" = fused_layer_norm_affine_fwd_12[2];  fused_layer_norm_affine_fwd_12 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_12: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_92, 2, '12')
        wait_tensor_25: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_12);  all_gather_into_tensor_12 = None
        t_24: "f32[768, 1152]" = torch.ops.aten.t.default(primals_76);  primals_76 = None
        view_122: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_25, [12288, 768]);  wait_tensor_25 = None
        mm_24: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_122, t_24);  view_122 = t_24 = None
        view_123: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_24, [1536, 8, 1152]);  mm_24 = None
        add_37: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_123, primals_77);  view_123 = primals_77 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_124: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_37, [1536, 8, 12, 96]);  add_37 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(view_124, [32, 32, 32], 3);  view_124 = None
        getitem_95: "f32[1536, 8, 12, 32]" = split_with_sizes_6[0]
        getitem_96: "f32[1536, 8, 12, 32]" = split_with_sizes_6[1]
        getitem_97: "f32[1536, 8, 12, 32]" = split_with_sizes_6[2];  split_with_sizes_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_125: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_95, [1536, 8, 12, 32]);  getitem_95 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_126: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_125, [1536, 96, 32]);  view_125 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_127: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_96, [1536, 96, -1]);  getitem_96 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_18: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_25: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_126, 0, 1);  view_126 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_26: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_127, 0, 1);  view_127 = None
        transpose_27: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_26, 1, 2);  transpose_26 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_6: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_18, transpose_25, transpose_27, beta = 0.0, alpha = 0.17677669529663687);  empty_18 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_128: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_6, [8, 12, 1536, 1536]);  baddbmm_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_6: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_128, _to_copy_1, -10000.0);  view_128 = None
        view_129: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_6, [96, 1536, 1536]);  masked_fill_6 = None
        view_130: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_129, [8, 12, 1536, 1536]);  view_129 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_6: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_130, -1, False);  view_130 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_19 = torch.ops.aten.native_dropout.default(_softmax_6, 0.1, True)
        getitem_98: "f32[8, 12, 1536, 1536]" = native_dropout_19[0]
        getitem_99: "b8[8, 12, 1536, 1536]" = native_dropout_19[1];  native_dropout_19 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_132: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_97, [1536, 96, -1]);  getitem_97 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_133: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_98, [96, 1536, -1]);  getitem_98 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_28: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_132, 0, 1);  view_132 = None
        bmm_6: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_133, transpose_28)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_134: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_6, [8, 12, 1536, 32]);  bmm_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_6: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_134, [2, 0, 1, 3]);  view_134 = None
        clone_9: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_135: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_9, [1536, 8, 384]);  clone_9 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_25: "f32[384, 768]" = torch.ops.aten.t.default(primals_78)
        view_136: "f32[12288, 384]" = torch.ops.aten.view.default(view_135, [12288, 384])
        mm_25: "f32[12288, 768]" = torch.ops.aten.mm.default(view_136, t_25);  view_136 = t_25 = None
        view_137: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_25, [1536, 8, 768]);  mm_25 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_19: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_12: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_137, 'sum', 2, '12');  view_137 = None
        wait_tensor_26: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_12);  reduce_scatter_tensor_12 = None
        copy_13: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_19, wait_tensor_26);  empty_19 = wait_tensor_26 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_38: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_13, primals_79);  copy_13 = primals_79 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_20 = torch.ops.aten.native_dropout.default(add_38, 0.1, True);  add_38 = None
        getitem_100: "f32[768, 8, 768]" = native_dropout_20[0]
        getitem_101: "b8[768, 8, 768]" = native_dropout_20[1];  native_dropout_20 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_39: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_36, getitem_100);  getitem_100 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_13 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_39, primals_80, primals_81, [768], 1e-05)
        getitem_102: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_13[0]
        getitem_103: "f32[6144]" = fused_layer_norm_affine_fwd_13[1]
        getitem_104: "f32[6144]" = fused_layer_norm_affine_fwd_13[2];  fused_layer_norm_affine_fwd_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_13: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_102, 2, '12')
        wait_tensor_27: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_13);  all_gather_into_tensor_13 = None
        t_26: "f32[768, 1536]" = torch.ops.aten.t.default(primals_82);  primals_82 = None
        view_138: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_27, [12288, 768]);  wait_tensor_27 = None
        mm_26: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_138, t_26);  view_138 = t_26 = None
        view_139: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_26, [1536, 8, 1536]);  mm_26 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_40: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_139, primals_83);  view_139 = primals_83 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_6: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_40)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_27: "f32[1536, 768]" = torch.ops.aten.t.default(primals_84)
        view_140: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_6, [12288, 1536])
        mm_27: "f32[12288, 768]" = torch.ops.aten.mm.default(view_140, t_27);  view_140 = t_27 = None
        view_141: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_27, [1536, 8, 768]);  mm_27 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_20: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_13: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_141, 'sum', 2, '12');  view_141 = None
        wait_tensor_28: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_13);  reduce_scatter_tensor_13 = None
        copy_14: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_20, wait_tensor_28);  empty_20 = wait_tensor_28 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_41: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_14, primals_85);  copy_14 = primals_85 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_21 = torch.ops.aten.native_dropout.default(add_41, 0.1, True);  add_41 = None
        getitem_105: "f32[768, 8, 768]" = native_dropout_21[0]
        getitem_106: "b8[768, 8, 768]" = native_dropout_21[1];  native_dropout_21 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_42: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_39, getitem_105);  getitem_105 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_14 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_42, primals_86, primals_87, [768], 1e-05)
        getitem_107: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_14[0]
        getitem_108: "f32[6144]" = fused_layer_norm_affine_fwd_14[1]
        getitem_109: "f32[6144]" = fused_layer_norm_affine_fwd_14[2];  fused_layer_norm_affine_fwd_14 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_14: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_107, 2, '12')
        wait_tensor_29: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_14);  all_gather_into_tensor_14 = None
        t_28: "f32[768, 1152]" = torch.ops.aten.t.default(primals_88);  primals_88 = None
        view_142: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_29, [12288, 768]);  wait_tensor_29 = None
        mm_28: "f32[12288, 1152]" = torch.ops.aten.mm.default(view_142, t_28);  view_142 = t_28 = None
        view_143: "f32[1536, 8, 1152]" = torch.ops.aten.view.default(mm_28, [1536, 8, 1152]);  mm_28 = None
        add_43: "f32[1536, 8, 1152]" = torch.ops.aten.add.Tensor(view_143, primals_89);  view_143 = primals_89 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_144: "f32[1536, 8, 12, 96]" = torch.ops.aten.view.default(add_43, [1536, 8, 12, 96]);  add_43 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(view_144, [32, 32, 32], 3);  view_144 = None
        getitem_110: "f32[1536, 8, 12, 32]" = split_with_sizes_7[0]
        getitem_111: "f32[1536, 8, 12, 32]" = split_with_sizes_7[1]
        getitem_112: "f32[1536, 8, 12, 32]" = split_with_sizes_7[2];  split_with_sizes_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_145: "f32[1536, 8, 12, 32]" = torch.ops.aten.view.default(getitem_110, [1536, 8, 12, 32]);  getitem_110 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_146: "f32[1536, 96, 32]" = torch.ops.aten.view.default(view_145, [1536, 96, 32]);  view_145 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_147: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_111, [1536, 96, -1]);  getitem_111 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_21: "f32[96, 1536, 1536]" = torch.ops.aten.empty.memory_format([96, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_29: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_146, 0, 1);  view_146 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_30: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_147, 0, 1);  view_147 = None
        transpose_31: "f32[96, 32, 1536]" = torch.ops.aten.transpose.int(transpose_30, 1, 2);  transpose_30 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_7: "f32[96, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_21, transpose_29, transpose_31, beta = 0.0, alpha = 0.17677669529663687);  empty_21 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_148: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_7, [8, 12, 1536, 1536]);  baddbmm_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_7: "f32[8, 12, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_148, _to_copy_1, -10000.0);  view_148 = None
        view_149: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_7, [96, 1536, 1536]);  masked_fill_7 = None
        view_150: "f32[8, 12, 1536, 1536]" = torch.ops.aten.view.default(view_149, [8, 12, 1536, 1536]);  view_149 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_7: "f32[8, 12, 1536, 1536]" = torch.ops.aten._softmax.default(view_150, -1, False);  view_150 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_22 = torch.ops.aten.native_dropout.default(_softmax_7, 0.1, True)
        getitem_113: "f32[8, 12, 1536, 1536]" = native_dropout_22[0]
        getitem_114: "b8[8, 12, 1536, 1536]" = native_dropout_22[1];  native_dropout_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_152: "f32[1536, 96, 32]" = torch.ops.aten.view.default(getitem_112, [1536, 96, -1]);  getitem_112 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_153: "f32[96, 1536, 1536]" = torch.ops.aten.view.default(getitem_113, [96, 1536, -1]);  getitem_113 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_32: "f32[96, 1536, 32]" = torch.ops.aten.transpose.int(view_152, 0, 1);  view_152 = None
        bmm_7: "f32[96, 1536, 32]" = torch.ops.aten.bmm.default(view_153, transpose_32)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_154: "f32[8, 12, 1536, 32]" = torch.ops.aten.view.default(bmm_7, [8, 12, 1536, 32]);  bmm_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_7: "f32[1536, 8, 12, 32]" = torch.ops.aten.permute.default(view_154, [2, 0, 1, 3]);  view_154 = None
        clone_10: "f32[1536, 8, 12, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_155: "f32[1536, 8, 384]" = torch.ops.aten.view.default(clone_10, [1536, 8, 384]);  clone_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_29: "f32[384, 768]" = torch.ops.aten.t.default(primals_90)
        view_156: "f32[12288, 384]" = torch.ops.aten.view.default(view_155, [12288, 384])
        mm_29: "f32[12288, 768]" = torch.ops.aten.mm.default(view_156, t_29);  view_156 = t_29 = None
        view_157: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_29, [1536, 8, 768]);  mm_29 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_22: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_14: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_157, 'sum', 2, '12');  view_157 = None
        wait_tensor_30: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_14);  reduce_scatter_tensor_14 = None
        copy_15: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_22, wait_tensor_30);  empty_22 = wait_tensor_30 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_44: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_15, primals_91);  copy_15 = primals_91 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_23 = torch.ops.aten.native_dropout.default(add_44, 0.1, True);  add_44 = None
        getitem_115: "f32[768, 8, 768]" = native_dropout_23[0]
        getitem_116: "b8[768, 8, 768]" = native_dropout_23[1];  native_dropout_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_45: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_42, getitem_115);  getitem_115 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_15 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_45, primals_92, primals_93, [768], 1e-05)
        getitem_117: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_15[0]
        getitem_118: "f32[6144]" = fused_layer_norm_affine_fwd_15[1]
        getitem_119: "f32[6144]" = fused_layer_norm_affine_fwd_15[2];  fused_layer_norm_affine_fwd_15 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_15: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_117, 2, '12')
        wait_tensor_31: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_15);  all_gather_into_tensor_15 = None
        t_30: "f32[768, 1536]" = torch.ops.aten.t.default(primals_94);  primals_94 = None
        view_158: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_31, [12288, 768]);  wait_tensor_31 = None
        mm_30: "f32[12288, 1536]" = torch.ops.aten.mm.default(view_158, t_30);  view_158 = t_30 = None
        view_159: "f32[1536, 8, 1536]" = torch.ops.aten.view.default(mm_30, [1536, 8, 1536]);  mm_30 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_46: "f32[1536, 8, 1536]" = torch.ops.aten.add.Tensor(view_159, primals_95);  view_159 = primals_95 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_7: "f32[1536, 8, 1536]" = torch.ops.aten.gelu.default(add_46)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_31: "f32[1536, 768]" = torch.ops.aten.t.default(primals_96)
        view_160: "f32[12288, 1536]" = torch.ops.aten.view.default(gelu_7, [12288, 1536])
        mm_31: "f32[12288, 768]" = torch.ops.aten.mm.default(view_160, t_31);  view_160 = t_31 = None
        view_161: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_31, [1536, 8, 768]);  mm_31 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_23: "f32[768, 8, 768]" = torch.ops.aten.empty.memory_format([768, 8, 768], dtype = torch.float32, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_15: "f32[768, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_161, 'sum', 2, '12');  view_161 = None
        wait_tensor_32: "f32[768, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_15);  reduce_scatter_tensor_15 = None
        copy_16: "f32[768, 8, 768]" = torch.ops.aten.copy.default(empty_23, wait_tensor_32);  empty_23 = wait_tensor_32 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_47: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(copy_16, primals_97);  copy_16 = primals_97 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_24 = torch.ops.aten.native_dropout.default(add_47, 0.1, True);  add_47 = None
        getitem_120: "f32[768, 8, 768]" = native_dropout_24[0]
        getitem_121: "b8[768, 8, 768]" = native_dropout_24[1];  native_dropout_24 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_48: "f32[768, 8, 768]" = torch.ops.aten.add.Tensor(add_45, getitem_120);  getitem_120 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_16 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_48, primals_98, primals_99, [768], 1e-05)
        getitem_122: "f32[768, 8, 768]" = fused_layer_norm_affine_fwd_16[0]
        getitem_123: "f32[6144]" = fused_layer_norm_affine_fwd_16[1]
        getitem_124: "f32[6144]" = fused_layer_norm_affine_fwd_16[2];  fused_layer_norm_affine_fwd_16 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_16: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_122, 2, '12')
        wait_tensor_33: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_16);  all_gather_into_tensor_16 = None
        t_32: "f32[768, 49152]" = torch.ops.aten.t.default(primals_100);  primals_100 = None
        view_162: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_33, [12288, 768]);  wait_tensor_33 = None
        mm_32: "f32[12288, 49152]" = torch.ops.aten.mm.default(view_162, t_32);  view_162 = t_32 = None
        view_163: "f32[1536, 8, 49152]" = torch.ops.aten.view.default(mm_32, [1536, 8, 49152]);  mm_32 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/language_module/language_module.py:77 in compute_language_model_loss, code: labels = labels.transpose(0, 1).contiguous()
        transpose_33: "i64[1536, 8]" = torch.ops.aten.transpose.int(_to_copy_3, 0, 1);  _to_copy_3 = None
        clone_11: "i64[1536, 8]" = torch.ops.aten.clone.default(transpose_33, memory_format = torch.contiguous_format);  transpose_33 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:245 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        max_1 = torch.ops.aten.max.dim(view_163, -1)
        getitem_125: "f32[1536, 8]" = max_1[0];  max_1 = None
        all_reduce_1: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_125, 'max', '12')
        wait_tensor_34: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        copy_17: "f32[1536, 8]" = torch.ops.aten.copy.default(getitem_125, wait_tensor_34);  getitem_125 = wait_tensor_34 = None
        clone_12: "f32[1536, 8, 49152]" = torch.ops.aten.clone.default(view_163);  view_163 = None
        unsqueeze_1: "f32[1536, 8, 1]" = torch.ops.aten.unsqueeze.default(copy_17, -1);  copy_17 = None
        sub_1: "f32[1536, 8, 49152]" = torch.ops.aten.sub.Tensor(clone_12, unsqueeze_1);  clone_12 = unsqueeze_1 = None
        lt_1: "b8[1536, 8]" = torch.ops.aten.lt.Scalar(clone_11, 49152)
        ge_1: "b8[1536, 8]" = torch.ops.aten.ge.Scalar(clone_11, 98304)
        bitwise_or_1: "b8[1536, 8]" = torch.ops.aten.bitwise_or.Tensor(lt_1, ge_1);  lt_1 = ge_1 = None
        clone_13: "i64[1536, 8]" = torch.ops.aten.clone.default(clone_11);  clone_11 = None
        sub_2: "i64[1536, 8]" = torch.ops.aten.sub.Tensor(clone_13, 49152);  clone_13 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        index_put_2: "i64[1536, 8]" = torch.ops.aten.index_put.default(sub_2, [bitwise_or_1], lift_fresh_copy_2);  sub_2 = lift_fresh_copy_2 = None
        arange: "i64[12288]" = torch.ops.aten.arange.start(0, 12288, device = device(type='cuda', index=1), pin_memory = False)
        view_166: "f32[12288, 49152]" = torch.ops.aten.view.default(sub_1, [-1, 49152])
        view_167: "i64[12288]" = torch.ops.aten.view.default(index_put_2, [-1]);  index_put_2 = None
        index_1: "f32[12288]" = torch.ops.aten.index.Tensor(view_166, [arange, view_167]);  view_166 = arange = None
        clone_14: "f32[12288]" = torch.ops.aten.clone.default(index_1);  index_1 = None
        view_168: "f32[1536, 8]" = torch.ops.aten.view.default(clone_14, [1536, 8]);  clone_14 = None
        _tensor_constant3 = self._tensor_constant3
        lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        index_put_3: "f32[1536, 8]" = torch.ops.aten.index_put.default(view_168, [bitwise_or_1], lift_fresh_copy_3);  view_168 = lift_fresh_copy_3 = None
        view_169: "f32[12288]" = torch.ops.aten.view.default(index_put_3, [12288]);  index_put_3 = None
        view_170: "f32[1536, 8]" = torch.ops.aten.view.default(view_169, [1536, 8]);  view_169 = None
        exp: "f32[1536, 8, 49152]" = torch.ops.aten.exp.default(sub_1)
        copy_18: "f32[1536, 8, 49152]" = torch.ops.aten.copy.default(sub_1, exp);  sub_1 = exp = None
        sum_1: "f32[1536, 8]" = torch.ops.aten.sum.dim_IntList(copy_18, [-1])
        all_reduce_2: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(view_170, 'sum', '12')
        wait_tensor_35: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = None
        copy_19: "f32[1536, 8]" = torch.ops.aten.copy.default(view_170, wait_tensor_35);  view_170 = wait_tensor_35 = None
        view_171: "f32[12288]" = torch.ops.aten.view.default(copy_19, [12288]);  copy_19 = None
        view_172: "f32[1536, 8]" = torch.ops.aten.view.default(view_171, [1536, 8]);  view_171 = None
        all_reduce_3: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(sum_1, 'sum', '12')
        wait_tensor_36: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = None
        copy_20: "f32[1536, 8]" = torch.ops.aten.copy.default(sum_1, wait_tensor_36);  sum_1 = wait_tensor_36 = None
        log: "f32[1536, 8]" = torch.ops.aten.log.default(copy_20)
        sub_3: "f32[1536, 8]" = torch.ops.aten.sub.Tensor(log, view_172);  log = view_172 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_34: "f32[8, 1536]" = torch.ops.aten.transpose.int(sub_3, 0, 1);  sub_3 = None
        clone_15: "f32[8, 1536]" = torch.ops.aten.clone.default(transpose_34, memory_format = torch.contiguous_format);  transpose_34 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:98 in loss_func, code: loss_mask = loss_mask.view(-1).float()
        view_173: "f32[12288]" = torch.ops.aten.view.default(_to_copy_4, [-1]);  _to_copy_4 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:99 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        view_174: "f32[12288]" = torch.ops.aten.view.default(clone_15, [-1]);  clone_15 = None
        mul: "f32[12288]" = torch.ops.aten.mul.Tensor(view_174, view_173);  view_174 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(view_173)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_2, sum_3);  sum_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:295 in forward_step, code: output_tensor /= num_microbatches
        div_2: "f32[]" = torch.ops.aten.div.Tensor(div_1, 1);  div_1 = None
        return [div_2, div_2, zeros, primals_2, primals_3, primals_6, primals_8, primals_9, primals_12, primals_14, primals_15, primals_18, primals_20, primals_21, primals_24, primals_26, primals_27, primals_30, primals_32, primals_33, primals_36, primals_38, primals_39, primals_42, primals_44, primals_45, primals_48, primals_50, primals_51, primals_54, primals_56, primals_57, primals_60, primals_62, primals_63, primals_66, primals_68, primals_69, primals_72, primals_74, primals_75, primals_78, primals_80, primals_81, primals_84, primals_86, primals_87, primals_90, primals_92, primals_93, primals_96, primals_98, primals_99, _to_copy_1, _to_copy_2, bitwise_or, index_put, getitem, getitem_1, getitem_2, getitem_3, getitem_4, transpose_1, transpose_3, _softmax, getitem_9, view_13, transpose_4, view_15, getitem_11, add_3, getitem_12, getitem_13, getitem_14, add_4, gelu, getitem_16, add_6, getitem_17, getitem_18, getitem_19, transpose_5, transpose_7, _softmax_1, getitem_24, view_33, transpose_8, view_35, getitem_26, add_9, getitem_27, getitem_28, getitem_29, add_10, gelu_1, getitem_31, add_12, getitem_32, getitem_33, getitem_34, transpose_9, transpose_11, _softmax_2, getitem_39, view_53, transpose_12, view_55, getitem_41, add_15, getitem_42, getitem_43, getitem_44, add_16, gelu_2, getitem_46, add_18, getitem_47, getitem_48, getitem_49, transpose_13, transpose_15, _softmax_3, getitem_54, view_73, transpose_16, view_75, getitem_56, add_21, getitem_57, getitem_58, getitem_59, add_22, gelu_3, getitem_61, add_24, getitem_62, getitem_63, getitem_64, transpose_17, transpose_19, _softmax_4, getitem_69, view_93, transpose_20, view_95, getitem_71, add_27, getitem_72, getitem_73, getitem_74, add_28, gelu_4, getitem_76, add_30, getitem_77, getitem_78, getitem_79, transpose_21, transpose_23, _softmax_5, getitem_84, view_113, transpose_24, view_115, getitem_86, add_33, getitem_87, getitem_88, getitem_89, add_34, gelu_5, getitem_91, add_36, getitem_92, getitem_93, getitem_94, transpose_25, transpose_27, _softmax_6, getitem_99, view_133, transpose_28, view_135, getitem_101, add_39, getitem_102, getitem_103, getitem_104, add_40, gelu_6, getitem_106, add_42, getitem_107, getitem_108, getitem_109, transpose_29, transpose_31, _softmax_7, getitem_114, view_153, transpose_32, view_155, getitem_116, add_45, getitem_117, getitem_118, getitem_119, add_46, gelu_7, getitem_121, add_48, getitem_122, getitem_123, getitem_124, bitwise_or_1, view_167, copy_18, copy_20, view_173, sum_3]
        

