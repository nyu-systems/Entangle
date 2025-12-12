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


# Graph[rank=3](fw, gid=0)
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[12288, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[288, 768]", primals_5: "f32[288]", primals_6: "f32[768, 96]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[384, 768]", primals_11: "f32[384]", primals_12: "f32[768, 384]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[12288, 768]", primals_17: "f32[1536, 768]", primals_18: "i64[8, 1536]", primals_19: "b8[8, 1, 1536, 1536]", primals_20: "i64[8, 1536]", primals_21: "i64[8, 1536]", primals_22: "f32[8, 1536]"):
        # File: /opt/tiger/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:456 in forward_backward_no_pipelining, code: total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        zeros: "i32[]" = torch.ops.aten.zeros.default([], dtype = torch.int32, device = device(type='cuda'), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:106 in forward_step_func, code: tokens = data['tokens'].to(device)
        _to_copy: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_18, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=3));  primals_18 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:107 in forward_step_func, code: attention_mask = data['attention_mask'].to(device)
        _to_copy_1: "b8[8, 1, 1536, 1536]" = torch.ops.aten._to_copy.default(primals_19, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=3));  primals_19 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:108 in forward_step_func, code: position_ids = data['position_ids'].to(device)
        _to_copy_2: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_20, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=3));  primals_20 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:109 in forward_step_func, code: labels = data['labels'].to(device)
        _to_copy_3: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_21, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=3));  primals_21 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:110 in forward_step_func, code: loss_mask = data['loss_mask'].to(device)
        _to_copy_4: "f32[8, 1536]" = torch.ops.aten._to_copy.default(primals_22, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=3));  primals_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:252 in forward, code: input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        lt: "b8[8, 1536]" = torch.ops.aten.lt.Scalar(_to_copy, 36864)
        ge: "b8[8, 1536]" = torch.ops.aten.ge.Scalar(_to_copy, 49152)
        bitwise_or: "b8[8, 1536]" = torch.ops.aten.bitwise_or.Tensor(lt, ge);  lt = ge = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:254 in forward, code: masked_input = input_.clone() - self.vocab_start_index
        clone: "i64[8, 1536]" = torch.ops.aten.clone.default(_to_copy);  _to_copy = None
        sub: "i64[8, 1536]" = torch.ops.aten.sub.Tensor(clone, 36864);  clone = None
        
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
        all_reduce: "f32[8, 1536, 768]" = torch.ops._c10d_functional.all_reduce.default(slice_scatter, 'sum', '42')
        wait_tensor: "f32[8, 1536, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        copy: "f32[8, 1536, 768]" = torch.ops.aten.copy.default(slice_scatter, wait_tensor);  slice_scatter = wait_tensor = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding: "f32[8, 1536, 768]" = torch.ops.aten.embedding.default(primals_17, _to_copy_2);  primals_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:112 in forward, code: embeddings = word_embeddings + position_embeddings
        view_1: "f32[8, 1536, 768]" = torch.ops.aten.view.default(copy, [8, 1536, 768]);  copy = None
        add: "f32[8, 1536, 768]" = torch.ops.aten.add.Tensor(view_1, embedding);  view_1 = embedding = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:118 in forward, code: embeddings = embeddings.transpose(0, 1).contiguous()
        transpose: "f32[1536, 8, 768]" = torch.ops.aten.transpose.int(add, 0, 1);  add = None
        clone_1: "f32[1536, 8, 768]" = torch.ops.aten.clone.default(transpose, memory_format = torch.contiguous_format);  transpose = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:499 in scatter_to_sequence_parallel_region, code: return _ScatterToSequenceParallelRegion.apply(input_)
        slice_4: "f32[192, 8, 768]" = torch.ops.aten.slice.Tensor(clone_1, 0, 576, 768);  clone_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:140 in forward, code: embeddings = embeddings.clone()
        clone_2: "f32[192, 8, 768]" = torch.ops.aten.clone.default(slice_4);  slice_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:142 in forward, code: embeddings = self.embedding_dropout(embeddings)
        native_dropout = torch.ops.aten.native_dropout.default(clone_2, 0.1, True);  clone_2 = None
        getitem: "f32[192, 8, 768]" = native_dropout[0]
        getitem_1: "b8[192, 8, 768]" = native_dropout[1];  native_dropout = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd = torch.ops.apex.fused_layer_norm_affine_fwd.default(getitem, primals_2, primals_3, [768], 1e-05)
        getitem_2: "f32[192, 8, 768]" = fused_layer_norm_affine_fwd[0]
        getitem_3: "f32[1536]" = fused_layer_norm_affine_fwd[1]
        getitem_4: "f32[1536]" = fused_layer_norm_affine_fwd[2];  fused_layer_norm_affine_fwd = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_2, 8, '42')
        wait_tensor_1: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        t: "f32[768, 288]" = torch.ops.aten.t.default(primals_4);  primals_4 = None
        view_2: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_1, [12288, 768]);  wait_tensor_1 = None
        mm: "f32[12288, 288]" = torch.ops.aten.mm.default(view_2, t);  view_2 = t = None
        view_3: "f32[1536, 8, 288]" = torch.ops.aten.view.default(mm, [1536, 8, 288]);  mm = None
        add_1: "f32[1536, 8, 288]" = torch.ops.aten.add.Tensor(view_3, primals_5);  view_3 = primals_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_4: "f32[1536, 8, 3, 96]" = torch.ops.aten.view.default(add_1, [1536, 8, 3, 96]);  add_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_4, [32, 32, 32], 3);  view_4 = None
        getitem_5: "f32[1536, 8, 3, 32]" = split_with_sizes[0]
        getitem_6: "f32[1536, 8, 3, 32]" = split_with_sizes[1]
        getitem_7: "f32[1536, 8, 3, 32]" = split_with_sizes[2];  split_with_sizes = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_5: "f32[1536, 8, 3, 32]" = torch.ops.aten.view.default(getitem_5, [1536, 8, 3, 32]);  getitem_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_6: "f32[1536, 24, 32]" = torch.ops.aten.view.default(view_5, [1536, 24, 32]);  view_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_7: "f32[1536, 24, 32]" = torch.ops.aten.view.default(getitem_6, [1536, 24, -1]);  getitem_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty: "f32[24, 1536, 1536]" = torch.ops.aten.empty.memory_format([24, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=3), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_1: "f32[24, 1536, 32]" = torch.ops.aten.transpose.int(view_6, 0, 1);  view_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_2: "f32[24, 1536, 32]" = torch.ops.aten.transpose.int(view_7, 0, 1);  view_7 = None
        transpose_3: "f32[24, 32, 1536]" = torch.ops.aten.transpose.int(transpose_2, 1, 2);  transpose_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm: "f32[24, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty, transpose_1, transpose_3, beta = 0.0, alpha = 0.17677669529663687);  empty = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_8: "f32[8, 3, 1536, 1536]" = torch.ops.aten.view.default(baddbmm, [8, 3, 1536, 1536]);  baddbmm = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill: "f32[8, 3, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_8, _to_copy_1, -10000.0);  view_8 = None
        view_9: "f32[24, 1536, 1536]" = torch.ops.aten.view.default(masked_fill, [24, 1536, 1536]);  masked_fill = None
        view_10: "f32[8, 3, 1536, 1536]" = torch.ops.aten.view.default(view_9, [8, 3, 1536, 1536]);  view_9 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax: "f32[8, 3, 1536, 1536]" = torch.ops.aten._softmax.default(view_10, -1, False);  view_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_1 = torch.ops.aten.native_dropout.default(_softmax, 0.1, True)
        getitem_8: "f32[8, 3, 1536, 1536]" = native_dropout_1[0]
        getitem_9: "b8[8, 3, 1536, 1536]" = native_dropout_1[1];  native_dropout_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_12: "f32[1536, 24, 32]" = torch.ops.aten.view.default(getitem_7, [1536, 24, -1]);  getitem_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_13: "f32[24, 1536, 1536]" = torch.ops.aten.view.default(getitem_8, [24, 1536, -1]);  getitem_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_4: "f32[24, 1536, 32]" = torch.ops.aten.transpose.int(view_12, 0, 1);  view_12 = None
        bmm: "f32[24, 1536, 32]" = torch.ops.aten.bmm.default(view_13, transpose_4)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_14: "f32[8, 3, 1536, 32]" = torch.ops.aten.view.default(bmm, [8, 3, 1536, 32]);  bmm = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute: "f32[1536, 8, 3, 32]" = torch.ops.aten.permute.default(view_14, [2, 0, 1, 3]);  view_14 = None
        clone_3: "f32[1536, 8, 3, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_15: "f32[1536, 8, 96]" = torch.ops.aten.view.default(clone_3, [1536, 8, 96]);  clone_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_1: "f32[96, 768]" = torch.ops.aten.t.default(primals_6)
        view_16: "f32[12288, 96]" = torch.ops.aten.view.default(view_15, [12288, 96])
        mm_1: "f32[12288, 768]" = torch.ops.aten.mm.default(view_16, t_1);  view_16 = t_1 = None
        view_17: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_1, [1536, 8, 768]);  mm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_1: "f32[192, 8, 768]" = torch.ops.aten.empty.memory_format([192, 8, 768], dtype = torch.float32, device = device(type='cuda', index=3), pin_memory = False)
        reduce_scatter_tensor: "f32[192, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_17, 'sum', 8, '42');  view_17 = None
        wait_tensor_2: "f32[192, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        copy_1: "f32[192, 8, 768]" = torch.ops.aten.copy.default(empty_1, wait_tensor_2);  empty_1 = wait_tensor_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_2: "f32[192, 8, 768]" = torch.ops.aten.add.Tensor(copy_1, primals_7);  copy_1 = primals_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_2 = torch.ops.aten.native_dropout.default(add_2, 0.1, True);  add_2 = None
        getitem_10: "f32[192, 8, 768]" = native_dropout_2[0]
        getitem_11: "b8[192, 8, 768]" = native_dropout_2[1];  native_dropout_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_3: "f32[192, 8, 768]" = torch.ops.aten.add.Tensor(getitem, getitem_10);  getitem_10 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_1 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_3, primals_8, primals_9, [768], 1e-05)
        getitem_12: "f32[192, 8, 768]" = fused_layer_norm_affine_fwd_1[0]
        getitem_13: "f32[1536]" = fused_layer_norm_affine_fwd_1[1]
        getitem_14: "f32[1536]" = fused_layer_norm_affine_fwd_1[2];  fused_layer_norm_affine_fwd_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_1: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_12, 8, '42')
        wait_tensor_3: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        t_2: "f32[768, 384]" = torch.ops.aten.t.default(primals_10);  primals_10 = None
        view_18: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_3, [12288, 768]);  wait_tensor_3 = None
        mm_2: "f32[12288, 384]" = torch.ops.aten.mm.default(view_18, t_2);  view_18 = t_2 = None
        view_19: "f32[1536, 8, 384]" = torch.ops.aten.view.default(mm_2, [1536, 8, 384]);  mm_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_4: "f32[1536, 8, 384]" = torch.ops.aten.add.Tensor(view_19, primals_11);  view_19 = primals_11 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu: "f32[1536, 8, 384]" = torch.ops.aten.gelu.default(add_4)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_3: "f32[384, 768]" = torch.ops.aten.t.default(primals_12)
        view_20: "f32[12288, 384]" = torch.ops.aten.view.default(gelu, [12288, 384])
        mm_3: "f32[12288, 768]" = torch.ops.aten.mm.default(view_20, t_3);  view_20 = t_3 = None
        view_21: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_3, [1536, 8, 768]);  mm_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:519 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_2: "f32[192, 8, 768]" = torch.ops.aten.empty.memory_format([192, 8, 768], dtype = torch.float32, device = device(type='cuda', index=3), pin_memory = False)
        reduce_scatter_tensor_1: "f32[192, 8, 768]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(view_21, 'sum', 8, '42');  view_21 = None
        wait_tensor_4: "f32[192, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        copy_2: "f32[192, 8, 768]" = torch.ops.aten.copy.default(empty_2, wait_tensor_4);  empty_2 = wait_tensor_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_5: "f32[192, 8, 768]" = torch.ops.aten.add.Tensor(copy_2, primals_13);  copy_2 = primals_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_3 = torch.ops.aten.native_dropout.default(add_5, 0.1, True);  add_5 = None
        getitem_15: "f32[192, 8, 768]" = native_dropout_3[0]
        getitem_16: "b8[192, 8, 768]" = native_dropout_3[1];  native_dropout_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_6: "f32[192, 8, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_15);  getitem_15 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_2 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_6, primals_14, primals_15, [768], 1e-05)
        getitem_17: "f32[192, 8, 768]" = fused_layer_norm_affine_fwd_2[0]
        getitem_18: "f32[1536]" = fused_layer_norm_affine_fwd_2[1]
        getitem_19: "f32[1536]" = fused_layer_norm_affine_fwd_2[2];  fused_layer_norm_affine_fwd_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_2: "f32[1536, 8, 768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_17, 8, '42')
        wait_tensor_5: "f32[1536, 8, 768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        t_4: "f32[768, 12288]" = torch.ops.aten.t.default(primals_16);  primals_16 = None
        view_22: "f32[12288, 768]" = torch.ops.aten.view.default(wait_tensor_5, [12288, 768]);  wait_tensor_5 = None
        mm_4: "f32[12288, 12288]" = torch.ops.aten.mm.default(view_22, t_4);  view_22 = t_4 = None
        view_23: "f32[1536, 8, 12288]" = torch.ops.aten.view.default(mm_4, [1536, 8, 12288]);  mm_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/language_module/language_module.py:77 in compute_language_model_loss, code: labels = labels.transpose(0, 1).contiguous()
        transpose_5: "i64[1536, 8]" = torch.ops.aten.transpose.int(_to_copy_3, 0, 1);  _to_copy_3 = None
        clone_4: "i64[1536, 8]" = torch.ops.aten.clone.default(transpose_5, memory_format = torch.contiguous_format);  transpose_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:245 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        max_1 = torch.ops.aten.max.dim(view_23, -1)
        getitem_20: "f32[1536, 8]" = max_1[0];  max_1 = None
        all_reduce_1: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_20, 'max', '42')
        wait_tensor_6: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        copy_3: "f32[1536, 8]" = torch.ops.aten.copy.default(getitem_20, wait_tensor_6);  getitem_20 = wait_tensor_6 = None
        clone_5: "f32[1536, 8, 12288]" = torch.ops.aten.clone.default(view_23);  view_23 = None
        unsqueeze_1: "f32[1536, 8, 1]" = torch.ops.aten.unsqueeze.default(copy_3, -1);  copy_3 = None
        sub_1: "f32[1536, 8, 12288]" = torch.ops.aten.sub.Tensor(clone_5, unsqueeze_1);  clone_5 = unsqueeze_1 = None
        lt_1: "b8[1536, 8]" = torch.ops.aten.lt.Scalar(clone_4, 36864)
        ge_1: "b8[1536, 8]" = torch.ops.aten.ge.Scalar(clone_4, 49152)
        bitwise_or_1: "b8[1536, 8]" = torch.ops.aten.bitwise_or.Tensor(lt_1, ge_1);  lt_1 = ge_1 = None
        clone_6: "i64[1536, 8]" = torch.ops.aten.clone.default(clone_4);  clone_4 = None
        sub_2: "i64[1536, 8]" = torch.ops.aten.sub.Tensor(clone_6, 36864);  clone_6 = None
        _tensor_constant2 = self._tensor_constant2
        lift_fresh_copy_2: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        index_put_2: "i64[1536, 8]" = torch.ops.aten.index_put.default(sub_2, [bitwise_or_1], lift_fresh_copy_2);  sub_2 = lift_fresh_copy_2 = None
        arange: "i64[12288]" = torch.ops.aten.arange.start(0, 12288, device = device(type='cuda', index=3), pin_memory = False)
        view_26: "f32[12288, 12288]" = torch.ops.aten.view.default(sub_1, [-1, 12288])
        view_27: "i64[12288]" = torch.ops.aten.view.default(index_put_2, [-1]);  index_put_2 = None
        index_1: "f32[12288]" = torch.ops.aten.index.Tensor(view_26, [arange, view_27]);  view_26 = arange = None
        clone_7: "f32[12288]" = torch.ops.aten.clone.default(index_1);  index_1 = None
        view_28: "f32[1536, 8]" = torch.ops.aten.view.default(clone_7, [1536, 8]);  clone_7 = None
        _tensor_constant3 = self._tensor_constant3
        lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        index_put_3: "f32[1536, 8]" = torch.ops.aten.index_put.default(view_28, [bitwise_or_1], lift_fresh_copy_3);  view_28 = lift_fresh_copy_3 = None
        view_29: "f32[12288]" = torch.ops.aten.view.default(index_put_3, [12288]);  index_put_3 = None
        view_30: "f32[1536, 8]" = torch.ops.aten.view.default(view_29, [1536, 8]);  view_29 = None
        exp: "f32[1536, 8, 12288]" = torch.ops.aten.exp.default(sub_1)
        copy_4: "f32[1536, 8, 12288]" = torch.ops.aten.copy.default(sub_1, exp);  sub_1 = exp = None
        sum_1: "f32[1536, 8]" = torch.ops.aten.sum.dim_IntList(copy_4, [-1])
        all_reduce_2: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(view_30, 'sum', '42')
        wait_tensor_7: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = None
        copy_5: "f32[1536, 8]" = torch.ops.aten.copy.default(view_30, wait_tensor_7);  view_30 = wait_tensor_7 = None
        view_31: "f32[12288]" = torch.ops.aten.view.default(copy_5, [12288]);  copy_5 = None
        view_32: "f32[1536, 8]" = torch.ops.aten.view.default(view_31, [1536, 8]);  view_31 = None
        all_reduce_3: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(sum_1, 'sum', '42')
        wait_tensor_8: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = None
        copy_6: "f32[1536, 8]" = torch.ops.aten.copy.default(sum_1, wait_tensor_8);  sum_1 = wait_tensor_8 = None
        log: "f32[1536, 8]" = torch.ops.aten.log.default(copy_6)
        sub_3: "f32[1536, 8]" = torch.ops.aten.sub.Tensor(log, view_32);  log = view_32 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_6: "f32[8, 1536]" = torch.ops.aten.transpose.int(sub_3, 0, 1);  sub_3 = None
        clone_8: "f32[8, 1536]" = torch.ops.aten.clone.default(transpose_6, memory_format = torch.contiguous_format);  transpose_6 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:98 in loss_func, code: loss_mask = loss_mask.view(-1).float()
        view_33: "f32[12288]" = torch.ops.aten.view.default(_to_copy_4, [-1]);  _to_copy_4 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:99 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        view_34: "f32[12288]" = torch.ops.aten.view.default(clone_8, [-1]);  clone_8 = None
        mul: "f32[12288]" = torch.ops.aten.mul.Tensor(view_34, view_33);  view_34 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(view_33)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_2, sum_3);  sum_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:295 in forward_step, code: output_tensor /= num_microbatches
        div_2: "f32[]" = torch.ops.aten.div.Tensor(div_1, 1);  div_1 = None
        return [div_2, div_2, zeros, primals_2, primals_3, primals_6, primals_8, primals_9, primals_12, primals_14, primals_15, _to_copy_1, _to_copy_2, bitwise_or, index_put, getitem, getitem_1, getitem_2, getitem_3, getitem_4, transpose_1, transpose_3, _softmax, getitem_9, view_13, transpose_4, view_15, getitem_11, add_3, getitem_12, getitem_13, getitem_14, add_4, gelu, getitem_16, add_6, getitem_17, getitem_18, getitem_19, bitwise_or_1, view_27, copy_4, copy_6, view_33, sum_3]
        

