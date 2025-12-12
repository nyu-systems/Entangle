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


# Graph[rank=0](fw, gid=0)
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[98304, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[2304, 768]", primals_5: "f32[2304]", primals_6: "f32[768, 768]", primals_7: "f32[768]", primals_8: "f32[768]", primals_9: "f32[768]", primals_10: "f32[3072, 768]", primals_11: "f32[3072]", primals_12: "f32[768, 3072]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[2304, 768]", primals_17: "f32[2304]", primals_18: "f32[768, 768]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[3072, 768]", primals_23: "f32[3072]", primals_24: "f32[768, 3072]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[768]", primals_28: "f32[98304, 768]", primals_29: "f32[1536, 768]", primals_30: "i64[8, 1536]", primals_31: "b8[8, 1, 1536, 1536]", primals_32: "i64[8, 1536]", primals_33: "i64[8, 1536]", primals_34: "f32[8, 1536]"):
        # File: /opt/tiger/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:456 in forward_backward_no_pipelining, code: total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        zeros: "i32[]" = torch.ops.aten.zeros.default([], dtype = torch.int32, device = device(type='cuda'), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:106 in forward_step_func, code: tokens = data['tokens'].to(device)
        _to_copy: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_30, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  primals_30 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:107 in forward_step_func, code: attention_mask = data['attention_mask'].to(device)
        _to_copy_1: "b8[8, 1, 1536, 1536]" = torch.ops.aten._to_copy.default(primals_31, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0));  primals_31 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:108 in forward_step_func, code: position_ids = data['position_ids'].to(device)
        _to_copy_2: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_32, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  primals_32 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:109 in forward_step_func, code: labels = data['labels'].to(device)
        _to_copy_3: "i64[8, 1536]" = torch.ops.aten._to_copy.default(primals_33, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0));  primals_33 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:110 in forward_step_func, code: loss_mask = data['loss_mask'].to(device)
        _to_copy_4: "f32[8, 1536]" = torch.ops.aten._to_copy.default(primals_34, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  primals_34 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:260 in forward, code: output_parallel = self.weight[masked_input]
        index: "f32[8, 1536, 768]" = torch.ops.aten.index.Tensor(primals_1, [_to_copy]);  primals_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:484 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        view: "f32[8, 1536, 768]" = torch.ops.aten.view.default(index, [8, 1536, 768]);  index = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding: "f32[8, 1536, 768]" = torch.ops.aten.embedding.default(primals_29, _to_copy_2);  primals_29 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:112 in forward, code: embeddings = word_embeddings + position_embeddings
        add: "f32[8, 1536, 768]" = torch.ops.aten.add.Tensor(view, embedding);  view = embedding = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:118 in forward, code: embeddings = embeddings.transpose(0, 1).contiguous()
        transpose: "f32[1536, 8, 768]" = torch.ops.aten.transpose.int(add, 0, 1);  add = None
        clone: "f32[1536, 8, 768]" = torch.ops.aten.clone.default(transpose, memory_format = torch.contiguous_format);  transpose = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:147 in forward, code: embeddings = self.embedding_dropout(embeddings)
        native_dropout = torch.ops.aten.native_dropout.default(clone, 0.1, True);  clone = None
        getitem: "f32[1536, 8, 768]" = native_dropout[0]
        getitem_1: "b8[1536, 8, 768]" = native_dropout[1];  native_dropout = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd = torch.ops.apex.fused_layer_norm_affine_fwd.default(getitem, primals_2, primals_3, [768], 1e-05)
        getitem_2: "f32[1536, 8, 768]" = fused_layer_norm_affine_fwd[0]
        getitem_3: "f32[12288]" = fused_layer_norm_affine_fwd[1]
        getitem_4: "f32[12288]" = fused_layer_norm_affine_fwd[2];  fused_layer_norm_affine_fwd = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:479 in copy_to_tensor_model_parallel_region, code: return _CopyToModelParallelRegion.apply(input_)
        view_1: "f32[1536, 8, 768]" = torch.ops.aten.view.default(getitem_2, [1536, 8, 768]);  getitem_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t: "f32[768, 2304]" = torch.ops.aten.t.default(primals_4)
        view_2: "f32[12288, 768]" = torch.ops.aten.view.default(view_1, [12288, 768])
        mm: "f32[12288, 2304]" = torch.ops.aten.mm.default(view_2, t);  view_2 = t = None
        view_3: "f32[1536, 8, 2304]" = torch.ops.aten.view.default(mm, [1536, 8, 2304]);  mm = None
        add_1: "f32[1536, 8, 2304]" = torch.ops.aten.add.Tensor(view_3, primals_5);  view_3 = primals_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_4: "f32[1536, 8, 24, 96]" = torch.ops.aten.view.default(add_1, [1536, 8, 24, 96]);  add_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_4, [32, 32, 32], 3);  view_4 = None
        getitem_5: "f32[1536, 8, 24, 32]" = split_with_sizes[0]
        getitem_6: "f32[1536, 8, 24, 32]" = split_with_sizes[1]
        getitem_7: "f32[1536, 8, 24, 32]" = split_with_sizes[2];  split_with_sizes = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_5: "f32[1536, 8, 24, 32]" = torch.ops.aten.view.default(getitem_5, [1536, 8, 24, 32]);  getitem_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_6: "f32[1536, 192, 32]" = torch.ops.aten.view.default(view_5, [1536, 192, 32]);  view_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_7: "f32[1536, 192, 32]" = torch.ops.aten.view.default(getitem_6, [1536, 192, -1]);  getitem_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty: "f32[192, 1536, 1536]" = torch.ops.aten.empty.memory_format([192, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_1: "f32[192, 1536, 32]" = torch.ops.aten.transpose.int(view_6, 0, 1);  view_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_2: "f32[192, 1536, 32]" = torch.ops.aten.transpose.int(view_7, 0, 1);  view_7 = None
        transpose_3: "f32[192, 32, 1536]" = torch.ops.aten.transpose.int(transpose_2, 1, 2);  transpose_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm: "f32[192, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty, transpose_1, transpose_3, beta = 0.0, alpha = 0.17677669529663687);  empty = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_8: "f32[8, 24, 1536, 1536]" = torch.ops.aten.view.default(baddbmm, [8, 24, 1536, 1536]);  baddbmm = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill: "f32[8, 24, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_8, _to_copy_1, -10000.0);  view_8 = None
        view_9: "f32[192, 1536, 1536]" = torch.ops.aten.view.default(masked_fill, [192, 1536, 1536]);  masked_fill = None
        view_10: "f32[8, 24, 1536, 1536]" = torch.ops.aten.view.default(view_9, [8, 24, 1536, 1536]);  view_9 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax: "f32[8, 24, 1536, 1536]" = torch.ops.aten._softmax.default(view_10, -1, False);  view_10 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_1 = torch.ops.aten.native_dropout.default(_softmax, 0.1, True)
        getitem_8: "f32[8, 24, 1536, 1536]" = native_dropout_1[0]
        getitem_9: "b8[8, 24, 1536, 1536]" = native_dropout_1[1];  native_dropout_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_12: "f32[1536, 192, 32]" = torch.ops.aten.view.default(getitem_7, [1536, 192, -1]);  getitem_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_13: "f32[192, 1536, 1536]" = torch.ops.aten.view.default(getitem_8, [192, 1536, -1]);  getitem_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_4: "f32[192, 1536, 32]" = torch.ops.aten.transpose.int(view_12, 0, 1);  view_12 = None
        bmm: "f32[192, 1536, 32]" = torch.ops.aten.bmm.default(view_13, transpose_4)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_14: "f32[8, 24, 1536, 32]" = torch.ops.aten.view.default(bmm, [8, 24, 1536, 32]);  bmm = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute: "f32[1536, 8, 24, 32]" = torch.ops.aten.permute.default(view_14, [2, 0, 1, 3]);  view_14 = None
        clone_1: "f32[1536, 8, 24, 32]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_15: "f32[1536, 8, 768]" = torch.ops.aten.view.default(clone_1, [1536, 8, 768]);  clone_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_1: "f32[768, 768]" = torch.ops.aten.t.default(primals_6)
        view_16: "f32[12288, 768]" = torch.ops.aten.view.default(view_15, [12288, 768])
        mm_1: "f32[12288, 768]" = torch.ops.aten.mm.default(view_16, t_1);  view_16 = t_1 = None
        view_17: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_1, [1536, 8, 768]);  mm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:484 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        view_18: "f32[1536, 8, 768]" = torch.ops.aten.view.default(view_17, [1536, 8, 768]);  view_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_2: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(view_18, primals_7);  view_18 = primals_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_2 = torch.ops.aten.native_dropout.default(add_2, 0.1, True);  add_2 = None
        getitem_10: "f32[1536, 8, 768]" = native_dropout_2[0]
        getitem_11: "b8[1536, 8, 768]" = native_dropout_2[1];  native_dropout_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_3: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(getitem, getitem_10);  getitem_10 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_1 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_3, primals_8, primals_9, [768], 1e-05)
        getitem_12: "f32[1536, 8, 768]" = fused_layer_norm_affine_fwd_1[0]
        getitem_13: "f32[12288]" = fused_layer_norm_affine_fwd_1[1]
        getitem_14: "f32[12288]" = fused_layer_norm_affine_fwd_1[2];  fused_layer_norm_affine_fwd_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:479 in copy_to_tensor_model_parallel_region, code: return _CopyToModelParallelRegion.apply(input_)
        view_19: "f32[1536, 8, 768]" = torch.ops.aten.view.default(getitem_12, [1536, 8, 768]);  getitem_12 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_2: "f32[768, 3072]" = torch.ops.aten.t.default(primals_10)
        view_20: "f32[12288, 768]" = torch.ops.aten.view.default(view_19, [12288, 768])
        mm_2: "f32[12288, 3072]" = torch.ops.aten.mm.default(view_20, t_2);  view_20 = t_2 = None
        view_21: "f32[1536, 8, 3072]" = torch.ops.aten.view.default(mm_2, [1536, 8, 3072]);  mm_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_4: "f32[1536, 8, 3072]" = torch.ops.aten.add.Tensor(view_21, primals_11);  view_21 = primals_11 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu: "f32[1536, 8, 3072]" = torch.ops.aten.gelu.default(add_4)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_3: "f32[3072, 768]" = torch.ops.aten.t.default(primals_12)
        view_22: "f32[12288, 3072]" = torch.ops.aten.view.default(gelu, [12288, 3072])
        mm_3: "f32[12288, 768]" = torch.ops.aten.mm.default(view_22, t_3);  view_22 = t_3 = None
        view_23: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_3, [1536, 8, 768]);  mm_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:484 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        view_24: "f32[1536, 8, 768]" = torch.ops.aten.view.default(view_23, [1536, 8, 768]);  view_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_5: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(view_24, primals_13);  view_24 = primals_13 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_3 = torch.ops.aten.native_dropout.default(add_5, 0.1, True);  add_5 = None
        getitem_15: "f32[1536, 8, 768]" = native_dropout_3[0]
        getitem_16: "b8[1536, 8, 768]" = native_dropout_3[1];  native_dropout_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_6: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(add_3, getitem_15);  getitem_15 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_2 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_6, primals_14, primals_15, [768], 1e-05)
        getitem_17: "f32[1536, 8, 768]" = fused_layer_norm_affine_fwd_2[0]
        getitem_18: "f32[12288]" = fused_layer_norm_affine_fwd_2[1]
        getitem_19: "f32[12288]" = fused_layer_norm_affine_fwd_2[2];  fused_layer_norm_affine_fwd_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:479 in copy_to_tensor_model_parallel_region, code: return _CopyToModelParallelRegion.apply(input_)
        view_25: "f32[1536, 8, 768]" = torch.ops.aten.view.default(getitem_17, [1536, 8, 768]);  getitem_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_4: "f32[768, 2304]" = torch.ops.aten.t.default(primals_16)
        view_26: "f32[12288, 768]" = torch.ops.aten.view.default(view_25, [12288, 768])
        mm_4: "f32[12288, 2304]" = torch.ops.aten.mm.default(view_26, t_4);  view_26 = t_4 = None
        view_27: "f32[1536, 8, 2304]" = torch.ops.aten.view.default(mm_4, [1536, 8, 2304]);  mm_4 = None
        add_7: "f32[1536, 8, 2304]" = torch.ops.aten.add.Tensor(view_27, primals_17);  view_27 = primals_17 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_28: "f32[1536, 8, 24, 96]" = torch.ops.aten.view.default(add_7, [1536, 8, 24, 96]);  add_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(view_28, [32, 32, 32], 3);  view_28 = None
        getitem_20: "f32[1536, 8, 24, 32]" = split_with_sizes_1[0]
        getitem_21: "f32[1536, 8, 24, 32]" = split_with_sizes_1[1]
        getitem_22: "f32[1536, 8, 24, 32]" = split_with_sizes_1[2];  split_with_sizes_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_29: "f32[1536, 8, 24, 32]" = torch.ops.aten.view.default(getitem_20, [1536, 8, 24, 32]);  getitem_20 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_30: "f32[1536, 192, 32]" = torch.ops.aten.view.default(view_29, [1536, 192, 32]);  view_29 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_31: "f32[1536, 192, 32]" = torch.ops.aten.view.default(getitem_21, [1536, 192, -1]);  getitem_21 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty_1: "f32[192, 1536, 1536]" = torch.ops.aten.empty.memory_format([192, 1536, 1536], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_5: "f32[192, 1536, 32]" = torch.ops.aten.transpose.int(view_30, 0, 1);  view_30 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_6: "f32[192, 1536, 32]" = torch.ops.aten.transpose.int(view_31, 0, 1);  view_31 = None
        transpose_7: "f32[192, 32, 1536]" = torch.ops.aten.transpose.int(transpose_6, 1, 2);  transpose_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm_1: "f32[192, 1536, 1536]" = torch.ops.aten.baddbmm.default(empty_1, transpose_5, transpose_7, beta = 0.0, alpha = 0.17677669529663687);  empty_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_32: "f32[8, 24, 1536, 1536]" = torch.ops.aten.view.default(baddbmm_1, [8, 24, 1536, 1536]);  baddbmm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill_1: "f32[8, 24, 1536, 1536]" = torch.ops.aten.masked_fill.Scalar(view_32, _to_copy_1, -10000.0);  view_32 = None
        view_33: "f32[192, 1536, 1536]" = torch.ops.aten.view.default(masked_fill_1, [192, 1536, 1536]);  masked_fill_1 = None
        view_34: "f32[8, 24, 1536, 1536]" = torch.ops.aten.view.default(view_33, [8, 24, 1536, 1536]);  view_33 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/nn/modules/activation.py:1554 in forward, code: return F.softmax(input, self.dim, _stacklevel=5)
        _softmax_1: "f32[8, 24, 1536, 1536]" = torch.ops.aten._softmax.default(view_34, -1, False);  view_34 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:181 in forward, code: attention_probs = self.attention_dropout(attention_probs)
        native_dropout_4 = torch.ops.aten.native_dropout.default(_softmax_1, 0.1, True)
        getitem_23: "f32[8, 24, 1536, 1536]" = native_dropout_4[0]
        getitem_24: "b8[8, 24, 1536, 1536]" = native_dropout_4[1];  native_dropout_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_36: "f32[1536, 192, 32]" = torch.ops.aten.view.default(getitem_22, [1536, 192, -1]);  getitem_22 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_37: "f32[192, 1536, 1536]" = torch.ops.aten.view.default(getitem_23, [192, 1536, -1]);  getitem_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_8: "f32[192, 1536, 32]" = torch.ops.aten.transpose.int(view_36, 0, 1);  view_36 = None
        bmm_1: "f32[192, 1536, 32]" = torch.ops.aten.bmm.default(view_37, transpose_8)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_38: "f32[8, 24, 1536, 32]" = torch.ops.aten.view.default(bmm_1, [8, 24, 1536, 32]);  bmm_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_1: "f32[1536, 8, 24, 32]" = torch.ops.aten.permute.default(view_38, [2, 0, 1, 3]);  view_38 = None
        clone_2: "f32[1536, 8, 24, 32]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_39: "f32[1536, 8, 768]" = torch.ops.aten.view.default(clone_2, [1536, 8, 768]);  clone_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_5: "f32[768, 768]" = torch.ops.aten.t.default(primals_18)
        view_40: "f32[12288, 768]" = torch.ops.aten.view.default(view_39, [12288, 768])
        mm_5: "f32[12288, 768]" = torch.ops.aten.mm.default(view_40, t_5);  view_40 = t_5 = None
        view_41: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_5, [1536, 8, 768]);  mm_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:484 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        view_42: "f32[1536, 8, 768]" = torch.ops.aten.view.default(view_41, [1536, 8, 768]);  view_41 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_8: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(view_42, primals_19);  view_42 = primals_19 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_5 = torch.ops.aten.native_dropout.default(add_8, 0.1, True);  add_8 = None
        getitem_25: "f32[1536, 8, 768]" = native_dropout_5[0]
        getitem_26: "b8[1536, 8, 768]" = native_dropout_5[1];  native_dropout_5 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_9: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(add_6, getitem_25);  getitem_25 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_3 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_9, primals_20, primals_21, [768], 1e-05)
        getitem_27: "f32[1536, 8, 768]" = fused_layer_norm_affine_fwd_3[0]
        getitem_28: "f32[12288]" = fused_layer_norm_affine_fwd_3[1]
        getitem_29: "f32[12288]" = fused_layer_norm_affine_fwd_3[2];  fused_layer_norm_affine_fwd_3 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:479 in copy_to_tensor_model_parallel_region, code: return _CopyToModelParallelRegion.apply(input_)
        view_43: "f32[1536, 8, 768]" = torch.ops.aten.view.default(getitem_27, [1536, 8, 768]);  getitem_27 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_6: "f32[768, 3072]" = torch.ops.aten.t.default(primals_22)
        view_44: "f32[12288, 768]" = torch.ops.aten.view.default(view_43, [12288, 768])
        mm_6: "f32[12288, 3072]" = torch.ops.aten.mm.default(view_44, t_6);  view_44 = t_6 = None
        view_45: "f32[1536, 8, 3072]" = torch.ops.aten.view.default(mm_6, [1536, 8, 3072]);  mm_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_10: "f32[1536, 8, 3072]" = torch.ops.aten.add.Tensor(view_45, primals_23);  view_45 = primals_23 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_1: "f32[1536, 8, 3072]" = torch.ops.aten.gelu.default(add_10)
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_7: "f32[3072, 768]" = torch.ops.aten.t.default(primals_24)
        view_46: "f32[12288, 3072]" = torch.ops.aten.view.default(gelu_1, [12288, 3072])
        mm_7: "f32[12288, 768]" = torch.ops.aten.mm.default(view_46, t_7);  view_46 = t_7 = None
        view_47: "f32[1536, 8, 768]" = torch.ops.aten.view.default(mm_7, [1536, 8, 768]);  mm_7 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:484 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        view_48: "f32[1536, 8, 768]" = torch.ops.aten.view.default(view_47, [1536, 8, 768]);  view_47 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:31 in _bias_dropout_add_func, code: x = x + bias
        add_11: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(view_48, primals_25);  view_48 = primals_25 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:32 in _bias_dropout_add_func, code: out = torch.nn.functional.dropout(x, p=prob, training=training)
        native_dropout_6 = torch.ops.aten.native_dropout.default(add_11, 0.1, True);  add_11 = None
        getitem_30: "f32[1536, 8, 768]" = native_dropout_6[0]
        getitem_31: "b8[1536, 8, 768]" = native_dropout_6[1];  native_dropout_6 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: out = residual + out
        add_12: "f32[1536, 8, 768]" = torch.ops.aten.add.Tensor(add_9, getitem_30);  getitem_30 = None
        
        # File: /opt/tiger/miniconda3/lib/python3.12/site-packages/torch/_library/custom_ops.py:506 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_4 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_12, primals_26, primals_27, [768], 1e-05)
        getitem_32: "f32[1536, 8, 768]" = fused_layer_norm_affine_fwd_4[0]
        getitem_33: "f32[12288]" = fused_layer_norm_affine_fwd_4[1]
        getitem_34: "f32[12288]" = fused_layer_norm_affine_fwd_4[2];  fused_layer_norm_affine_fwd_4 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/mappings.py:479 in copy_to_tensor_model_parallel_region, code: return _CopyToModelParallelRegion.apply(input_)
        view_49: "f32[1536, 8, 768]" = torch.ops.aten.view.default(getitem_32, [1536, 8, 768]);  getitem_32 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/layers.py:697 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        t_8: "f32[768, 98304]" = torch.ops.aten.t.default(primals_28)
        view_50: "f32[12288, 768]" = torch.ops.aten.view.default(view_49, [12288, 768])
        mm_8: "f32[12288, 98304]" = torch.ops.aten.mm.default(view_50, t_8);  view_50 = t_8 = None
        view_51: "f32[1536, 8, 98304]" = torch.ops.aten.view.default(mm_8, [1536, 8, 98304]);  mm_8 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/language_module/language_module.py:77 in compute_language_model_loss, code: labels = labels.transpose(0, 1).contiguous()
        transpose_9: "i64[1536, 8]" = torch.ops.aten.transpose.int(_to_copy_3, 0, 1);  _to_copy_3 = None
        clone_3: "i64[1536, 8]" = torch.ops.aten.clone.default(transpose_9, memory_format = torch.contiguous_format);  transpose_9 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:245 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        max_1 = torch.ops.aten.max.dim(view_51, -1)
        getitem_35: "f32[1536, 8]" = max_1[0];  max_1 = None
        all_reduce: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(getitem_35, 'max', '7')
        wait_tensor: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        copy: "f32[1536, 8]" = torch.ops.aten.copy.default(getitem_35, wait_tensor);  getitem_35 = wait_tensor = None
        clone_4: "f32[1536, 8, 98304]" = torch.ops.aten.clone.default(view_51);  view_51 = None
        unsqueeze_1: "f32[1536, 8, 1]" = torch.ops.aten.unsqueeze.default(copy, -1);  copy = None
        sub: "f32[1536, 8, 98304]" = torch.ops.aten.sub.Tensor(clone_4, unsqueeze_1);  clone_4 = unsqueeze_1 = None
        lt: "b8[1536, 8]" = torch.ops.aten.lt.Scalar(clone_3, 0)
        ge: "b8[1536, 8]" = torch.ops.aten.ge.Scalar(clone_3, 98304)
        bitwise_or: "b8[1536, 8]" = torch.ops.aten.bitwise_or.Tensor(lt, ge);  lt = ge = None
        clone_5: "i64[1536, 8]" = torch.ops.aten.clone.default(clone_3);  clone_3 = None
        sub_1: "i64[1536, 8]" = torch.ops.aten.sub.Tensor(clone_5, 0);  clone_5 = None
        _tensor_constant0 = self._tensor_constant0
        lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        index_put: "i64[1536, 8]" = torch.ops.aten.index_put.default(sub_1, [bitwise_or], lift_fresh_copy);  sub_1 = lift_fresh_copy = None
        arange: "i64[12288]" = torch.ops.aten.arange.start(0, 12288, device = device(type='cuda', index=0), pin_memory = False)
        view_54: "f32[12288, 98304]" = torch.ops.aten.view.default(sub, [-1, 98304])
        view_55: "i64[12288]" = torch.ops.aten.view.default(index_put, [-1]);  index_put = None
        index_1: "f32[12288]" = torch.ops.aten.index.Tensor(view_54, [arange, view_55]);  view_54 = arange = None
        clone_6: "f32[12288]" = torch.ops.aten.clone.default(index_1);  index_1 = None
        view_56: "f32[1536, 8]" = torch.ops.aten.view.default(clone_6, [1536, 8]);  clone_6 = None
        _tensor_constant1 = self._tensor_constant1
        lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        index_put_1: "f32[1536, 8]" = torch.ops.aten.index_put.default(view_56, [bitwise_or], lift_fresh_copy_1);  view_56 = lift_fresh_copy_1 = None
        view_57: "f32[12288]" = torch.ops.aten.view.default(index_put_1, [12288]);  index_put_1 = None
        view_58: "f32[1536, 8]" = torch.ops.aten.view.default(view_57, [1536, 8]);  view_57 = None
        exp: "f32[1536, 8, 98304]" = torch.ops.aten.exp.default(sub)
        copy_1: "f32[1536, 8, 98304]" = torch.ops.aten.copy.default(sub, exp);  sub = exp = None
        sum_1: "f32[1536, 8]" = torch.ops.aten.sum.dim_IntList(copy_1, [-1])
        all_reduce_1: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(view_58, 'sum', '7')
        wait_tensor_1: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        copy_2: "f32[1536, 8]" = torch.ops.aten.copy.default(view_58, wait_tensor_1);  view_58 = wait_tensor_1 = None
        view_59: "f32[12288]" = torch.ops.aten.view.default(copy_2, [12288]);  copy_2 = None
        view_60: "f32[1536, 8]" = torch.ops.aten.view.default(view_59, [1536, 8]);  view_59 = None
        all_reduce_2: "f32[1536, 8]" = torch.ops._c10d_functional.all_reduce.default(sum_1, 'sum', '7')
        wait_tensor_2: "f32[1536, 8]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = None
        copy_3: "f32[1536, 8]" = torch.ops.aten.copy.default(sum_1, wait_tensor_2);  sum_1 = wait_tensor_2 = None
        log: "f32[1536, 8]" = torch.ops.aten.log.default(copy_3)
        sub_2: "f32[1536, 8]" = torch.ops.aten.sub.Tensor(log, view_60);  log = view_60 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_10: "f32[8, 1536]" = torch.ops.aten.transpose.int(sub_2, 0, 1);  sub_2 = None
        clone_7: "f32[8, 1536]" = torch.ops.aten.clone.default(transpose_10, memory_format = torch.contiguous_format);  transpose_10 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:98 in loss_func, code: loss_mask = loss_mask.view(-1).float()
        view_61: "f32[12288]" = torch.ops.aten.view.default(_to_copy_4, [-1]);  _to_copy_4 = None
        
        # File: /opt/tiger/Megatron-LM/examples/simple_gpt.py:99 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        view_62: "f32[12288]" = torch.ops.aten.view.default(clone_7, [-1]);  clone_7 = None
        mul: "f32[12288]" = torch.ops.aten.mul.Tensor(view_62, view_61);  view_62 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(view_61)
        div_1: "f32[]" = torch.ops.aten.div.Tensor(sum_2, sum_3);  sum_2 = None
        
        # File: /opt/tiger/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:295 in forward_step, code: output_tensor /= num_microbatches
        div_2: "f32[]" = torch.ops.aten.div.Tensor(div_1, 1);  div_1 = None
        return [div_2, div_2, zeros, primals_2, primals_3, primals_4, primals_6, primals_8, primals_9, primals_10, primals_12, primals_14, primals_15, primals_16, primals_18, primals_20, primals_21, primals_22, primals_24, primals_26, primals_27, primals_28, _to_copy, _to_copy_1, _to_copy_2, getitem, getitem_1, getitem_3, getitem_4, view_1, transpose_1, transpose_3, _softmax, getitem_9, view_13, transpose_4, view_15, getitem_11, add_3, getitem_13, getitem_14, view_19, add_4, gelu, getitem_16, add_6, getitem_18, getitem_19, view_25, transpose_5, transpose_7, _softmax_1, getitem_24, view_37, transpose_8, view_39, getitem_26, add_9, getitem_28, getitem_29, view_43, add_10, gelu_1, getitem_31, add_12, getitem_33, getitem_34, view_49, bitwise_or, view_55, copy_1, copy_3, view_61, sum_3]
        

