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


# Graph[rank=0](both, gid=0)
class joint_helper(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "i64[32, 1024]"; primals_2: "i64[32, 1024]"; primals_3: "f32[32, 1024]"; primals_4: "b8[32, 1, 1024, 1024]"; primals_5: "i64[32, 1024]"; primals_6: "f16[49152, 1536]"; primals_7: "f16[2048, 1536]"; primals_8: "f16[1536]"; primals_9: "f16[1536]"; primals_10: "f16[1536, 768]"; primals_11: "f16[2304, 1536]"; primals_12: "f16[48]"; primals_13: "f16[48]"; primals_14: "f16[48]"; primals_15: "f16[48]"; primals_16: "f16[1536]"; primals_17: "f16[1536]"; primals_18: "f16[1, 1536]"; primals_19: "f16[3072, 1536]"; primals_20: "f16[1536, 3072]"; primals_21: "f16[1536]"; primals_22: "f16[1536]"; tangents_1: "f16[49152, 1536]"; tangents_2: "f16[2048, 1536]"; tangents_3: "f16[1536]"; tangents_4: "f16[1536]"; tangents_5: "f16[1536, 768]"; tangents_6: "f16[2304, 1536]"; tangents_7: "f16[48]"; tangents_8: "f16[48]"; tangents_9: "f16[48]"; tangents_10: "f16[48]"; tangents_11: "f16[1536]"; tangents_12: "f16[1536]"; tangents_13: "f16[1, 1536]"; tangents_14: "f16[3072, 1536]"; tangents_15: "f16[1536, 3072]"; tangents_16: "f16[1536]"; tangents_17: "f16[1536]"; tangents_18: "f32[1]"; 
    
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        # No stacktrace found for following nodes
        clone: "f16[49152, 1536]" = torch.ops.aten.clone.default(primals_6);  primals_6 = None
        clone_1: "f16[2048, 1536]" = torch.ops.aten.clone.default(primals_7);  primals_7 = None
        clone_2: "f16[1536]" = torch.ops.aten.clone.default(primals_8);  primals_8 = None
        clone_3: "f16[1536]" = torch.ops.aten.clone.default(primals_9);  primals_9 = None
        clone_4: "f16[1536, 768]" = torch.ops.aten.clone.default(primals_10);  primals_10 = None
        clone_5: "f16[2304, 1536]" = torch.ops.aten.clone.default(primals_11);  primals_11 = None
        clone_6: "f16[48]" = torch.ops.aten.clone.default(primals_12);  primals_12 = None
        clone_7: "f16[48]" = torch.ops.aten.clone.default(primals_13);  primals_13 = None
        clone_8: "f16[48]" = torch.ops.aten.clone.default(primals_14);  primals_14 = None
        clone_9: "f16[48]" = torch.ops.aten.clone.default(primals_15);  primals_15 = None
        clone_10: "f16[1536]" = torch.ops.aten.clone.default(primals_16);  primals_16 = None
        clone_11: "f16[1536]" = torch.ops.aten.clone.default(primals_17);  primals_17 = None
        clone_12: "f16[1, 1536]" = torch.ops.aten.clone.default(primals_18);  primals_18 = None
        clone_13: "f16[3072, 1536]" = torch.ops.aten.clone.default(primals_19);  primals_19 = None
        clone_14: "f16[1536, 3072]" = torch.ops.aten.clone.default(primals_20);  primals_20 = None
        clone_15: "f16[1536]" = torch.ops.aten.clone.default(primals_21);  primals_21 = None
        clone_16: "f16[1536]" = torch.ops.aten.clone.default(primals_22);  primals_22 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:459 in forward_backward_no_pipelining, code: total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        zeros: "i32[]" = torch.ops.aten.zeros.default([], dtype = torch.int32, device = device(type='cuda'), pin_memory = False)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:457 in get_batch_on_this_tp_rank, code: tokens = data["tokens"].cuda(non_blocking = True)
        _to_copy: "i64[32, 1024]" = torch.ops.aten._to_copy.default(primals_1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), non_blocking = True);  primals_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:458 in get_batch_on_this_tp_rank, code: labels = data["labels"].cuda(non_blocking = True)
        _to_copy_1: "i64[32, 1024]" = torch.ops.aten._to_copy.default(primals_2, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), non_blocking = True);  primals_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:459 in get_batch_on_this_tp_rank, code: loss_mask = data["loss_mask"].cuda(non_blocking = True)
        _to_copy_2: "f32[32, 1024]" = torch.ops.aten._to_copy.default(primals_3, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), non_blocking = True);  primals_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:460 in get_batch_on_this_tp_rank, code: attention_mask = None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True)
        _to_copy_3: "b8[32, 1, 1024, 1024]" = torch.ops.aten._to_copy.default(primals_4, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), non_blocking = True);  primals_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:461 in get_batch_on_this_tp_rank, code: position_ids = data["position_ids"].cuda(non_blocking = True)
        _to_copy_4: "i64[32, 1024]" = torch.ops.aten._to_copy.default(primals_5, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), non_blocking = True);  primals_5 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        log_tensor: "i64[32, 1024]" = torch.ops.tg.log_tensor.default(_to_copy, 'tokens');  _to_copy = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        log_tensor_1: "i64[32, 1024]" = torch.ops.tg.log_tensor.default(_to_copy_1, 'labels');  _to_copy_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        log_tensor_2: "f32[32, 1024]" = torch.ops.tg.log_tensor.default(_to_copy_2, 'loss_mask');  _to_copy_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        log_tensor_3: "b8[32, 1, 1024, 1024]" = torch.ops.tg.log_tensor.default(_to_copy_3, 'attention_mask');  _to_copy_3 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        log_tensor_4: "i64[32, 1024]" = torch.ops.tg.log_tensor.default(_to_copy_4, 'position_ids');  _to_copy_4 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:148 in broadcast, code: tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
        broadcast: "i64[32, 1024]" = torch.ops._c10d_functional.broadcast.default(log_tensor, 0, '12');  log_tensor = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor: "i64[32, 1024]" = torch.ops._c10d_functional.wait_tensor.default(broadcast);  broadcast = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:148 in broadcast, code: tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
        broadcast_1: "i64[32, 1024]" = torch.ops._c10d_functional.broadcast.default(log_tensor_1, 0, '12');  log_tensor_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_1: "i64[32, 1024]" = torch.ops._c10d_functional.wait_tensor.default(broadcast_1);  broadcast_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:148 in broadcast, code: tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
        broadcast_2: "f32[32, 1024]" = torch.ops._c10d_functional.broadcast.default(log_tensor_2, 0, '12');  log_tensor_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_2: "f32[32, 1024]" = torch.ops._c10d_functional.wait_tensor.default(broadcast_2);  broadcast_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:148 in broadcast, code: tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
        broadcast_3: "b8[32, 1, 1024, 1024]" = torch.ops._c10d_functional.broadcast.default(log_tensor_3, 0, '12');  log_tensor_3 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_3: "b8[32, 1, 1024, 1024]" = torch.ops._c10d_functional.wait_tensor.default(broadcast_3);  broadcast_3 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:148 in broadcast, code: tensor = torch.ops._c10d_functional.broadcast(self, src, group_name)
        broadcast_4: "i64[32, 1024]" = torch.ops._c10d_functional.broadcast.default(log_tensor_4, 0, '12');  log_tensor_4 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_4: "i64[32, 1024]" = torch.ops._c10d_functional.wait_tensor.default(broadcast_4);  broadcast_4 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.embedding.word_embeddings.weight', _t_base_index = 0, _all_bases = [clone]);  clone = None
        getitem_1: "f16[49152, 1536]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        auto_functionalized_v2_1 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.embedding.position_embeddings.weight', _t_base_index = 0, _all_bases = [clone_1]);  clone_1 = None
        getitem_3: "f16[2048, 1536]" = auto_functionalized_v2_1[1];  auto_functionalized_v2_1 = None
        auto_functionalized_v2_2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.input_layernorm.weight', _t_base_index = 0, _all_bases = [clone_2]);  clone_2 = None
        getitem_5: "f16[1536]" = auto_functionalized_v2_2[1];  auto_functionalized_v2_2 = None
        auto_functionalized_v2_3 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.input_layernorm.bias', _t_base_index = 0, _all_bases = [clone_3]);  clone_3 = None
        getitem_7: "f16[1536]" = auto_functionalized_v2_3[1];  auto_functionalized_v2_3 = None
        auto_functionalized_v2_4 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.linear_proj.weight', _t_base_index = 0, _all_bases = [clone_4]);  clone_4 = None
        getitem_9: "f16[1536, 768]" = auto_functionalized_v2_4[1];  auto_functionalized_v2_4 = None
        auto_functionalized_v2_5 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.linear_qkv.weight', _t_base_index = 0, _all_bases = [clone_5]);  clone_5 = None
        getitem_11: "f16[2304, 1536]" = auto_functionalized_v2_5[1];  auto_functionalized_v2_5 = None
        auto_functionalized_v2_6 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.q_layernorm.weight', _t_base_index = 0, _all_bases = [clone_6]);  clone_6 = None
        getitem_13: "f16[48]" = auto_functionalized_v2_6[1];  auto_functionalized_v2_6 = None
        auto_functionalized_v2_7 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.q_layernorm.bias', _t_base_index = 0, _all_bases = [clone_7]);  clone_7 = None
        getitem_15: "f16[48]" = auto_functionalized_v2_7[1];  auto_functionalized_v2_7 = None
        auto_functionalized_v2_8 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.k_layernorm.weight', _t_base_index = 0, _all_bases = [clone_8]);  clone_8 = None
        getitem_17: "f16[48]" = auto_functionalized_v2_8[1];  auto_functionalized_v2_8 = None
        auto_functionalized_v2_9 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.k_layernorm.bias', _t_base_index = 0, _all_bases = [clone_9]);  clone_9 = None
        getitem_19: "f16[48]" = auto_functionalized_v2_9[1];  auto_functionalized_v2_9 = None
        auto_functionalized_v2_10 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.pre_mlp_layernorm.weight', _t_base_index = 0, _all_bases = [clone_10]);  clone_10 = None
        getitem_21: "f16[1536]" = auto_functionalized_v2_10[1];  auto_functionalized_v2_10 = None
        auto_functionalized_v2_11 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.pre_mlp_layernorm.bias', _t_base_index = 0, _all_bases = [clone_11]);  clone_11 = None
        getitem_23: "f16[1536]" = auto_functionalized_v2_11[1];  auto_functionalized_v2_11 = None
        auto_functionalized_v2_12 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.router.weight', _t_base_index = 0, _all_bases = [clone_12]);  clone_12 = None
        getitem_25: "f16[1, 1536]" = auto_functionalized_v2_12[1];  auto_functionalized_v2_12 = None
        auto_functionalized_v2_13 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.local_experts.0.dense_h_to_4h.weight', _t_base_index = 0, _all_bases = [clone_13]);  clone_13 = None
        getitem_27: "f16[3072, 1536]" = auto_functionalized_v2_13[1];  auto_functionalized_v2_13 = None
        auto_functionalized_v2_14 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.local_experts.0.dense_4h_to_h.weight', _t_base_index = 0, _all_bases = [clone_14]);  clone_14 = None
        getitem_29: "f16[1536, 3072]" = auto_functionalized_v2_14[1];  auto_functionalized_v2_14 = None
        auto_functionalized_v2_15 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.final_layernorm.weight', _t_base_index = 0, _all_bases = [clone_15]);  clone_15 = None
        getitem_31: "f16[1536]" = auto_functionalized_v2_15[1];  auto_functionalized_v2_15 = None
        auto_functionalized_v2_16 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.final_layernorm.bias', _t_base_index = 0, _all_bases = [clone_16]);  clone_16 = None
        getitem_33: "f16[1536]" = auto_functionalized_v2_16[1];  auto_functionalized_v2_16 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:252 in forward, code: input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        lt: "b8[32, 1024]" = torch.ops.aten.lt.Scalar(wait_tensor, 0)
        ge: "b8[32, 1024]" = torch.ops.aten.ge.Scalar(wait_tensor, 49152)
        bitwise_or: "b8[32, 1024]" = torch.ops.aten.bitwise_or.Tensor(lt, ge);  lt = ge = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:254 in forward, code: masked_input = input_.clone() - self.vocab_start_index
        clone_17: "i64[32, 1024]" = torch.ops.aten.clone.default(wait_tensor);  wait_tensor = None
        sub: "i64[32, 1024]" = torch.ops.aten.sub.Tensor(clone_17, 0);  clone_17 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:255 in forward, code: masked_input[input_mask] = 0
        _tensor_constant0: "i64[]" = self._tensor_constant0
        lift_fresh_copy: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        index_put: "i64[32, 1024]" = torch.ops.aten.index_put.default(sub, [bitwise_or], lift_fresh_copy);  sub = lift_fresh_copy = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        embedding: "f16[32, 1024, 1536]" = torch.ops.aten.embedding.default(getitem_1, index_put)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:266 in forward, code: output_parallel[input_mask, :] = 0.0
        _tensor_constant1: "f16[]" = self._tensor_constant1
        lift_fresh_copy_1: "f16[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        index_put_1: "f16[32, 1024, 1536]" = torch.ops.aten.index_put.default(embedding, [bitwise_or], lift_fresh_copy_1);  embedding = lift_fresh_copy_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:485 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
        all_reduce: "f16[32, 1024, 1536]" = torch.ops._c10d_functional.all_reduce.default(index_put_1, 'sum', '12')
        wait_tensor_5: "f16[32, 1024, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        copy: "f16[32, 1024, 1536]" = torch.ops.aten.copy.default(index_put_1, wait_tensor_5);  index_put_1 = wait_tensor_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_1: "f16[32, 1024, 1536]" = torch.ops.aten.embedding.default(getitem_3, wait_tensor_4)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:112 in forward, code: embeddings = word_embeddings + position_embeddings
        view_1: "f16[32, 1024, 1536]" = torch.ops.aten.view.default(copy, [32, 1024, 1536]);  copy = None
        add: "f16[32, 1024, 1536]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:118 in forward, code: embeddings = embeddings.transpose(0, 1).contiguous()
        transpose: "f16[1024, 32, 1536]" = torch.ops.aten.transpose.int(add, 0, 1);  add = None
        clone_18: "f16[1024, 32, 1536]" = torch.ops.aten.clone.default(transpose, memory_format = torch.contiguous_format);  transpose = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:500 in scatter_to_sequence_parallel_region, code: return _ScatterToSequenceParallelRegion.apply(input_)
        slice_1: "f16[512, 32, 1536]" = torch.ops.aten.slice.Tensor(clone_18, 0, 0, 512);  clone_18 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:140 in forward, code: embeddings = embeddings.clone()
        clone_19: "f16[512, 32, 1536]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd = torch.ops.apex.fused_layer_norm_affine_fwd.default(clone_19, getitem_5, getitem_7, [1536], 1e-05)
        getitem_34: "f16[512, 32, 1536]" = fused_layer_norm_affine_fwd[0]
        getitem_35: "f32[16384]" = fused_layer_norm_affine_fwd[1]
        getitem_36: "f32[16384]" = fused_layer_norm_affine_fwd[2];  fused_layer_norm_affine_fwd = None
        detach: "f32[16384]" = torch.ops.aten.detach.default(getitem_35);  getitem_35 = None
        detach_1: "f32[16384]" = torch.ops.aten.detach.default(detach);  detach = None
        detach_2: "f32[16384]" = torch.ops.aten.detach.default(getitem_36);  getitem_36 = None
        detach_3: "f32[16384]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_34, 2, '12')
        wait_tensor_6: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        view_2: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_6, [32768, 1536]);  wait_tensor_6 = None
        t_1: "f16[1536, 2304]" = torch.ops.aten.t.default(getitem_11)
        mm: "f16[32768, 2304]" = torch.ops.aten.mm.default(view_2, t_1);  view_2 = t_1 = None
        _unsafe_view: "f16[1024, 32, 2304]" = torch.ops.aten._unsafe_view.default(mm, [1024, 32, 2304]);  mm = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_3: "f16[1024, 32, 16, 144]" = torch.ops.aten.view.default(_unsafe_view, [1024, 32, 16, 144]);  _unsafe_view = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [48, 48, 48], 3);  view_3 = None
        getitem_37: "f16[1024, 32, 16, 48]" = split_with_sizes[0]
        getitem_38: "f16[1024, 32, 16, 48]" = split_with_sizes[1]
        getitem_39: "f16[1024, 32, 16, 48]" = split_with_sizes[2];  split_with_sizes = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_4: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(getitem_37, [1024, 32, 16, 48]);  getitem_37 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_1 = torch.ops.apex.fused_layer_norm_affine_fwd.default(view_4, getitem_13, getitem_15, [48], 1e-05)
        getitem_40: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_fwd_1[0]
        getitem_41: "f32[524288]" = fused_layer_norm_affine_fwd_1[1]
        getitem_42: "f32[524288]" = fused_layer_norm_affine_fwd_1[2];  fused_layer_norm_affine_fwd_1 = None
        clone_20: "f16[1024, 32, 16, 48]" = torch.ops.aten.clone.default(view_4, memory_format = torch.contiguous_format);  view_4 = None
        detach_4: "f32[524288]" = torch.ops.aten.detach.default(getitem_41);  getitem_41 = None
        detach_5: "f32[524288]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
        detach_6: "f32[524288]" = torch.ops.aten.detach.default(getitem_42);  getitem_42 = None
        detach_7: "f32[524288]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_2 = torch.ops.apex.fused_layer_norm_affine_fwd.default(getitem_38, getitem_17, getitem_19, [48], 1e-05)
        getitem_43: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_fwd_2[0]
        getitem_44: "f32[524288]" = fused_layer_norm_affine_fwd_2[1]
        getitem_45: "f32[524288]" = fused_layer_norm_affine_fwd_2[2];  fused_layer_norm_affine_fwd_2 = None
        clone_21: "f16[1024, 32, 16, 48]" = torch.ops.aten.clone.default(getitem_38, memory_format = torch.contiguous_format);  getitem_38 = None
        detach_8: "f32[524288]" = torch.ops.aten.detach.default(getitem_44);  getitem_44 = None
        detach_9: "f32[524288]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
        detach_10: "f32[524288]" = torch.ops.aten.detach.default(getitem_45);  getitem_45 = None
        detach_11: "f32[524288]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_5: "f16[1024, 512, 48]" = torch.ops.aten.view.default(getitem_40, [1024, 512, -1]);  getitem_40 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_6: "f16[1024, 512, 48]" = torch.ops.aten.view.default(getitem_43, [1024, 512, -1]);  getitem_43 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty: "f16[512, 1024, 1024]" = torch.ops.aten.empty.memory_format([512, 1024, 1024], dtype = torch.float16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_1: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(view_5, 0, 1);  view_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_2: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(view_6, 0, 1);  view_6 = None
        transpose_3: "f16[512, 48, 1024]" = torch.ops.aten.transpose.int(transpose_2, 1, 2);  transpose_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        baddbmm: "f16[512, 1024, 1024]" = torch.ops.aten.baddbmm.default(empty, transpose_1, transpose_3, beta = 0.0, alpha = 0.14433756729740646);  empty = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:165 in forward, code: attention_scores = matmul_result.view(*output_size)
        view_7: "f16[32, 16, 1024, 1024]" = torch.ops.aten.view.default(baddbmm, [32, 16, 1024, 1024]);  baddbmm = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        masked_fill: "f16[32, 16, 1024, 1024]" = torch.ops.aten.masked_fill.Scalar(view_7, wait_tensor_3, -10000.0);  view_7 = None
        view_8: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(masked_fill, [512, 1024, 1024]);  masked_fill = None
        view_9: "f16[32, 16, 1024, 1024]" = torch.ops.aten.view.default(view_8, [32, 16, 1024, 1024]);  view_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_softmax.py:208 in forward_torch_softmax, code: probs = torch.nn.Softmax(dim=-1)(mask_output)
        _softmax: "f16[32, 16, 1024, 1024]" = torch.ops.aten._softmax.default(view_9, -1, False);  view_9 = None
        detach_12: "f16[32, 16, 1024, 1024]" = torch.ops.aten.detach.default(_softmax)
        detach_13: "f16[32, 16, 1024, 1024]" = torch.ops.aten.detach.default(detach_12);  detach_12 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_11: "f16[1024, 512, 48]" = torch.ops.aten.view.default(getitem_39, [1024, 512, -1]);  getitem_39 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_12: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(_softmax, [512, 1024, -1]);  _softmax = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_4: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(view_11, 0, 1);  view_11 = None
        bmm: "f16[512, 1024, 48]" = torch.ops.aten.bmm.default(view_12, transpose_4)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_13: "f16[32, 16, 1024, 48]" = torch.ops.aten.view.default(bmm, [32, 16, 1024, 48]);  bmm = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute: "f16[1024, 32, 16, 48]" = torch.ops.aten.permute.default(view_13, [2, 0, 1, 3]);  view_13 = None
        clone_22: "f16[1024, 32, 16, 48]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_14: "f16[1024, 32, 768]" = torch.ops.aten.view.default(clone_22, [1024, 32, 768]);  clone_22 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_15: "f16[32768, 768]" = torch.ops.aten.view.default(view_14, [32768, 768])
        t_3: "f16[768, 1536]" = torch.ops.aten.t.default(getitem_9)
        mm_1: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_15, t_3);  view_15 = t_3 = None
        _unsafe_view_1: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_1, [1024, 32, 1536]);  mm_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:520 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        reduce_scatter_tensor: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_1, 'sum', 2, '12');  _unsafe_view_1 = None
        wait_tensor_7: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:45 in _bias_dropout_add_func, code: out = residual + out
        add_1: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(clone_19, wait_tensor_7);  wait_tensor_7 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_3 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_1, getitem_21, getitem_23, [1536], 1e-05)
        getitem_46: "f16[512, 32, 1536]" = fused_layer_norm_affine_fwd_3[0]
        getitem_47: "f32[16384]" = fused_layer_norm_affine_fwd_3[1]
        getitem_48: "f32[16384]" = fused_layer_norm_affine_fwd_3[2];  fused_layer_norm_affine_fwd_3 = None
        detach_14: "f32[16384]" = torch.ops.aten.detach.default(getitem_47);  getitem_47 = None
        detach_15: "f32[16384]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
        detach_16: "f32[16384]" = torch.ops.aten.detach.default(getitem_48);  getitem_48 = None
        detach_17: "f32[16384]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:262 in forward, code: route = self.router(hidden_states)
        view_16: "f16[16384, 1536]" = torch.ops.aten.view.default(getitem_46, [16384, 1536])
        t_5: "f16[1536, 1]" = torch.ops.aten.t.default(getitem_25)
        mm_2: "f16[16384, 1]" = torch.ops.aten.mm.default(view_16, t_5);  t_5 = None
        _unsafe_view_2: "f16[512, 32, 1]" = torch.ops.aten._unsafe_view.default(mm_2, [512, 32, 1]);  mm_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:263 in forward, code: route = route.view(-1, args.num_experts)
        view_17: "f16[16384, 1]" = torch.ops.aten.view.default(_unsafe_view_2, [-1, 1]);  _unsafe_view_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:270 in forward, code: sinkroute = sinkhorn(route.detach().to(dtype=torch.float32))
        detach_18: "f16[16384, 1]" = torch.ops.aten.detach.default(view_17)
        detach_19: "f16[16384, 1]" = torch.ops.aten.detach.default(detach_18);  detach_18 = None
        detach_20: "f16[16384, 1]" = torch.ops.aten.detach.default(detach_19);  detach_19 = None
        _to_copy_5: "f32[16384, 1]" = torch.ops.aten._to_copy.default(detach_20, dtype = torch.float32);  detach_20 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        sinkhorn: "f32[16384, 1]" = torch.ops.megatron.sinkhorn.default(_to_copy_5);  _to_copy_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:271 in forward, code: _, max_ind = torch.max(sinkroute, dim=1)
        max_1 = torch.ops.aten.max.dim(sinkhorn, 1);  sinkhorn = None
        getitem_50: "i64[16384]" = max_1[1];  max_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:272 in forward, code: route = torch.sigmoid(route)
        sigmoid: "f16[16384, 1]" = torch.ops.aten.sigmoid.default(view_17);  view_17 = None
        detach_21: "f16[16384, 1]" = torch.ops.aten.detach.default(sigmoid)
        detach_22: "f16[16384, 1]" = torch.ops.aten.detach.default(detach_21);  detach_21 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:273 in forward, code: max_prob = route[torch.arange(route.size(0)), max_ind]
        arange: "i64[16384]" = torch.ops.aten.arange.default(16384, device = device(type='cpu'), pin_memory = False)
        index: "f16[16384]" = torch.ops.aten.index.Tensor(sigmoid, [arange, getitem_50]);  sigmoid = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:278 in forward, code: max_prob = torch.unsqueeze(max_prob, 1)
        unsqueeze: "f16[16384, 1]" = torch.ops.aten.unsqueeze.default(index, 1);  index = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:279 in forward, code: hidden_states = hidden_states.view(-1, hidden_states.size(2))
        view_18: "f16[16384, 1536]" = torch.ops.aten.view.default(getitem_46, [-1, 1536]);  getitem_46 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:511 in gather_from_sequence_parallel_region, code: return _GatherFromSequenceParallelRegion.apply(
        empty_1: "f16[32768, 1536]" = torch.ops.aten.empty.memory_format([32768, 1536], dtype = torch.float16, device = device(type='cuda', index=0), pin_memory = False)
        all_gather_into_tensor_1: "f16[32768, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(view_18, 2, '25');  view_18 = None
        wait_tensor_8: "f16[32768, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        copy_1: "f16[32768, 1536]" = torch.ops.aten.copy.default(empty_1, wait_tensor_8);  empty_1 = wait_tensor_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:249 in gather_indices, code: output = torch.empty(dim_size, dtype=local_indices.dtype,
        empty_2: "i64[32768]" = torch.ops.aten.empty.memory_format([32768], dtype = torch.int64, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:199 in all_gather_tensor, code: tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        all_gather_into_tensor_2: "i64[32768]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_50, 2, '25')
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_9: "i64[32768]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:1064 in all_gather_tensor_inplace, code: return output_tensor.copy_(all_gather_tensor(input_tensor, gather_dim, group, tag))
        copy_2: "i64[32768]" = torch.ops.aten.copy.default(empty_2, wait_tensor_9);  empty_2 = wait_tensor_9 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:292 in forward, code: output_total = torch.zeros_like(global_hidden_states)
        zeros_like: "f16[32768, 1536]" = torch.ops.aten.zeros_like.default(copy_1, pin_memory = False)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:298 in forward, code: local_indices = (global_indices == local_expert_index).nonzero()
        eq: "b8[32768]" = torch.ops.aten.eq.Scalar(copy_2, 0);  copy_2 = None
        nonzero: "i64[u0, 1]" = torch.ops.aten.nonzero.default(eq);  eq = None
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(nonzero, 0)
        ge_2: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u0 >= 0 on node 'ge_1'");  ge_2 = _assert_scalar = None
        le: "Sym(u0 <= 32768)" = sym_size_int <= 32768
        _assert_scalar_1 = torch.ops.aten._assert_scalar.default(le, "Runtime assertion failed for expression u0 <= 32768 on node 'le'");  le = _assert_scalar_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:299 in forward, code: hidden = global_hidden_states[local_indices, :]
        index_1: "f16[u0, 1, 1536]" = torch.ops.aten.index.Tensor(copy_1, [nonzero]);  copy_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_19: "f16[u0, 1536]" = torch.ops.aten.view.default(index_1, [sym_size_int, 1536])
        t_7: "f16[1536, 3072]" = torch.ops.aten.t.default(getitem_27)
        mm_3: "f16[u0, 3072]" = torch.ops.aten.mm.default(view_19, t_7);  view_19 = t_7 = None
        _unsafe_view_3: "f16[u0, 1, 3072]" = torch.ops.aten._unsafe_view.default(mm_3, [sym_size_int, 1, 3072]);  mm_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:172 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu: "f16[u0, 1, 3072]" = torch.ops.aten.gelu.default(_unsafe_view_3)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_20: "f16[u0, 3072]" = torch.ops.aten.view.default(gelu, [sym_size_int, 3072])
        t_9: "f16[3072, 1536]" = torch.ops.aten.t.default(getitem_29)
        mm_4: "f16[u0, 1536]" = torch.ops.aten.mm.default(view_20, t_9);  view_20 = t_9 = None
        _unsafe_view_4: "f16[u0, 1, 1536]" = torch.ops.aten._unsafe_view.default(mm_4, [sym_size_int, 1, 1536]);  mm_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:301 in forward, code: output_total[local_indices, :] = output
        index_put_2: "f16[32768, 1536]" = torch.ops.aten.index_put.default(zeros_like, [nonzero], _unsafe_view_4);  zeros_like = _unsafe_view_4 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:278 in reduce_scatter_tensor, code: tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        reduce_scatter_tensor_1: "f16[16384, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(index_put_2, 'sum', 2, '25');  index_put_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_10: "f16[16384, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:318 in forward, code: output_total = output_total*max_prob
        mul_38: "f16[16384, 1536]" = torch.ops.aten.mul.Tensor(wait_tensor_10, unsqueeze);  unsqueeze = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:319 in forward, code: output_total = output_total.view(s, b, h)
        view_21: "f16[512, 32, 1536]" = torch.ops.aten.view.default(mul_38, [512, 32, 1536]);  mul_38 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:45 in _bias_dropout_add_func, code: out = residual + out
        add_40: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(add_1, view_21);  view_21 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_3: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_40, 2, '12')
        wait_tensor_11: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        view_22: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_11, [32768, 1536]);  wait_tensor_11 = None
        t_11: "f16[1536, 49152]" = torch.ops.aten.t.default(getitem_1)
        mm_5: "f16[32768, 49152]" = torch.ops.aten.mm.default(view_22, t_11);  view_22 = t_11 = None
        _unsafe_view_5: "f16[1024, 32, 49152]" = torch.ops.aten._unsafe_view.default(mm_5, [1024, 32, 49152]);  mm_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:77 in compute_language_model_loss, code: labels = labels.transpose(0, 1).contiguous()
        transpose_5: "i64[1024, 32]" = torch.ops.aten.transpose.int(wait_tensor_1, 0, 1);  wait_tensor_1 = None
        clone_23: "i64[1024, 32]" = torch.ops.aten.clone.default(transpose_5, memory_format = torch.contiguous_format);  transpose_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:236 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        _to_copy_6: "f32[1024, 32, 49152]" = torch.ops.aten._to_copy.default(_unsafe_view_5, dtype = torch.float32);  _unsafe_view_5 = None
        max_2 = torch.ops.aten.max.dim(_to_copy_6, -1)
        getitem_51: "f32[1024, 32]" = max_2[0];  max_2 = None
        all_reduce_1: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(getitem_51, 'max', '12')
        wait_tensor_12: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        copy_3: "f32[1024, 32]" = torch.ops.aten.copy.default(getitem_51, wait_tensor_12);  getitem_51 = wait_tensor_12 = None
        unsqueeze_2: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(copy_3, -1);  copy_3 = None
        sub_12: "f32[1024, 32, 49152]" = torch.ops.aten.sub.Tensor(_to_copy_6, unsqueeze_2);  _to_copy_6 = unsqueeze_2 = None
        lt_7: "b8[1024, 32]" = torch.ops.aten.lt.Scalar(clone_23, 0)
        ge_5: "b8[1024, 32]" = torch.ops.aten.ge.Scalar(clone_23, 49152)
        bitwise_or_1: "b8[1024, 32]" = torch.ops.aten.bitwise_or.Tensor(lt_7, ge_5);  lt_7 = ge_5 = None
        clone_24: "i64[1024, 32]" = torch.ops.aten.clone.default(clone_23);  clone_23 = None
        sub_13: "i64[1024, 32]" = torch.ops.aten.sub.Tensor(clone_24, 0);  clone_24 = None
        _tensor_constant2: "i64[]" = self._tensor_constant2
        lift_fresh_copy_2: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        index_put_3: "i64[1024, 32]" = torch.ops.aten.index_put.default(sub_13, [bitwise_or_1], lift_fresh_copy_2);  sub_13 = lift_fresh_copy_2 = None
        arange_1: "i64[32768]" = torch.ops.aten.arange.start(0, 32768, device = device(type='cuda', index=0), pin_memory = False)
        view_25: "f32[32768, 49152]" = torch.ops.aten.view.default(sub_12, [-1, 49152])
        view_26: "i64[32768]" = torch.ops.aten.view.default(index_put_3, [-1]);  index_put_3 = None
        index_2: "f32[32768]" = torch.ops.aten.index.Tensor(view_25, [arange_1, view_26]);  view_25 = arange_1 = None
        clone_25: "f32[32768]" = torch.ops.aten.clone.default(index_2);  index_2 = None
        view_27: "f32[1024, 32]" = torch.ops.aten.view.default(clone_25, [1024, 32]);  clone_25 = None
        _tensor_constant3: "f32[]" = self._tensor_constant3
        lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        index_put_4: "f32[1024, 32]" = torch.ops.aten.index_put.default(view_27, [bitwise_or_1], lift_fresh_copy_3);  view_27 = lift_fresh_copy_3 = None
        view_28: "f32[32768]" = torch.ops.aten.view.default(index_put_4, [32768]);  index_put_4 = None
        view_29: "f32[1024, 32]" = torch.ops.aten.view.default(view_28, [1024, 32]);  view_28 = None
        exp: "f32[1024, 32, 49152]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_1: "f32[1024, 32]" = torch.ops.aten.sum.dim_IntList(exp, [-1])
        all_reduce_2: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(view_29, 'sum', '12')
        wait_tensor_13: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = None
        copy_4: "f32[1024, 32]" = torch.ops.aten.copy.default(view_29, wait_tensor_13);  view_29 = wait_tensor_13 = None
        view_30: "f32[32768]" = torch.ops.aten.view.default(copy_4, [32768]);  copy_4 = None
        view_31: "f32[1024, 32]" = torch.ops.aten.view.default(view_30, [1024, 32]);  view_30 = None
        all_reduce_3: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(sum_1, 'sum', '12')
        wait_tensor_14: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = None
        copy_5: "f32[1024, 32]" = torch.ops.aten.copy.default(sum_1, wait_tensor_14);  sum_1 = wait_tensor_14 = None
        log: "f32[1024, 32]" = torch.ops.aten.log.default(copy_5)
        sub_14: "f32[1024, 32]" = torch.ops.aten.sub.Tensor(log, view_31);  log = view_31 = None
        unsqueeze_4: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(copy_5, -1);  copy_5 = None
        div: "f32[1024, 32, 49152]" = torch.ops.aten.div.Tensor(exp, unsqueeze_4);  exp = unsqueeze_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_6: "f32[32, 1024]" = torch.ops.aten.transpose.int(sub_14, 0, 1);  sub_14 = None
        clone_26: "f32[32, 1024]" = torch.ops.aten.clone.default(transpose_6, memory_format = torch.contiguous_format);  transpose_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:171 in loss_func, code: loss_mask = loss_mask.view(-1).float()
        view_32: "f32[32768]" = torch.ops.aten.view.default(wait_tensor_2, [-1]);  wait_tensor_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:172 in loss_func, code: total_tokens = loss_mask.sum().view(1)
        sum_2: "f32[]" = torch.ops.aten.sum.default(view_32)
        view_33: "f32[1]" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:173 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask).view(1)
        view_34: "f32[32768]" = torch.ops.aten.view.default(clone_26, [-1]);  clone_26 = None
        mul_39: "f32[32768]" = torch.ops.aten.mul.Tensor(view_34, view_32);  view_34 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(mul_39);  mul_39 = None
        view_35: "f32[1]" = torch.ops.aten.view.default(sum_3, [1]);  sum_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:210 in loss_func, code: reporting_loss = loss.clone().detach()
        clone_27: "f32[1]" = torch.ops.aten.clone.default(view_35)
        detach_23: "f32[1]" = torch.ops.aten.detach.default(clone_27);  clone_27 = None
        detach_24: "f32[1]" = torch.ops.aten.detach.default(detach_23);  detach_23 = None
        detach_25: "f32[1]" = torch.ops.aten.detach.default(detach_24);  detach_24 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:211 in loss_func, code: reporting_total_tokens = total_tokens.clone().detach()
        clone_28: "f32[1]" = torch.ops.aten.clone.default(view_33)
        detach_26: "f32[1]" = torch.ops.aten.detach.default(clone_28);  clone_28 = None
        detach_27: "f32[1]" = torch.ops.aten.detach.default(detach_26);  detach_26 = None
        detach_28: "f32[1]" = torch.ops.aten.detach.default(detach_27);  detach_27 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:170 in all_reduce, code: tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
        all_reduce_4: "f32[1]" = torch.ops._c10d_functional.all_reduce.default(detach_25, 'sum', '1')
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_15: "f32[1]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_4);  all_reduce_4 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:1112 in all_reduce_inplace, code: return tensor.copy_(all_reduce(tensor, op, group, tag))
        copy_6: "f32[1]" = torch.ops.aten.copy.default(detach_25, wait_tensor_15);  detach_25 = wait_tensor_15 = None
        detach_30: "f32[1]" = torch.ops.aten.detach.default(copy_6);  copy_6 = None
        detach_31: "f32[1]" = torch.ops.aten.detach.default(detach_30);  detach_30 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:170 in all_reduce, code: tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
        all_reduce_5: "f32[1]" = torch.ops._c10d_functional.all_reduce.default(detach_28, 'sum', '1')
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_16: "f32[1]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_5);  all_reduce_5 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:1112 in all_reduce_inplace, code: return tensor.copy_(all_reduce(tensor, op, group, tag))
        copy_7: "f32[1]" = torch.ops.aten.copy.default(detach_28, wait_tensor_16);  detach_28 = wait_tensor_16 = None
        detach_33: "f32[1]" = torch.ops.aten.detach.default(copy_7);  copy_7 = None
        detach_34: "f32[1]" = torch.ops.aten.detach.default(detach_33);  detach_33 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:215 in loss_func, code: local_num_tokens = total_tokens.clone().detach().to(torch.int)
        clone_29: "f32[1]" = torch.ops.aten.clone.default(view_33);  view_33 = None
        detach_35: "f32[1]" = torch.ops.aten.detach.default(clone_29);  clone_29 = None
        detach_36: "f32[1]" = torch.ops.aten.detach.default(detach_35);  detach_35 = None
        detach_37: "f32[1]" = torch.ops.aten.detach.default(detach_36);  detach_36 = None
        _to_copy_7: "i32[1]" = torch.ops.aten._to_copy.default(detach_37, dtype = torch.int32);  detach_37 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:218 in loss_func, code: loss * args.context_parallel_size,
        mul_40: "f32[1]" = torch.ops.aten.mul.Tensor(view_35, 1);  view_35 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:289 in forward_step, code: output_tensor /= num_tokens
        div_1: "f32[1]" = torch.ops.aten.div.Tensor(mul_40, _to_copy_7);  mul_40 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:290 in forward_step, code: output_tensor /= num_microbatches
        div_2: "f32[1]" = torch.ops.aten.div.Tensor(div_1, 1);  div_1 = None
        div_3: "f32[1]" = torch.ops.aten.div.Tensor(tangents_18, 1);  tangents_18 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:289 in forward_step, code: output_tensor /= num_tokens
        div_4: "f32[1]" = torch.ops.aten.div.Tensor(div_3, _to_copy_7);  div_3 = _to_copy_7 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:218 in loss_func, code: loss * args.context_parallel_size,
        mul_41: "f32[1]" = torch.ops.aten.mul.Tensor(div_4, 1);  div_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:173 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask).view(1)
        view_36: "f32[]" = torch.ops.aten.view.default(mul_41, []);  mul_41 = None
        expand: "f32[32768]" = torch.ops.aten.expand.default(view_36, [32768]);  view_36 = None
        mul_42: "f32[32768]" = torch.ops.aten.mul.Tensor(expand, view_32);  expand = view_32 = None
        view_37: "f32[32, 1024]" = torch.ops.aten.view.default(mul_42, [32, 1024]);  mul_42 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_7: "f32[1024, 32]" = torch.ops.aten.transpose.int(view_37, 0, 1);  view_37 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:236 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        arange_2: "i64[32768]" = torch.ops.aten.arange.start(0, 32768, device = device(type='cuda', index=0), pin_memory = False)
        view_39: "b8[32768]" = torch.ops.aten.view.default(bitwise_or_1, [-1]);  bitwise_or_1 = None
        _to_copy_8: "f32[32768]" = torch.ops.aten._to_copy.default(view_39, dtype = torch.float32);  view_39 = None
        rsub: "f32[32768]" = torch.ops.aten.rsub.Scalar(_to_copy_8, 1.0);  _to_copy_8 = None
        view_40: "f32[32768, 49152]" = torch.ops.aten.view.default(div, [-1, 49152]);  div = None
        index_3: "f32[32768]" = torch.ops.aten.index.Tensor(view_40, [arange_2, view_26])
        sub_15: "f32[32768]" = torch.ops.aten.sub.Tensor(index_3, rsub);  index_3 = rsub = None
        index_put_5: "f32[32768, 49152]" = torch.ops.aten.index_put.default(view_40, [arange_2, view_26], sub_15);  view_40 = arange_2 = view_26 = sub_15 = None
        view_41: "f32[1024, 32, 49152]" = torch.ops.aten.view.default(index_put_5, [1024, 32, 49152]);  index_put_5 = None
        unsqueeze_5: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(transpose_7, -1);  transpose_7 = None
        mul_43: "f32[1024, 32, 49152]" = torch.ops.aten.mul.Tensor(view_41, unsqueeze_5);  view_41 = unsqueeze_5 = None
        _to_copy_9: "f16[1024, 32, 49152]" = torch.ops.aten._to_copy.default(mul_43, dtype = torch.float16);  mul_43 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_4: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_40, 2, '12');  add_40 = None
        wait_tensor_17: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        wait_tensor_18: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_17);  wait_tensor_17 = None
        view_43: "f16[32768, 49152]" = torch.ops.aten.view.default(_to_copy_9, [32768, 49152])
        mm_6: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_43, getitem_1);  view_43 = None
        _unsafe_view_6: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_6, [1024, 32, 1536]);  mm_6 = None
        view_44: "f16[32768, 49152]" = torch.ops.aten.view.default(_to_copy_9, [32768, 49152]);  _to_copy_9 = None
        view_45: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_18, [32768, 1536]);  wait_tensor_18 = None
        reduce_scatter_tensor_2: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_6, 'sum', 2, '12');  _unsafe_view_6 = None
        wait_tensor_19: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        t_12: "f16[49152, 32768]" = torch.ops.aten.t.default(view_44);  view_44 = None
        mm_7: "f16[49152, 1536]" = torch.ops.aten.mm.default(t_12, view_45);  t_12 = view_45 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_41: "f16[49152, 1536]" = torch.ops.aten.add.Tensor(tangents_1, mm_7);  tangents_1 = mm_7 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:319 in forward, code: output_total = output_total.view(s, b, h)
        view_46: "f16[16384, 1536]" = torch.ops.aten.view.default(wait_tensor_19, [16384, 1536])
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:318 in forward, code: output_total = output_total*max_prob
        mul_44: "f16[16384, 1536]" = torch.ops.aten.mul.Tensor(view_46, wait_tensor_10);  view_46 = wait_tensor_10 = None
        sum_4: "f16[16384, 1]" = torch.ops.aten.sum.dim_IntList(mul_44, [1], True);  mul_44 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        zeros_1: "f16[u0, 1, 1536]" = torch.ops.aten.zeros.default([sym_size_int, 1, 1536], dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0))
        view_47: "f16[u0, 1536]" = torch.ops.aten.view.default(zeros_1, [sym_size_int, 1536])
        mm_8: "f16[u0, 3072]" = torch.ops.aten.mm.default(view_47, getitem_29);  view_47 = None
        _unsafe_view_7: "f16[u0, 1, 3072]" = torch.ops.aten._unsafe_view.default(mm_8, [sym_size_int, 1, 3072]);  mm_8 = None
        view_48: "f16[u0, 1536]" = torch.ops.aten.view.default(zeros_1, [sym_size_int, 1536]);  zeros_1 = None
        view_49: "f16[u0, 3072]" = torch.ops.aten.view.default(gelu, [sym_size_int, 3072]);  gelu = None
        t_13: "f16[1536, u0]" = torch.ops.aten.t.default(view_48);  view_48 = None
        mm_9: "f16[1536, 3072]" = torch.ops.aten.mm.default(t_13, view_49);  t_13 = view_49 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_42: "f16[1536, 3072]" = torch.ops.aten.add.Tensor(tangents_15, mm_9);  tangents_15 = mm_9 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:172 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_backward: "f16[u0, 1, 3072]" = torch.ops.aten.gelu_backward.default(_unsafe_view_7, _unsafe_view_3);  _unsafe_view_7 = _unsafe_view_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_50: "f16[u0, 3072]" = torch.ops.aten.view.default(gelu_backward, [sym_size_int, 3072])
        mm_10: "f16[u0, 1536]" = torch.ops.aten.mm.default(view_50, getitem_27);  view_50 = None
        _unsafe_view_8: "f16[u0, 1, 1536]" = torch.ops.aten._unsafe_view.default(mm_10, [sym_size_int, 1, 1536]);  mm_10 = None
        view_51: "f16[u0, 3072]" = torch.ops.aten.view.default(gelu_backward, [sym_size_int, 3072]);  gelu_backward = None
        view_52: "f16[u0, 1536]" = torch.ops.aten.view.default(index_1, [sym_size_int, 1536]);  index_1 = sym_size_int = None
        t_14: "f16[3072, u0]" = torch.ops.aten.t.default(view_51);  view_51 = None
        mm_11: "f16[3072, 1536]" = torch.ops.aten.mm.default(t_14, view_52);  t_14 = view_52 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_43: "f16[3072, 1536]" = torch.ops.aten.add.Tensor(tangents_14, mm_11);  tangents_14 = mm_11 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:299 in forward, code: hidden = global_hidden_states[local_indices, :]
        new_zeros: "f16[32768, 1536]" = torch.ops.aten.new_zeros.default(_unsafe_view_8, [32768, 1536], dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0))
        index_put_6: "f16[32768, 1536]" = torch.ops.aten.index_put.default(new_zeros, [nonzero], _unsafe_view_8, True);  new_zeros = nonzero = _unsafe_view_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:511 in gather_from_sequence_parallel_region, code: return _GatherFromSequenceParallelRegion.apply(
        reduce_scatter_tensor_3: "f16[16384, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(index_put_6, 'sum', 2, '25');  index_put_6 = None
        wait_tensor_20: "f16[16384, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_3);  reduce_scatter_tensor_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:279 in forward, code: hidden_states = hidden_states.view(-1, hidden_states.size(2))
        view_53: "f16[512, 32, 1536]" = torch.ops.aten.view.default(wait_tensor_20, [512, 32, 1536]);  wait_tensor_20 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:278 in forward, code: max_prob = torch.unsqueeze(max_prob, 1)
        squeeze: "f16[16384]" = torch.ops.aten.squeeze.dim(sum_4, 1);  sum_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:273 in forward, code: max_prob = route[torch.arange(route.size(0)), max_ind]
        new_zeros_1: "f16[16384, 1]" = torch.ops.aten.new_zeros.default(squeeze, [16384, 1], dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0))
        index_put_7: "f16[16384, 1]" = torch.ops.aten.index_put.default(new_zeros_1, [arange, getitem_50], squeeze, True);  new_zeros_1 = arange = getitem_50 = squeeze = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:272 in forward, code: route = torch.sigmoid(route)
        detach_38: "f16[16384, 1]" = torch.ops.aten.detach.default(detach_22);  detach_22 = None
        detach_39: "f16[16384, 1]" = torch.ops.aten.detach.default(detach_38);  detach_38 = None
        sigmoid_backward: "f16[16384, 1]" = torch.ops.aten.sigmoid_backward.default(index_put_7, detach_39);  index_put_7 = detach_39 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:263 in forward, code: route = route.view(-1, args.num_experts)
        view_54: "f16[512, 32, 1]" = torch.ops.aten.view.default(sigmoid_backward, [512, 32, 1]);  sigmoid_backward = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:262 in forward, code: route = self.router(hidden_states)
        view_55: "f16[16384, 1]" = torch.ops.aten.view.default(view_54, [16384, 1]);  view_54 = None
        t_15: "f16[1, 16384]" = torch.ops.aten.t.default(view_55)
        mm_12: "f16[1, 1536]" = torch.ops.aten.mm.default(t_15, view_16);  t_15 = view_16 = None
        t_16: "f16[1536, 1]" = torch.ops.aten.t.default(mm_12);  mm_12 = None
        t_18: "f16[1536, 1]" = torch.ops.aten.t.default(getitem_25)
        t_19: "f16[1, 1536]" = torch.ops.aten.t.default(t_18);  t_18 = None
        mm_13: "f16[16384, 1536]" = torch.ops.aten.mm.default(view_55, t_19);  view_55 = t_19 = None
        view_56: "f16[512, 32, 1536]" = torch.ops.aten.view.default(mm_13, [512, 32, 1536]);  mm_13 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:262 in forward, code: route = self.router(hidden_states)
        add_44: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(view_53, view_56);  view_53 = view_56 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:262 in forward, code: route = self.router(hidden_states)
        t_20: "f16[1, 1536]" = torch.ops.aten.t.default(t_16);  t_16 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/legacy/model/transformer.py:262 in forward, code: route = self.router(hidden_states)
        add_45: "f16[1, 1536]" = torch.ops.aten.add.Tensor(tangents_13, t_20);  tangents_13 = t_20 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_40: "f32[16384]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
        detach_41: "f32[16384]" = torch.ops.aten.detach.default(detach_40);  detach_40 = None
        detach_42: "f32[16384]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
        detach_43: "f32[16384]" = torch.ops.aten.detach.default(detach_42);  detach_42 = None
        fused_layer_norm_affine_bwd = torch.ops.apex.fused_layer_norm_affine_bwd.default(add_44, detach_41, detach_43, add_1, [1536], getitem_21, getitem_23, 1e-05);  add_44 = detach_41 = detach_43 = add_1 = None
        getitem_53: "f16[512, 32, 1536]" = fused_layer_norm_affine_bwd[0]
        getitem_54: "f16[1536]" = fused_layer_norm_affine_bwd[1]
        getitem_55: "f16[1536]" = fused_layer_norm_affine_bwd[2];  fused_layer_norm_affine_bwd = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_46: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(wait_tensor_19, getitem_53);  wait_tensor_19 = getitem_53 = None
        add_47: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_11, getitem_54);  tangents_11 = getitem_54 = None
        add_48: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_12, getitem_55);  tangents_12 = getitem_55 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:520 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_4: "f16[1024, 32, 1536]" = torch.ops.aten.empty.memory_format([1024, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=0), pin_memory = False)
        all_gather_into_tensor_5: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_46, 2, '12')
        wait_tensor_21: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        copy_8: "f16[1024, 32, 1536]" = torch.ops.aten.copy.default(empty_4, wait_tensor_21);  empty_4 = wait_tensor_21 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_58: "f16[32768, 1536]" = torch.ops.aten.view.default(copy_8, [32768, 1536])
        mm_14: "f16[32768, 768]" = torch.ops.aten.mm.default(view_58, getitem_9);  view_58 = None
        _unsafe_view_9: "f16[1024, 32, 768]" = torch.ops.aten._unsafe_view.default(mm_14, [1024, 32, 768]);  mm_14 = None
        view_60: "f16[32768, 768]" = torch.ops.aten.view.default(view_14, [32768, 768]);  view_14 = None
        view_61: "f16[32768, 1536]" = torch.ops.aten.view.default(copy_8, [32768, 1536]);  copy_8 = None
        t_22: "f16[1536, 32768]" = torch.ops.aten.t.default(view_61);  view_61 = None
        mm_15: "f16[1536, 768]" = torch.ops.aten.mm.default(t_22, view_60);  t_22 = view_60 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_49: "f16[1536, 768]" = torch.ops.aten.add.Tensor(tangents_5, mm_15);  tangents_5 = mm_15 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_62: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(_unsafe_view_9, [1024, 32, 16, 48]);  _unsafe_view_9 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_1: "f16[32, 16, 1024, 48]" = torch.ops.aten.permute.default(view_62, [1, 2, 0, 3]);  view_62 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_63: "f16[512, 1024, 48]" = torch.ops.aten.view.default(permute_1, [512, 1024, 48]);  permute_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_8: "f16[512, 1024, 1024]" = torch.ops.aten.transpose.int(view_12, 1, 2);  view_12 = None
        bmm_1: "f16[512, 1024, 48]" = torch.ops.aten.bmm.default(transpose_8, view_63);  transpose_8 = None
        transpose_9: "f16[512, 48, 1024]" = torch.ops.aten.transpose.int(transpose_4, 1, 2);  transpose_4 = None
        bmm_2: "f16[512, 1024, 1024]" = torch.ops.aten.bmm.default(view_63, transpose_9);  view_63 = transpose_9 = None
        transpose_10: "f16[1024, 512, 48]" = torch.ops.aten.transpose.int(bmm_1, 0, 1);  bmm_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_64: "f16[32, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [32, 16, 1024, 1024]);  bmm_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_65: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(transpose_10, [1024, 32, 16, 48]);  transpose_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_softmax.py:208 in forward_torch_softmax, code: probs = torch.nn.Softmax(dim=-1)(mask_output)
        detach_44: "f16[32, 16, 1024, 1024]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
        detach_45: "f16[32, 16, 1024, 1024]" = torch.ops.aten.detach.default(detach_44);  detach_44 = None
        _softmax_backward_data: "f16[32, 16, 1024, 1024]" = torch.ops.aten._softmax_backward_data.default(view_64, detach_45, -1, torch.float16);  view_64 = detach_45 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        view_66: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(_softmax_backward_data, [512, 1024, 1024]);  _softmax_backward_data = None
        new_empty_strided: "f16[512, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(view_66, [512, 1024, 1024], [1048576, 1024, 1])
        copy_9: "f16[512, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided, view_66);  new_empty_strided = view_66 = None
        view_68: "f16[32, 16, 1024, 1024]" = torch.ops.aten.view.default(copy_9, [32, 16, 1024, 1024]);  copy_9 = None
        clone_30: "f16[32, 16, 1024, 1024]" = torch.ops.aten.clone.default(view_68, memory_format = torch.contiguous_format)
        masked_fill_1: "f16[32, 16, 1024, 1024]" = torch.ops.aten.masked_fill.Scalar(clone_30, wait_tensor_3, 0);  clone_30 = wait_tensor_3 = None
        copy_10: "f16[32, 16, 1024, 1024]" = torch.ops.aten.copy.default(view_68, masked_fill_1);  view_68 = masked_fill_1 = None
        view_69: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(copy_10, [512, 1024, 1024]);  copy_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        transpose_11: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(transpose_3, 1, 2);  transpose_3 = None
        bmm_3: "f16[512, 1024, 48]" = torch.ops.aten.bmm.default(view_69, transpose_11);  transpose_11 = None
        mul_46: "f16[512, 1024, 48]" = torch.ops.aten.mul.Scalar(bmm_3, 0.14433756729740646);  bmm_3 = None
        transpose_12: "f16[512, 48, 1024]" = torch.ops.aten.transpose.int(transpose_1, 1, 2);  transpose_1 = None
        bmm_4: "f16[512, 48, 1024]" = torch.ops.aten.bmm.default(transpose_12, view_69);  transpose_12 = view_69 = None
        mul_47: "f16[512, 48, 1024]" = torch.ops.aten.mul.Scalar(bmm_4, 0.14433756729740646);  bmm_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_13: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(mul_47, 1, 2);  mul_47 = None
        transpose_14: "f16[1024, 512, 48]" = torch.ops.aten.transpose.int(transpose_13, 0, 1);  transpose_13 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_15: "f16[1024, 512, 48]" = torch.ops.aten.transpose.int(mul_46, 0, 1);  mul_46 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_71: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(transpose_14, [1024, 32, 16, 48]);  transpose_14 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_72: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(transpose_15, [1024, 32, 16, 48]);  transpose_15 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_46: "f32[524288]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
        detach_47: "f32[524288]" = torch.ops.aten.detach.default(detach_46);  detach_46 = None
        detach_48: "f32[524288]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
        detach_49: "f32[524288]" = torch.ops.aten.detach.default(detach_48);  detach_48 = None
        fused_layer_norm_affine_bwd_1 = torch.ops.apex.fused_layer_norm_affine_bwd.default(view_71, detach_47, detach_49, clone_21, [48], getitem_17, getitem_19, 1e-05);  view_71 = detach_47 = detach_49 = clone_21 = None
        getitem_56: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_bwd_1[0]
        getitem_57: "f16[48]" = fused_layer_norm_affine_bwd_1[1]
        getitem_58: "f16[48]" = fused_layer_norm_affine_bwd_1[2];  fused_layer_norm_affine_bwd_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_50: "f16[48]" = torch.ops.aten.add.Tensor(tangents_9, getitem_57);  tangents_9 = getitem_57 = None
        add_51: "f16[48]" = torch.ops.aten.add.Tensor(tangents_10, getitem_58);  tangents_10 = getitem_58 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_50: "f32[524288]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
        detach_51: "f32[524288]" = torch.ops.aten.detach.default(detach_50);  detach_50 = None
        detach_52: "f32[524288]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
        detach_53: "f32[524288]" = torch.ops.aten.detach.default(detach_52);  detach_52 = None
        fused_layer_norm_affine_bwd_2 = torch.ops.apex.fused_layer_norm_affine_bwd.default(view_72, detach_51, detach_53, clone_20, [48], getitem_13, getitem_15, 1e-05);  view_72 = detach_51 = detach_53 = clone_20 = None
        getitem_59: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_bwd_2[0]
        getitem_60: "f16[48]" = fused_layer_norm_affine_bwd_2[1]
        getitem_61: "f16[48]" = fused_layer_norm_affine_bwd_2[2];  fused_layer_norm_affine_bwd_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_52: "f16[48]" = torch.ops.aten.add.Tensor(tangents_7, getitem_60);  tangents_7 = getitem_60 = None
        add_53: "f16[48]" = torch.ops.aten.add.Tensor(tangents_8, getitem_61);  tangents_8 = getitem_61 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_73: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(getitem_59, [1024, 32, 16, 48]);  getitem_59 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        cat: "f16[1024, 32, 16, 144]" = torch.ops.aten.cat.default([view_73, getitem_56, view_65], 3);  view_73 = getitem_56 = view_65 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_74: "f16[1024, 32, 2304]" = torch.ops.aten.view.default(cat, [1024, 32, 2304]);  cat = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_6: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_34, 2, '12');  getitem_34 = None
        wait_tensor_22: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        wait_tensor_23: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_22);  wait_tensor_22 = None
        view_75: "f16[32768, 2304]" = torch.ops.aten.view.default(view_74, [32768, 2304])
        mm_16: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_75, getitem_11);  view_75 = None
        _unsafe_view_10: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_16, [1024, 32, 1536]);  mm_16 = None
        view_76: "f16[32768, 2304]" = torch.ops.aten.view.default(view_74, [32768, 2304]);  view_74 = None
        view_77: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_23, [32768, 1536]);  wait_tensor_23 = None
        reduce_scatter_tensor_4: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_10, 'sum', 2, '12');  _unsafe_view_10 = None
        wait_tensor_24: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_4);  reduce_scatter_tensor_4 = None
        t_23: "f16[2304, 32768]" = torch.ops.aten.t.default(view_76);  view_76 = None
        mm_17: "f16[2304, 1536]" = torch.ops.aten.mm.default(t_23, view_77);  t_23 = view_77 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_54: "f16[2304, 1536]" = torch.ops.aten.add.Tensor(tangents_6, mm_17);  tangents_6 = mm_17 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_54: "f32[16384]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        detach_55: "f32[16384]" = torch.ops.aten.detach.default(detach_54);  detach_54 = None
        detach_56: "f32[16384]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
        detach_57: "f32[16384]" = torch.ops.aten.detach.default(detach_56);  detach_56 = None
        fused_layer_norm_affine_bwd_3 = torch.ops.apex.fused_layer_norm_affine_bwd.default(wait_tensor_24, detach_55, detach_57, clone_19, [1536], getitem_5, getitem_7, 1e-05);  wait_tensor_24 = detach_55 = detach_57 = clone_19 = None
        getitem_62: "f16[512, 32, 1536]" = fused_layer_norm_affine_bwd_3[0]
        getitem_63: "f16[1536]" = fused_layer_norm_affine_bwd_3[1]
        getitem_64: "f16[1536]" = fused_layer_norm_affine_bwd_3[2];  fused_layer_norm_affine_bwd_3 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_55: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(add_46, getitem_62);  add_46 = getitem_62 = None
        add_56: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_3, getitem_63);  tangents_3 = getitem_63 = None
        add_57: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_4, getitem_64);  tangents_4 = getitem_64 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:500 in scatter_to_sequence_parallel_region, code: return _ScatterToSequenceParallelRegion.apply(input_)
        empty_6: "f16[1024, 32, 1536]" = torch.ops.aten.empty.memory_format([1024, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=0), pin_memory = False)
        all_gather_into_tensor_7: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_55, 2, '12');  add_55 = None
        wait_tensor_25: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        copy_11: "f16[1024, 32, 1536]" = torch.ops.aten.copy.default(empty_6, wait_tensor_25);  empty_6 = wait_tensor_25 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        transpose_17: "f16[32, 1024, 1536]" = torch.ops.aten.transpose.int(copy_11, 0, 1);  copy_11 = None
        embedding_dense_backward: "f16[2048, 1536]" = torch.ops.aten.embedding_dense_backward.default(transpose_17, wait_tensor_4, 2048, -1, False);  wait_tensor_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        add_58: "f16[2048, 1536]" = torch.ops.aten.add.Tensor(tangents_2, embedding_dense_backward);  tangents_2 = embedding_dense_backward = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:266 in forward, code: output_parallel[input_mask, :] = 0.0
        zeros_10: "f16[]" = torch.ops.aten.zeros.default([], dtype = torch.float16, layout = torch.strided, device = device(type='cpu'))
        index_put_8: "f16[32, 1024, 1536]" = torch.ops.aten.index_put.default(transpose_17, [bitwise_or], zeros_10);  transpose_17 = bitwise_or = zeros_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        embedding_dense_backward_1: "f16[49152, 1536]" = torch.ops.aten.embedding_dense_backward.default(index_put_8, index_put, 49152, -1, False);  index_put_8 = index_put = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        add_59: "f16[49152, 1536]" = torch.ops.aten.add.Tensor(add_41, embedding_dense_backward_1);  add_41 = embedding_dense_backward_1 = None
        return pytree.tree_unflatten([getitem_1, getitem_3, getitem_5, getitem_7, getitem_9, getitem_11, getitem_13, getitem_15, getitem_17, getitem_19, getitem_21, getitem_23, getitem_25, getitem_27, getitem_29, getitem_31, getitem_33, div_2, detach_31, detach_34, zeros, None, None, None, None, None, add_59, add_58, add_56, add_57, add_49, add_54, add_52, add_53, add_50, add_51, add_47, add_48, add_45, add_43, add_42, tangents_16, tangents_17], self._out_spec)
        

