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


# Graph[rank=1](both, gid=0)
class joint_helper(torch.nn.Module):
    def forward(self, primals, tangents):
        primals_1: "i64[32, 1024]"; primals_2: "i64[32, 1024]"; primals_3: "f32[32, 1024]"; primals_4: "b8[32, 1, 1024, 1024]"; primals_5: "i64[32, 1024]"; primals_6: "f16[49152, 1536]"; primals_7: "f16[2048, 1536]"; primals_8: "f16[1536]"; primals_9: "f16[1536]"; primals_10: "f16[1536, 768]"; primals_11: "f16[1536]"; primals_12: "f16[2304, 1536]"; primals_13: "f16[2304]"; primals_14: "f16[48]"; primals_15: "f16[48]"; primals_16: "f16[48]"; primals_17: "f16[48]"; primals_18: "f16[1536]"; primals_19: "f16[1536]"; primals_20: "f16[3072, 1536]"; primals_21: "f16[3072]"; primals_22: "f16[1536, 3072]"; primals_23: "f16[1536]"; primals_24: "f16[1536]"; primals_25: "f16[1536]"; tangents_1: "f16[49152, 1536]"; tangents_2: "f16[2048, 1536]"; tangents_3: "f16[1536]"; tangents_4: "f16[1536]"; tangents_5: "f16[1536, 768]"; tangents_6: "f16[1536]"; tangents_7: "f16[2304, 1536]"; tangents_8: "f16[2304]"; tangents_9: "f16[48]"; tangents_10: "f16[48]"; tangents_11: "f16[48]"; tangents_12: "f16[48]"; tangents_13: "f16[1536]"; tangents_14: "f16[1536]"; tangents_15: "f16[3072, 1536]"; tangents_16: "f16[3072]"; tangents_17: "f16[1536, 3072]"; tangents_18: "f16[1536]"; tangents_19: "f16[1536]"; tangents_20: "f16[1536]"; tangents_21: "f32[1]"; 
    
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, tangents_1, tangents_2, tangents_3, tangents_4, tangents_5, tangents_6, tangents_7, tangents_8, tangents_9, tangents_10, tangents_11, tangents_12, tangents_13, tangents_14, tangents_15, tangents_16, tangents_17, tangents_18, tangents_19, tangents_20, tangents_21, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        # No stacktrace found for following nodes
        clone: "f16[49152, 1536]" = torch.ops.aten.clone.default(primals_6);  primals_6 = None
        clone_1: "f16[2048, 1536]" = torch.ops.aten.clone.default(primals_7);  primals_7 = None
        clone_2: "f16[1536]" = torch.ops.aten.clone.default(primals_8);  primals_8 = None
        clone_3: "f16[1536]" = torch.ops.aten.clone.default(primals_9);  primals_9 = None
        clone_4: "f16[1536, 768]" = torch.ops.aten.clone.default(primals_10);  primals_10 = None
        clone_5: "f16[1536]" = torch.ops.aten.clone.default(primals_11);  primals_11 = None
        clone_6: "f16[2304, 1536]" = torch.ops.aten.clone.default(primals_12);  primals_12 = None
        clone_7: "f16[2304]" = torch.ops.aten.clone.default(primals_13);  primals_13 = None
        clone_8: "f16[48]" = torch.ops.aten.clone.default(primals_14);  primals_14 = None
        clone_9: "f16[48]" = torch.ops.aten.clone.default(primals_15);  primals_15 = None
        clone_10: "f16[48]" = torch.ops.aten.clone.default(primals_16);  primals_16 = None
        clone_11: "f16[48]" = torch.ops.aten.clone.default(primals_17);  primals_17 = None
        clone_12: "f16[1536]" = torch.ops.aten.clone.default(primals_18);  primals_18 = None
        clone_13: "f16[1536]" = torch.ops.aten.clone.default(primals_19);  primals_19 = None
        clone_14: "f16[3072, 1536]" = torch.ops.aten.clone.default(primals_20);  primals_20 = None
        clone_15: "f16[3072]" = torch.ops.aten.clone.default(primals_21);  primals_21 = None
        clone_16: "f16[1536, 3072]" = torch.ops.aten.clone.default(primals_22);  primals_22 = None
        clone_17: "f16[1536]" = torch.ops.aten.clone.default(primals_23);  primals_23 = None
        clone_18: "f16[1536]" = torch.ops.aten.clone.default(primals_24);  primals_24 = None
        clone_19: "f16[1536]" = torch.ops.aten.clone.default(primals_25);  primals_25 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:458 in forward_backward_no_pipelining, code: total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
        zeros: "i32[]" = torch.ops.aten.zeros.default([], dtype = torch.int32, device = device(type='cuda'), pin_memory = False)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:508 in get_batch_on_this_tp_rank, code: tokens = data["tokens"].cuda(non_blocking = True)
        _to_copy: "i64[32, 1024]" = torch.ops.aten._to_copy.default(primals_1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=1), non_blocking = True);  primals_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:509 in get_batch_on_this_tp_rank, code: labels = data["labels"].cuda(non_blocking = True)
        _to_copy_1: "i64[32, 1024]" = torch.ops.aten._to_copy.default(primals_2, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=1), non_blocking = True);  primals_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:510 in get_batch_on_this_tp_rank, code: loss_mask = data["loss_mask"].cuda(non_blocking = True)
        _to_copy_2: "f32[32, 1024]" = torch.ops.aten._to_copy.default(primals_3, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=1), non_blocking = True);  primals_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:511 in get_batch_on_this_tp_rank, code: attention_mask = None if "attention_mask" not in data else data["attention_mask"].cuda(non_blocking = True)
        _to_copy_3: "b8[32, 1, 1024, 1024]" = torch.ops.aten._to_copy.default(primals_4, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=1), non_blocking = True);  primals_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/training/utils.py:512 in get_batch_on_this_tp_rank, code: position_ids = data["position_ids"].cuda(non_blocking = True)
        _to_copy_4: "i64[32, 1024]" = torch.ops.aten._to_copy.default(primals_5, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=1), non_blocking = True);  primals_5 = None
        
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
        auto_functionalized_v2_5 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.linear_proj.bias', _t_base_index = 0, _all_bases = [clone_5]);  clone_5 = None
        getitem_11: "f16[1536]" = auto_functionalized_v2_5[1];  auto_functionalized_v2_5 = None
        auto_functionalized_v2_6 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.linear_qkv.weight', _t_base_index = 0, _all_bases = [clone_6]);  clone_6 = None
        getitem_13: "f16[2304, 1536]" = auto_functionalized_v2_6[1];  auto_functionalized_v2_6 = None
        auto_functionalized_v2_7 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.linear_qkv.bias', _t_base_index = 0, _all_bases = [clone_7]);  clone_7 = None
        getitem_15: "f16[2304]" = auto_functionalized_v2_7[1];  auto_functionalized_v2_7 = None
        auto_functionalized_v2_8 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.q_layernorm.weight', _t_base_index = 0, _all_bases = [clone_8]);  clone_8 = None
        getitem_17: "f16[48]" = auto_functionalized_v2_8[1];  auto_functionalized_v2_8 = None
        auto_functionalized_v2_9 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.q_layernorm.bias', _t_base_index = 0, _all_bases = [clone_9]);  clone_9 = None
        getitem_19: "f16[48]" = auto_functionalized_v2_9[1];  auto_functionalized_v2_9 = None
        auto_functionalized_v2_10 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.k_layernorm.weight', _t_base_index = 0, _all_bases = [clone_10]);  clone_10 = None
        getitem_21: "f16[48]" = auto_functionalized_v2_10[1];  auto_functionalized_v2_10 = None
        auto_functionalized_v2_11 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.self_attention.k_layernorm.bias', _t_base_index = 0, _all_bases = [clone_11]);  clone_11 = None
        getitem_23: "f16[48]" = auto_functionalized_v2_11[1];  auto_functionalized_v2_11 = None
        auto_functionalized_v2_12 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.pre_mlp_layernorm.weight', _t_base_index = 0, _all_bases = [clone_12]);  clone_12 = None
        getitem_25: "f16[1536]" = auto_functionalized_v2_12[1];  auto_functionalized_v2_12 = None
        auto_functionalized_v2_13 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.pre_mlp_layernorm.bias', _t_base_index = 0, _all_bases = [clone_13]);  clone_13 = None
        getitem_27: "f16[1536]" = auto_functionalized_v2_13[1];  auto_functionalized_v2_13 = None
        auto_functionalized_v2_14 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.linear_fc1.weight', _t_base_index = 0, _all_bases = [clone_14]);  clone_14 = None
        getitem_29: "f16[3072, 1536]" = auto_functionalized_v2_14[1];  auto_functionalized_v2_14 = None
        auto_functionalized_v2_15 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.linear_fc1.bias', _t_base_index = 0, _all_bases = [clone_15]);  clone_15 = None
        getitem_31: "f16[3072]" = auto_functionalized_v2_15[1];  auto_functionalized_v2_15 = None
        auto_functionalized_v2_16 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.linear_fc2.weight', _t_base_index = 0, _all_bases = [clone_16]);  clone_16 = None
        getitem_33: "f16[1536, 3072]" = auto_functionalized_v2_16[1];  auto_functionalized_v2_16 = None
        auto_functionalized_v2_17 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.layers.0.mlp.linear_fc2.bias', _t_base_index = 0, _all_bases = [clone_17]);  clone_17 = None
        getitem_35: "f16[1536]" = auto_functionalized_v2_17[1];  auto_functionalized_v2_17 = None
        auto_functionalized_v2_18 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.final_layernorm.weight', _t_base_index = 0, _all_bases = [clone_18]);  clone_18 = None
        getitem_37: "f16[1536]" = auto_functionalized_v2_18[1];  auto_functionalized_v2_18 = None
        auto_functionalized_v2_19 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.tg.inplace_log_tensor.default, s = 'module.module.decoder.final_layernorm.bias', _t_base_index = 0, _all_bases = [clone_19]);  clone_19 = None
        getitem_39: "f16[1536]" = auto_functionalized_v2_19[1];  auto_functionalized_v2_19 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:252 in forward, code: input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        lt: "b8[32, 1024]" = torch.ops.aten.lt.Scalar(wait_tensor, 49152)
        ge: "b8[32, 1024]" = torch.ops.aten.ge.Scalar(wait_tensor, 98304)
        bitwise_or: "b8[32, 1024]" = torch.ops.aten.bitwise_or.Tensor(lt, ge);  lt = ge = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:254 in forward, code: masked_input = input_.clone() - self.vocab_start_index
        clone_20: "i64[32, 1024]" = torch.ops.aten.clone.default(wait_tensor);  wait_tensor = None
        sub: "i64[32, 1024]" = torch.ops.aten.sub.Tensor(clone_20, 49152);  clone_20 = None
        
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
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:482 in reduce_from_tensor_model_parallel_region, code: return _ReduceFromModelParallelRegion.apply(input_)
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
        clone_21: "f16[1024, 32, 1536]" = torch.ops.aten.clone.default(transpose, memory_format = torch.contiguous_format);  transpose = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:497 in scatter_to_sequence_parallel_region, code: return _ScatterToSequenceParallelRegion.apply(input_)
        slice_1: "f16[512, 32, 1536]" = torch.ops.aten.slice.Tensor(clone_21, 0, 512, 1024);  clone_21 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:140 in forward, code: embeddings = embeddings.clone()
        clone_22: "f16[512, 32, 1536]" = torch.ops.aten.clone.default(slice_1);  slice_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd = torch.ops.apex.fused_layer_norm_affine_fwd.default(clone_22, getitem_5, getitem_7, [1536], 1e-05)
        getitem_40: "f16[512, 32, 1536]" = fused_layer_norm_affine_fwd[0]
        getitem_41: "f32[16384]" = fused_layer_norm_affine_fwd[1]
        getitem_42: "f32[16384]" = fused_layer_norm_affine_fwd[2];  fused_layer_norm_affine_fwd = None
        detach: "f32[16384]" = torch.ops.aten.detach.default(getitem_41);  getitem_41 = None
        detach_1: "f32[16384]" = torch.ops.aten.detach.default(detach);  detach = None
        detach_2: "f32[16384]" = torch.ops.aten.detach.default(getitem_42);  getitem_42 = None
        detach_3: "f32[16384]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_40, 2, '12')
        wait_tensor_6: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor);  all_gather_into_tensor = None
        view_2: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_6, [32768, 1536]);  wait_tensor_6 = None
        t_1: "f16[1536, 2304]" = torch.ops.aten.t.default(getitem_13)
        mm: "f16[32768, 2304]" = torch.ops.aten.mm.default(view_2, t_1);  view_2 = t_1 = None
        _unsafe_view: "f16[1024, 32, 2304]" = torch.ops.aten._unsafe_view.default(mm, [1024, 32, 2304]);  mm = None
        add_1: "f16[1024, 32, 2304]" = torch.ops.aten.add.Tensor(_unsafe_view, getitem_15);  _unsafe_view = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_3: "f16[1024, 32, 16, 144]" = torch.ops.aten.view.default(add_1, [1024, 32, 16, 144]);  add_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_3, [48, 48, 48], 3);  view_3 = None
        getitem_43: "f16[1024, 32, 16, 48]" = split_with_sizes[0]
        getitem_44: "f16[1024, 32, 16, 48]" = split_with_sizes[1]
        getitem_45: "f16[1024, 32, 16, 48]" = split_with_sizes[2];  split_with_sizes = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_4: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(getitem_43, [1024, 32, 16, 48]);  getitem_43 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_1 = torch.ops.apex.fused_layer_norm_affine_fwd.default(view_4, getitem_17, getitem_19, [48], 1e-05)
        getitem_46: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_fwd_1[0]
        getitem_47: "f32[524288]" = fused_layer_norm_affine_fwd_1[1]
        getitem_48: "f32[524288]" = fused_layer_norm_affine_fwd_1[2];  fused_layer_norm_affine_fwd_1 = None
        clone_23: "f16[1024, 32, 16, 48]" = torch.ops.aten.clone.default(view_4, memory_format = torch.contiguous_format);  view_4 = None
        detach_4: "f32[524288]" = torch.ops.aten.detach.default(getitem_47);  getitem_47 = None
        detach_5: "f32[524288]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
        detach_6: "f32[524288]" = torch.ops.aten.detach.default(getitem_48);  getitem_48 = None
        detach_7: "f32[524288]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_2 = torch.ops.apex.fused_layer_norm_affine_fwd.default(getitem_44, getitem_21, getitem_23, [48], 1e-05)
        getitem_49: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_fwd_2[0]
        getitem_50: "f32[524288]" = fused_layer_norm_affine_fwd_2[1]
        getitem_51: "f32[524288]" = fused_layer_norm_affine_fwd_2[2];  fused_layer_norm_affine_fwd_2 = None
        clone_24: "f16[1024, 32, 16, 48]" = torch.ops.aten.clone.default(getitem_44, memory_format = torch.contiguous_format);  getitem_44 = None
        detach_8: "f32[524288]" = torch.ops.aten.detach.default(getitem_50);  getitem_50 = None
        detach_9: "f32[524288]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
        detach_10: "f32[524288]" = torch.ops.aten.detach.default(getitem_51);  getitem_51 = None
        detach_11: "f32[524288]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_5: "f16[1024, 512, 48]" = torch.ops.aten.view.default(getitem_46, [1024, 512, -1]);  getitem_46 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_6: "f16[1024, 512, 48]" = torch.ops.aten.view.default(getitem_49, [1024, 512, -1]);  getitem_49 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:149 in forward, code: matmul_input_buffer = torch.empty((output_size[0] * output_size[1], output_size[2], output_size[3]), dtype=query.dtype, device=query.device)
        empty: "f16[512, 1024, 1024]" = torch.ops.aten.empty.memory_format([512, 1024, 1024], dtype = torch.float16, device = device(type='cuda', index=1), pin_memory = False)
        
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
        view_11: "f16[1024, 512, 48]" = torch.ops.aten.view.default(getitem_45, [1024, 512, -1]);  getitem_45 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_12: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(_softmax, [512, 1024, -1]);  _softmax = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_4: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(view_11, 0, 1);  view_11 = None
        bmm: "f16[512, 1024, 48]" = torch.ops.aten.bmm.default(view_12, transpose_4)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_13: "f16[32, 16, 1024, 48]" = torch.ops.aten.view.default(bmm, [32, 16, 1024, 48]);  bmm = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute: "f16[1024, 32, 16, 48]" = torch.ops.aten.permute.default(view_13, [2, 0, 1, 3]);  view_13 = None
        clone_25: "f16[1024, 32, 16, 48]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_14: "f16[1024, 32, 768]" = torch.ops.aten.view.default(clone_25, [1024, 32, 768]);  clone_25 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_15: "f16[32768, 768]" = torch.ops.aten.view.default(view_14, [32768, 768])
        t_3: "f16[768, 1536]" = torch.ops.aten.t.default(getitem_9)
        mm_1: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_15, t_3);  view_15 = t_3 = None
        _unsafe_view_1: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_1, [1024, 32, 1536]);  mm_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:517 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_1: "f16[512, 32, 1536]" = torch.ops.aten.empty.memory_format([512, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_1, 'sum', 2, '12');  _unsafe_view_1 = None
        wait_tensor_7: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor);  reduce_scatter_tensor = None
        copy_1: "f16[512, 32, 1536]" = torch.ops.aten.copy.default(empty_1, wait_tensor_7);  empty_1 = wait_tensor_7 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        add_2: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(copy_1, getitem_11);  copy_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:38 in _bias_dropout_add_func, code: out = residual + out
        add_3: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(clone_22, add_2);  add_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        fused_layer_norm_affine_fwd_3 = torch.ops.apex.fused_layer_norm_affine_fwd.default(add_3, getitem_25, getitem_27, [1536], 1e-05)
        getitem_52: "f16[512, 32, 1536]" = fused_layer_norm_affine_fwd_3[0]
        getitem_53: "f32[16384]" = fused_layer_norm_affine_fwd_3[1]
        getitem_54: "f32[16384]" = fused_layer_norm_affine_fwd_3[2];  fused_layer_norm_affine_fwd_3 = None
        detach_14: "f32[16384]" = torch.ops.aten.detach.default(getitem_53);  getitem_53 = None
        detach_15: "f32[16384]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
        detach_16: "f32[16384]" = torch.ops.aten.detach.default(getitem_54);  getitem_54 = None
        detach_17: "f32[16384]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_1: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_52, 2, '12')
        wait_tensor_8: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_1);  all_gather_into_tensor_1 = None
        view_16: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_8, [32768, 1536]);  wait_tensor_8 = None
        t_5: "f16[1536, 3072]" = torch.ops.aten.t.default(getitem_29)
        mm_2: "f16[32768, 3072]" = torch.ops.aten.mm.default(view_16, t_5);  view_16 = t_5 = None
        _unsafe_view_2: "f16[1024, 32, 3072]" = torch.ops.aten._unsafe_view.default(mm_2, [1024, 32, 3072]);  mm_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_4: "f16[1024, 32, 3072]" = torch.ops.aten.add.Tensor(_unsafe_view_2, getitem_31);  _unsafe_view_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu: "f16[1024, 32, 3072]" = torch.ops.aten.gelu.default(add_4)
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_17: "f16[32768, 3072]" = torch.ops.aten.view.default(gelu, [32768, 3072])
        t_7: "f16[3072, 1536]" = torch.ops.aten.t.default(getitem_33)
        mm_3: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_17, t_7);  view_17 = t_7 = None
        _unsafe_view_3: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_3, [1024, 32, 1536]);  mm_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:517 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_2: "f16[512, 32, 1536]" = torch.ops.aten.empty.memory_format([512, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=1), pin_memory = False)
        reduce_scatter_tensor_1: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_3, 'sum', 2, '12');  _unsafe_view_3 = None
        wait_tensor_9: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_1);  reduce_scatter_tensor_1 = None
        copy_2: "f16[512, 32, 1536]" = torch.ops.aten.copy.default(empty_2, wait_tensor_9);  empty_2 = wait_tensor_9 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        add_5: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(copy_2, getitem_35);  copy_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:38 in _bias_dropout_add_func, code: out = residual + out
        add_6: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(add_3, add_5);  add_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_2: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_6, 2, '12')
        wait_tensor_10: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_2);  all_gather_into_tensor_2 = None
        view_18: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_10, [32768, 1536]);  wait_tensor_10 = None
        t_9: "f16[1536, 49152]" = torch.ops.aten.t.default(getitem_1)
        mm_4: "f16[32768, 49152]" = torch.ops.aten.mm.default(view_18, t_9);  view_18 = t_9 = None
        _unsafe_view_4: "f16[1024, 32, 49152]" = torch.ops.aten._unsafe_view.default(mm_4, [1024, 32, 49152]);  mm_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:77 in compute_language_model_loss, code: labels = labels.transpose(0, 1).contiguous()
        transpose_5: "i64[1024, 32]" = torch.ops.aten.transpose.int(wait_tensor_1, 0, 1);  wait_tensor_1 = None
        clone_26: "i64[1024, 32]" = torch.ops.aten.clone.default(transpose_5, memory_format = torch.contiguous_format);  transpose_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:235 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        _to_copy_5: "f32[1024, 32, 49152]" = torch.ops.aten._to_copy.default(_unsafe_view_4, dtype = torch.float32);  _unsafe_view_4 = None
        max_1 = torch.ops.aten.max.dim(_to_copy_5, -1)
        getitem_55: "f32[1024, 32]" = max_1[0];  max_1 = None
        all_reduce_1: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(getitem_55, 'max', '12')
        wait_tensor_11: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        copy_3: "f32[1024, 32]" = torch.ops.aten.copy.default(getitem_55, wait_tensor_11);  getitem_55 = wait_tensor_11 = None
        unsqueeze_1: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(copy_3, -1);  copy_3 = None
        sub_1: "f32[1024, 32, 49152]" = torch.ops.aten.sub.Tensor(_to_copy_5, unsqueeze_1);  _to_copy_5 = unsqueeze_1 = None
        lt_1: "b8[1024, 32]" = torch.ops.aten.lt.Scalar(clone_26, 49152)
        ge_1: "b8[1024, 32]" = torch.ops.aten.ge.Scalar(clone_26, 98304)
        bitwise_or_1: "b8[1024, 32]" = torch.ops.aten.bitwise_or.Tensor(lt_1, ge_1);  lt_1 = ge_1 = None
        clone_27: "i64[1024, 32]" = torch.ops.aten.clone.default(clone_26);  clone_26 = None
        sub_2: "i64[1024, 32]" = torch.ops.aten.sub.Tensor(clone_27, 49152);  clone_27 = None
        _tensor_constant2: "i64[]" = self._tensor_constant2
        lift_fresh_copy_2: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
        index_put_2: "i64[1024, 32]" = torch.ops.aten.index_put.default(sub_2, [bitwise_or_1], lift_fresh_copy_2);  sub_2 = lift_fresh_copy_2 = None
        arange: "i64[32768]" = torch.ops.aten.arange.start(0, 32768, device = device(type='cuda', index=1), pin_memory = False)
        view_21: "f32[32768, 49152]" = torch.ops.aten.view.default(sub_1, [-1, 49152])
        view_22: "i64[32768]" = torch.ops.aten.view.default(index_put_2, [-1]);  index_put_2 = None
        index: "f32[32768]" = torch.ops.aten.index.Tensor(view_21, [arange, view_22]);  view_21 = arange = None
        clone_28: "f32[32768]" = torch.ops.aten.clone.default(index);  index = None
        view_23: "f32[1024, 32]" = torch.ops.aten.view.default(clone_28, [1024, 32]);  clone_28 = None
        _tensor_constant3: "f32[]" = self._tensor_constant3
        lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
        index_put_3: "f32[1024, 32]" = torch.ops.aten.index_put.default(view_23, [bitwise_or_1], lift_fresh_copy_3);  view_23 = lift_fresh_copy_3 = None
        view_24: "f32[32768]" = torch.ops.aten.view.default(index_put_3, [32768]);  index_put_3 = None
        view_25: "f32[1024, 32]" = torch.ops.aten.view.default(view_24, [1024, 32]);  view_24 = None
        exp: "f32[1024, 32, 49152]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[1024, 32]" = torch.ops.aten.sum.dim_IntList(exp, [-1])
        all_reduce_2: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(view_25, 'sum', '12')
        wait_tensor_12: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = None
        copy_4: "f32[1024, 32]" = torch.ops.aten.copy.default(view_25, wait_tensor_12);  view_25 = wait_tensor_12 = None
        view_26: "f32[32768]" = torch.ops.aten.view.default(copy_4, [32768]);  copy_4 = None
        view_27: "f32[1024, 32]" = torch.ops.aten.view.default(view_26, [1024, 32]);  view_26 = None
        all_reduce_3: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(sum_1, 'sum', '12')
        wait_tensor_13: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = None
        copy_5: "f32[1024, 32]" = torch.ops.aten.copy.default(sum_1, wait_tensor_13);  sum_1 = wait_tensor_13 = None
        log: "f32[1024, 32]" = torch.ops.aten.log.default(copy_5)
        sub_3: "f32[1024, 32]" = torch.ops.aten.sub.Tensor(log, view_27);  log = view_27 = None
        unsqueeze_3: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(copy_5, -1);  copy_5 = None
        div: "f32[1024, 32, 49152]" = torch.ops.aten.div.Tensor(exp, unsqueeze_3);  exp = unsqueeze_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_6: "f32[32, 1024]" = torch.ops.aten.transpose.int(sub_3, 0, 1);  sub_3 = None
        clone_29: "f32[32, 1024]" = torch.ops.aten.clone.default(transpose_6, memory_format = torch.contiguous_format);  transpose_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:171 in loss_func, code: loss_mask = loss_mask.view(-1).float()
        view_28: "f32[32768]" = torch.ops.aten.view.default(wait_tensor_2, [-1]);  wait_tensor_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:172 in loss_func, code: total_tokens = loss_mask.sum().view(1)
        sum_2: "f32[]" = torch.ops.aten.sum.default(view_28)
        view_29: "f32[1]" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:173 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask).view(1)
        view_30: "f32[32768]" = torch.ops.aten.view.default(clone_29, [-1]);  clone_29 = None
        mul: "f32[32768]" = torch.ops.aten.mul.Tensor(view_30, view_28);  view_30 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
        view_31: "f32[1]" = torch.ops.aten.view.default(sum_3, [1]);  sum_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:210 in loss_func, code: reporting_loss = loss.clone().detach()
        clone_30: "f32[1]" = torch.ops.aten.clone.default(view_31)
        detach_18: "f32[1]" = torch.ops.aten.detach.default(clone_30);  clone_30 = None
        detach_19: "f32[1]" = torch.ops.aten.detach.default(detach_18);  detach_18 = None
        detach_20: "f32[1]" = torch.ops.aten.detach.default(detach_19);  detach_19 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:211 in loss_func, code: reporting_total_tokens = total_tokens.clone().detach()
        clone_31: "f32[1]" = torch.ops.aten.clone.default(view_29)
        detach_21: "f32[1]" = torch.ops.aten.detach.default(clone_31);  clone_31 = None
        detach_22: "f32[1]" = torch.ops.aten.detach.default(detach_21);  detach_21 = None
        detach_23: "f32[1]" = torch.ops.aten.detach.default(detach_22);  detach_22 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:170 in all_reduce, code: tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
        all_reduce_4: "f32[1]" = torch.ops._c10d_functional.all_reduce.default(detach_20, 'sum', '3')
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_14: "f32[1]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_4);  all_reduce_4 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:1112 in all_reduce_inplace, code: return tensor.copy_(all_reduce(tensor, op, group, tag))
        copy_6: "f32[1]" = torch.ops.aten.copy.default(detach_20, wait_tensor_14);  detach_20 = wait_tensor_14 = None
        detach_25: "f32[1]" = torch.ops.aten.detach.default(copy_6);  copy_6 = None
        detach_26: "f32[1]" = torch.ops.aten.detach.default(detach_25);  detach_25 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:170 in all_reduce, code: tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
        all_reduce_5: "f32[1]" = torch.ops._c10d_functional.all_reduce.default(detach_23, 'sum', '3')
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_15: "f32[1]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_5);  all_reduce_5 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:1112 in all_reduce_inplace, code: return tensor.copy_(all_reduce(tensor, op, group, tag))
        copy_7: "f32[1]" = torch.ops.aten.copy.default(detach_23, wait_tensor_15);  detach_23 = wait_tensor_15 = None
        detach_28: "f32[1]" = torch.ops.aten.detach.default(copy_7);  copy_7 = None
        detach_29: "f32[1]" = torch.ops.aten.detach.default(detach_28);  detach_28 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:215 in loss_func, code: local_num_tokens = total_tokens.clone().detach().to(torch.int)
        clone_32: "f32[1]" = torch.ops.aten.clone.default(view_29);  view_29 = None
        detach_30: "f32[1]" = torch.ops.aten.detach.default(clone_32);  clone_32 = None
        detach_31: "f32[1]" = torch.ops.aten.detach.default(detach_30);  detach_30 = None
        detach_32: "f32[1]" = torch.ops.aten.detach.default(detach_31);  detach_31 = None
        _to_copy_6: "i32[1]" = torch.ops.aten._to_copy.default(detach_32, dtype = torch.int32);  detach_32 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:218 in loss_func, code: loss * args.context_parallel_size,
        mul_1: "f32[1]" = torch.ops.aten.mul.Tensor(view_31, 1);  view_31 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:289 in forward_step, code: output_tensor /= num_tokens
        div_1: "f32[1]" = torch.ops.aten.div.Tensor(mul_1, _to_copy_6);  mul_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:290 in forward_step, code: output_tensor /= num_microbatches
        div_2: "f32[1]" = torch.ops.aten.div.Tensor(div_1, 1);  div_1 = None
        div_3: "f32[1]" = torch.ops.aten.div.Tensor(tangents_21, 1);  tangents_21 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:289 in forward_step, code: output_tensor /= num_tokens
        div_4: "f32[1]" = torch.ops.aten.div.Tensor(div_3, _to_copy_6);  div_3 = _to_copy_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:218 in loss_func, code: loss * args.context_parallel_size,
        mul_2: "f32[1]" = torch.ops.aten.mul.Tensor(div_4, 1);  div_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:173 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask).view(1)
        view_32: "f32[]" = torch.ops.aten.view.default(mul_2, []);  mul_2 = None
        expand: "f32[32768]" = torch.ops.aten.expand.default(view_32, [32768]);  view_32 = None
        mul_3: "f32[32768]" = torch.ops.aten.mul.Tensor(expand, view_28);  expand = view_28 = None
        view_33: "f32[32, 1024]" = torch.ops.aten.view.default(mul_3, [32, 1024]);  mul_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_7: "f32[1024, 32]" = torch.ops.aten.transpose.int(view_33, 0, 1);  view_33 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:235 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        arange_1: "i64[32768]" = torch.ops.aten.arange.start(0, 32768, device = device(type='cuda', index=1), pin_memory = False)
        view_35: "b8[32768]" = torch.ops.aten.view.default(bitwise_or_1, [-1]);  bitwise_or_1 = None
        _to_copy_7: "f32[32768]" = torch.ops.aten._to_copy.default(view_35, dtype = torch.float32);  view_35 = None
        rsub: "f32[32768]" = torch.ops.aten.rsub.Scalar(_to_copy_7, 1.0);  _to_copy_7 = None
        view_36: "f32[32768, 49152]" = torch.ops.aten.view.default(div, [-1, 49152]);  div = None
        index_1: "f32[32768]" = torch.ops.aten.index.Tensor(view_36, [arange_1, view_22])
        sub_4: "f32[32768]" = torch.ops.aten.sub.Tensor(index_1, rsub);  index_1 = rsub = None
        index_put_4: "f32[32768, 49152]" = torch.ops.aten.index_put.default(view_36, [arange_1, view_22], sub_4);  view_36 = arange_1 = view_22 = sub_4 = None
        view_37: "f32[1024, 32, 49152]" = torch.ops.aten.view.default(index_put_4, [1024, 32, 49152]);  index_put_4 = None
        unsqueeze_4: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(transpose_7, -1);  transpose_7 = None
        mul_4: "f32[1024, 32, 49152]" = torch.ops.aten.mul.Tensor(view_37, unsqueeze_4);  view_37 = unsqueeze_4 = None
        _to_copy_8: "f16[1024, 32, 49152]" = torch.ops.aten._to_copy.default(mul_4, dtype = torch.float16);  mul_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_3: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_6, 2, '12');  add_6 = None
        wait_tensor_16: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_3);  all_gather_into_tensor_3 = None
        wait_tensor_17: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_16);  wait_tensor_16 = None
        view_39: "f16[32768, 49152]" = torch.ops.aten.view.default(_to_copy_8, [32768, 49152])
        mm_5: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_39, getitem_1);  view_39 = None
        _unsafe_view_5: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_5, [1024, 32, 1536]);  mm_5 = None
        view_40: "f16[32768, 49152]" = torch.ops.aten.view.default(_to_copy_8, [32768, 49152]);  _to_copy_8 = None
        view_41: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_17, [32768, 1536]);  wait_tensor_17 = None
        reduce_scatter_tensor_2: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_5, 'sum', 2, '12');  _unsafe_view_5 = None
        wait_tensor_18: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_2);  reduce_scatter_tensor_2 = None
        t_10: "f16[49152, 32768]" = torch.ops.aten.t.default(view_40);  view_40 = None
        mm_6: "f16[49152, 1536]" = torch.ops.aten.mm.default(t_10, view_41);  t_10 = view_41 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_7: "f16[49152, 1536]" = torch.ops.aten.add.Tensor(tangents_1, mm_6);  tangents_1 = mm_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        sum_4: "f16[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(wait_tensor_18, [0, 1], True)
        view_42: "f16[1536]" = torch.ops.aten.view.default(sum_4, [1536]);  sum_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        add_8: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_18, view_42);  tangents_18 = view_42 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:517 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_4: "f16[1024, 32, 1536]" = torch.ops.aten.empty.memory_format([1024, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=1), pin_memory = False)
        all_gather_into_tensor_4: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(wait_tensor_18, 2, '12')
        wait_tensor_19: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_4);  all_gather_into_tensor_4 = None
        copy_8: "f16[1024, 32, 1536]" = torch.ops.aten.copy.default(empty_4, wait_tensor_19);  empty_4 = wait_tensor_19 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_44: "f16[32768, 1536]" = torch.ops.aten.view.default(copy_8, [32768, 1536])
        mm_7: "f16[32768, 3072]" = torch.ops.aten.mm.default(view_44, getitem_33);  view_44 = None
        _unsafe_view_6: "f16[1024, 32, 3072]" = torch.ops.aten._unsafe_view.default(mm_7, [1024, 32, 3072]);  mm_7 = None
        view_46: "f16[32768, 3072]" = torch.ops.aten.view.default(gelu, [32768, 3072]);  gelu = None
        view_47: "f16[32768, 1536]" = torch.ops.aten.view.default(copy_8, [32768, 1536]);  copy_8 = None
        t_12: "f16[1536, 32768]" = torch.ops.aten.t.default(view_47);  view_47 = None
        mm_8: "f16[1536, 3072]" = torch.ops.aten.mm.default(t_12, view_46);  t_12 = view_46 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_9: "f16[1536, 3072]" = torch.ops.aten.add.Tensor(tangents_17, mm_8);  tangents_17 = mm_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_backward: "f16[1024, 32, 3072]" = torch.ops.aten.gelu_backward.default(_unsafe_view_6, add_4);  _unsafe_view_6 = add_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        sum_5: "f16[1, 1, 3072]" = torch.ops.aten.sum.dim_IntList(gelu_backward, [0, 1], True)
        view_48: "f16[3072]" = torch.ops.aten.view.default(sum_5, [3072]);  sum_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_10: "f16[3072]" = torch.ops.aten.add.Tensor(tangents_16, view_48);  tangents_16 = view_48 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_5: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_52, 2, '12');  getitem_52 = None
        wait_tensor_20: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_5);  all_gather_into_tensor_5 = None
        wait_tensor_21: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_20);  wait_tensor_20 = None
        view_49: "f16[32768, 3072]" = torch.ops.aten.view.default(gelu_backward, [32768, 3072])
        mm_9: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_49, getitem_29);  view_49 = None
        _unsafe_view_7: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_9, [1024, 32, 1536]);  mm_9 = None
        view_50: "f16[32768, 3072]" = torch.ops.aten.view.default(gelu_backward, [32768, 3072]);  gelu_backward = None
        view_51: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_21, [32768, 1536]);  wait_tensor_21 = None
        reduce_scatter_tensor_3: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_7, 'sum', 2, '12');  _unsafe_view_7 = None
        wait_tensor_22: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_3);  reduce_scatter_tensor_3 = None
        t_13: "f16[3072, 32768]" = torch.ops.aten.t.default(view_50);  view_50 = None
        mm_10: "f16[3072, 1536]" = torch.ops.aten.mm.default(t_13, view_51);  t_13 = view_51 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_11: "f16[3072, 1536]" = torch.ops.aten.add.Tensor(tangents_15, mm_10);  tangents_15 = mm_10 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_33: "f32[16384]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
        detach_34: "f32[16384]" = torch.ops.aten.detach.default(detach_33);  detach_33 = None
        detach_35: "f32[16384]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
        detach_36: "f32[16384]" = torch.ops.aten.detach.default(detach_35);  detach_35 = None
        fused_layer_norm_affine_bwd = torch.ops.apex.fused_layer_norm_affine_bwd.default(wait_tensor_22, detach_34, detach_36, add_3, [1536], getitem_25, getitem_27, 1e-05);  wait_tensor_22 = detach_34 = detach_36 = add_3 = None
        getitem_57: "f16[512, 32, 1536]" = fused_layer_norm_affine_bwd[0]
        getitem_58: "f16[1536]" = fused_layer_norm_affine_bwd[1]
        getitem_59: "f16[1536]" = fused_layer_norm_affine_bwd[2];  fused_layer_norm_affine_bwd = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_12: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(wait_tensor_18, getitem_57);  wait_tensor_18 = getitem_57 = None
        add_13: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_13, getitem_58);  tangents_13 = getitem_58 = None
        add_14: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_14, getitem_59);  tangents_14 = getitem_59 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        sum_6: "f16[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1], True)
        view_52: "f16[1536]" = torch.ops.aten.view.default(sum_6, [1536]);  sum_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        add_15: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_6, view_52);  tangents_6 = view_52 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:517 in reduce_scatter_to_sequence_parallel_region, code: return _ReduceScatterToSequenceParallelRegion.apply(
        empty_6: "f16[1024, 32, 1536]" = torch.ops.aten.empty.memory_format([1024, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=1), pin_memory = False)
        all_gather_into_tensor_6: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_12, 2, '12')
        wait_tensor_23: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_6);  all_gather_into_tensor_6 = None
        copy_9: "f16[1024, 32, 1536]" = torch.ops.aten.copy.default(empty_6, wait_tensor_23);  empty_6 = wait_tensor_23 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_54: "f16[32768, 1536]" = torch.ops.aten.view.default(copy_9, [32768, 1536])
        mm_11: "f16[32768, 768]" = torch.ops.aten.mm.default(view_54, getitem_9);  view_54 = None
        _unsafe_view_8: "f16[1024, 32, 768]" = torch.ops.aten._unsafe_view.default(mm_11, [1024, 32, 768]);  mm_11 = None
        view_56: "f16[32768, 768]" = torch.ops.aten.view.default(view_14, [32768, 768]);  view_14 = None
        view_57: "f16[32768, 1536]" = torch.ops.aten.view.default(copy_9, [32768, 1536]);  copy_9 = None
        t_15: "f16[1536, 32768]" = torch.ops.aten.t.default(view_57);  view_57 = None
        mm_12: "f16[1536, 768]" = torch.ops.aten.mm.default(t_15, view_56);  t_15 = view_56 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_16: "f16[1536, 768]" = torch.ops.aten.add.Tensor(tangents_5, mm_12);  tangents_5 = mm_12 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_58: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(_unsafe_view_8, [1024, 32, 16, 48]);  _unsafe_view_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_1: "f16[32, 16, 1024, 48]" = torch.ops.aten.permute.default(view_58, [1, 2, 0, 3]);  view_58 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_59: "f16[512, 1024, 48]" = torch.ops.aten.view.default(permute_1, [512, 1024, 48]);  permute_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_8: "f16[512, 1024, 1024]" = torch.ops.aten.transpose.int(view_12, 1, 2);  view_12 = None
        bmm_1: "f16[512, 1024, 48]" = torch.ops.aten.bmm.default(transpose_8, view_59);  transpose_8 = None
        transpose_9: "f16[512, 48, 1024]" = torch.ops.aten.transpose.int(transpose_4, 1, 2);  transpose_4 = None
        bmm_2: "f16[512, 1024, 1024]" = torch.ops.aten.bmm.default(view_59, transpose_9);  view_59 = transpose_9 = None
        transpose_10: "f16[1024, 512, 48]" = torch.ops.aten.transpose.int(bmm_1, 0, 1);  bmm_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_60: "f16[32, 16, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [32, 16, 1024, 1024]);  bmm_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_61: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(transpose_10, [1024, 32, 16, 48]);  transpose_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_softmax.py:208 in forward_torch_softmax, code: probs = torch.nn.Softmax(dim=-1)(mask_output)
        detach_37: "f16[32, 16, 1024, 1024]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
        detach_38: "f16[32, 16, 1024, 1024]" = torch.ops.aten.detach.default(detach_37);  detach_37 = None
        _softmax_backward_data: "f16[32, 16, 1024, 1024]" = torch.ops.aten._softmax_backward_data.default(view_60, detach_38, -1, torch.float16);  view_60 = detach_38 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        view_62: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(_softmax_backward_data, [512, 1024, 1024]);  _softmax_backward_data = None
        new_empty_strided: "f16[512, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(view_62, [512, 1024, 1024], [1048576, 1024, 1])
        copy_10: "f16[512, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided, view_62);  new_empty_strided = view_62 = None
        view_64: "f16[32, 16, 1024, 1024]" = torch.ops.aten.view.default(copy_10, [32, 16, 1024, 1024]);  copy_10 = None
        clone_33: "f16[32, 16, 1024, 1024]" = torch.ops.aten.clone.default(view_64, memory_format = torch.contiguous_format)
        masked_fill_1: "f16[32, 16, 1024, 1024]" = torch.ops.aten.masked_fill.Scalar(clone_33, wait_tensor_3, 0);  clone_33 = wait_tensor_3 = None
        copy_11: "f16[32, 16, 1024, 1024]" = torch.ops.aten.copy.default(view_64, masked_fill_1);  view_64 = masked_fill_1 = None
        view_65: "f16[512, 1024, 1024]" = torch.ops.aten.view.default(copy_11, [512, 1024, 1024]);  copy_11 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        transpose_11: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(transpose_3, 1, 2);  transpose_3 = None
        bmm_3: "f16[512, 1024, 48]" = torch.ops.aten.bmm.default(view_65, transpose_11);  transpose_11 = None
        mul_5: "f16[512, 1024, 48]" = torch.ops.aten.mul.Scalar(bmm_3, 0.14433756729740646);  bmm_3 = None
        transpose_12: "f16[512, 48, 1024]" = torch.ops.aten.transpose.int(transpose_1, 1, 2);  transpose_1 = None
        bmm_4: "f16[512, 48, 1024]" = torch.ops.aten.bmm.default(transpose_12, view_65);  transpose_12 = view_65 = None
        mul_6: "f16[512, 48, 1024]" = torch.ops.aten.mul.Scalar(bmm_4, 0.14433756729740646);  bmm_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_13: "f16[512, 1024, 48]" = torch.ops.aten.transpose.int(mul_6, 1, 2);  mul_6 = None
        transpose_14: "f16[1024, 512, 48]" = torch.ops.aten.transpose.int(transpose_13, 0, 1);  transpose_13 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_15: "f16[1024, 512, 48]" = torch.ops.aten.transpose.int(mul_5, 0, 1);  mul_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_67: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(transpose_14, [1024, 32, 16, 48]);  transpose_14 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_68: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(transpose_15, [1024, 32, 16, 48]);  transpose_15 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_39: "f32[524288]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
        detach_40: "f32[524288]" = torch.ops.aten.detach.default(detach_39);  detach_39 = None
        detach_41: "f32[524288]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
        detach_42: "f32[524288]" = torch.ops.aten.detach.default(detach_41);  detach_41 = None
        fused_layer_norm_affine_bwd_1 = torch.ops.apex.fused_layer_norm_affine_bwd.default(view_67, detach_40, detach_42, clone_24, [48], getitem_21, getitem_23, 1e-05);  view_67 = detach_40 = detach_42 = clone_24 = None
        getitem_60: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_bwd_1[0]
        getitem_61: "f16[48]" = fused_layer_norm_affine_bwd_1[1]
        getitem_62: "f16[48]" = fused_layer_norm_affine_bwd_1[2];  fused_layer_norm_affine_bwd_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_17: "f16[48]" = torch.ops.aten.add.Tensor(tangents_11, getitem_61);  tangents_11 = getitem_61 = None
        add_18: "f16[48]" = torch.ops.aten.add.Tensor(tangents_12, getitem_62);  tangents_12 = getitem_62 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_43: "f32[524288]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
        detach_44: "f32[524288]" = torch.ops.aten.detach.default(detach_43);  detach_43 = None
        detach_45: "f32[524288]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
        detach_46: "f32[524288]" = torch.ops.aten.detach.default(detach_45);  detach_45 = None
        fused_layer_norm_affine_bwd_2 = torch.ops.apex.fused_layer_norm_affine_bwd.default(view_68, detach_44, detach_46, clone_23, [48], getitem_17, getitem_19, 1e-05);  view_68 = detach_44 = detach_46 = clone_23 = None
        getitem_63: "f16[1024, 32, 16, 48]" = fused_layer_norm_affine_bwd_2[0]
        getitem_64: "f16[48]" = fused_layer_norm_affine_bwd_2[1]
        getitem_65: "f16[48]" = fused_layer_norm_affine_bwd_2[2];  fused_layer_norm_affine_bwd_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_19: "f16[48]" = torch.ops.aten.add.Tensor(tangents_9, getitem_64);  tangents_9 = getitem_64 = None
        add_20: "f16[48]" = torch.ops.aten.add.Tensor(tangents_10, getitem_65);  tangents_10 = getitem_65 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_69: "f16[1024, 32, 16, 48]" = torch.ops.aten.view.default(getitem_63, [1024, 32, 16, 48]);  getitem_63 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        cat: "f16[1024, 32, 16, 144]" = torch.ops.aten.cat.default([view_69, getitem_60, view_61], 3);  view_69 = getitem_60 = view_61 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_70: "f16[1024, 32, 2304]" = torch.ops.aten.view.default(cat, [1024, 32, 2304]);  cat = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        all_gather_into_tensor_7: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(getitem_40, 2, '12');  getitem_40 = None
        wait_tensor_24: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_7);  all_gather_into_tensor_7 = None
        wait_tensor_25: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(wait_tensor_24);  wait_tensor_24 = None
        view_71: "f16[32768, 2304]" = torch.ops.aten.view.default(view_70, [32768, 2304])
        mm_13: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_71, getitem_13);  view_71 = None
        _unsafe_view_9: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_13, [1024, 32, 1536]);  mm_13 = None
        view_72: "f16[32768, 2304]" = torch.ops.aten.view.default(view_70, [32768, 2304]);  view_70 = None
        view_73: "f16[32768, 1536]" = torch.ops.aten.view.default(wait_tensor_25, [32768, 1536]);  wait_tensor_25 = None
        reduce_scatter_tensor_4: "f16[512, 32, 1536]" = torch.ops._c10d_functional.reduce_scatter_tensor.default(_unsafe_view_9, 'sum', 2, '12');  _unsafe_view_9 = None
        wait_tensor_26: "f16[512, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(reduce_scatter_tensor_4);  reduce_scatter_tensor_4 = None
        t_16: "f16[2304, 32768]" = torch.ops.aten.t.default(view_72)
        mm_14: "f16[2304, 1536]" = torch.ops.aten.mm.default(t_16, view_73);  t_16 = view_73 = None
        sum_7: "f16[2304]" = torch.ops.aten.sum.dim_IntList(view_72, [0]);  view_72 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_21: "f16[2304, 1536]" = torch.ops.aten.add.Tensor(tangents_7, mm_14);  tangents_7 = mm_14 = None
        add_22: "f16[2304]" = torch.ops.aten.add.Tensor(tangents_8, sum_7);  tangents_8 = sum_7 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_47: "f32[16384]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        detach_48: "f32[16384]" = torch.ops.aten.detach.default(detach_47);  detach_47 = None
        detach_49: "f32[16384]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
        detach_50: "f32[16384]" = torch.ops.aten.detach.default(detach_49);  detach_49 = None
        fused_layer_norm_affine_bwd_3 = torch.ops.apex.fused_layer_norm_affine_bwd.default(wait_tensor_26, detach_48, detach_50, clone_22, [1536], getitem_5, getitem_7, 1e-05);  wait_tensor_26 = detach_48 = detach_50 = clone_22 = None
        getitem_66: "f16[512, 32, 1536]" = fused_layer_norm_affine_bwd_3[0]
        getitem_67: "f16[1536]" = fused_layer_norm_affine_bwd_3[1]
        getitem_68: "f16[1536]" = fused_layer_norm_affine_bwd_3[2];  fused_layer_norm_affine_bwd_3 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_23: "f16[512, 32, 1536]" = torch.ops.aten.add.Tensor(add_12, getitem_66);  add_12 = getitem_66 = None
        add_24: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_3, getitem_67);  tangents_3 = getitem_67 = None
        add_25: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_4, getitem_68);  tangents_4 = getitem_68 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/mappings.py:497 in scatter_to_sequence_parallel_region, code: return _ScatterToSequenceParallelRegion.apply(input_)
        empty_8: "f16[1024, 32, 1536]" = torch.ops.aten.empty.memory_format([1024, 32, 1536], dtype = torch.float16, device = device(type='cuda', index=1), pin_memory = False)
        all_gather_into_tensor_8: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.all_gather_into_tensor.default(add_23, 2, '12');  add_23 = None
        wait_tensor_27: "f16[1024, 32, 1536]" = torch.ops._c10d_functional.wait_tensor.default(all_gather_into_tensor_8);  all_gather_into_tensor_8 = None
        copy_12: "f16[1024, 32, 1536]" = torch.ops.aten.copy.default(empty_8, wait_tensor_27);  empty_8 = wait_tensor_27 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        transpose_17: "f16[32, 1024, 1536]" = torch.ops.aten.transpose.int(copy_12, 0, 1);  copy_12 = None
        embedding_dense_backward: "f16[2048, 1536]" = torch.ops.aten.embedding_dense_backward.default(transpose_17, wait_tensor_4, 2048, -1, False);  wait_tensor_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        add_26: "f16[2048, 1536]" = torch.ops.aten.add.Tensor(tangents_2, embedding_dense_backward);  tangents_2 = embedding_dense_backward = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:266 in forward, code: output_parallel[input_mask, :] = 0.0
        zeros_9: "f16[]" = torch.ops.aten.zeros.default([], dtype = torch.float16, layout = torch.strided, device = device(type='cpu'))
        index_put_5: "f16[32, 1024, 1536]" = torch.ops.aten.index_put.default(transpose_17, [bitwise_or], zeros_9);  transpose_17 = bitwise_or = zeros_9 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        embedding_dense_backward_1: "f16[49152, 1536]" = torch.ops.aten.embedding_dense_backward.default(index_put_5, index_put, 49152, -1, False);  index_put_5 = index_put = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        add_27: "f16[49152, 1536]" = torch.ops.aten.add.Tensor(add_7, embedding_dense_backward_1);  add_7 = embedding_dense_backward_1 = None
        return pytree.tree_unflatten([getitem_1, getitem_3, getitem_5, getitem_7, getitem_9, getitem_11, getitem_13, getitem_15, getitem_17, getitem_19, getitem_21, getitem_23, getitem_25, getitem_27, getitem_29, getitem_31, getitem_33, getitem_35, getitem_37, getitem_39, div_2, detach_26, detach_29, zeros, None, None, None, None, None, add_27, add_26, add_24, add_25, add_16, add_15, add_21, add_22, add_19, add_20, add_17, add_18, add_13, add_14, add_11, add_10, add_9, add_8, tangents_19, tangents_20], self._out_spec)
        

