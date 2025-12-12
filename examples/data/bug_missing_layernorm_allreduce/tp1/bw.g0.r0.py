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


# Graph[rank=0](bw, gid=0)
class GraphModule(torch.nn.Module):
    def forward(self, wait_tensor: "i64[32, 1024]", wait_tensor_3: "b8[32, 1, 1024, 1024]", wait_tensor_4: "i64[32, 1024]", getitem_1: "f16[98304, 1536]", getitem_5: "f16[1536]", getitem_7: "f16[1536]", getitem_9: "f16[1536, 1536]", getitem_13: "f16[4608, 1536]", getitem_17: "f16[48]", getitem_19: "f16[48]", getitem_21: "f16[48]", getitem_23: "f16[48]", getitem_25: "f16[1536]", getitem_27: "f16[1536]", getitem_29: "f16[6144, 1536]", getitem_33: "f16[1536, 6144]", clone_20: "f16[1024, 32, 1536]", getitem_41: "f32[32768]", getitem_42: "f32[32768]", view_1: "f16[1024, 32, 1536]", getitem_44: "f16[1024, 32, 32, 48]", view_4: "f16[1024, 32, 32, 48]", getitem_47: "f32[1048576]", getitem_48: "f32[1048576]", getitem_50: "f32[1048576]", getitem_51: "f32[1048576]", transpose_1: "f16[1024, 1024, 48]", transpose_3: "f16[1024, 48, 1024]", _softmax: "f16[32, 32, 1024, 1024]", view_12: "f16[1024, 1024, 1024]", transpose_4: "f16[1024, 1024, 48]", view_14: "f16[1024, 32, 1536]", add_3: "f16[1024, 32, 1536]", getitem_53: "f32[32768]", getitem_54: "f32[32768]", view_17: "f16[1024, 32, 1536]", add_4: "f16[1024, 32, 6144]", gelu: "f16[1024, 32, 6144]", view_21: "f16[1024, 32, 1536]", getitem_55: "f32[1024, 32]", bitwise_or: "b8[1024, 32]", view_26: "i64[32768]", _tensor_constant1: "f32[]", exp: "f32[1024, 32, 98304]", copy_2: "f32[1024, 32]", view_32: "f32[32768]", _to_copy_6: "i32[1]", tangents_1: "f16[98304, 1536]", tangents_2: "f16[2048, 1536]", tangents_3: "f16[1536]", tangents_4: "f16[1536]", tangents_5: "f16[1536, 1536]", tangents_6: "f16[1536]", tangents_7: "f16[4608, 1536]", tangents_8: "f16[4608]", tangents_9: "f16[48]", tangents_10: "f16[48]", tangents_11: "f16[48]", tangents_12: "f16[48]", tangents_13: "f16[1536]", tangents_14: "f16[1536]", tangents_15: "f16[6144, 1536]", tangents_16: "f16[6144]", tangents_17: "f16[1536, 6144]", tangents_18: "f16[1536]", tangents_19: "f16[1536]", tangents_20: "f16[1536]", tangents_21: "f32[1]"):
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach: "f32[32768]" = torch.ops.aten.detach.default(getitem_41);  getitem_41 = None
        detach_1: "f32[32768]" = torch.ops.aten.detach.default(detach);  detach = None
        detach_2: "f32[32768]" = torch.ops.aten.detach.default(getitem_42);  getitem_42 = None
        detach_3: "f32[32768]" = torch.ops.aten.detach.default(detach_2);  detach_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        clone_21: "f16[1024, 32, 32, 48]" = torch.ops.aten.clone.default(view_4, memory_format = torch.contiguous_format);  view_4 = None
        detach_4: "f32[1048576]" = torch.ops.aten.detach.default(getitem_47);  getitem_47 = None
        detach_5: "f32[1048576]" = torch.ops.aten.detach.default(detach_4);  detach_4 = None
        detach_6: "f32[1048576]" = torch.ops.aten.detach.default(getitem_48);  getitem_48 = None
        detach_7: "f32[1048576]" = torch.ops.aten.detach.default(detach_6);  detach_6 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        clone_22: "f16[1024, 32, 32, 48]" = torch.ops.aten.clone.default(getitem_44, memory_format = torch.contiguous_format);  getitem_44 = None
        detach_8: "f32[1048576]" = torch.ops.aten.detach.default(getitem_50);  getitem_50 = None
        detach_9: "f32[1048576]" = torch.ops.aten.detach.default(detach_8);  detach_8 = None
        detach_10: "f32[1048576]" = torch.ops.aten.detach.default(getitem_51);  getitem_51 = None
        detach_11: "f32[1048576]" = torch.ops.aten.detach.default(detach_10);  detach_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_softmax.py:208 in forward_torch_softmax, code: probs = torch.nn.Softmax(dim=-1)(mask_output)
        detach_12: "f16[32, 32, 1024, 1024]" = torch.ops.aten.detach.default(_softmax);  _softmax = None
        detach_13: "f16[32, 32, 1024, 1024]" = torch.ops.aten.detach.default(detach_12);  detach_12 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_14: "f32[32768]" = torch.ops.aten.detach.default(getitem_53);  getitem_53 = None
        detach_15: "f32[32768]" = torch.ops.aten.detach.default(detach_14);  detach_14 = None
        detach_16: "f32[32768]" = torch.ops.aten.detach.default(getitem_54);  getitem_54 = None
        detach_17: "f32[32768]" = torch.ops.aten.detach.default(detach_16);  detach_16 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_22: "f16[32768, 1536]" = torch.ops.aten.view.default(view_21, [32768, 1536])
        t_9: "f16[1536, 98304]" = torch.ops.aten.t.default(getitem_1)
        mm_4: "f16[32768, 98304]" = torch.ops.aten.mm.default(view_22, t_9);  view_22 = t_9 = None
        _unsafe_view_4: "f16[1024, 32, 98304]" = torch.ops.aten._unsafe_view.default(mm_4, [1024, 32, 98304]);  mm_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:235 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        _to_copy_5: "f32[1024, 32, 98304]" = torch.ops.aten._to_copy.default(_unsafe_view_4, dtype = torch.float32);  _unsafe_view_4 = None
        all_reduce: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(getitem_55, 'max', '7')
        wait_tensor_5: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
        copy: "f32[1024, 32]" = torch.ops.aten.copy.default(getitem_55, wait_tensor_5);  getitem_55 = wait_tensor_5 = None
        unsqueeze_1: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(copy, -1);  copy = None
        sub: "f32[1024, 32, 98304]" = torch.ops.aten.sub.Tensor(_to_copy_5, unsqueeze_1);  _to_copy_5 = unsqueeze_1 = None
        arange: "i64[32768]" = torch.ops.aten.arange.start(0, 32768, device = device(type='cuda', index=0), pin_memory = False)
        view_25: "f32[32768, 98304]" = torch.ops.aten.view.default(sub, [-1, 98304]);  sub = None
        index: "f32[32768]" = torch.ops.aten.index.Tensor(view_25, [arange, view_26]);  view_25 = arange = None
        clone_26: "f32[32768]" = torch.ops.aten.clone.default(index);  index = None
        view_27: "f32[1024, 32]" = torch.ops.aten.view.default(clone_26, [1024, 32]);  clone_26 = None
        lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        index_put_1: "f32[1024, 32]" = torch.ops.aten.index_put.default(view_27, [bitwise_or], lift_fresh_copy_1);  view_27 = lift_fresh_copy_1 = None
        view_28: "f32[32768]" = torch.ops.aten.view.default(index_put_1, [32768]);  index_put_1 = None
        view_29: "f32[1024, 32]" = torch.ops.aten.view.default(view_28, [1024, 32]);  view_28 = None
        sum_1: "f32[1024, 32]" = torch.ops.aten.sum.dim_IntList(exp, [-1])
        all_reduce_1: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(view_29, 'sum', '7')
        wait_tensor_6: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_1);  all_reduce_1 = None
        copy_1: "f32[1024, 32]" = torch.ops.aten.copy.default(view_29, wait_tensor_6);  view_29 = wait_tensor_6 = None
        view_30: "f32[32768]" = torch.ops.aten.view.default(copy_1, [32768]);  copy_1 = None
        view_31: "f32[1024, 32]" = torch.ops.aten.view.default(view_30, [1024, 32]);  view_30 = None
        all_reduce_2: "f32[1024, 32]" = torch.ops._c10d_functional.all_reduce.default(sum_1, 'sum', '7');  sum_1 = None
        wait_tensor_7: "f32[1024, 32]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_2);  all_reduce_2 = wait_tensor_7 = None
        log: "f32[1024, 32]" = torch.ops.aten.log.default(copy_2)
        sub_2: "f32[1024, 32]" = torch.ops.aten.sub.Tensor(log, view_31);  log = view_31 = None
        unsqueeze_3: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(copy_2, -1);  copy_2 = None
        div: "f32[1024, 32, 98304]" = torch.ops.aten.div.Tensor(exp, unsqueeze_3);  exp = unsqueeze_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_6: "f32[32, 1024]" = torch.ops.aten.transpose.int(sub_2, 0, 1);  sub_2 = None
        clone_27: "f32[32, 1024]" = torch.ops.aten.clone.default(transpose_6, memory_format = torch.contiguous_format);  transpose_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:172 in loss_func, code: total_tokens = loss_mask.sum().view(1)
        sum_2: "f32[]" = torch.ops.aten.sum.default(view_32)
        view_33: "f32[1]" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:173 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask).view(1)
        view_34: "f32[32768]" = torch.ops.aten.view.default(clone_27, [-1]);  clone_27 = None
        mul: "f32[32768]" = torch.ops.aten.mul.Tensor(view_34, view_32);  view_34 = None
        sum_3: "f32[]" = torch.ops.aten.sum.default(mul);  mul = None
        view_35: "f32[1]" = torch.ops.aten.view.default(sum_3, [1]);  sum_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:210 in loss_func, code: reporting_loss = loss.clone().detach()
        clone_28: "f32[1]" = torch.ops.aten.clone.default(view_35);  view_35 = None
        detach_18: "f32[1]" = torch.ops.aten.detach.default(clone_28);  clone_28 = None
        detach_19: "f32[1]" = torch.ops.aten.detach.default(detach_18);  detach_18 = None
        detach_20: "f32[1]" = torch.ops.aten.detach.default(detach_19);  detach_19 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:211 in loss_func, code: reporting_total_tokens = total_tokens.clone().detach()
        clone_29: "f32[1]" = torch.ops.aten.clone.default(view_33);  view_33 = None
        detach_21: "f32[1]" = torch.ops.aten.detach.default(clone_29);  clone_29 = None
        detach_22: "f32[1]" = torch.ops.aten.detach.default(detach_21);  detach_21 = None
        detach_23: "f32[1]" = torch.ops.aten.detach.default(detach_22);  detach_22 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:170 in all_reduce, code: tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
        all_reduce_3: "f32[1]" = torch.ops._c10d_functional.all_reduce.default(detach_20, 'sum', '1');  detach_20 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_8: "f32[1]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_3);  all_reduce_3 = wait_tensor_8 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:170 in all_reduce, code: tensor = torch.ops._c10d_functional.all_reduce(self, reduceOp.lower(), group_name)
        all_reduce_4: "f32[1]" = torch.ops._c10d_functional.all_reduce.default(detach_23, 'sum', '1');  detach_23 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/distributed/_functional_collectives.py:135 in wait_tensor, code: return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]
        wait_tensor_9: "f32[1]" = torch.ops._c10d_functional.wait_tensor.default(all_reduce_4);  all_reduce_4 = wait_tensor_9 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:290 in forward_step, code: output_tensor /= num_microbatches
        div_3: "f32[1]" = torch.ops.aten.div.Tensor(tangents_21, 1);  tangents_21 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/pipeline_parallel/schedules.py:289 in forward_step, code: output_tensor /= num_tokens
        div_4: "f32[1]" = torch.ops.aten.div.Tensor(div_3, _to_copy_6);  div_3 = _to_copy_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:218 in loss_func, code: loss * args.context_parallel_size,
        mul_2: "f32[1]" = torch.ops.aten.mul.Tensor(div_4, 1);  div_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/pretrain_gpt.py:173 in loss_func, code: loss = torch.sum(losses.view(-1) * loss_mask).view(1)
        view_36: "f32[]" = torch.ops.aten.view.default(mul_2, []);  mul_2 = None
        expand: "f32[32768]" = torch.ops.aten.expand.default(view_36, [32768]);  view_36 = None
        mul_3: "f32[32768]" = torch.ops.aten.mul.Tensor(expand, view_32);  expand = view_32 = None
        view_37: "f32[32, 1024]" = torch.ops.aten.view.default(mul_3, [32, 1024]);  mul_3 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/language_module/language_module.py:84 in compute_language_model_loss, code: loss = loss.transpose(0, 1).contiguous()
        transpose_7: "f32[1024, 32]" = torch.ops.aten.transpose.int(view_37, 0, 1);  view_37 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/cross_entropy.py:235 in vocab_parallel_cross_entropy, code: return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
        arange_1: "i64[32768]" = torch.ops.aten.arange.start(0, 32768, device = device(type='cuda', index=0), pin_memory = False)
        view_39: "b8[32768]" = torch.ops.aten.view.default(bitwise_or, [-1]);  bitwise_or = None
        _to_copy_7: "f32[32768]" = torch.ops.aten._to_copy.default(view_39, dtype = torch.float32);  view_39 = None
        rsub: "f32[32768]" = torch.ops.aten.rsub.Scalar(_to_copy_7, 1.0);  _to_copy_7 = None
        view_40: "f32[32768, 98304]" = torch.ops.aten.view.default(div, [-1, 98304]);  div = None
        index_1: "f32[32768]" = torch.ops.aten.index.Tensor(view_40, [arange_1, view_26])
        sub_3: "f32[32768]" = torch.ops.aten.sub.Tensor(index_1, rsub);  index_1 = rsub = None
        index_put_2: "f32[32768, 98304]" = torch.ops.aten.index_put.default(view_40, [arange_1, view_26], sub_3);  view_40 = arange_1 = view_26 = sub_3 = None
        view_41: "f32[1024, 32, 98304]" = torch.ops.aten.view.default(index_put_2, [1024, 32, 98304]);  index_put_2 = None
        unsqueeze_4: "f32[1024, 32, 1]" = torch.ops.aten.unsqueeze.default(transpose_7, -1);  transpose_7 = None
        mul_4: "f32[1024, 32, 98304]" = torch.ops.aten.mul.Tensor(view_41, unsqueeze_4);  view_41 = unsqueeze_4 = None
        _to_copy_8: "f16[1024, 32, 98304]" = torch.ops.aten._to_copy.default(mul_4, dtype = torch.float16);  mul_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_43: "f16[32768, 98304]" = torch.ops.aten.view.default(_to_copy_8, [32768, 98304])
        mm_5: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_43, getitem_1);  view_43 = getitem_1 = None
        _unsafe_view_5: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_5, [1024, 32, 1536]);  mm_5 = None
        view_44: "f16[32768, 98304]" = torch.ops.aten.view.default(_to_copy_8, [32768, 98304]);  _to_copy_8 = None
        view_45: "f16[32768, 1536]" = torch.ops.aten.view.default(view_21, [32768, 1536]);  view_21 = None
        t_10: "f16[98304, 32768]" = torch.ops.aten.t.default(view_44);  view_44 = None
        mm_6: "f16[98304, 1536]" = torch.ops.aten.mm.default(t_10, view_45);  t_10 = view_45 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_7: "f16[98304, 1536]" = torch.ops.aten.add.Tensor(tangents_1, mm_6);  tangents_1 = mm_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        sum_4: "f16[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(_unsafe_view_5, [0, 1], True)
        view_46: "f16[1536]" = torch.ops.aten.view.default(sum_4, [1536]);  sum_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        add_8: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_18, view_46);  tangents_18 = view_46 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_47: "f16[32768, 1536]" = torch.ops.aten.view.default(_unsafe_view_5, [32768, 1536])
        mm_7: "f16[32768, 6144]" = torch.ops.aten.mm.default(view_47, getitem_33);  view_47 = getitem_33 = None
        _unsafe_view_6: "f16[1024, 32, 6144]" = torch.ops.aten._unsafe_view.default(mm_7, [1024, 32, 6144]);  mm_7 = None
        view_48: "f16[32768, 1536]" = torch.ops.aten.view.default(_unsafe_view_5, [32768, 1536])
        view_49: "f16[32768, 6144]" = torch.ops.aten.view.default(gelu, [32768, 6144]);  gelu = None
        t_11: "f16[1536, 32768]" = torch.ops.aten.t.default(view_48);  view_48 = None
        mm_8: "f16[1536, 6144]" = torch.ops.aten.mm.default(t_11, view_49);  t_11 = view_49 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_9: "f16[1536, 6144]" = torch.ops.aten.add.Tensor(tangents_17, mm_8);  tangents_17 = mm_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:125 in forward, code: intermediate_parallel = self.activation_func(intermediate_parallel)
        gelu_backward: "f16[1024, 32, 6144]" = torch.ops.aten.gelu_backward.default(_unsafe_view_6, add_4);  _unsafe_view_6 = add_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        sum_5: "f16[1, 1, 6144]" = torch.ops.aten.sum.dim_IntList(gelu_backward, [0, 1], True)
        view_50: "f16[6144]" = torch.ops.aten.view.default(sum_5, [6144]);  sum_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/mlp.py:116 in forward, code: intermediate_parallel = intermediate_parallel + bias_parallel
        add_10: "f16[6144]" = torch.ops.aten.add.Tensor(tangents_16, view_50);  tangents_16 = view_50 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_51: "f16[32768, 6144]" = torch.ops.aten.view.default(gelu_backward, [32768, 6144])
        mm_9: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_51, getitem_29);  view_51 = getitem_29 = None
        _unsafe_view_7: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_9, [1024, 32, 1536]);  mm_9 = None
        view_52: "f16[32768, 6144]" = torch.ops.aten.view.default(gelu_backward, [32768, 6144]);  gelu_backward = None
        view_53: "f16[32768, 1536]" = torch.ops.aten.view.default(view_17, [32768, 1536]);  view_17 = None
        t_12: "f16[6144, 32768]" = torch.ops.aten.t.default(view_52);  view_52 = None
        mm_10: "f16[6144, 1536]" = torch.ops.aten.mm.default(t_12, view_53);  t_12 = view_53 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_11: "f16[6144, 1536]" = torch.ops.aten.add.Tensor(tangents_15, mm_10);  tangents_15 = mm_10 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_33: "f32[32768]" = torch.ops.aten.detach.default(detach_15);  detach_15 = None
        detach_34: "f32[32768]" = torch.ops.aten.detach.default(detach_33);  detach_33 = None
        detach_35: "f32[32768]" = torch.ops.aten.detach.default(detach_17);  detach_17 = None
        detach_36: "f32[32768]" = torch.ops.aten.detach.default(detach_35);  detach_35 = None
        fused_layer_norm_affine_bwd = torch.ops.apex.fused_layer_norm_affine_bwd.default(_unsafe_view_7, detach_34, detach_36, add_3, [1536], getitem_25, getitem_27, 1e-05);  _unsafe_view_7 = detach_34 = detach_36 = add_3 = getitem_25 = getitem_27 = None
        getitem_57: "f16[1024, 32, 1536]" = fused_layer_norm_affine_bwd[0]
        getitem_58: "f16[1536]" = fused_layer_norm_affine_bwd[1]
        getitem_59: "f16[1536]" = fused_layer_norm_affine_bwd[2];  fused_layer_norm_affine_bwd = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_12: "f16[1024, 32, 1536]" = torch.ops.aten.add.Tensor(_unsafe_view_5, getitem_57);  _unsafe_view_5 = getitem_57 = None
        add_13: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_13, getitem_58);  tangents_13 = getitem_58 = None
        add_14: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_14, getitem_59);  tangents_14 = getitem_59 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        sum_6: "f16[1, 1, 1536]" = torch.ops.aten.sum.dim_IntList(add_12, [0, 1], True)
        view_54: "f16[1536]" = torch.ops.aten.view.default(sum_6, [1536]);  sum_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_bias_dropout.py:33 in _bias_dropout_add_func, code: x = x + bias
        add_15: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_6, view_54);  tangents_6 = view_54 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_55: "f16[32768, 1536]" = torch.ops.aten.view.default(add_12, [32768, 1536])
        mm_11: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_55, getitem_9);  view_55 = getitem_9 = None
        _unsafe_view_8: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_11, [1024, 32, 1536]);  mm_11 = None
        view_56: "f16[32768, 1536]" = torch.ops.aten.view.default(add_12, [32768, 1536])
        view_57: "f16[32768, 1536]" = torch.ops.aten.view.default(view_14, [32768, 1536]);  view_14 = None
        t_13: "f16[1536, 32768]" = torch.ops.aten.t.default(view_56);  view_56 = None
        mm_12: "f16[1536, 1536]" = torch.ops.aten.mm.default(t_13, view_57);  t_13 = view_57 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_16: "f16[1536, 1536]" = torch.ops.aten.add.Tensor(tangents_5, mm_12);  tangents_5 = mm_12 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:210 in forward, code: context = context.view(*new_context_shape)
        view_58: "f16[1024, 32, 32, 48]" = torch.ops.aten.view.default(_unsafe_view_8, [1024, 32, 32, 48]);  _unsafe_view_8 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:206 in forward, code: context = context.permute(2, 0, 1, 3).contiguous()
        permute_1: "f16[32, 32, 1024, 48]" = torch.ops.aten.permute.default(view_58, [1, 2, 0, 3]);  view_58 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:203 in forward, code: context = context.view(*output_size)
        view_59: "f16[1024, 1024, 48]" = torch.ops.aten.view.default(permute_1, [1024, 1024, 48]);  permute_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:200 in forward, code: context = torch.bmm(attention_probs, value.transpose(0, 1))
        transpose_8: "f16[1024, 1024, 1024]" = torch.ops.aten.transpose.int(view_12, 1, 2);  view_12 = None
        bmm_1: "f16[1024, 1024, 48]" = torch.ops.aten.bmm.default(transpose_8, view_59);  transpose_8 = None
        transpose_9: "f16[1024, 48, 1024]" = torch.ops.aten.transpose.int(transpose_4, 1, 2);  transpose_4 = None
        bmm_2: "f16[1024, 1024, 1024]" = torch.ops.aten.bmm.default(view_59, transpose_9);  view_59 = transpose_9 = None
        transpose_10: "f16[1024, 1024, 48]" = torch.ops.aten.transpose.int(bmm_1, 0, 1);  bmm_1 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:197 in forward, code: attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        view_60: "f16[32, 32, 1024, 1024]" = torch.ops.aten.view.default(bmm_2, [32, 32, 1024, 1024]);  bmm_2 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:194 in forward, code: value = value.view(value.size(0), output_size[0] * output_size[1], -1)
        view_61: "f16[1024, 32, 32, 48]" = torch.ops.aten.view.default(transpose_10, [1024, 32, 32, 48]);  transpose_10 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/fusions/fused_softmax.py:208 in forward_torch_softmax, code: probs = torch.nn.Softmax(dim=-1)(mask_output)
        detach_37: "f16[32, 32, 1024, 1024]" = torch.ops.aten.detach.default(detach_13);  detach_13 = None
        detach_38: "f16[32, 32, 1024, 1024]" = torch.ops.aten.detach.default(detach_37);  detach_37 = None
        _softmax_backward_data: "f16[32, 32, 1024, 1024]" = torch.ops.aten._softmax_backward_data.default(view_60, detach_38, -1, torch.float16);  view_60 = detach_38 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/utils.py:36 in attention_mask_func, code: attention_scores.masked_fill_(attention_mask, -10000.0)
        view_62: "f16[1024, 1024, 1024]" = torch.ops.aten.view.default(_softmax_backward_data, [1024, 1024, 1024]);  _softmax_backward_data = None
        new_empty_strided: "f16[1024, 1024, 1024]" = torch.ops.aten.new_empty_strided.default(view_62, [1024, 1024, 1024], [1048576, 1024, 1])
        copy_5: "f16[1024, 1024, 1024]" = torch.ops.aten.copy.default(new_empty_strided, view_62);  new_empty_strided = view_62 = None
        view_64: "f16[32, 32, 1024, 1024]" = torch.ops.aten.view.default(copy_5, [32, 32, 1024, 1024]);  copy_5 = None
        clone_31: "f16[32, 32, 1024, 1024]" = torch.ops.aten.clone.default(view_64, memory_format = torch.contiguous_format)
        masked_fill_1: "f16[32, 32, 1024, 1024]" = torch.ops.aten.masked_fill.Scalar(clone_31, wait_tensor_3, 0);  clone_31 = wait_tensor_3 = None
        copy_6: "f16[32, 32, 1024, 1024]" = torch.ops.aten.copy.default(view_64, masked_fill_1);  view_64 = masked_fill_1 = None
        view_65: "f16[1024, 1024, 1024]" = torch.ops.aten.view.default(copy_6, [1024, 1024, 1024]);  copy_6 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:156 in forward, code: matmul_result = torch.baddbmm(
        transpose_11: "f16[1024, 1024, 48]" = torch.ops.aten.transpose.int(transpose_3, 1, 2);  transpose_3 = None
        bmm_3: "f16[1024, 1024, 48]" = torch.ops.aten.bmm.default(view_65, transpose_11);  transpose_11 = None
        mul_5: "f16[1024, 1024, 48]" = torch.ops.aten.mul.Scalar(bmm_3, 0.14433756729740646);  bmm_3 = None
        transpose_12: "f16[1024, 48, 1024]" = torch.ops.aten.transpose.int(transpose_1, 1, 2);  transpose_1 = None
        bmm_4: "f16[1024, 48, 1024]" = torch.ops.aten.bmm.default(transpose_12, view_65);  transpose_12 = view_65 = None
        mul_6: "f16[1024, 48, 1024]" = torch.ops.aten.mul.Scalar(bmm_4, 0.14433756729740646);  bmm_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:159 in forward, code: key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        transpose_13: "f16[1024, 1024, 48]" = torch.ops.aten.transpose.int(mul_6, 1, 2);  mul_6 = None
        transpose_14: "f16[1024, 1024, 48]" = torch.ops.aten.transpose.int(transpose_13, 0, 1);  transpose_13 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:158 in forward, code: query.transpose(0, 1),  # [b * np, sq, hn]
        transpose_15: "f16[1024, 1024, 48]" = torch.ops.aten.transpose.int(mul_5, 0, 1);  mul_5 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:145 in forward, code: key = key.view(output_size[3], output_size[0] * output_size[1], -1)
        view_67: "f16[1024, 32, 32, 48]" = torch.ops.aten.view.default(transpose_14, [1024, 32, 32, 48]);  transpose_14 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/dot_product_attention.py:143 in forward, code: query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        view_68: "f16[1024, 32, 32, 48]" = torch.ops.aten.view.default(transpose_15, [1024, 32, 32, 48]);  transpose_15 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_39: "f32[1048576]" = torch.ops.aten.detach.default(detach_9);  detach_9 = None
        detach_40: "f32[1048576]" = torch.ops.aten.detach.default(detach_39);  detach_39 = None
        detach_41: "f32[1048576]" = torch.ops.aten.detach.default(detach_11);  detach_11 = None
        detach_42: "f32[1048576]" = torch.ops.aten.detach.default(detach_41);  detach_41 = None
        fused_layer_norm_affine_bwd_1 = torch.ops.apex.fused_layer_norm_affine_bwd.default(view_67, detach_40, detach_42, clone_22, [48], getitem_21, getitem_23, 1e-05);  view_67 = detach_40 = detach_42 = clone_22 = getitem_21 = getitem_23 = None
        getitem_60: "f16[1024, 32, 32, 48]" = fused_layer_norm_affine_bwd_1[0]
        getitem_61: "f16[48]" = fused_layer_norm_affine_bwd_1[1]
        getitem_62: "f16[48]" = fused_layer_norm_affine_bwd_1[2];  fused_layer_norm_affine_bwd_1 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_17: "f16[48]" = torch.ops.aten.add.Tensor(tangents_11, getitem_61);  tangents_11 = getitem_61 = None
        add_18: "f16[48]" = torch.ops.aten.add.Tensor(tangents_12, getitem_62);  tangents_12 = getitem_62 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_43: "f32[1048576]" = torch.ops.aten.detach.default(detach_5);  detach_5 = None
        detach_44: "f32[1048576]" = torch.ops.aten.detach.default(detach_43);  detach_43 = None
        detach_45: "f32[1048576]" = torch.ops.aten.detach.default(detach_7);  detach_7 = None
        detach_46: "f32[1048576]" = torch.ops.aten.detach.default(detach_45);  detach_45 = None
        fused_layer_norm_affine_bwd_2 = torch.ops.apex.fused_layer_norm_affine_bwd.default(view_68, detach_44, detach_46, clone_21, [48], getitem_17, getitem_19, 1e-05);  view_68 = detach_44 = detach_46 = clone_21 = getitem_17 = getitem_19 = None
        getitem_63: "f16[1024, 32, 32, 48]" = fused_layer_norm_affine_bwd_2[0]
        getitem_64: "f16[48]" = fused_layer_norm_affine_bwd_2[1]
        getitem_65: "f16[48]" = fused_layer_norm_affine_bwd_2[2];  fused_layer_norm_affine_bwd_2 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_19: "f16[48]" = torch.ops.aten.add.Tensor(tangents_9, getitem_64);  tangents_9 = getitem_64 = None
        add_20: "f16[48]" = torch.ops.aten.add.Tensor(tangents_10, getitem_65);  tangents_10 = getitem_65 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:654 in get_query_key_value_tensors, code: query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        view_69: "f16[1024, 32, 32, 48]" = torch.ops.aten.view.default(getitem_63, [1024, 32, 32, 48]);  getitem_63 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:651 in get_query_key_value_tensors, code: (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)
        cat: "f16[1024, 32, 32, 144]" = torch.ops.aten.cat.default([view_69, getitem_60, view_61], 3);  view_69 = getitem_60 = view_61 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/transformer/attention.py:630 in get_query_key_value_tensors, code: mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        view_70: "f16[1024, 32, 4608]" = torch.ops.aten.view.default(cat, [1024, 32, 4608]);  cat = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        view_71: "f16[32768, 4608]" = torch.ops.aten.view.default(view_70, [32768, 4608])
        mm_13: "f16[32768, 1536]" = torch.ops.aten.mm.default(view_71, getitem_13);  view_71 = getitem_13 = None
        _unsafe_view_9: "f16[1024, 32, 1536]" = torch.ops.aten._unsafe_view.default(mm_13, [1024, 32, 1536]);  mm_13 = None
        view_72: "f16[32768, 4608]" = torch.ops.aten.view.default(view_70, [32768, 4608]);  view_70 = None
        view_73: "f16[32768, 1536]" = torch.ops.aten.view.default(view_1, [32768, 1536]);  view_1 = None
        t_14: "f16[4608, 32768]" = torch.ops.aten.t.default(view_72)
        mm_14: "f16[4608, 1536]" = torch.ops.aten.mm.default(t_14, view_73);  t_14 = view_73 = None
        sum_7: "f16[4608]" = torch.ops.aten.sum.dim_IntList(view_72, [0]);  view_72 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:698 in linear_with_grad_accumulation_and_async_allreduce, code: return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        add_21: "f16[4608, 1536]" = torch.ops.aten.add.Tensor(tangents_7, mm_14);  tangents_7 = mm_14 = None
        add_22: "f16[4608]" = torch.ops.aten.add.Tensor(tangents_8, sum_7);  tangents_8 = sum_7 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        detach_47: "f32[32768]" = torch.ops.aten.detach.default(detach_1);  detach_1 = None
        detach_48: "f32[32768]" = torch.ops.aten.detach.default(detach_47);  detach_47 = None
        detach_49: "f32[32768]" = torch.ops.aten.detach.default(detach_3);  detach_3 = None
        detach_50: "f32[32768]" = torch.ops.aten.detach.default(detach_49);  detach_49 = None
        fused_layer_norm_affine_bwd_3 = torch.ops.apex.fused_layer_norm_affine_bwd.default(_unsafe_view_9, detach_48, detach_50, clone_20, [1536], getitem_5, getitem_7, 1e-05);  _unsafe_view_9 = detach_48 = detach_50 = clone_20 = getitem_5 = getitem_7 = None
        getitem_66: "f16[1024, 32, 1536]" = fused_layer_norm_affine_bwd_3[0]
        getitem_67: "f16[1536]" = fused_layer_norm_affine_bwd_3[1]
        getitem_68: "f16[1536]" = fused_layer_norm_affine_bwd_3[2];  fused_layer_norm_affine_bwd_3 = None
        
         # File: /dfs/scratch0/shirwu/anaconda3/envs/megatron/lib/python3.12/site-packages/torch/_library/custom_ops.py:675 in __call__, code: return self._opoverload(*args, **kwargs)
        add_23: "f16[1024, 32, 1536]" = torch.ops.aten.add.Tensor(add_12, getitem_66);  add_12 = getitem_66 = None
        add_24: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_3, getitem_67);  tangents_3 = getitem_67 = None
        add_25: "f16[1536]" = torch.ops.aten.add.Tensor(tangents_4, getitem_68);  tangents_4 = getitem_68 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:118 in forward, code: embeddings = embeddings.transpose(0, 1).contiguous()
        transpose_16: "f16[32, 1024, 1536]" = torch.ops.aten.transpose.int(add_23, 0, 1);  add_23 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        embedding_dense_backward: "f16[2048, 1536]" = torch.ops.aten.embedding_dense_backward.default(transpose_16, wait_tensor_4, 2048, -1, False);  wait_tensor_4 = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/models/common/embeddings/language_model_embedding.py:111 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        add_26: "f16[2048, 1536]" = torch.ops.aten.add.Tensor(tangents_2, embedding_dense_backward);  tangents_2 = embedding_dense_backward = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        embedding_dense_backward_1: "f16[98304, 1536]" = torch.ops.aten.embedding_dense_backward.default(transpose_16, wait_tensor, 98304, -1, False);  transpose_16 = wait_tensor = None
        
         # File: /dfs/project/kgrlm/shirwu/hank/Megatron-LM/megatron/core/tensor_parallel/layers.py:263 in forward, code: output_parallel = F.embedding(masked_input, self.weight)
        add_27: "f16[98304, 1536]" = torch.ops.aten.add.Tensor(add_7, embedding_dense_backward_1);  add_7 = embedding_dense_backward_1 = None
        return (None, None, None, None, None, add_27, add_26, add_24, add_25, add_16, add_15, add_21, add_22, add_19, add_20, add_17, add_18, add_13, add_14, add_11, add_10, add_9, add_8, tangents_19, tangents_20)
        

