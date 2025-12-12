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
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[32768]", arg1_1: "bf16[151936, 1536]", arg2_1: "bf16[1536]", arg3_1: "bf16[2048]", arg4_1: "bf16[2048, 1536]", arg5_1: "bf16[32768, 128]", arg6_1: "i64[32768]", arg7_1: "bf16[1536, 1536]", arg8_1: "bf16[1536]", arg9_1: "bf16[17920, 1536]", arg10_1: "bf16[1536, 8960]", arg11_1: "bf16[1536]"):
         # File: /opt/tiger/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py:47 in embedding, code: return F.embedding(input_, layer.weight)
        embedding: "bf16[32768, 1536]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/layernorm.py:94 in forward_cuda, code: out = torch.empty_like(x)
        empty_like: "bf16[32768, 1536]" = torch.ops.aten.empty_like.default(embedding, pin_memory = False)
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/layernorm.py:98 in forward_cuda, code: self.weight.data,
        detach: "bf16[1536]" = torch.ops.aten.detach.default(arg2_1);  arg2_1 = None
        
         # File: /opt/tiger/vllm/vllm/_custom_ops.py:153 in rms_norm, code: torch.ops._C.rms_norm(out, input, weight, epsilon)
        auto_functionalized = torch.ops.higher_order.auto_functionalized(torch.ops._C.rms_norm.default, result = empty_like, input = embedding, weight = detach, epsilon = 1e-06);  empty_like = detach = None
        getitem_1: "bf16[32768, 1536]" = auto_functionalized[1];  auto_functionalized = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/linear.py:142 in apply, code: return F.linear(x, layer.weight, bias)
        t: "bf16[1536, 2048]" = torch.ops.aten.t.default(arg4_1);  arg4_1 = None
        addmm: "bf16[32768, 2048]" = torch.ops.aten.addmm.default(arg3_1, getitem_1, t);  arg3_1 = getitem_1 = t = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/models/qwen2.py:177 in forward, code: q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        split_with_sizes = torch.ops.aten.split_with_sizes.default(addmm, [1536, 256, 256], -1)
        getitem_2: "bf16[32768, 1536]" = split_with_sizes[0]
        getitem_3: "bf16[32768, 256]" = split_with_sizes[1];  split_with_sizes = None
        
         # File: /opt/tiger/vllm/vllm/_custom_ops.py:136 in rotary_embedding, code: torch.ops._C.rotary_embedding(positions, query, key, head_size,
        auto_functionalized_1 = torch.ops.higher_order.auto_functionalized(torch.ops._C.rotary_embedding.default, positions = arg6_1, query = getitem_2, key = getitem_3, head_size = 128, cos_sin_cache = arg5_1, is_neox = True);  arg6_1 = getitem_2 = getitem_3 = None
        getitem_6: "bf16[32768, 1536]" = auto_functionalized_1[1]
        getitem_7: "bf16[32768, 256]" = auto_functionalized_1[2];  auto_functionalized_1 = None
        slice_scatter: "bf16[32768, 2048]" = torch.ops.aten.slice_scatter.default(addmm, getitem_6, 1, 0, 1536);  addmm = getitem_6 = None
        slice_scatter_1: "bf16[32768, 2048]" = torch.ops.aten.slice_scatter.default(slice_scatter, getitem_7, 1, 1536, 1792);  slice_scatter = getitem_7 = None
        
         # File: /opt/tiger/vllm/vllm/attention/layer.py:167 in forward, code: output = torch.empty_like(query)
        split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(slice_scatter_1, [1536, 256, 256], -1)
        getitem_14: "bf16[32768, 1536]" = split_with_sizes_3[0];  split_with_sizes_3 = None
        empty_like_1: "bf16[32768, 1536]" = torch.ops.aten.empty_like.default(getitem_14, pin_memory = False);  getitem_14 = None
        
         # File: /opt/tiger/vllm/vllm/attention/layer.py:173 in forward, code: output = output.view(-1, self.num_heads, self.head_size)
        view_1: "bf16[32768, 12, 128]" = torch.ops.aten.view.default(empty_like_1, [-1, 12, 128]);  empty_like_1 = None
        
         # File: /opt/tiger/vllm/vllm/attention/layer.py:190 in forward, code: torch.ops.vllm.unified_attention_with_output(
        split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(slice_scatter_1, [1536, 256, 256], -1)
        getitem_17: "bf16[32768, 1536]" = split_with_sizes_4[0];  split_with_sizes_4 = None
        view_4: "bf16[32768, 12, 128]" = torch.ops.aten.view.default(getitem_17, [-1, 12, 128]);  getitem_17 = None
        split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(slice_scatter_1, [1536, 256, 256], -1)
        getitem_21: "bf16[32768, 256]" = split_with_sizes_5[1];  split_with_sizes_5 = None
        view_5: "bf16[32768, 2, 128]" = torch.ops.aten.view.default(getitem_21, [-1, 2, 128]);  getitem_21 = None
        split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(slice_scatter_1, [1536, 256, 256], -1);  slice_scatter_1 = None
        getitem_25: "bf16[32768, 256]" = split_with_sizes_6[2];  split_with_sizes_6 = None
        view_6: "bf16[32768, 2, 128]" = torch.ops.aten.view.default(getitem_25, [-1, 2, 128]);  getitem_25 = None
        auto_functionalized_2 = torch.ops.higher_order.auto_functionalized(torch.ops.vllm.unified_attention_with_output.default, query = view_4, key = view_5, value = view_6, output = view_1, layer_name = 'model.layers.0.self_attn.attn');  view_4 = view_5 = view_6 = view_1 = None
        getitem_27: "bf16[32768, 12, 128]" = auto_functionalized_2[1];  auto_functionalized_2 = None
        view_7: "bf16[32768, 1536]" = torch.ops.aten.view.default(getitem_27, [32768, 1536]);  getitem_27 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/linear.py:142 in apply, code: return F.linear(x, layer.weight, bias)
        t_1: "bf16[1536, 1536]" = torch.ops.aten.t.default(arg7_1);  arg7_1 = None
        view_10: "bf16[32768, 12, 128]" = torch.ops.aten.view.default(view_7, [-1, 12, 128]);  view_7 = None
        view_11: "bf16[32768, 1536]" = torch.ops.aten.view.default(view_10, [-1, 1536]);  view_10 = None
        mm: "bf16[32768, 1536]" = torch.ops.aten.mm.default(view_11, t_1);  view_11 = t_1 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/layernorm.py:90 in forward_cuda, code: self.weight.data,
        detach_1: "bf16[1536]" = torch.ops.aten.detach.default(arg8_1);  arg8_1 = None
        
         # File: /opt/tiger/vllm/vllm/_custom_ops.py:158 in fused_add_rms_norm, code: torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
        auto_functionalized_3 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = mm, residual = embedding, weight = detach_1, epsilon = 1e-06);  mm = embedding = detach_1 = None
        getitem_29: "bf16[32768, 1536]" = auto_functionalized_3[1]
        getitem_30: "bf16[32768, 1536]" = auto_functionalized_3[2];  auto_functionalized_3 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/linear.py:142 in apply, code: return F.linear(x, layer.weight, bias)
        t_2: "bf16[1536, 17920]" = torch.ops.aten.t.default(arg9_1);  arg9_1 = None
        mm_1: "bf16[32768, 17920]" = torch.ops.aten.mm.default(getitem_29, t_2);  getitem_29 = t_2 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/activation.py:81 in forward_cuda, code: out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        empty: "bf16[32768, 8960]" = torch.ops.aten.empty.memory_format([32768, 8960], dtype = torch.bfloat16, device = device(type='cuda', index=0), pin_memory = False)
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/activation.py:82 in forward_cuda, code: self.op(out, x)
        auto_functionalized_4 = torch.ops.higher_order.auto_functionalized(torch.ops._C.silu_and_mul.default, out = empty, input = mm_1);  empty = mm_1 = None
        getitem_32: "bf16[32768, 8960]" = auto_functionalized_4[1];  auto_functionalized_4 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/linear.py:142 in apply, code: return F.linear(x, layer.weight, bias)
        t_3: "bf16[8960, 1536]" = torch.ops.aten.t.default(arg10_1);  arg10_1 = None
        mm_2: "bf16[32768, 1536]" = torch.ops.aten.mm.default(getitem_32, t_3);  getitem_32 = t_3 = None
        
         # File: /opt/tiger/vllm/vllm/model_executor/layers/layernorm.py:90 in forward_cuda, code: self.weight.data,
        detach_2: "bf16[1536]" = torch.ops.aten.detach.default(arg11_1);  arg11_1 = None
        
         # File: /opt/tiger/vllm/vllm/_custom_ops.py:158 in fused_add_rms_norm, code: torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
        auto_functionalized_5 = torch.ops.higher_order.auto_functionalized(torch.ops._C.fused_add_rms_norm.default, input = mm_2, residual = getitem_30, weight = detach_2, epsilon = 1e-06);  mm_2 = getitem_30 = detach_2 = None
        getitem_34: "bf16[32768, 1536]" = auto_functionalized_5[1];  auto_functionalized_5 = None
        return (getitem_34, arg5_1)
        

