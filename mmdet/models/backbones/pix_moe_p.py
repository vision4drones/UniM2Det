import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import os

class Taskprompt(nn.Module):
    def __init__(self, in_dim, atom_num=32, atom_dim=256):
        super(Taskprompt, self).__init__()
        hidden_dim = 64
        self.CondNet = nn.Sequential(nn.Conv2d(in_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 3, 3), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, hidden_dim, 1), nn.LeakyReLU(0.1, True),
                                     nn.Conv2d(hidden_dim, 32, 1))
        self.lastOut = nn.Linear(32, atom_num)
        self.act = nn.GELU()
        self.dictionary = nn.Parameter(torch.randn(atom_num, atom_dim), requires_grad=True)
    def forward(self, x):
        out = self.CondNet(x)
        out = nn.AdaptiveAvgPool2d(1)(out)
        out = out.view(out.size(0), -1)
        out = self.lastOut(out)
        logits = F.softmax(out, -1)
        out = logits @ self.dictionary
        out = self.act(out)

        return out


class Expertset(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
    def es(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)
    def ee(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        ensemble = zeros.index_add(0, self._batch_index, stitched.float())
        ensemble[ensemble == 0] = np.finfo(float).eps
        return ensemble.log()
    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Mlp(nn.Module):
    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_feat = out_feat or in_feat
        h_feat = h_feat or in_feat
        self.fc1 = nn.Linear(in_feat, h_feat)
        self.act = act_layer()
        self.fc2 = nn.Linear(h_feat, out_feat)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return x

class MultiModalPixMoE(nn.Module):
    def __init__(self, input_size, output_size, mlp_ratio, modalities, num_experts, no=True, use_experts=2):
        super(MultiModalPixMoE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.no = no
        self.k = use_experts  # 4选2中的2
        self.num_experts_per_modality = num_experts
        self.modalities = modalities
        # 模态映射表，方便处理 list 类型的 source
        self.mod_map = {m: i for i, m in enumerate(self.modalities)}

        # 1. 共享专家 (Shared Expert) - 始终激活
        self.share_expert = Mlp(input_size, h_feat=int(input_size * mlp_ratio), out_feat=output_size)
        self.share_norm = nn.LayerNorm(output_size)
        self.share_proj = nn.Linear(output_size, output_size)

        # 2. 模态特定的动态专家池 (ModuleDict 包含 3x4 个专家)
        self.expert_pools = nn.ModuleDict({
            m: nn.ModuleList([
                Mlp(input_size, h_feat=int(input_size * mlp_ratio), out_feat=output_size)
                for _ in range(self.num_experts_per_modality)
            ]) for m in self.modalities
        })

        # 3. 模态特定的路由参数 (ParameterDict)
        # 每个模态的输入 (x+prompt) 映射到其对应的 4 个专家
        self.w_g = nn.ParameterDict({
            m: nn.Parameter(torch.randn(2*input_size, self.num_experts_per_modality)) for m in self.modalities
        })
        self.w_n = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(2*input_size, self.num_experts_per_modality)) for m in self.modalities
        })

        # 4. 各模态动态分支的归一化和投影
        self.dynamic_norm = nn.ModuleDict({m: nn.LayerNorm(output_size) for m in self.modalities})
        self.dynamic_proj = nn.ModuleDict({m: nn.Linear(output_size, output_size) for m in self.modalities})

        self.softmax = nn.Softmax(1)
        self.sp = nn.Softplus()
        self.final_norm = nn.LayerNorm(output_size)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        # 统计相关
        self.stats_data = {}
        self.record_stats = True

    def balance(self, x):
        eps = 1e-10
        if x.shape[0] <= 1: return torch.tensor([0], device=x.device)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def noisyk_w_modality(self, x, modality, train, noise_epsilon=1e-2):
        """针对特定模态的 4选2 路由逻辑"""
        w_g = self.w_g[modality]
        w_n = self.w_n[modality]

        clean_logits = x @ w_g
        if self.no and train:
            raw_noise_stddev = x @ w_n
            noise_stddev = ((self.sp(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # 执行 Top-K (k=2)
        top_logits, top_indices = logits.topk(self.k, dim=1)
        top_k_gates = self.softmax(top_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        w = zeros.scatter(1, top_indices, top_k_gates)

        # 负载均衡统计 s
        s = (w > 0).sum(0)
        return w, s

    def forward(self, x, prompt, source_list):
        """
        source: 'visible', 'thermal', 或 'fusion'
        """
        B, C, H, W = x.shape
        prompt = prompt.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        x = rearrange(x, 'b c h w -> (b h w) c')
        prompt = rearrange(prompt, 'b c h w -> (b h w) c')
        x_p = torch.cat((x, prompt), dim=1)

        # 将 source_list 映射为 Token 级别的掩码
        # 1. 生成 B 长度的 ID 向量
        source_ids = torch.tensor([self.mod_map[s] for s in source_list], device=x.device)
        # 2. 扩展到所有 Token (B*H*W)
        token_mod_ids = source_ids.view(B, 1, 1).expand(B, H, W).reshape(-1)

        # --- 分支 1: 共享专家 (全量处理) ---
        share_y = self.share_expert(x)
        share_y = self.share_norm(share_y)
        share_y = self.share_proj(share_y)

        # --- 分支 2: 动态路由专家 (分组处理) ---
        dynamic_out_full = torch.zeros_like(x)
        total_loss = 0.0

        for mod_name in self.modalities:
            mod_id = self.mod_map[mod_name]
            # 找到当前模态对应的 Token 索引
            indices = (token_mod_ids == mod_id).nonzero(as_tuple=True)[0]

            if indices.numel() == 0:
                continue

            # 提取该模态的 Token
            x_mod_p = x_p[indices]
            x_mod = x[indices]

            # 路由计算 (针对该模态子集)
            w, s = self.noisyk_w_modality(x_mod_p, mod_name, self.training, noise_epsilon=1e-2)

            # 计算局部负载均衡 Loss
            total_loss += (self.balance(w.sum(0)) + self.balance(s))

            # 调用该模态的专家池
            expertset = Expertset(self.num_experts_per_modality, w)
            expert_inputs = expertset.es(x_mod)
            current_pool = self.expert_pools[mod_name]
            expert_outputs = [current_pool[i](expert_inputs[i]) for i in range(self.num_experts_per_modality)]

            # 合并输出并写回对应位置 (使用 exp() 抵消 Expertset 内部的 log())
            mod_out = expertset.ee(expert_outputs)#.exp()
            mod_out = self.dynamic_norm[mod_name](mod_out)
            # mod_out = self.dynamic_proj[mod_name](mod_out)

            dynamic_out_full[indices] = mod_out.to(dynamic_out_full.dtype)

        # --- 最终融合 ---
        y = share_y + dynamic_out_full
        y = self.final_norm(y)
        y = rearrange(y, '(b h w) c -> b c h w', b=B, h=H, w=W)

        return y, total_loss


class TaskMoE_MultiModal_prompt(nn.Module):
    def __init__(self, atom_dim, dim, ffn_expansion_factor, modalities, num_experts, use_experts):
        super(TaskMoE_MultiModal_prompt, self).__init__()
        # 初始化新的多模态动态模型
        self.fc = nn.Linear(atom_dim, dim)
        self.model = MultiModalPixMoE(
            input_size=dim,
            output_size=dim,
            mlp_ratio=ffn_expansion_factor,
            modalities=modalities,
            num_experts=num_experts,
            no=True,
            use_experts=use_experts
        )

    def forward(self, x, prompt, source):
        # 进入模型，source 决定了使用哪个专家池和路由
        d = self.fc(prompt)
        out, loss = self.model(x, d, source)
        return out + x, loss