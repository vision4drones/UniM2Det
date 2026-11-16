import torch
import torch.nn as nn
from einops import rearrange


def l2_normalize(x, dim=-1, eps=1e-6):
    return x / (x.norm(dim=dim, keepdim=True).clamp_min(eps))


class Mlp(nn.Module):
    def __init__(self, in_feat, h_feat=None, out_feat=None, act_layer=nn.GELU,
                 drop=0.):
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


class MultiModalSoftMoE(nn.Module):
    """Full-image Soft MoE with per-modality expert pools and slot embeddings.

    Dispatch softmax is over all spatial tokens of one image. Combine softmax
    is over all slots. Each expert is hard-wired to its own slots.
    """

    def __init__(self,
                 dim,
                 mlp_ratio,
                 modalities,
                 num_experts,
                 num_slots=1,
                 use_share_expert=False,
                 drop=0.):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.modalities = list(modalities)
        self.mod_map = {m: i for i, m in enumerate(self.modalities)}
        self.use_share_expert = use_share_expert

        hidden = int(dim * mlp_ratio)
        self.expert_pools = nn.ModuleDict({
            m: nn.ModuleList([
                Mlp(dim, h_feat=hidden, out_feat=dim, drop=drop)
                for _ in range(num_experts)
            ])
            for m in self.modalities
        })
        self.slot_embeds = nn.ParameterDict({
            m: nn.Parameter(torch.randn(num_experts, num_slots, dim) * 0.02)
            for m in self.modalities
        })
        self.slot_scales = nn.ParameterDict({
            m: nn.Parameter(torch.ones(1)) for m in self.modalities
        })

        if use_share_expert:
            self.share_expert = Mlp(
                dim, h_feat=hidden, out_feat=dim, drop=drop)
        else:
            self.share_expert = None

    def _soft_moe_one_modality(self, x, modality):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            y: [B, C, H, W]
        """
        b, c, h, w = x.shape
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        # Routing softmax over all spatial tokens is unstable in fp16/AMP.
        tokens_f = tokens.float()
        tokens_n = l2_normalize(tokens_f, dim=-1)
        phi = self.slot_embeds[modality].float()
        phi_n = self.slot_scales[modality].float() * l2_normalize(phi, dim=-1)

        # logits: [B, M, E, S]
        logits = torch.einsum('bmd,esd->bmes', tokens_n, phi_n)
        dispatch = logits.softmax(dim=1)
        combine = rearrange(logits, 'b m e s -> b m (e s)').softmax(dim=-1)

        # slots: [B, E, S, D], mixture of original tokens
        slots = torch.einsum('bmd,bmes->besd', tokens_f, dispatch).to(dtype=x.dtype)

        expert_outs = []
        experts = self.expert_pools[modality]
        for e, expert in enumerate(experts):
            expert_outs.append(expert(slots[:, e]))
        expert_outs = torch.stack(expert_outs, dim=1)
        expert_outs = rearrange(expert_outs, 'b e s d -> b (e s) d')

        y = torch.einsum('bkd,bmk->bmd', expert_outs.float(), combine)
        if self.share_expert is not None:
            y = y + self.share_expert(tokens).float()
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
        return y.to(dtype=x.dtype)

    def forward(self, x, source_list):
        """
        Args:
            x: [B, C, H, W]
            source_list: list of modality names, length B
        """
        out = torch.zeros_like(x)
        for mod_name in self.modalities:
            indices = [i for i, s in enumerate(source_list) if s == mod_name]
            if len(indices) == 0:
                continue
            idx = torch.tensor(indices, device=x.device)
            y_mod = self._soft_moe_one_modality(x[idx], mod_name)
            out[idx] = y_mod.to(dtype=out.dtype)
        return out


class TaskSoftMoE_MultiModal(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor,
                 modalities,
                 num_experts,
                 num_slots=1,
                 use_share_expert=False):
        super().__init__()
        self.model = MultiModalSoftMoE(
            dim=dim,
            mlp_ratio=ffn_expansion_factor,
            modalities=modalities,
            num_experts=num_experts,
            num_slots=num_slots,
            use_share_expert=use_share_expert,
        )

    def forward(self, x, source_list):
        out = self.model(x, source_list)
        return out + x, None
