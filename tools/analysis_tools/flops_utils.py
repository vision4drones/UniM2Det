# Copyright (c) OpenMMLab. All rights reserved.
"""Analytical FLOPs helpers for GroundingDINO-style detectors."""
from typing import Dict, List, Sequence, Tuple, Union

import torch.nn as nn
from mmcv.ops import MultiScaleDeformableAttention
from torch import Tensor


def _get_attn_cfg(attn_module: nn.Module) -> Dict:
    """Extract config-like fields from an attention module."""
    cfg = {}
    for key in ('embed_dims', 'num_heads', 'num_levels', 'num_points'):
        if hasattr(attn_module, key):
            cfg[key] = getattr(attn_module, key)
    if 'embed_dims' not in cfg and hasattr(attn_module, 'embed_dim'):
        cfg['embed_dims'] = attn_module.embed_dim
    return cfg


def count_linear_flops(in_features: int, out_features: int,
                       num_tokens: int) -> int:
    """FLOPs for nn.Linear: 2 * tokens * in * out."""
    return 2 * num_tokens * in_features * out_features


def count_mha_flops(num_queries: int, num_keys: int, embed_dims: int,
                    num_heads: int) -> int:
    """FLOPs for standard multi-head attention (Q/K/V + attn + out proj)."""
    # Q, K, V projections
    flops = 3 * count_linear_flops(embed_dims, embed_dims, num_queries)
    flops += count_linear_flops(embed_dims, embed_dims, num_keys)
    # attention matrix + weighted sum
    head_dim = embed_dims // num_heads
    flops += 2 * num_heads * num_queries * num_keys * head_dim
    flops += 2 * num_heads * num_queries * num_keys * head_dim
    # output projection
    flops += count_linear_flops(embed_dims, embed_dims, num_queries)
    return flops


def count_ffn_flops(num_tokens: int, embed_dims: int,
                    ffn_channels: int) -> int:
    """FLOPs for a 2-layer FFN."""
    return (count_linear_flops(embed_dims, ffn_channels, num_tokens) +
            count_linear_flops(ffn_channels, embed_dims, num_tokens))


def count_ms_deform_attn_flops(
    num_queries: int,
    num_keys: int,
    embed_dims: int,
    num_heads: int = 8,
    num_levels: int = 4,
    num_points: int = 4,
) -> int:
    """Analytical FLOPs for MultiScaleDeformableAttention (full cost)."""
    head_dim = embed_dims // num_heads
    # value / output projections
    flops = count_linear_flops(embed_dims, embed_dims, num_keys)
    flops += count_linear_flops(embed_dims, embed_dims, num_queries)
    # sampling offset & attention weight projections
    flops += count_linear_flops(
        embed_dims, num_heads * num_levels * num_points * 2, num_queries)
    flops += count_linear_flops(
        embed_dims, num_heads * num_levels * num_points, num_queries)
    # bilinear sampling + weighted aggregation
    flops += count_ms_deform_sampling_flops(
        num_queries, embed_dims, num_heads, num_levels, num_points)
    return flops


def count_ms_deform_sampling_flops(
    num_queries: int,
    embed_dims: int,
    num_heads: int = 8,
    num_levels: int = 4,
    num_points: int = 4,
) -> int:
    """FLOPs for the CUDA sampling kernel missed by JIT tracing."""
    head_dim = embed_dims // num_heads
    return num_queries * num_heads * num_levels * num_points * head_dim * 15


def count_bert_flops(seq_len: int, hidden_size: int = 768,
                     intermediate_size: int = 3072,
                     num_layers: int = 12, num_heads: int = 12) -> int:
    """Analytical FLOPs for BERT encoder (12 layers by default)."""
    flops = 0
    for _ in range(num_layers):
        # self-attention
        flops += 3 * count_linear_flops(hidden_size, hidden_size, seq_len)
        flops += 2 * num_heads * seq_len * seq_len * (
            hidden_size // num_heads)
        flops += 2 * num_heads * seq_len * seq_len * (
            hidden_size // num_heads)
        flops += count_linear_flops(hidden_size, hidden_size, seq_len)
        # FFN
        flops += count_ffn_flops(seq_len, hidden_size, intermediate_size)
    return flops


def count_pix_moe_flops(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    mlp_ratio: float,
    num_experts: int,
    top_k: int,
) -> int:
    """Active FLOPs for MultiModalPixMoE (shared + top-k experts + router)."""
    num_tokens = batch_size * height * width
    hidden = int(channels * mlp_ratio)
    # shared expert MLP
    flops = count_ffn_flops(num_tokens, channels, hidden)
    # router
    flops += 2 * num_tokens * (2 * channels) * num_experts
    # top-k active sparse experts
    expert_flops = count_ffn_flops(num_tokens, channels, hidden)
    flops += top_k * expert_flops
    return flops


def find_ms_deform_attn_modules(model: nn.Module) -> List[nn.Module]:
    """Find all MultiScaleDeformableAttention modules in a model."""
    modules = []
    for module in model.modules():
        if isinstance(module, MultiScaleDeformableAttention):
            modules.append(module)
    return modules


def count_model_ms_deform_attn_flops(
    model: nn.Module,
    encoder_num_tokens: int,
    decoder_num_queries: int,
    num_keys: int,
) -> Tuple[int, int, Dict[str, int]]:
    """Count MSDeformAttn full & sampling-only FLOPs."""
    encoder_modules = []
    decoder_modules = []
    if hasattr(model, 'encoder'):
        encoder_modules = find_ms_deform_attn_modules(model.encoder)
    if hasattr(model, 'decoder'):
        decoder_modules = find_ms_deform_attn_modules(model.decoder)

    detail = {
        'encoder_ms_deform': 0,
        'decoder_ms_deform': 0,
        'encoder_sampling': 0,
        'decoder_sampling': 0,
    }
    full_total = 0
    sampling_total = 0

    for module in encoder_modules:
        cfg = _get_attn_cfg(module)
        embed_dims = cfg.get('embed_dims', 256)
        num_heads = cfg.get('num_heads', 8)
        num_levels = cfg.get('num_levels', 4)
        num_points = cfg.get('num_points', 4)
        layer_full = count_ms_deform_attn_flops(
            num_queries=encoder_num_tokens,
            num_keys=num_keys,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        layer_sampling = count_ms_deform_sampling_flops(
            num_queries=encoder_num_tokens,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        detail['encoder_ms_deform'] += layer_full
        detail['encoder_sampling'] += layer_sampling
        full_total += layer_full
        sampling_total += layer_sampling

    for module in decoder_modules:
        cfg = _get_attn_cfg(module)
        embed_dims = cfg.get('embed_dims', 256)
        num_heads = cfg.get('num_heads', 8)
        num_levels = cfg.get('num_levels', 4)
        num_points = cfg.get('num_points', 4)
        layer_full = count_ms_deform_attn_flops(
            num_queries=decoder_num_queries,
            num_keys=num_keys,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        layer_sampling = count_ms_deform_sampling_flops(
            num_queries=decoder_num_queries,
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
        )
        detail['decoder_ms_deform'] += layer_full
        detail['decoder_sampling'] += layer_sampling
        full_total += layer_full
        sampling_total += layer_sampling

    return full_total, sampling_total, detail


def format_flops(flops: Union[int, float]) -> str:
    """Format FLOPs count."""
    flops = int(flops)
    if flops >= 1e12:
        return f'{flops / 1e12:.2f} TFLOPs'
    if flops >= 1e9:
        return f'{flops / 1e9:.2f} GFLOPs'
    if flops >= 1e6:
        return f'{flops / 1e6:.2f} MFLOPs'
    return f'{flops} FLOPs'
