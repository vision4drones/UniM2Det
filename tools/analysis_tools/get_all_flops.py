# Copyright (c) OpenMMLab. All rights reserved.
"""Comprehensive FLOPs / Params statistics for GroundingDINO detectors.

This script combines:
1) JIT-traced FLOPs via mmengine (baseline, may undercount custom ops)
2) Analytical supplements for MultiScaleDeformableAttention
3) Component-wise breakdown (backbone+neck, text encoder, transformer+head)

Example:
    python tools/analysis_tools/get_all_flops.py \\
        configs/mm_grounding_dino/my_cfg/grounding_dino_swin-t_pretrain_multi_same_pe_pixmoe_p.py \\
        --num-images 1 --device cpu
"""
import argparse
import os
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.utils import digit_version

sys.path.insert(0, os.path.dirname(__file__))
from flops_utils import (
    count_bert_flops,
    count_model_ms_deform_attn_flops,
    count_pix_moe_flops,
    format_flops,
)

from mmdet.registry import MODELS

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError:
    raise ImportError('Please upgrade mmengine >= 0.6.0')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Comprehensive FLOPs/Params statistics')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--num-images',
        type=int,
        default=1,
        help='number of images used for FLOPs averaging')
    parser.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='device for FLOPs tracing (cpu avoids OOM)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config file')
    return parser.parse_args()


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_params(num: int) -> str:
    return _format_size(num)


class BackboneNeckWrapper(nn.Module):
    """Trace backbone + neck with fixed source_list."""

    def __init__(self, backbone, neck, source_list: List[Optional[str]]):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.source_list = source_list

    def forward(self, x: torch.Tensor):
        feats, _ = self.backbone(x, self.source_list)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats


class BertEncoderWrapper(nn.Module):
    """Trace BERT encoder with fixed tokenized inputs."""

    def __init__(self, language_model, tokenizer_input: Dict):
        super().__init__()
        self.language_backbone = language_model.language_backbone
        self.register_buffer(
            'input_ids',
            tokenizer_input['input_ids'],
            persistent=False)
        self.register_buffer(
            'attention_mask',
            tokenizer_input['attention_mask'].long(),
            persistent=False)
        if tokenizer_input.get('position_ids') is not None:
            self.register_buffer(
                'position_ids',
                tokenizer_input['position_ids'].long(),
                persistent=False)
        else:
            self.position_ids = None
        if tokenizer_input.get('token_type_ids') is not None:
            self.register_buffer(
                'token_type_ids',
                tokenizer_input['token_type_ids'].long(),
                persistent=False)
        else:
            self.token_type_ids = None

    def forward(self, _: torch.Tensor):
        inputs = {
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'position_ids': self.position_ids,
            'token_type_ids': self.token_type_ids,
        }
        return self.language_backbone(inputs)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def jit_flops(module: nn.Module, inputs: torch.Tensor) -> int:
    """Run JIT-based FLOPs counting for a submodule."""
    outputs = get_model_complexity_info(
        module,
        None,
        inputs=inputs,
        show_table=False,
        show_arch=False)
    return int(outputs['flops'])


def prepare_text_inputs(model, data_samples) -> Tuple[List[str], Dict]:
    """Prepare text prompts and BERT trace inputs from data samples."""
    text_prompts = [ds.text for ds in data_samples]
    if hasattr(model, 'get_tokens_positive_and_prompts'):
        custom_entities = getattr(data_samples[0], 'custom_entities', False)
        enhanced_text_prompts = []
        tokens_positives = []
        for ds in data_samples:
            enhanced_text_prompts.append(
                getattr(ds, 'caption_prompt', None))
            tokens_positives.append(ds.get('tokens_positive', None))
        if len(text_prompts) == 1:
            outputs = [
                model.get_tokens_positive_and_prompts(
                    text_prompts[0], custom_entities,
                    enhanced_text_prompts[0], tokens_positives[0])
            ] * len(data_samples)
        else:
            outputs = [
                model.get_tokens_positive_and_prompts(
                    tp, custom_entities, ep, tp_pos)
                for tp, ep, tp_pos in zip(
                    text_prompts, enhanced_text_prompts, tokens_positives)
            ]
        _, text_prompts, _, _ = zip(*outputs)
        text_prompts = list(text_prompts)

    device = next(model.parameters()).device
    tokenized = model.language_model.tokenizer(
        list(text_prompts),
        padding='max_length'
        if model.language_model.pad_to_max else 'longest',
        return_special_tokens_mask=True,
        return_tensors='pt',
        truncation=True).to(device)

    if model.language_model.use_sub_sentence_represent:
        from mmdet.models.language_models.bert import (
            generate_masks_with_special_tokens_and_transfer_map)
        attention_mask, position_ids = (
            generate_masks_with_special_tokens_and_transfer_map(
                tokenized, model.language_model.special_tokens))
        token_type_ids = tokenized['token_type_ids']
    else:
        attention_mask = tokenized['attention_mask']
        position_ids = None
        token_type_ids = None

    tokenizer_input = {
        'input_ids': tokenized['input_ids'],
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'token_type_ids': token_type_ids,
    }
    return list(text_prompts), tokenizer_input


@torch.no_grad()
def collect_forward_meta(model, inputs: torch.Tensor,
                         data_samples) -> Dict:
    """Collect runtime tensor shapes for analytical FLOPs."""
    source_list = []
    for sample in data_samples:
        source = getattr(sample, 'source', None)
        if hasattr(model, 'source_names') and source not in model.source_names:
            source = None
        source_list.append(source)

    text_prompts, tokenizer_input = prepare_text_inputs(model, data_samples)
    text_dict = model.language_model(text_prompts)
    if getattr(model, 'text_feat_map', None) is not None:
        text_dict['embedded'] = model.text_feat_map(text_dict['embedded'])

    visual_feats, _ = model.extract_feat(inputs, source_list)
    encoder_inputs, _ = model.pre_transformer(visual_feats, data_samples)

    spatial_shapes = encoder_inputs['spatial_shapes']
    num_feat_points = int(encoder_inputs['feat'].shape[1])
    if spatial_shapes.dim() == 2:
        num_keys = int(
            (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum().item())
    else:
        num_keys = num_feat_points

    meta = {
        'source_list': source_list,
        'text_prompts': text_prompts,
        'tokenizer_input': tokenizer_input,
        'text_len': int(text_dict['embedded'].shape[1]),
        'spatial_shapes': spatial_shapes.cpu().tolist(),
        'num_feat_points': num_feat_points,
        'num_keys': num_keys,
        'num_queries': int(model.num_queries),
        'input_shape': tuple(inputs.shape),
        'pad_shape': tuple(inputs.shape[-2:]),
    }

    # PixMoE shape: applied right after patch embed (H/4 x W/4)
    _, _, h, w = inputs.shape
    meta['moe_spatial'] = (h // 4, w // 4)
    if hasattr(model.backbone, 'embed_dims'):
        meta['moe_channels'] = int(model.backbone.embed_dims)
    else:
        meta['moe_channels'] = 96

    return meta


def estimate_pix_moe_supplement(model, meta: Dict) -> int:
    """Estimate PixMoE supplement if backbone JIT undercounts routing ops."""
    backbone = model.backbone
    if not hasattr(backbone, 'moe_cfg') or backbone.moe_cfg is None:
        return 0
    moe_cfg = backbone.moe_cfg
    b = meta['input_shape'][0]
    h, w = meta['moe_spatial']
    c = meta['moe_channels']
    return count_pix_moe_flops(
        batch_size=b,
        channels=c,
        height=h,
        width=w,
        mlp_ratio=moe_cfg.get('ffn_expansion_factor', 2),
        num_experts=moe_cfg.get('num_experts', 4),
        top_k=moe_cfg.get('use_experts', 2),
    )


def run_statistics(args, logger) -> Dict:
    if digit_version(torch.__version__) < digit_version('1.12'):
        logger.warning(
            'torch<1.12 may have compatibility issues with torch.jit tracing.')

    cfg = Config.fromfile(args.config)
    cfg.val_dataloader.batch_size = 1
    cfg.work_dir = tempfile.TemporaryDirectory().name
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    device = resolve_device(args.device)

    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model).to(device)
    model = revert_sync_batchnorm(model)
    model.eval()

    result = {
        'component_flops': {},
        'component_flops_gflops': {},
        'supplements': {},
        'supplements_gflops': {},
    }
    jit_totals = []

    _forward = model.forward
    for idx, data_batch in enumerate(data_loader):
        if idx == args.num_images:
            break

        data = model.data_preprocessor(data_batch)
        inputs = data['inputs'].to(device)
        data_samples = data['data_samples']

        meta = collect_forward_meta(model, inputs, data_samples)
        result['input_shape'] = meta['input_shape']
        result['pad_shape'] = meta['pad_shape']
        result['text_len'] = meta['text_len']
        result['num_feat_points'] = meta['num_feat_points']
        result['num_queries'] = meta['num_queries']
        result['spatial_shapes'] = meta['spatial_shapes']
        result['text_prompts'] = meta['text_prompts']

        # --- 1) Full-model JIT FLOPs (baseline) ---
        model.forward = partial(_forward, data_samples=data_samples)
        jit_full = jit_flops(model, inputs)
        jit_totals.append(jit_full)
        model.forward = _forward

        # --- 2) Component JIT FLOPs ---
        backbone_neck = BackboneNeckWrapper(
            model.backbone, model.neck, meta['source_list']).to(device).eval()
        jit_backbone_neck = jit_flops(backbone_neck, inputs)

        bert_wrapper = BertEncoderWrapper(
            model.language_model,
            meta['tokenizer_input']).to(device).eval()
        dummy = torch.zeros(1, device=device)
        jit_bert = jit_flops(bert_wrapper, dummy)

        jit_transformer_head = max(
            jit_full - jit_backbone_neck - jit_bert, 0)

        result['component_flops']['backbone_neck_jit'] = jit_backbone_neck
        result['component_flops']['text_encoder_jit'] = jit_bert
        result['component_flops']['transformer_head_jit'] = jit_transformer_head
        result['component_flops']['full_model_jit'] = jit_full

        # --- 3) Analytical supplements ---
        ms_deform_full, ms_deform_sampling, ms_deform_detail = (
            count_model_ms_deform_attn_flops(
                model,
                encoder_num_tokens=meta['num_feat_points'],
                decoder_num_queries=meta['num_queries'],
                num_keys=meta['num_keys'],
            ))
        ms_deform_supplement = ms_deform_sampling

        bert_body = model.language_model.language_backbone[0]
        bert_analytical = count_bert_flops(
            seq_len=meta['text_len'],
            hidden_size=bert_body.model.config.hidden_size,
            intermediate_size=bert_body.model.config.intermediate_size,
            num_layers=bert_body.model.config.num_hidden_layers,
            num_heads=bert_body.model.config.num_attention_heads,
        )
        bert_supplement = max(bert_analytical - jit_bert, 0)

        pix_moe_supplement = estimate_pix_moe_supplement(model, meta)
        # Only add MoE supplement if backbone JIT seems too low
        if jit_backbone_neck < pix_moe_supplement:
            pix_moe_supplement = int(pix_moe_supplement * 0.3)
        else:
            pix_moe_supplement = 0

        result['supplements']['ms_deform_attn'] = ms_deform_supplement
        result['supplements']['ms_deform_detail'] = ms_deform_detail
        result['supplements']['bert'] = bert_supplement
        result['supplements']['pix_moe'] = pix_moe_supplement
        result['supplements']['ms_deform_analytical_total'] = ms_deform_full
        result['supplements']['bert_analytical_total'] = bert_analytical

    del data_loader

    jit_mean = int(np.mean(jit_totals))
    supplement_sum = sum(
        v for k, v in result['supplements'].items()
        if k not in ('ms_deform_detail', 'ms_deform_analytical_total',
                     'bert_analytical_total') and isinstance(v, int))
    corrected = jit_mean + supplement_sum

    result['jit_flops'] = jit_mean
    result['supplement_flops'] = supplement_sum
    result['corrected_flops'] = corrected

    # Params
    total_params = count_params(model)
    text_params = count_params(model.language_model)
    result['params_total'] = total_params
    result['params_text'] = text_params
    result['params_visual'] = total_params - text_params

    for key, val in result['component_flops'].items():
        result['component_flops_gflops'][key] = val / 1e9
    for key, val in result['supplements'].items():
        if isinstance(val, int):
            result['supplements_gflops'][key] = val / 1e9

    result['jit_flops_gflops'] = jit_mean / 1e9
    result['supplement_flops_gflops'] = supplement_sum / 1e9
    result['corrected_flops_gflops'] = corrected / 1e9

    return result


def print_report(result: Dict):
    split = '=' * 50
    print(split)
    print('Comprehensive FLOPs / Params Report')
    print(split)
    print(f"Input shape      : {result['input_shape']}")
    print(f"Pad shape        : {result['pad_shape']}")
    print(f"Text length      : {result['text_len']}")
    print(f"Feature points   : {result['num_feat_points']}")
    print(f"Num queries      : {result['num_queries']}")
    print(f"Spatial shapes   : {result['spatial_shapes']}")
    print(f"Text prompts     : {result['text_prompts']}")
    print(split)
    print('[Params]')
    print(f"  Total          : {format_params(result['params_total'])}")
    print(f"  Text (BERT)    : {format_params(result['params_text'])}")
    print(f"  Visual (no BERT): {format_params(result['params_visual'])}")
    print(split)
    print('[FLOPs - JIT traced (baseline, may underestimate)]')
    for name, flops in result['component_flops'].items():
        print(f"  {name:<26}: {format_flops(flops)}")
    print(split)
    print('[FLOPs - Analytical supplements]')
    print(f"  ms_deform sampling kernel"
          f"      : {format_flops(result['supplements']['ms_deform_attn'])}")
    detail = result['supplements']['ms_deform_detail']
    print(f"    - encoder sampling            "
          f": {format_flops(detail['encoder_sampling'])}")
    print(f"    - decoder sampling            "
          f": {format_flops(detail['decoder_sampling'])}")
    print(f"    - encoder MSDeform (full)     "
          f": {format_flops(detail['encoder_ms_deform'])}")
    print(f"    - decoder MSDeform (full)     "
          f": {format_flops(detail['decoder_ms_deform'])}")
    print(f"  bert (analytical - JIT)       "
          f": {format_flops(result['supplements']['bert'])}")
    print(f"  pix_moe (if undercounted)     "
          f": {format_flops(result['supplements']['pix_moe'])}")
    print(split)
    print('[FLOPs - Summary]')
    print(f"  JIT baseline     : {format_flops(result['jit_flops'])}")
    print(f"  + Supplements    : {format_flops(result['supplement_flops'])}")
    print(f"  Corrected total  : {format_flops(result['corrected_flops'])}")
    print(split)
    print('Notes:')
    print('  1. JIT baseline undercounts mmcv MultiScaleDeformableAttention.')
    print('  2. MoE reports active FLOPs (top-k experts), not dense FLOPs.')
    print('  3. Use the same input resolution / text length when comparing.')
    print('  4. For papers, verify against profiler or manual calculation.')
    print(split)


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='MMLogger')
    if not Path(args.config).exists():
        raise FileNotFoundError(f'{args.config} not found.')
    result = run_statistics(args, logger)
    print_report(result)


if __name__ == '__main__':
    main()
