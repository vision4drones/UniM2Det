#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Get model Params and FLOPs for MMDet / GroundingDINO configs.

Adapted from mmgdDINO/tools/others/get_params_flops.py with multi-modal
(source / channel) support for GroundingDINO_multi_MoE.

Example:
    python tools/analysis_tools/get_params_flops.py \\
        configs/mm_grounding_dino/my_cfg/grounding_dino_swin-t_pretrain_multi_same_pe_pixmoe_p.py \\
        --device cpu --exclude-text-backbone
"""
import argparse
import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.model import revert_sync_batchnorm
from mmengine.registry import init_default_scope
from mmengine.runner import Runner

# Force priority to current repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules

try:
    from mmengine.analysis import get_model_complexity_info
    from mmengine.analysis.print_helper import _format_size
except ImportError as e:
    raise ImportError('Please upgrade mmengine >= 0.6.0') from e


def parse_size_to_float_and_unit(size_str):
    """Convert size string like '0.408T' to (0.408, 'T')."""
    s = size_str.strip()
    unit = s[-1]
    val = float(s[:-1])
    return val, unit


def format_with_same_unit(value, unit):
    return f'{value:.3f}{unit}'


def format_params_m(num_params):
    """Format parameter count in millions with 3 decimal places."""
    return f'{float(num_params) / 1e6:.3f}M'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get model params and FLOPs for MMDet config.')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--mode',
        choices=['auto', 'dataloader', 'dummy'],
        default='auto',
        help=('How to construct inputs for FLOPs. '
              '"auto": try dataloader first then fallback to dummy input.'))
    parser.add_argument(
        '--num-images',
        type=int,
        default=20,
        help='Number of images for averaging FLOPs in dataloader mode.')
    parser.add_argument(
        '--shape',
        nargs=2,
        type=int,
        metavar=('H', 'W'),
        default=[800, 1333],
        help='Dummy input shape (H W), used in dummy/auto-fallback mode.')
    parser.add_argument(
        '--channels',
        type=int,
        default=None,
        help=('Input channels for dummy mode. '
              'Default: 6 for fusion source, 3 otherwise.'))
    parser.add_argument(
        '--source',
        type=str,
        default='visible',
        help='Modality source for MoE backbone in dummy mode.')
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device for model complexity calculation.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in config, key=value style.')
    parser.add_argument(
        '--exclude-text-backbone',
        action='store_true',
        help=('Exclude text tokenizer/backbone/text mapping branches from '
              'Params/FLOPs. This keeps detection + multimodal transformer '
              'paths, but replaces language model and text mapping modules '
              'with parameter-free placeholders.'))
    parser.add_argument(
        '--analysis-text',
        type=str,
        default='drone . car .',
        help=('Reference text used in dummy mode and for inferring the token '
              'sequence length when excluding the text backbone.'))
    return parser.parse_args()


def build_model(cfg, device):
    register_all_modules(init_default_scope=False)
    # Ensure custom detectors in this repo are registered.
    import mmdet.models.detectors.grounding_dino_multi_moe  # noqa: F401
    import mmdet.models.detectors.grounding_dino_multi  # noqa: F401
    import mmdet.models.detectors.grounding_dino  # noqa: F401

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)
    model = revert_sync_batchnorm(model)
    if device == 'cuda' and torch.cuda.is_available():
        model = model.cuda()
        actual_device = 'cuda'
    else:
        model = model.cpu()
        actual_device = 'cpu'
    model.eval()
    return model, actual_device


class _DummyLanguageModel(nn.Module):
    """Parameter-free language model placeholder for FLOPs analysis."""

    def __init__(self, embed_dims=256, seq_len=256):
        super().__init__()
        self.embed_dims = embed_dims
        self.seq_len = seq_len
        self.register_buffer('_dev_anchor', torch.zeros(1), persistent=False)

    def forward(self, text_prompts):
        bs = len(text_prompts)
        dev = self._dev_anchor.device
        embedded = torch.zeros(
            (bs, self.seq_len, self.embed_dims),
            dtype=torch.float32,
            device=dev)
        text_token_mask = torch.ones(
            (bs, self.seq_len), dtype=torch.bool, device=dev)
        masks = torch.zeros(
            (bs, self.seq_len, self.seq_len),
            dtype=torch.bool,
            device=dev)
        position_ids = torch.arange(
            self.seq_len, dtype=torch.long, device=dev).unsqueeze(0).repeat(
                bs, 1)
        return dict(
            embedded=embedded,
            text_token_mask=text_token_mask,
            masks=masks,
            position_ids=position_ids)


def infer_seq_len_from_tokenizer(model, reference_text):
    """Infer effective token length from the original tokenizer."""
    if not hasattr(model, 'language_model'):
        return 256
    language_model = model.language_model
    tokenizer = getattr(language_model, 'tokenizer', None)
    max_tokens = getattr(language_model, 'max_tokens', 256)
    if tokenizer is None:
        return max_tokens

    tokenized = tokenizer([reference_text], return_tensors='pt')
    seq_len = int(tokenized.input_ids.shape[1])
    return min(seq_len, max_tokens)


def strip_text_backbone_for_analysis(model, reference_text):
    """Remove text tokenizer/backbone/mapping params and their FLOPs."""
    if not hasattr(model, 'embed_dims'):
        return
    seq_len = infer_seq_len_from_tokenizer(model, reference_text)

    model.language_model = _DummyLanguageModel(
        embed_dims=int(model.embed_dims), seq_len=seq_len).to(
            next(model.parameters()).device)
    model.text_feat_map = None
    if hasattr(model, 'specific_maps'):
        model.specific_maps = nn.ModuleDict()
    patch_forward_without_text(model)


def patch_forward_without_text(model):
    """Bypass tokenizer/text preprocessing after replacing language_model."""

    def _forward_wo_text(self, batch_inputs, batch_data_samples=None):
        source_list = []
        source_names = getattr(self, 'source_names', [])
        for sample in batch_data_samples:
            source = getattr(sample, 'source', None)
            if source_names and source not in source_names:
                source = None
            source_list.append(source)

        visual_feats, _ = self.extract_feat(batch_inputs, source_list)
        text_prompts = [''] * batch_inputs.shape[0]
        text_dict = self.language_model(text_prompts)
        head_inputs_dict = self.forward_transformer(
            visual_feats, text_dict, batch_data_samples)
        return self.bbox_head.forward(**head_inputs_dict)

    import types
    model._forward = types.MethodType(_forward_wo_text, model)


def disable_checkpoint_for_analysis(cfg):
    """Disable checkpoint settings that break JIT-based FLOPs tracing."""
    model_cfg = cfg.model
    if 'encoder' in model_cfg and isinstance(model_cfg.encoder, dict):
        if 'num_cp' in model_cfg.encoder:
            model_cfg.encoder.num_cp = 0
    if 'backbone' in model_cfg and isinstance(model_cfg.backbone, dict):
        if 'with_cp' in model_cfg.backbone:
            model_cfg.backbone.with_cp = False


def resolve_dummy_channels(args, model=None):
    if args.channels is not None:
        return args.channels
    # PixMoE backbone task_prompt expects 6-channel input in this repo.
    if model is not None and hasattr(model.backbone, 'task_prompt'):
        return 6
    return 6 if args.source == 'fusion' else 3


def build_dummy_data_sample(h, w, text, source):
    """Build a minimal DetDataSample for dummy-mode FLOPs tracing."""
    data_sample = DetDataSample()
    data_sample.set_metainfo(
        dict(
            img_shape=(h, w),
            ori_shape=(h, w),
            pad_shape=(h, w),
            batch_input_shape=(h, w)))
    data_sample.text = text
    data_sample.custom_entities = True
    data_sample.source = source
    return data_sample


def compute_with_dataloader(cfg, model, num_images):
    avg_flops = []
    result = {}
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    model_forward = model.forward
    params = None
    input_channels = None

    for idx, data_batch in enumerate(data_loader):
        if idx >= num_images:
            break
        data = model.data_preprocessor(data_batch)
        data_samples = data.get('data_samples', None)
        result['ori_shape'] = data_samples[0].ori_shape
        result['pad_shape'] = getattr(data_samples[0], 'batch_input_shape',
                                      data_samples[0].pad_shape)
        input_channels = int(data['inputs'].shape[1])
        model.forward = partial(model_forward, data_samples=data_samples)
        outputs = get_model_complexity_info(
            model,
            None,
            inputs=data['inputs'],
            show_table=False,
            show_arch=False)
        avg_flops.append(outputs['flops'])
        params = outputs['params']

    model.forward = model_forward
    if not avg_flops:
        raise RuntimeError('No samples were loaded from val_dataloader.')

    result['compute_type'] = 'dataloader'
    result['flops'] = _format_size(int(np.average(avg_flops)))
    result['params'] = format_params_m(params)
    result['input_channels'] = input_channels
    return result


def compute_with_dummy(model, device, shape_hw, channels, source, text):
    h, w = shape_hw
    dev = torch.device('cuda' if device == 'cuda' else 'cpu')
    dummy = torch.randn(1, channels, h, w, device=dev)

    data_sample = build_dummy_data_sample(h, w, text, source)

    model_forward = model.forward
    model.forward = partial(model_forward, data_samples=[data_sample])
    outputs = get_model_complexity_info(
        model, None, inputs=dummy, show_table=False, show_arch=False)
    model.forward = model_forward

    return {
        'compute_type': 'dummy_input',
        'ori_shape': (h, w),
        'pad_shape': (h, w),
        'flops': _format_size(int(outputs['flops'])),
        'params': format_params_m(outputs['params']),
        'input_channels': channels,
        'source': source,
    }


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='complexity_logger')

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f'Config not found: {config_path}')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    disable_checkpoint_for_analysis(cfg)
    cfg.work_dir = tempfile.TemporaryDirectory().name
    if hasattr(cfg, 'val_dataloader'):
        cfg.val_dataloader.batch_size = 1

    model, actual_device = build_model(cfg, args.device)
    model._analysis_text = args.analysis_text
    if args.exclude_text_backbone:
        strip_text_backbone_for_analysis(model, args.analysis_text)

    result = None
    if args.mode in ('auto', 'dataloader'):
        try:
            result = compute_with_dataloader(cfg, model, args.num_images)
        except Exception as e:
            if args.mode == 'dataloader':
                raise
            logger.warning(
                f'Dataloader mode failed, fallback to dummy mode. Cause: {e}')

    if result is None:
        channels = resolve_dummy_channels(args, model)
        result = compute_with_dummy(
            model,
            actual_device,
            args.shape,
            channels,
            args.source,
            args.analysis_text)

    split_line = '=' * 36
    print(split_line)
    print(f'Config: {args.config}')
    print(f'Device: {actual_device}')
    print(f'Compute type: {result["compute_type"]}')
    print(f'Exclude text backbone: {args.exclude_text_backbone}')
    print(f'Analysis text: {args.analysis_text}')
    if result.get('input_channels') is not None:
        print(f'Input channels: {result["input_channels"]}')
    if result.get('source') is not None:
        print(f'Source modality: {result["source"]}')
    print(f'Input shape: {result["pad_shape"]}')
    print(f'FLOPs: {result["flops"]}')
    flops_val, flops_unit = parse_size_to_float_and_unit(result['flops'])
    print(f'FLOPs (x2 MAC convention): '
          f'{format_with_same_unit(flops_val * 2.0, flops_unit)}')
    print(f'Params: {result["params"]}')
    print(split_line)
    print('Note: FLOPs may differ by input construction and unsupported ops.')
    print('      Use --exclude-text-backbone for visual-only Params/FLOPs.')


if __name__ == '__main__':
    main()
