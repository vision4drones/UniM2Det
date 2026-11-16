# Copyright (c) OpenMMLab. All rights reserved.
"""Count model parameters with and without the text backbone (BERT).

Example:
    python tools/analysis_tools/get_params_wo_text.py \\
        configs/mm_grounding_dino/my_cfg/grounding_dino_swin-t_pretrain_multi_same_pe_pixmoe_p.py
"""
import argparse
from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope

from mmdet.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Count detector params with/without text backbone')
    parser.add_argument('config', help='model config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override settings in the config file')
    parser.add_argument(
        '--show-breakdown',
        action='store_true',
        help='print per-module parameter breakdown')
    return parser.parse_args()


def count_params(module, trainable_only: bool = False) -> int:
    """Count parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_params(num: int) -> str:
    """Format parameter count in human-readable units."""
    if num >= 1e9:
        return f'{num / 1e9:.2f} G ({num:,})'
    if num >= 1e6:
        return f'{num / 1e6:.2f} M ({num:,})'
    if num >= 1e3:
        return f'{num / 1e3:.2f} K ({num:,})'
    return f'{num} ({num:,})'


def get_visual_submodules(model):
    """Return major visual submodules for breakdown reporting."""
    candidates = [
        ('backbone', getattr(model, 'backbone', None)),
        ('neck', getattr(model, 'neck', None)),
        ('encoder', getattr(model, 'encoder', None)),
        ('decoder', getattr(model, 'decoder', None)),
        ('bbox_head', getattr(model, 'bbox_head', None)),
        ('text_feat_map', getattr(model, 'text_feat_map', None)),
        ('positional_encoding', getattr(model, 'positional_encoding', None)),
        ('query_embedding', getattr(model, 'query_embedding', None)),
    ]
    return [(name, module) for name, module in candidates if module is not None]


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f'{config_path} not found.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)

    total_params = count_params(model)
    total_trainable = count_params(model, trainable_only=True)

    language_model = getattr(model, 'language_model', None)
    if language_model is None:
        text_params = 0
        text_trainable = 0
    else:
        text_params = count_params(language_model)
        text_trainable = count_params(language_model, trainable_only=True)

    visual_params = total_params - text_params
    visual_trainable = total_trainable - text_trainable

    split_line = '=' * 40
    print(split_line)
    print(f'Config: {config_path}')
    print(f'Model type: {cfg.model.type}')
    print(split_line)
    print('Total parameters')
    print(f'  all:        {format_params(total_params)}')
    print(f'  trainable:  {format_params(total_trainable)}')
    print(split_line)
    print('Text backbone (language_model / BERT)')
    print(f'  all:        {format_params(text_params)}')
    print(f'  trainable:  {format_params(text_trainable)}')
    print(split_line)
    print('Visual parameters (total - text backbone)')
    print(f'  all:        {format_params(visual_params)}')
    print(f'  trainable:  {format_params(visual_trainable)}')
    print(split_line)

    if args.show_breakdown:
        print('Visual submodule breakdown')
        accounted = 0
        for name, module in get_visual_submodules(model):
            num = count_params(module)
            accounted += num
            print(f'  {name:<22} {format_params(num)}')
        other_params = visual_params - accounted
        print(f'  {"other (misc)":<22} {format_params(other_params)}')
        print(split_line)

    print('Note: text_feat_map is counted in visual params; '
          'only language_model is excluded.')


if __name__ == '__main__':
    main()
