#!/usr/bin/env python3
# Copyright (c) OpenMMLab. All rights reserved.
"""Measure GPU and CPU memory for MMDet / GroundingDINO configs.

Reports:
    - GPU allocated / reserved after model load
    - GPU peak allocated / reserved during inference
    - CPU RSS after model load and after inference
    - CPU peak RSS (VmHWM)

Example:
    python tools/analysis_tools/get_memory.py \\
        configs/mm_grounding_dino/my_cfg/grounding_dino_swin-t_pretrain_multi_same_pe_pixmoe_p.py \\
        --device cuda --shape 800 1333
"""
import argparse
import gc
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.logging import MMLogger
from mmengine.runner import Runner

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from get_params_flops import (  # noqa: E402
    build_dummy_data_sample, build_model, disable_checkpoint_for_analysis,
    resolve_dummy_channels, strip_text_backbone_for_analysis)

MB = 1024.0 ** 2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Measure GPU/CPU memory for MMDet config.')
    parser.add_argument('config', help='Config file path')
    parser.add_argument(
        '--mode',
        choices=['auto', 'dataloader', 'dummy'],
        default='dummy',
        help=('How to construct inputs. '
              '"dummy" is the default for memory measurement.'))
    parser.add_argument(
        '--num-images',
        type=int,
        default=10,
        help='Number of measured inference iterations.')
    parser.add_argument(
        '--num-warmup',
        type=int,
        default=3,
        help='Warmup iterations before measuring peak memory.')
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
              'Default: 6 for fusion / PixMoE, 3 otherwise.'))
    parser.add_argument(
        '--source',
        type=str,
        default='visible',
        help='Modality source for MoE backbone in dummy mode.')
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for memory measurement.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override settings in config, key=value style.')
    parser.add_argument(
        '--exclude-text-backbone',
        action='store_true',
        help=('Exclude text tokenizer/backbone/text mapping branches. '
              'Uses _forward instead of predict.'))
    parser.add_argument(
        '--analysis-text',
        type=str,
        default='drone . car .',
        help='Reference text used in dummy mode.')
    parser.add_argument(
        '--forward-only',
        action='store_true',
        help='Use model._forward instead of predict (no post-processing).')
    return parser.parse_args()


def format_mb(value, digits=3):
    if value is None:
        return 'N/A'
    return f'{float(value):.{digits}f} MB'


def read_proc_status():
    """Read current / peak process RSS from /proc (Linux)."""
    status_path = Path('/proc/self/status')
    rss_mb = None
    peak_mb = None
    if not status_path.exists():
        return rss_mb, peak_mb
    for line in status_path.read_text().splitlines():
        if line.startswith('VmRSS:'):
            rss_mb = float(line.split()[1]) / 1024.0
        elif line.startswith('VmHWM:'):
            peak_mb = float(line.split()[1]) / 1024.0
    return rss_mb, peak_mb


def cpu_rss_mb():
    rss_mb, _ = read_proc_status()
    if rss_mb is not None:
        return rss_mb
    try:
        import psutil
        return psutil.Process().memory_info().rss / MB
    except Exception:
        return None


def cpu_peak_rss_mb():
    _, peak_mb = read_proc_status()
    if peak_mb is not None:
        return peak_mb
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes.
        if sys.platform == 'darwin':
            return ru / MB
        return ru / 1024.0
    except Exception:
        return None


def gpu_stats(device):
    if device != 'cuda' or not torch.cuda.is_available():
        return dict(
            allocated=None,
            reserved=None,
            peak_allocated=None,
            peak_reserved=None)
    torch.cuda.synchronize()
    return dict(
        allocated=torch.cuda.memory_allocated() / MB,
        reserved=torch.cuda.memory_reserved() / MB,
        peak_allocated=torch.cuda.max_memory_allocated() / MB,
        peak_reserved=torch.cuda.max_memory_reserved() / MB)


def nvidia_smi_used_mb():
    if not torch.cuda.is_available():
        return None, None
    try:
        idx = torch.cuda.current_device()
        out = subprocess.check_output(
            [
                'nvidia-smi', f'--id={idx}',
                '--query-gpu=memory.used,memory.total',
                '--format=csv,nounits,noheader'
            ],
            text=True,
            stderr=subprocess.DEVNULL)
        used, total = [float(x.strip()) for x in out.strip().split(',')]
        return used, total
    except Exception:
        return None, None


def reset_gpu_peak(device):
    if device != 'cuda' or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def run_one_forward(model, inputs, data_samples, use_forward):
    if use_forward:
        return model._forward(inputs, data_samples)
    return model.predict(inputs, data_samples, rescale=False)


def measure_dummy(model, device, shape_hw, channels, source, text, warmup,
                  n_iters, use_forward):
    h, w = shape_hw
    dev = torch.device('cuda' if device == 'cuda' else 'cpu')
    dummy = torch.randn(1, channels, h, w, device=dev)
    data_sample = build_dummy_data_sample(h, w, text, source)

    with torch.no_grad():
        for _ in range(max(warmup, 0)):
            run_one_forward(model, dummy, [data_sample], use_forward)
            if device == 'cuda':
                torch.cuda.synchronize()

        reset_gpu_peak(device)
        for _ in range(max(n_iters, 1)):
            run_one_forward(model, dummy, [data_sample], use_forward)
            if device == 'cuda':
                torch.cuda.synchronize()

    return dict(
        compute_type='dummy_input',
        pad_shape=(h, w),
        input_channels=channels,
        source=source)


def measure_dataloader(cfg, model, device, warmup, n_iters, use_forward):
    data_loader = Runner.build_dataloader(cfg.val_dataloader)
    batches = []
    total_need = max(warmup, 0) + max(n_iters, 1)
    for idx, data_batch in enumerate(data_loader):
        if idx >= total_need:
            break
        batches.append(data_batch)
    if not batches:
        raise RuntimeError('No samples were loaded from val_dataloader.')

    pad_shape = None
    input_channels = None

    def _prep(data_batch):
        nonlocal pad_shape, input_channels
        data = model.data_preprocessor(data_batch)
        samples = data['data_samples']
        pad_shape = getattr(samples[0], 'batch_input_shape', samples[0].pad_shape)
        input_channels = int(data['inputs'].shape[1])
        return data['inputs'], samples

    with torch.no_grad():
        for i in range(max(warmup, 0)):
            inputs, samples = _prep(batches[i % len(batches)])
            run_one_forward(model, inputs, samples, use_forward)
            if device == 'cuda':
                torch.cuda.synchronize()

        reset_gpu_peak(device)
        start = max(warmup, 0)
        for i in range(max(n_iters, 1)):
            inputs, samples = _prep(batches[(start + i) % len(batches)])
            run_one_forward(model, inputs, samples, use_forward)
            if device == 'cuda':
                torch.cuda.synchronize()

    source = getattr(samples[0], 'source', None)
    return dict(
        compute_type='dataloader',
        pad_shape=pad_shape,
        input_channels=input_channels,
        source=source)


def main():
    args = parse_args()
    logger = MMLogger.get_instance(name='memory_logger')

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

    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning('CUDA is not available, fallback to CPU.')
        args.device = 'cpu'

    model, actual_device = build_model(cfg, args.device)
    if args.exclude_text_backbone:
        strip_text_backbone_for_analysis(model, args.analysis_text)

    use_forward = args.forward_only or args.exclude_text_backbone
    model.eval()

    if actual_device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    mem_after_load = gpu_stats(actual_device)
    cpu_after_load = cpu_rss_mb()
    smi_after_load, smi_total = nvidia_smi_used_mb()

    result = None
    if args.mode in ('auto', 'dataloader'):
        try:
            result = measure_dataloader(
                cfg, model, actual_device, args.num_warmup, args.num_images,
                use_forward)
        except Exception as e:
            if args.mode == 'dataloader':
                raise
            logger.warning(
                f'Dataloader mode failed, fallback to dummy mode. Cause: {e}')

    if result is None:
        channels = resolve_dummy_channels(args, model)
        result = measure_dummy(
            model,
            actual_device,
            args.shape,
            channels,
            args.source,
            args.analysis_text,
            args.num_warmup,
            args.num_images,
            use_forward)

    mem_after_infer = gpu_stats(actual_device)
    cpu_after_infer = cpu_rss_mb()
    cpu_peak = cpu_peak_rss_mb()
    smi_after_infer, _ = nvidia_smi_used_mb()

    split_line = '=' * 36
    print(split_line)
    print(f'Config: {args.config}')
    print(f'Device: {actual_device}')
    print(f'Compute type: {result["compute_type"]}')
    print(f'Forward type: {"_forward" if use_forward else "predict"}')
    print(f'Exclude text backbone: {args.exclude_text_backbone}')
    print(f'Analysis text: {args.analysis_text}')
    if result.get('input_channels') is not None:
        print(f'Input channels: {result["input_channels"]}')
    if result.get('source') is not None:
        print(f'Source modality: {result["source"]}')
    print(f'Input shape: {result["pad_shape"]}')
    print(f'Warmup / Iters: {args.num_warmup} / {args.num_images}')
    print(split_line)
    print('GPU memory (PyTorch)')
    print(f'  After load allocated: {format_mb(mem_after_load["allocated"])}')
    print(f'  After load reserved:  {format_mb(mem_after_load["reserved"])}')
    print(f'  Peak allocated:       {format_mb(mem_after_infer["peak_allocated"])}')
    print(f'  Peak reserved:        {format_mb(mem_after_infer["peak_reserved"])}')
    if smi_after_load is not None:
        print('GPU memory (nvidia-smi)')
        print(f'  After load used:      {format_mb(smi_after_load)}'
              f' / {format_mb(smi_total)}')
        print(f'  After infer used:     {format_mb(smi_after_infer)}'
              f' / {format_mb(smi_total)}')
    print('CPU memory')
    print(f'  RSS after load:       {format_mb(cpu_after_load)}')
    print(f'  RSS after infer:      {format_mb(cpu_after_infer)}')
    print(f'  Peak RSS (VmHWM):     {format_mb(cpu_peak)}')
    print(split_line)
    print('Note: Peak allocated is torch.cuda.max_memory_allocated() and is')
    print('      usually smaller than nvidia-smi. Dummy mode measures the')
    print('      network itself; dataloader mode includes preprocessing.')


if __name__ == '__main__':
    main()
