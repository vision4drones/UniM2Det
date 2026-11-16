#!/usr/bin/env python3
"""Diagnose full-image Soft MoE dispatch / combine (journal counterpart).

This script is standalone and does not modify training source files.
Dispatch and combine tensors are captured by wrapping MultiModalSoftMoE
at runtime.

It is NOT a drop-in replacement for analyze_hsms_moe_routing.py: that
script reads discrete Top-k gates. Soft MoE has no Top-k assignment.

Analyses (paired with the HSMS-MoE diagnostics):
  1. Object share of per-slot dispatch mass vs. object-area prior
  2. Intra-instance combine consistency (smooth mixing, not a mosaic)
  3. Qualitative dispatch / combine maps

Example:
    python tools/analysis_tools/analyze_soft_moe_routing.py \
        configs/mm_grounding_dino/my_cfg/grounding_dino_swin-t_pretrain_multi_same_pe_softmoe.py \
        train_out2/soft_moe_no_share/epoch_4.pth \
        --out-dir visual_out/soft_moe_routing \
        --max-per-split 200

Paper figures should use a checkpoint at a comparable epoch to
train_out2/epoch_15_best.pth, not an early snapshot.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import os.path as osp
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        'OpenCV is required. Use the training environment, e.g.\n'
        '  /home/hello/anaconda3/envs/mmgdDino2/bin/python '
        'tools/analysis_tools/analyze_soft_moe_routing.py ...'
    ) from exc
import numpy as np
import torch
from einops import rearrange
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet.models.backbones.soft_moe import l2_normalize
from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

_HSMS = Path(__file__).resolve().parent / 'analyze_hsms_moe_routing.py'
_spec = importlib.util.spec_from_file_location('analyze_hsms_moe_routing', _HSMS)
hsms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hsms)

SPLIT_ORDER = hsms.SPLIT_ORDER
EXPERT_COLORS = hsms.EXPERT_COLORS
EXPERT_BGR = hsms.EXPERT_BGR
OBJ_BG_COLORS = hsms.OBJ_BG_COLORS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize Soft MoE dispatch and combine maps.')
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/mm_grounding_dino/my_cfg/'
        'grounding_dino_swin-t_pretrain_multi_same_pe_softmoe.py')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default='train_out2/soft_moe_no_share/epoch_4.pth')
    parser.add_argument(
        '--out-dir', default='visual_out/soft_moe_routing')
    parser.add_argument('--max-per-split', type=int, default=200)
    parser.add_argument('--num-vis', type=int, default=6)
    parser.add_argument('--box-expand', type=int, default=2)
    parser.add_argument('--min-inst-tokens', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--splits', nargs='+', default=None)
    return parser.parse_args()


def find_softmoe(model):
    root = model.module if hasattr(model, 'module') else model
    backbone = getattr(root, 'backbone', None)
    if backbone is None:
        raise RuntimeError('Cannot find backbone on the detector.')
    block = getattr(backbone, 'taskMoE_block', None)
    if block is None:
        raise RuntimeError(
            'Cannot find taskMoE_block. This script is for Soft MoE.')
    moe = getattr(block, 'model', None)
    if moe is None or not hasattr(moe, 'slot_embeds'):
        raise RuntimeError(
            'Cannot find MultiModalSoftMoE. Use analyze_hsms_moe_routing.py '
            'for discrete PixMoE / HSMS-MoE.')
    return root, backbone, moe


class SoftMoECatcher:
    """Runtime wrapper around MultiModalSoftMoE. No source file is changed."""

    def __init__(self, moe):
        self.moe = moe
        self._orig_forward = moe.forward
        self._orig_one = moe._soft_moe_one_modality
        self.last = None
        self._partial = None

    def install(self):
        catcher = self
        orig_fwd = self._orig_forward
        orig_one = self._orig_one

        def wrapped_one(self_moe, x, modality):
            y = orig_one(x, modality)
            with torch.no_grad():
                b, c, h, w = x.shape
                tokens = rearrange(x, 'b c h w -> b (h w) c').float()
                tokens_n = l2_normalize(tokens, dim=-1)
                phi = self_moe.slot_embeds[modality].float()
                scale = self_moe.slot_scales[modality].float()
                phi_n = scale * l2_normalize(phi, dim=-1)
                logits = torch.einsum('bmd,esd->bmes', tokens_n, phi_n)
                dispatch = logits.softmax(dim=1)
                combine = rearrange(
                    logits, 'b m e s -> b m (e s)').softmax(dim=-1)
                catcher._partial[modality] = dict(
                    dispatch=dispatch.detach().cpu(),
                    combine=combine.detach().cpu(),
                    H=h, W=w, B=b)
            return y

        def wrapped_forward(self_moe, x, source_list):
            b, c, h, w = x.shape
            catcher._partial = {
                '_meta': dict(B=b, H=h, W=w, source_list=list(source_list))
            }
            out = orig_fwd(x, source_list)
            catcher.last = catcher._assemble(self_moe)
            catcher._partial = None
            return out

        self.moe._soft_moe_one_modality = types.MethodType(wrapped_one, self.moe)
        self.moe.forward = types.MethodType(wrapped_forward, self.moe)

    def restore(self):
        self.moe.forward = self._orig_forward
        self.moe._soft_moe_one_modality = self._orig_one

    def _assemble(self, moe):
        meta = self._partial['_meta']
        b, h, w = meta['B'], meta['H'], meta['W']
        n_exp = moe.num_experts
        n_slots = moe.num_slots
        n_tot = n_exp * n_slots
        source_list = meta['source_list']
        dispatch = torch.zeros((b, h, w, n_exp))
        combine = torch.zeros((b, h, w, n_tot))
        for mod_name, pack in self._partial.items():
            if mod_name == '_meta':
                continue
            indices = [i for i, s in enumerate(source_list) if s == mod_name]
            if not indices:
                continue
            disp = pack['dispatch']  # [b_mod, m, e, s]
            comb = pack['combine']   # [b_mod, m, e*s]
            bm = disp.shape[0]
            disp_e = disp.sum(dim=-1).view(bm, h, w, n_exp)
            comb_hw = comb.view(bm, h, w, n_tot)
            dispatch[indices] = disp_e
            combine[indices] = comb_hw
        return dict(
            dispatch=dispatch,
            combine=combine,
            source_list=source_list,
            H=h, W=w)


def neighbor_cosine(weight_hwc, ys, xs, valid, eps=1e-8):
    if len(ys) == 0:
        return float('nan')
    h, w = weight_hwc.shape[:2]
    acc = 0.0
    cnt = 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny = ys + dy
        nx = xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        if not inside.any():
            continue
        ny, nx = ny[inside], nx[inside]
        yy, xx = ys[inside], xs[inside]
        ok = valid[ny, nx]
        if not ok.any():
            continue
        v0 = weight_hwc[yy, xx][ok]
        v1 = weight_hwc[ny, nx][ok]
        num = (v0 * v1).sum(axis=-1)
        den = (np.linalg.norm(v0, axis=-1)
               * np.linalg.norm(v1, axis=-1) + eps)
        acc += float((num / den).sum())
        cnt += int(ok.sum())
    if cnt == 0:
        return float('nan')
    return acc / cnt


def plot_dispatch_object_mass(plt, stats, out_dir, n_exp, n_images=None):
    splits = [s for s in SPLIT_ORDER if s in stats]
    if not splits:
        splits = sorted(stats.keys())
    n = len(splits)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 6.2))
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(n_exp)
    for ax, split in zip(axes, splits):
        rec = stats[split]
        obj = rec['disp_obj']
        valid = rec['disp_valid']
        share = np.divide(obj, np.maximum(valid, 1e-12))
        ax.bar(x, 100.0 * share, color=EXPERT_COLORS[:n_exp])
        prior = rec['area_prior_sum'] / max(rec['n_images'], 1)
        ax.axhline(100.0 * prior, ls='--', c='gray', lw=1.0,
                   label=f'Area prior ({100.0 * prior:.1f}%)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'$E_{i}$' for i in range(n_exp)])
        ax.set_xlabel('Expert slot')
        ax.set_ylabel('Object share of dispatch mass (%)')
        ax.set_title(split, pad=6)
        ax.set_ylim(0, max(15.0, 100.0 * share.max() * 1.25,
                           100.0 * prior * 1.4))
        ax.legend(frameon=False, fontsize=8)
    for ax in axes[n:]:
        ax.axis('off')
    n_note = f' ($N={n_images}$ images)' if n_images else ''
    fig.suptitle(
        'Object share of Soft-MoE dispatch mass versus area prior' + n_note,
        y=1.0, fontsize=12)
    fig.tight_layout()
    hsms.save_fig(fig, out_dir, 'fig1_dispatch_object_mass')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    xs = np.arange(len(splits))
    prior = np.array([
        stats[s]['area_prior_sum'] / max(stats[s]['n_images'], 1)
        for s in splits])
    mean_obj = np.array([
        float(np.divide(stats[s]['disp_obj'],
                        np.maximum(stats[s]['disp_valid'], 1e-12)).mean())
        for s in splits])
    ax.bar(xs - 0.18, 100.0 * prior, 0.36, color='#BBBBBB',
           label='Object-area prior')
    ax.bar(xs + 0.18, 100.0 * mean_obj, 0.36, color='#0072B2',
           label='Mean object dispatch mass')
    ax.set_xticks(xs)
    ax.set_xticklabels(splits, rotation=20, ha='right')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Dataset')
    ax.set_title('Does dispatch concentrate on objects above the area prior?')
    ax.legend(frameon=False)
    fig.tight_layout()
    hsms.save_fig(fig, out_dir, 'fig1b_dispatch_vs_area_prior')
    plt.close(fig)


def plot_combine_consistency(plt, inst_rows, out_dir, min_tokens, n_images=None):
    splits = [s for s in SPLIT_ORDER if any(r['split'] == s for r in inst_rows)]
    if not splits:
        splits = sorted({r['split'] for r in inst_rows})
    rows_ok = [r for r in inst_rows if r['n_tokens'] >= min_tokens]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    x = np.arange(len(splits))
    width = 0.18
    palette = ['#0072B2', '#56B4E9', '#E69F00', '#D55E00']
    for k in range(1, 5):
        frac = []
        for split in splits:
            vals = [r['n_unique_argmax'] for r in rows_ok if r['split'] == split]
            frac.append(100.0 * np.mean([v == k for v in vals]) if vals else 0)
        ax1.bar(x + (k - 2.5) * width, frac, width, color=palette[k - 1],
                label=f'{k}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, rotation=20, ha='right')
    ax1.set_ylabel('Percentage of instances (%)')
    ax1.set_xlabel('Dataset')
    ax1.set_title('(a) Distinct argmax slots per instance')
    ax1.set_ylim(0, 128)
    ax1.legend(title='Number of slots', frameon=False, ncol=4, fontsize=8,
               loc='upper right')

    data = [[r['majority'] for r in rows_ok if r['split'] == split]
            for split in splits]
    try:
        bp = ax2.boxplot(data, tick_labels=splits, showfliers=False,
                         patch_artist=True)
    except TypeError:
        bp = ax2.boxplot(data, labels=splits, showfliers=False,
                         patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#56B4E9')
        patch.set_alpha(0.8)
    ax2.axhline(0.5, ls='--', c='gray', lw=0.8, label='Equal share (0.5)')
    ax2.set_ylabel('Majority-slot share of combine mass')
    ax2.set_xlabel('Dataset')
    ax2.set_title('(b) Instance-level majority combine share')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=20)
    ax2.legend(frameon=False, fontsize=8)

    data = [[r['neighbor_cos'] for r in rows_ok
             if r['split'] == split and np.isfinite(r['neighbor_cos'])]
            for split in splits]
    try:
        bp = ax3.boxplot(data, tick_labels=splits, showfliers=False,
                         patch_artist=True)
    except TypeError:
        bp = ax3.boxplot(data, labels=splits, showfliers=False,
                         patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#009E73')
        patch.set_alpha(0.8)
    ax3.axhline(1.0, ls='--', c='gray', lw=0.8)
    ax3.set_ylabel('Neighbor cosine similarity')
    ax3.set_xlabel('Dataset')
    ax3.set_title('(c) Spatial smoothness of combine weights')
    ax3.set_ylim(0, 1.05)
    ax3.tick_params(axis='x', rotation=20)

    data = [[r['mean_peak'] for r in rows_ok if r['split'] == split]
            for split in splits]
    try:
        bp = ax4.boxplot(data, tick_labels=splits, showfliers=False,
                         patch_artist=True)
    except TypeError:
        bp = ax4.boxplot(data, labels=splits, showfliers=False,
                         patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#E69F00')
        patch.set_alpha(0.8)
    ax4.set_ylabel(r'Mean peak combine weight $\max_e g_e$')
    ax4.set_xlabel('Dataset')
    ax4.set_title('(d) How peaked the per-token combine is')
    ax4.set_ylim(0, 1)
    ax4.tick_params(axis='x', rotation=20)

    n_note = f' ($N={n_images}$ images)' if n_images else ''
    fig.suptitle(
        'Intra-instance consistency of Soft-MoE combine weights' + n_note,
        y=1.01, fontsize=12)
    fig.tight_layout()
    hsms.save_fig(fig, out_dir, 'fig2_combine_consistency')
    plt.close(fig)


def _blank(h, w):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _panel_header(width, caption, caption_h=48, gap=16):
    cap = _blank(caption_h, width)
    cv2.putText(cap, caption, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (25, 25, 25), 2, cv2.LINE_AA)
    spacer = _blank(gap, width)
    return np.concatenate([cap, spacer], axis=0)


def _fit_size(img, h, w):
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return out


def _maps_grid(maps_hw, img_h, img_w, pad_h, pad_w, ori_h, ori_w,
               target_h, target_w, normalize='p99'):
    n = maps_hw.shape[-1]
    cell_h = target_h // 2
    cell_w = target_w // 2
    cells = []
    for e in range(n):
        m = hsms.align_map(
            maps_hw[..., e], img_h, img_w, pad_h, pad_w, ori_h, ori_w,
            nearest=False)
        if normalize == 'p99':
            p = np.percentile(m, 99.5) if m.size else 1.0
            m = np.clip(m / max(float(p), 1e-8), 0.0, 1.0)
        else:
            m = np.clip(m, 0.0, 1.0)
        u8 = (m * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
        color = cv2.resize(color, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        cv2.putText(color, f'E{e}', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cells.append(color)
    while len(cells) < 4:
        cells.append(_blank(cell_h, cell_w))
    top = np.concatenate(cells[:2], axis=1)
    bot = np.concatenate(cells[2:4], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    return _fit_size(grid, target_h, target_w)


def _draw_boxes(img, boxes_ori):
    out = img.copy()
    for box in boxes_ori:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)
    return out


def draw_examples(examples, out_dir):
    vis_dir = osp.join(out_dir, 'overlays')
    os.makedirs(vis_dir, exist_ok=True)
    for i, ex in enumerate(examples):
        img = cv2.imread(ex['img_path'])
        if img is None:
            continue
        ori_h, ori_w = img.shape[:2]
        boxed = _draw_boxes(img, ex['boxes_ori'])
        obj = hsms.align_map(
            ex['obj'].astype(np.float32), ex['img_h'], ex['img_w'],
            ex['pad_h'], ex['pad_w'], ori_h, ori_w, nearest=True)
        obj_vis = img.copy()
        red = np.zeros_like(obj_vis)
        red[:, :] = (0, 0, 220)
        mask = obj > 0.5
        if mask.any():
            obj_vis[mask] = cv2.addWeighted(
                img, 0.55, red, 0.45, 0)[mask]
        obj_vis = _draw_boxes(obj_vis, ex['boxes_ori'])

        disp_grid = _maps_grid(
            ex['dispatch'], ex['img_h'], ex['img_w'], ex['pad_h'], ex['pad_w'],
            ori_h, ori_w, ori_h, ori_w, normalize='p99')
        comb_grid = _maps_grid(
            ex['combine'], ex['img_h'], ex['img_w'], ex['pad_h'], ex['pad_w'],
            ori_h, ori_w, ori_h, ori_w, normalize='none')

        h1 = _panel_header(ori_w, '(a) Input image')
        h2 = _panel_header(ori_w, '(b) Dispatch mass (each slot, 99.5th-pct norm.)')
        h3 = _panel_header(ori_w, '(c) Object-token mask')
        h4 = _panel_header(ori_w, '(d) Combine weights')
        p1 = np.concatenate([h1, boxed], axis=0)
        p2 = np.concatenate([h2, disp_grid], axis=0)
        p3 = np.concatenate([h3, obj_vis], axis=0)
        p4 = np.concatenate([h4, comb_grid], axis=0)
        row1 = np.concatenate([p1, p2], axis=1)
        row2 = np.concatenate([p3, p4], axis=1)
        canvas = np.concatenate([row1, row2], axis=0)
        split_safe = ex['split'].replace('/', '-').replace(' ', '')
        name = (f'{split_safe}_{i:02d}_'
                f'{osp.splitext(osp.basename(ex["img_path"]))[0]}.jpg')
        cv2.imwrite(osp.join(vis_dir, name), canvas)


def summarize(stats, inst_rows, min_tokens):
    out = {}
    for split, rec in stats.items():
        inst = [r for r in inst_rows
                if r['split'] == split and r['n_tokens'] >= min_tokens]
        obj_share = np.divide(rec['disp_obj'],
                              np.maximum(rec['disp_valid'], 1e-12))
        out[split] = {
            'n_images': int(rec['n_images']),
            'n_instances': int(len(inst)),
            'area_prior': float(rec['area_prior_sum'] / max(rec['n_images'], 1)),
            'dispatch_object_share_per_expert': obj_share.tolist(),
            'mean_dispatch_object_share': float(obj_share.mean()),
            'mean_unique_argmax_per_instance':
                float(np.mean([r['n_unique_argmax'] for r in inst])) if inst else None,
            'mean_majority_combine':
                float(np.mean([r['majority'] for r in inst])) if inst else None,
            'mean_neighbor_cosine':
                float(np.nanmean([r['neighbor_cos'] for r in inst])) if inst else None,
            'mean_peak_combine':
                float(np.mean([r['mean_peak'] for r in inst])) if inst else None,
        }
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.load_from = None
    cfg.resume = False
    if 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    device = torch.device(
        args.device if torch.cuda.is_available() or 'cpu' in args.device
        else 'cpu')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()
    model.to(device)

    detector, backbone, moe = find_softmoe(model)
    stride = hsms.routing_stride(backbone)
    n_exp = int(moe.num_experts)
    n_slots = int(moe.num_slots)
    print(f'[Soft MoE] experts={n_exp}, slots/expert={n_slots}, '
          f'stride={stride}, layer={getattr(backbone, "l", None)}')

    catcher = SoftMoECatcher(moe)
    catcher.install()

    prefixes = list(cfg.dataset_prefixes)
    datasets = list(cfg.all_val_dataset)
    if args.splits:
        keep = set(args.splits)
        pairs = [(p, d) for p, d in zip(prefixes, datasets) if p in keep]
        if not pairs:
            raise ValueError(f'None of --splits {args.splits} match {prefixes}')
        prefixes, datasets = zip(*pairs)

    stats = defaultdict(lambda: dict(
        disp_obj=np.zeros(n_exp, dtype=np.float64),
        disp_valid=np.zeros(n_exp, dtype=np.float64),
        area_prior_sum=0.0,
        n_images=0,
    ))
    inst_rows = []
    examples = []
    max_per = args.max_per_split if args.max_per_split > 0 else 10 ** 9

    with torch.no_grad():
        for prefix, ds_cfg in zip(prefixes, datasets):
            loader = hsms.build_split_dataloader(cfg, ds_cfg, args.num_workers)
            n_done = 0
            print(f'==> split {prefix}, max={max_per}')
            for data in loader:
                if n_done >= max_per:
                    break
                data = detector.data_preprocessor(data, False)
                inputs = data['inputs']
                samples = data['data_samples']
                source_list = []
                for sample in samples:
                    src = getattr(sample, 'source', None)
                    if src not in detector.source_names:
                        src = None
                    source_list.append(src)
                _ = detector.extract_feat(inputs, source_list)
                packed = catcher.last
                if packed is None:
                    continue
                dispatch = packed['dispatch']
                combine = packed['combine']
                bsz = dispatch.shape[0]
                fh, fw = packed['H'], packed['W']
                for bi in range(bsz):
                    sample = samples[bi]
                    src = packed['source_list'][bi]
                    img_path = getattr(sample, 'img_path', '') or ''
                    split = hsms.infer_split(img_path, src)
                    meta = sample.metainfo if hasattr(sample, 'metainfo') else {}
                    img_h, img_w = (meta.get('img_shape') or sample.img_shape)[:2]
                    pad_h, pad_w = (meta.get('pad_shape') or
                                    (inputs.shape[-2], inputs.shape[-1]))[:2]
                    disp = dispatch[bi].numpy()
                    comb = combine[bi].numpy()
                    boxes_img, boxes_ori, _, _ = hsms.gt_boxes_in_img_and_ori(
                        sample, img_h, img_w)
                    valid, obj, bg, inst_tokens = hsms.token_masks(
                        fh, fw, stride, img_h, img_w, boxes_img,
                        expand=args.box_expand)
                    if not valid.any():
                        continue
                    stats[split]['disp_obj'] += disp[obj].sum(axis=0) if obj.any() \
                        else np.zeros(n_exp)
                    stats[split]['disp_valid'] += disp[valid].sum(axis=0)
                    stats[split]['area_prior_sum'] += float(
                        obj[valid].mean()) if valid.any() else 0.0
                    stats[split]['n_images'] += 1

                    argmax = comb.argmax(axis=-1)
                    for ys, xs, area in inst_tokens:
                        if len(ys) == 0:
                            continue
                        labs = argmax[ys, xs]
                        uniq = int(np.unique(labs).size)
                        vec = comb[ys, xs]
                        mean_vec = vec.mean(axis=0)
                        majority = float(mean_vec.max())
                        mean_peak = float(vec.max(axis=-1).mean())
                        neigh = neighbor_cosine(comb, ys, xs, valid)
                        inst_rows.append(dict(
                            split=split,
                            n_tokens=int(len(ys)),
                            n_unique_argmax=uniq,
                            majority=majority,
                            mean_peak=mean_peak,
                            neighbor_cos=neigh,
                            area=float(area),
                        ))

                    if (len([e for e in examples if e['split'] == split])
                            < args.num_vis and obj.any()):
                        examples.append(dict(
                            split=split,
                            img_path=img_path,
                            img_h=int(img_h),
                            img_w=int(img_w),
                            pad_h=int(pad_h),
                            pad_w=int(pad_w),
                            dispatch=disp.astype(np.float32),
                            combine=comb.astype(np.float32),
                            obj=obj,
                            boxes_ori=boxes_ori.numpy(),
                        ))
                n_done += bsz
                if n_done % 20 == 0:
                    print(f'    {prefix}: {n_done} images')
            print(f'    done {prefix}: {n_done} images')

    catcher.restore()

    plt = hsms.set_style()
    n_images = int(sum(rec['n_images'] for rec in stats.values()))
    if stats:
        plot_dispatch_object_mass(plt, stats, args.out_dir, n_exp, n_images)
    if inst_rows:
        plot_combine_consistency(
            plt, inst_rows, args.out_dir, args.min_inst_tokens, n_images)
    draw_examples(examples, args.out_dir)

    summary = summarize(stats, inst_rows, args.min_inst_tokens)
    summary_path = osp.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'checkpoint': osp.abspath(args.checkpoint),
            'config': osp.abspath(args.config),
            'num_experts': n_exp,
            'num_slots': n_slots,
            'stride': int(stride),
            'max_per_split': args.max_per_split,
            'n_images': n_images,
            'note': (
                'Dispatch object share should be compared with the area prior: '
                'if slots attend to objects, mean_dispatch_object_share > area_prior. '
                'Combine neighbor cosine near 1 indicates spatially smooth mixing, '
                'the counterpart of HSMS-MoE instance fragmentation. '
                'Token-count load CV is undefined: each expert always receives '
                'exactly num_slots vectors. Early checkpoints are not paper-ready.'
            ),
            'splits': summary,
        }, f, indent=2)

    print('\nSaved figures to', osp.abspath(args.out_dir))
    print('Summary:', summary_path)
    for split, rec in summary.items():
        print(f"  {split}: images={rec['n_images']}, "
              f"area_prior={rec['area_prior']:.3f}, "
              f"disp_obj={rec['mean_dispatch_object_share']:.3f}, "
              f"neigh_cos={rec['mean_neighbor_cosine']}")


if __name__ == '__main__':
    main()
