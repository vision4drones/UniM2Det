#!/usr/bin/env python3
"""Diagnose discrete Top-k routing of the conference HSMS-MoE (2/4).

This script is standalone: it does not modify training/inference source files.
Routing tensors are captured by temporarily wrapping PixMoE methods at runtime.

Four analyses (journal expansion, without contradicting the conference paper):
  1. Object vs background expert occupancy
  2. Intra-instance routing fragmentation
  3. Image-level / pseudo-batch load fluctuation (not only the mean)
  4. Top-1 vs second-best gate gap

Example:
    python tools/analysis_tools/analyze_hsms_moe_routing.py \
        configs/mm_grounding_dino/my_cfg/grounding_dino_swin-t_pretrain_multi_same_pe_pixmoe_p.py \
        train_out2/epoch_15_best.pth \
        --out-dir visual_out/hsms_moe_routing \
        --max-per-split 200
"""
from __future__ import annotations

import argparse
import copy
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
        'tools/analysis_tools/analyze_hsms_moe_routing.py ...'
    ) from exc
import numpy as np
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

# Okabe-Ito, colorblind-friendly. Matches Expert 0-3 in conference Fig. 3.
EXPERT_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
EXPERT_BGR = [(178, 114, 0), (0, 159, 230), (115, 158, 0), (0, 94, 213)]
OBJ_BG_COLORS = ['#4C78A8', '#F58518']  # object, background
SPLIT_ORDER = [
    'DV(RGB)', 'DV(IR)', 'DV(RGB-IR)', 'VisDrone', 'DP(IR)', 'DP(RGB-IR)'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize HSMS-MoE (2/4) discrete routing diagnostics.')
    parser.add_argument(
        'config',
        nargs='?',
        default='configs/mm_grounding_dino/my_cfg/'
        'grounding_dino_swin-t_pretrain_multi_same_pe_pixmoe_p.py',
        help='Config used to train the 2/4 model.')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default='train_out2/epoch_15_best.pth',
        help='Trained 2/4 weights.')
    parser.add_argument(
        '--out-dir',
        default='visual_out/hsms_moe_routing',
        help='Directory for figures, overlays, and JSON summaries.')
    parser.add_argument(
        '--max-per-split',
        type=int,
        default=200,
        help='Max validation images per dataset split. 0 means all.')
    parser.add_argument(
        '--num-vis',
        type=int,
        default=6,
        help='Qualitative overlays saved per split.')
    parser.add_argument(
        '--box-expand',
        type=int,
        default=2,
        help='Expand GT boxes by this many pixels on the resized image.')
    parser.add_argument(
        '--min-inst-tokens',
        type=int,
        default=4,
        help='Ignore instances with fewer tokens in fragmentation stats.')
    parser.add_argument(
        '--gap-thr',
        type=float,
        default=0.05,
        help='Low-confidence threshold on top1-top2 gate gap.')
    parser.add_argument(
        '--pseudo-batch-size',
        type=int,
        default=4,
        help='Group consecutive same-split images to mimic training batch=4.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config settings, key=value.')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=None,
        help='Optional subset of dataset_prefixes to run, e.g. visdrone.')
    return parser.parse_args()


def set_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    return plt


def infer_split(img_path, source):
    p = (img_path or '').replace('\\', '/').lower()
    src = source or 'unknown'
    if 'visdrone' in p:
        return 'VisDrone'
    if 'rgbtdroneperson' in p or 'droneperson' in p:
        return 'DP(RGB-IR)' if src == 'fusion' else 'DP(IR)'
    if 'dronevehicle' in p:
        if src == 'fusion':
            return 'DV(RGB-IR)'
        if src == 'thermal':
            return 'DV(IR)'
        return 'DV(RGB)'
    return f'other/{src}'


def find_pixmoe(model):
    root = model.module if hasattr(model, 'module') else model
    backbone = getattr(root, 'backbone', None)
    if backbone is None:
        raise RuntimeError('Cannot find backbone on the detector.')
    block = getattr(backbone, 'taskMoE_block', None)
    if block is None:
        raise RuntimeError(
            'Cannot find taskMoE_block. This script is for PixMoE-p HSMS-MoE.')
    moe = getattr(block, 'model', None)
    if moe is None:
        raise RuntimeError('Cannot find MultiModalPixMoE under taskMoE_block.')
    return root, backbone, moe


def routing_stride(backbone):
    """Pixel stride of the MoE feature map relative to the padded image."""
    patch = getattr(backbone.patch_embed, 'kernel_size', 4)
    if isinstance(patch, (tuple, list)):
        patch = patch[0]
    layer = getattr(backbone, 'l', -1)
    if layer is None or layer < 0:
        return int(patch)
    return int(patch) * (2 ** (layer + 1))


def boxes_to_xyxy(gt_instances, device='cpu'):
    if gt_instances is None or not hasattr(gt_instances, 'bboxes'):
        return torch.zeros((0, 4), dtype=torch.float32)
    bboxes = gt_instances.bboxes
    if hasattr(bboxes, 'convert_to'):
        try:
            bboxes = bboxes.convert_to('hbox')
        except Exception:
            pass
    if hasattr(bboxes, 'tensor'):
        tensor = bboxes.tensor
    else:
        tensor = torch.as_tensor(bboxes)
    if tensor.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    return tensor.detach().float().cpu()


def _scale_from_meta(meta, img_h, img_w):
    ori = meta.get('ori_shape', (img_h, img_w))
    ori_h, ori_w = int(ori[0]), int(ori[1])
    sf = meta.get('scale_factor', None)
    if sf is not None:
        sf = np.array(sf, dtype=np.float64).reshape(-1)
        sx = float(sf[0])
        sy = float(sf[1] if sf.size > 1 else sf[0])
    else:
        sx = img_w / max(ori_w, 1)
        sy = img_h / max(ori_h, 1)
    return sx, sy, ori_h, ori_w


def gt_boxes_in_img_and_ori(sample, img_h, img_w):
    """Return boxes on the resized canvas and on the original image.

    Val pipelines load COCO boxes after FixScaleResize, so packed gt boxes
    remain in original pixels. Token grids live on the resized/padded image.
    """
    meta = sample.metainfo if hasattr(sample, 'metainfo') else {}
    sx, sy, ori_h, ori_w = _scale_from_meta(meta, img_h, img_w)
    boxes = boxes_to_xyxy(getattr(sample, 'gt_instances', None))
    empty = torch.zeros((0, 4), dtype=torch.float32)
    if boxes.numel() == 0:
        return empty, empty, (ori_h, ori_w), (sx, sy)
    max_x = float(boxes[:, 2].max())
    max_y = float(boxes[:, 3].max())
    fits_img = max_x <= img_w + 2 and max_y <= img_h + 2
    fits_ori = max_x <= ori_w + 2 and max_y <= ori_h + 2
    rescaled = abs(sx - 1.0) > 1e-3 or abs(sy - 1.0) > 1e-3
    if fits_img and not fits_ori:
        boxes_img = boxes
        boxes_ori = boxes.clone()
        boxes_ori[:, [0, 2]] /= max(sx, 1e-6)
        boxes_ori[:, [1, 3]] /= max(sy, 1e-6)
    else:
        # Original frame, including the common "both fit because image was
        # upscaled" case.
        boxes_ori = boxes
        boxes_img = boxes.clone()
        boxes_img[:, [0, 2]] *= sx
        boxes_img[:, [1, 3]] *= sy
        _ = (fits_ori, rescaled)
    return boxes_img, boxes_ori, (ori_h, ori_w), (sx, sy)


def entropy_from_usage(usage, eps=1e-12):
    p = np.asarray(usage, dtype=np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    return float(-(p * np.log(p + eps)).sum())


def cv_from_usage(usage, eps=1e-12):
    p = np.asarray(usage, dtype=np.float64)
    mean = p.mean()
    if mean <= eps:
        return 0.0
    return float(p.std() / mean)


def align_map(feat_map, img_h, img_w, pad_h, pad_w, ori_h, ori_w, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    padded = cv2.resize(
        feat_map.astype(np.float32), (pad_w, pad_h), interpolation=interp)
    valid = padded[:img_h, :img_w]
    return cv2.resize(valid, (ori_w, ori_h), interpolation=interp)


def colorize_label_map(label_hw, overlay_bgr, alpha=0.45):
    h, w = overlay_bgr.shape[:2]
    color = np.zeros_like(overlay_bgr)
    for e, bgr in enumerate(EXPERT_BGR):
        color[label_hw == e] = bgr
    out = overlay_bgr.copy()
    mask = label_hw >= 0
    if mask.any():
        blended = cv2.addWeighted(overlay_bgr, 1.0 - alpha, color, alpha, 0)
        out[mask] = blended[mask]
    return out


class RoutingCatcher:
    """Runtime wrapper around MultiModalPixMoE. No source file is changed."""

    def __init__(self, moe):
        self.moe = moe
        self._orig_forward = moe.forward
        self._orig_noisyk = moe.noisyk_w_modality
        self.last = None
        self._partial = None

    def install(self):
        catcher = self
        orig_fwd = self._orig_forward
        orig_noisyk = self._orig_noisyk

        def wrapped_noisyk(self_moe, x, modality, train, noise_epsilon=1e-2):
            w, s = orig_noisyk(x, modality, train, noise_epsilon)
            with torch.no_grad():
                logits = x @ self_moe.w_g[modality]
            if catcher._partial is not None:
                catcher._partial[modality] = {
                    'w': w.detach(),
                    'logits': logits.detach(),
                }
            return w, s

        def wrapped_forward(self_moe, x, prompt, source_list):
            b, c, h, w = x.shape
            catcher._partial = {
                '_meta': dict(B=b, H=h, W=w, source_list=list(source_list))
            }
            out = orig_fwd(x, prompt, source_list)
            catcher.last = catcher._assemble(self_moe)
            catcher._partial = None
            return out

        self.moe.noisyk_w_modality = types.MethodType(wrapped_noisyk, self.moe)
        self.moe.forward = types.MethodType(wrapped_forward, self.moe)

    def restore(self):
        self.moe.forward = self._orig_forward
        self.moe.noisyk_w_modality = self._orig_noisyk

    def _assemble(self, moe):
        meta = self._partial['_meta']
        b, h, w = meta['B'], meta['H'], meta['W']
        n_exp = moe.num_experts_per_modality
        source_list = meta['source_list']
        device = moe.w_g[next(iter(moe.w_g))].device
        source_ids = torch.tensor(
            [moe.mod_map[s] for s in source_list], device=device)
        token_mod_ids = source_ids.view(b, 1, 1).expand(b, h, w).reshape(-1)
        gates = torch.zeros((b * h * w, n_exp), device=device)
        logits = torch.zeros((b * h * w, n_exp), device=device)
        for mod_name, pack in self._partial.items():
            if mod_name == '_meta':
                continue
            mod_id = moe.mod_map[mod_name]
            idx = (token_mod_ids == mod_id).nonzero(as_tuple=True)[0]
            gates[idx] = pack['w'].to(device)
            logits[idx] = pack['logits'].to(device)
        return {
            'gates': gates.view(b, h, w, n_exp).detach().cpu(),
            'logits': logits.view(b, h, w, n_exp).detach().cpu(),
            'source_list': source_list,
            'H': h,
            'W': w,
        }


def token_masks(h, w, stride, img_h, img_w, boxes_xyxy, expand=0):
    """Return valid/object masks and per-instance token index lists."""
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    # A token is valid if its top-left pixel is inside the unpadded image.
    valid = (ys * stride < img_h) & (xs * stride < img_w)
    obj = np.zeros((h, w), dtype=bool)
    inst_tokens = []
    if boxes_xyxy is not None and len(boxes_xyxy) > 0:
        boxes = boxes_xyxy.clone()
        if expand:
            boxes[:, 0] -= expand
            boxes[:, 1] -= expand
            boxes[:, 2] += expand
            boxes[:, 3] += expand
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            tx1 = max(0, int(math.floor(x1 / stride)))
            ty1 = max(0, int(math.floor(y1 / stride)))
            tx2 = min(w, int(math.ceil(x2 / stride)))
            ty2 = min(h, int(math.ceil(y2 / stride)))
            if tx2 <= tx1 or ty2 <= ty1:
                continue
            sl = np.zeros((h, w), dtype=bool)
            sl[ty1:ty2, tx1:tx2] = True
            sl &= valid
            obj |= sl
            yy, xx = np.where(sl)
            inst_tokens.append((yy, xx, (x2 - x1) * (y2 - y1)))
    bg = valid & (~obj)
    return valid, obj, bg, inst_tokens


def neighbor_agreement(top1, ys, xs, valid):
    if len(ys) == 0:
        return float('nan')
    h, w = top1.shape
    acc = 0.0
    cnt = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
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
            acc += (top1[ny, nx][ok] == top1[yy, xx][ok]).sum()
            cnt += int(ok.sum())
    if cnt == 0:
        return float('nan')
    return float(acc / cnt)


def build_split_dataloader(cfg, dataset_cfg, num_workers):
    loader_cfg = copy.deepcopy(cfg.val_dataloader)
    loader_cfg.dataset = dataset_cfg
    loader_cfg.num_workers = num_workers
    loader_cfg.persistent_workers = False
    loader_cfg.batch_size = 1
    return Runner.build_dataloader(loader_cfg)


def save_fig(fig, out_dir, name):
    png = osp.join(out_dir, f'{name}.png')
    pdf = osp.join(out_dir, f'{name}.pdf')
    fig.savefig(png, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    return png, pdf


def plot_object_vs_background(plt, stats, out_dir):
    splits = [s for s in SPLIT_ORDER if s in stats]
    if not splits:
        splits = sorted(stats.keys())
    n = len(splits)
    n_img = int(sum(stats[s]['n_images'] for s in splits))
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 6.2))
    axes = np.atleast_1d(axes).ravel()
    x = np.arange(4)
    width = 0.36
    bars = None
    for ax, split in zip(axes, splits):
        rec = stats[split]
        obj = rec['occ_obj'].astype(np.float64)
        bg = rec['occ_bg'].astype(np.float64)
        tot = obj + bg
        tot[tot == 0] = 1
        b1 = ax.bar(x - width / 2, 100 * obj / tot, width, color=OBJ_BG_COLORS[0])
        b2 = ax.bar(x + width / 2, 100 * bg / tot, width, color=OBJ_BG_COLORS[1])
        bars = (b1, b2)
        ax.set_xticks(x)
        ax.set_xticklabels([f'$E_{i}$' for i in range(4)])
        ax.set_xlabel('Expert')
        ax.set_ylabel('Proportion of assignments (%)')
        ax.set_title(split, pad=6)
        ax.set_ylim(0, 100)
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.legend(
        bars, ['Object tokens', 'Background tokens'],
        loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975),
        borderaxespad=0.0)
    fig.suptitle(
        f'Composition of Top-$k$ assignments per expert  '
        f'($N={n_img}$ images)',
        y=1.0, fontsize=12)
    save_fig(fig, out_dir, 'fig1_object_vs_background')
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 6.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, split in zip(axes, splits):
        rec = stats[split]
        b1 = ax.bar(x - width / 2, rec['occ_obj'], width, color=OBJ_BG_COLORS[0])
        b2 = ax.bar(x + width / 2, rec['occ_bg'], width, color=OBJ_BG_COLORS[1])
        bars = (b1, b2)
        ax.set_xticks(x)
        ax.set_xticklabels([f'$E_{i}$' for i in range(4)])
        ax.set_xlabel('Expert')
        ax.set_ylabel('Number of token–expert assignments')
        ax.set_title(split, pad=6)
        ax.set_yscale('log')
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    fig.legend(
        bars, ['Object tokens', 'Background tokens'],
        loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.975),
        borderaxespad=0.0)
    fig.suptitle(
        f'Absolute Top-$k$ assignment counts per expert  '
        f'($N={n_img}$ images; log scale)',
        y=1.0, fontsize=12)
    save_fig(fig, out_dir, 'fig1b_object_vs_background_counts')
    plt.close(fig)


def plot_fragmentation(plt, inst_rows, out_dir, min_tokens, n_images=None):
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
            vals = [r['n_unique_top1'] for r in rows_ok if r['split'] == split]
            frac.append(100.0 * np.mean([v == k for v in vals]) if vals else 0)
        ax1.bar(x + (k - 2.5) * width, frac, width, color=palette[k - 1],
                label=f'{k}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(splits, rotation=20, ha='right')
    ax1.set_ylabel('Percentage of instances (%)')
    ax1.set_xlabel('Dataset')
    ax1.set_title('(a) Distinct Top-1 experts per instance')
    ax1.set_ylim(0, 128)
    ax1.legend(title='Number of experts', frameon=False, ncol=4, fontsize=8,
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
    ax2.set_ylabel('Majority-expert share')
    ax2.set_xlabel('Dataset')
    ax2.set_title('(b) Majority-expert share within each instance')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=20)
    ax2.legend(frameon=False, fontsize=8)

    areas = np.array([r['area'] for r in rows_ok])
    uniq = np.array([r['n_unique_top1'] for r in rows_ok], dtype=np.float64)
    rng = np.random.RandomState(0)
    uniq_j = uniq + rng.uniform(-0.12, 0.12, size=len(uniq))
    if len(areas):
        ax3.scatter(np.sqrt(np.clip(areas, 0, None)), uniq_j, s=6, alpha=0.15,
                    c='#0072B2', linewidths=0)
    ax3.set_yticks([1, 2, 3, 4])
    ax3.set_xlabel(r'Instance scale $\sqrt{\mathrm{area}}$ (pixels)')
    ax3.set_ylabel('Number of distinct Top-1 experts')
    ax3.set_title('(c) Distinct experts versus instance scale')
    ax3.set_ylim(0.6, 4.5)

    labels = [r'Small ($<32^2$)', r'Medium ($32^2$–$96^2$)', r'Large ($\geq96^2$)']
    edges = [0, 32 ** 2, 96 ** 2, 1e12]
    means, ns = [], []
    for i in range(3):
        sel = [r['n_unique_top1'] for r in rows_ok
               if edges[i] <= r['area'] < edges[i + 1]]
        means.append(float(np.mean(sel)) if sel else 0)
        ns.append(len(sel))
    ax4.bar(np.arange(3), means, color=['#56B4E9', '#0072B2', '#003366'])
    ax4.set_xticks(np.arange(3))
    ax4.set_xticklabels(labels)
    ax4.set_ylabel('Mean number of distinct Top-1 experts')
    ax4.set_xlabel('COCO-style scale bin')
    ax4.set_ylim(0, 4.5)
    ax4.set_title('(d) Mean distinct experts by instance scale')
    for i, (m, ncnt) in enumerate(zip(means, ns)):
        ax4.text(i, m + 0.08, f'{m:.2f}\n$n={ncnt}$', ha='center', va='bottom',
                 fontsize=8)
    n_note = f' ($N={n_images}$ images)' if n_images else ''
    fig.suptitle(
        'Intra-instance routing fragmentation under discrete Top-$k$' + n_note,
        y=1.01, fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, 'fig2_instance_fragmentation')
    plt.close(fig)


def plot_load_fluctuation(plt, img_rows, out_dir, pseudo_batch_size, n_images=None):
    splits = [s for s in SPLIT_ORDER if any(r['split'] == s for r in img_rows)]
    if not splits:
        splits = sorted({r['split'] for r in img_rows})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    usages = np.stack([r['usage_topk'] for r in img_rows], axis=0)
    box_kwargs = dict(patch_artist=True, showfliers=False)
    try:
        bp = ax1.boxplot(
            [usages[:, e] for e in range(4)],
            tick_labels=[f'$E_{e}$' for e in range(4)],
            **box_kwargs)
    except TypeError:
        bp = ax1.boxplot(
            [usages[:, e] for e in range(4)],
            labels=[f'$E_{e}$' for e in range(4)],
            **box_kwargs)
    for patch, c in zip(bp['boxes'], EXPERT_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Per-image assignment share')
    ax1.set_xlabel('Expert')
    ax1.set_title('(a) Per-image expert assignment share')
    ax1.axhline(0.25, ls='--', lw=0.8, c='gray', label='Uniform share ($1/4$)')
    ax1.set_ylim(0, 0.45)
    ax1.legend(frameon=False, fontsize=8)

    for split in splits:
        cvs = [r['cv'] for r in img_rows if r['split'] == split]
        ax2.hist(cvs, bins=np.linspace(0, max(1.0, max(cvs) if cvs else 1), 16),
                 histtype='step', linewidth=1.8, density=True, label=split)
    ax2.set_xlabel('Coefficient of variation of expert usage')
    ax2.set_ylabel('Probability density')
    ax2.set_title('(b) Distribution of the usage coefficient of variation')
    ax2.legend(frameon=False, fontsize=8)

    pick = splits[0]
    seq = [r for r in img_rows if r['split'] == pick][:80]
    if seq:
        heat = np.stack([r['usage_topk'] for r in seq], axis=0).T
        im = ax3.imshow(heat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax3.set_yticks(range(4))
        ax3.set_yticklabels([f'$E_{e}$' for e in range(4)])
        ax3.set_xlabel('Image index')
        ax3.set_ylabel('Expert')
        ax3.set_title(f'(c) Expert assignment share across images ({pick})')
        cbar = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar.set_label('Assignment share')

    labels, img_cv, batch_cv = [], [], []
    for split in splits:
        rows = [r for r in img_rows if r['split'] == split]
        labels.append(split)
        img_cv.append(float(np.mean([r['cv'] for r in rows])) if rows else 0)
        grouped = []
        for i in range(0, len(rows) - pseudo_batch_size + 1, pseudo_batch_size):
            u = np.mean(
                [rows[j]['usage_topk'] for j in range(i, i + pseudo_batch_size)],
                axis=0)
            grouped.append(cv_from_usage(u))
        batch_cv.append(float(np.mean(grouped)) if grouped else 0)
    x = np.arange(len(labels))
    ax4.bar(x - 0.18, img_cv, 0.36, color='#0072B2', label='Per image')
    ax4.bar(x + 0.18, batch_cv, 0.36, color='#E69F00',
            label=f'Mean of {pseudo_batch_size} images')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=20, ha='right')
    ax4.set_ylabel('Mean coefficient of variation')
    ax4.set_xlabel('Dataset')
    ax4.set_title('(d) Per-image versus grouped-image load imbalance')
    ax4.legend(frameon=False, fontsize=8)
    n_note = f' ($N={n_images}$ images)' if n_images else ''
    fig.suptitle(
        'Image-wise fluctuation of discrete Top-$k$ expert load' + n_note,
        y=1.01, fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, 'fig3_load_fluctuation')
    plt.close(fig)


def plot_gap(plt, gap_rows, out_dir, gap_thr, n_images=None):
    splits = [s for s in SPLIT_ORDER if any(r['split'] == s for r in gap_rows)]
    if not splits:
        splits = sorted({r['split'] for r in gap_rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax1, ax2 = axes
    bins = np.linspace(0, 1, 41)
    obj_all, bg_all = [], []
    for split in splits:
        obj_list = [r['gap_obj'] for r in gap_rows
                    if r['split'] == split and len(r['gap_obj'])]
        bg_list = [r['gap_bg'] for r in gap_rows
                   if r['split'] == split and len(r['gap_bg'])]
        if obj_list:
            obj_all.append(np.concatenate(obj_list))
        if bg_list:
            bg_all.append(np.concatenate(bg_list))
    if obj_all:
        obj_cat = np.concatenate(obj_all)
        ax1.hist(obj_cat, bins=bins, density=True, color=OBJ_BG_COLORS[0],
                 alpha=0.55, label='Object')
        ax2.plot(
            np.sort(obj_cat),
            np.linspace(0, 1, len(obj_cat)),
            color=OBJ_BG_COLORS[0], label='Object')
    if bg_all:
        bg_cat = np.concatenate(bg_all)
        ax1.hist(bg_cat, bins=bins, density=True, color=OBJ_BG_COLORS[1],
                 alpha=0.45, label='Background')
        ax2.plot(
            np.sort(bg_cat),
            np.linspace(0, 1, len(bg_cat)),
            color=OBJ_BG_COLORS[1], label='Background')
    ax1.axvline(gap_thr, ls='--', c='gray', lw=1,
                label=fr'Near-tie ($g<{gap_thr:.2f}$)')
    ax1.set_xlabel(r'Routing gap $g=g_{(1)}-g_{(2)}$')
    ax1.set_ylabel('Probability density')
    ax1.set_title('(a) Distribution of the routing gap')
    ax1.set_xlim(0, 1)
    ax1.legend(frameon=False, fontsize=8, loc='upper left')
    ax2.axvline(gap_thr, ls='--', c='gray', lw=1)
    ax2.set_xlabel(r'Routing gap $g=g_{(1)}-g_{(2)}$')
    ax2.set_ylabel(r'Cumulative probability $P(G\leq g)$')
    ax2.set_title('(b) Empirical CDF of the routing gap')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False, fontsize=8, loc='lower right')
    n_note = f' ($N={n_images}$ images)' if n_images else ''
    fig.suptitle(
        r'Top-$k$ routing gap $g_{(1)}-g_{(2)}$' + n_note,
        y=1.02, fontsize=12)
    fig.tight_layout()
    save_fig(fig, out_dir, 'fig4_topk_gap')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(splits))
    obj_rate, bg_rate = [], []
    for split in splits:
        obj_list = [r['gap_obj'] for r in gap_rows
                    if r['split'] == split and len(r['gap_obj'])]
        bg_list = [r['gap_bg'] for r in gap_rows
                   if r['split'] == split and len(r['gap_bg'])]
        obj = np.concatenate(obj_list) if obj_list else np.array([])
        bg = np.concatenate(bg_list) if bg_list else np.array([])
        obj_rate.append(float((obj < gap_thr).mean()) if len(obj) else 0)
        bg_rate.append(float((bg < gap_thr).mean()) if len(bg) else 0)
    ax.bar(x - 0.18, 100 * np.array(obj_rate), 0.36, color=OBJ_BG_COLORS[0],
           label='Object')
    ax.bar(x + 0.18, 100 * np.array(bg_rate), 0.36, color=OBJ_BG_COLORS[1],
           label='Background')
    ax.set_xticks(x)
    ax.set_xticklabels(splits, rotation=20, ha='right')
    ax.set_ylabel(rf'Percentage of tokens with $g<{gap_thr:.2f}$ (%)')
    ax.set_xlabel('Dataset')
    ax.set_title(
        rf'Fraction of near-tie Top-$k$ assignments ($g<{gap_thr:.2f}$)'
        + (f' ($N={n_images}$ images)' if n_images else ''))
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, out_dir, 'fig4b_low_gap_rate')
    plt.close(fig)


def _blank(h, w):
    return np.ones((h, w, 3), dtype=np.uint8) * 255


def _paste_center(canvas, patch):
    ch, cw = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    if pw > cw or ph > ch:
        scale = min(cw / max(pw, 1), ch / max(ph, 1))
        patch = cv2.resize(
            patch, (max(1, int(pw * scale)), max(1, int(ph * scale))),
            interpolation=cv2.INTER_AREA)
        ph, pw = patch.shape[:2]
    y = max(0, (ch - ph) // 2)
    x = max(0, (cw - pw) // 2)
    canvas[y:y + ph, x:x + pw] = patch
    return canvas


def _panel_header(width, caption, legend_img=None, caption_h=54, legend_h=58, gap=22):
    """Caption, then the panel legend, then padding above the image."""
    cap = _blank(caption_h, width)
    cv2.putText(cap, caption, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (25, 25, 25), 2, cv2.LINE_AA)
    mid = _blank(legend_h, width)
    if legend_img is not None:
        _paste_center(mid, legend_img)
    spacer = _blank(gap, width)
    return np.concatenate([cap, mid, spacer], axis=0)


def _expert_legend_img():
    img = _blank(48, 460)
    cv2.putText(img, 'Experts', (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (40, 40, 40), 1, cv2.LINE_AA)
    for e, bgr in enumerate(EXPERT_BGR):
        x0 = 108 + e * 86
        cv2.rectangle(img, (x0, 12), (x0 + 22, 34), bgr, -1)
        cv2.rectangle(img, (x0, 12), (x0 + 22, 34), (30, 30, 30), 1)
        cv2.putText(img, f'E{e}', (x0 + 28, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def _gap_legend_img():
    img = _blank(48, 430)
    cv2.putText(img, '0', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, 'near-tie', (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (80, 80, 80), 1, cv2.LINE_AA)
    for i in range(240):
        c = cv2.applyColorMap(np.uint8([[i * 255 / 239]]), cv2.COLORMAP_VIRIDIS)[0, 0]
        img[10:30, 78 + i] = c
    cv2.putText(img, '1', (324, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(img, 'confident', (324, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (80, 80, 80), 1, cv2.LINE_AA)
    return img


def draw_examples(examples, out_dir):
    vis_dir = osp.join(out_dir, 'overlays')
    os.makedirs(vis_dir, exist_ok=True)
    expert_leg = _expert_legend_img()
    gap_leg = _gap_legend_img()
    for i, ex in enumerate(examples):
        img = cv2.imread(ex['img_path'])
        if img is None:
            continue
        ori_h, ori_w = img.shape[:2]
        top1 = align_map(
            ex['top1'], ex['img_h'], ex['img_w'], ex['pad_h'], ex['pad_w'],
            ori_h, ori_w, nearest=True)
        gap = align_map(
            ex['gap'], ex['img_h'], ex['img_w'], ex['pad_h'], ex['pad_w'],
            ori_h, ori_w, nearest=False)
        obj = align_map(
            ex['obj'].astype(np.float32), ex['img_h'], ex['img_w'],
            ex['pad_h'], ex['pad_w'], ori_h, ori_w, nearest=True)
        top1_i = np.round(top1).astype(np.int32)
        overlay = colorize_label_map(top1_i, img, alpha=0.5)
        boxed = overlay.copy()
        boxes_ori = ex.get('boxes_ori', ex['boxes'])
        for box in boxes_ori:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 255, 255), 2)
        gap_u8 = np.clip(gap * 255.0, 0, 255).astype(np.uint8)
        gap_color = cv2.applyColorMap(gap_u8, cv2.COLORMAP_VIRIDIS)
        obj_vis = img.copy()
        red = np.zeros_like(obj_vis)
        red[:, :] = (0, 0, 220)
        obj_mask = obj > 0.5
        obj_vis[obj_mask] = cv2.addWeighted(
            img, 0.55, red, 0.45, 0)[obj_mask]
        for box in boxes_ori:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(obj_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

        w = img.shape[1]
        h1 = _panel_header(w, '(a) Input image', None)
        h2 = _panel_header(w, '(b) Top-1 expert assignment', expert_leg)
        h3 = _panel_header(w, '(c) Routing gap g = g(1) - g(2)', gap_leg)
        h4 = _panel_header(w, '(d) Object-token mask', None)
        panels = [
            np.concatenate([h1, img], axis=0),
            np.concatenate([h2, boxed], axis=0),
            np.concatenate([h3, gap_color], axis=0),
            np.concatenate([h4, obj_vis], axis=0),
        ]
        canvas = np.concatenate(panels, axis=1)
        split_safe = ex['split'].replace('/', '-').replace(' ', '')
        name = f'{split_safe}_{i:02d}_{osp.splitext(osp.basename(ex["img_path"]))[0]}.jpg'
        cv2.imwrite(osp.join(vis_dir, name), canvas)


def summarize(stats, inst_rows, img_rows, gap_rows, gap_thr, min_tokens):
    out = {}
    for split, rec in stats.items():
        obj, bg = rec['occ_obj'], rec['occ_bg']
        tot = obj + bg
        inst = [r for r in inst_rows
                if r['split'] == split and r['n_tokens'] >= min_tokens]
        imgs = [r for r in img_rows if r['split'] == split]
        obj_list = [r['gap_obj'] for r in gap_rows
                    if r['split'] == split and len(r['gap_obj'])]
        bg_list = [r['gap_bg'] for r in gap_rows
                   if r['split'] == split and len(r['gap_bg'])]
        gaps_obj = np.concatenate(obj_list) if obj_list else np.array([])
        gaps_bg = np.concatenate(bg_list) if bg_list else np.array([])
        out[split] = {
            'n_images': int(rec['n_images']),
            'n_instances': len(inst),
            'object_token_share': float(obj.sum() / max(tot.sum(), 1)),
            'expert_object_share': (obj / np.maximum(tot, 1)).tolist(),
            'expert_load_topk': (tot / max(tot.sum(), 1)).tolist(),
            'mean_unique_top1_per_instance':
                float(np.mean([r['n_unique_top1'] for r in inst])) if inst else None,
            'mean_majority_agreement':
                float(np.mean([r['majority'] for r in inst])) if inst else None,
            'mean_neighbor_agreement':
                float(np.nanmean([r['neighbor'] for r in inst])) if inst else None,
            'mean_image_usage_cv':
                float(np.mean([r['cv'] for r in imgs])) if imgs else None,
            'mean_image_usage_entropy':
                float(np.mean([r['entropy'] for r in imgs])) if imgs else None,
            'uniform_entropy': float(math.log(4)),
            'mean_gap_object': float(np.mean(gaps_obj)) if len(gaps_obj) else None,
            'mean_gap_background': float(np.mean(gaps_bg)) if len(gaps_bg) else None,
            'low_gap_rate_object':
                float((gaps_obj < gap_thr).mean()) if len(gaps_obj) else None,
            'low_gap_rate_background':
                float((gaps_bg < gap_thr).mean()) if len(gaps_bg) else None,
        }
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    overlay_dir = osp.join(args.out_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # Do not re-load the original pretrain; we only need the 2/4 checkpoint.
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

    detector, backbone, moe = find_pixmoe(model)
    stride = routing_stride(backbone)
    n_exp = moe.num_experts_per_modality
    k = moe.k
    print(f'[HSMS-MoE] experts={n_exp}, top-k={k}, stride={stride}, '
          f'layer={getattr(backbone, "l", None)}')

    catcher = RoutingCatcher(moe)
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
        occ_obj=np.zeros(n_exp, dtype=np.float64),
        occ_bg=np.zeros(n_exp, dtype=np.float64),
        occ_obj_top1=np.zeros(n_exp, dtype=np.float64),
        occ_bg_top1=np.zeros(n_exp, dtype=np.float64),
        n_images=0,
    ))
    inst_rows = []
    img_rows = []
    gap_rows = []
    examples = []

    max_per = args.max_per_split if args.max_per_split > 0 else 10 ** 9

    with torch.no_grad():
        for prefix, ds_cfg in zip(prefixes, datasets):
            loader = build_split_dataloader(cfg, ds_cfg, args.num_workers)
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
                gates = packed['gates']  # [B,H,W,E]
                bsz, fh, fw, _ = gates.shape
                for bi in range(bsz):
                    sample = samples[bi]
                    src = packed['source_list'][bi]
                    img_path = getattr(sample, 'img_path', '') or ''
                    split = infer_split(img_path, src)
                    meta = sample.metainfo if hasattr(sample, 'metainfo') else {}
                    img_h, img_w = (meta.get('img_shape') or sample.img_shape)[:2]
                    pad_h, pad_w = (meta.get('pad_shape') or
                                    (inputs.shape[-2], inputs.shape[-1]))[:2]
                    g = gates[bi].numpy()
                    boxes_img, boxes_ori, _, _ = gt_boxes_in_img_and_ori(
                        sample, img_h, img_w)
                    valid, obj, bg, inst_tokens = token_masks(
                        fh, fw, stride, img_h, img_w,
                        boxes_img,
                        expand=args.box_expand)
                    if not valid.any():
                        continue
                    # Top-k occupancy: gates are already zero except k experts.
                    occ = g[valid]
                    obj_occ = g[obj] if obj.any() else np.zeros((0, n_exp))
                    bg_occ = g[bg] if bg.any() else np.zeros((0, n_exp))
                    stats[split]['occ_obj'] += obj_occ.sum(axis=0)
                    stats[split]['occ_bg'] += bg_occ.sum(axis=0)
                    top1 = g.argmax(axis=-1)
                    if obj.any():
                        hist, _ = np.histogram(
                            top1[obj], bins=np.arange(n_exp + 1))
                        stats[split]['occ_obj_top1'] += hist
                    if bg.any():
                        hist, _ = np.histogram(
                            top1[bg], bins=np.arange(n_exp + 1))
                        stats[split]['occ_bg_top1'] += hist
                    stats[split]['n_images'] += 1

                    usage = g[valid].sum(axis=0)
                    usage = usage / max(usage.sum(), 1e-12)
                    img_rows.append(dict(
                        split=split,
                        prefix=prefix,
                        usage_topk=usage.astype(np.float64),
                        cv=cv_from_usage(usage),
                        entropy=entropy_from_usage(usage),
                    ))

                    # Gap among the k selected experts. With k=2, g1+g2=1.
                    part = np.sort(g, axis=-1)
                    gap = part[..., -1] - part[..., -2]
                    def _sample_gaps(mask, n=4000):
                        vals = gap[mask]
                        if vals.size == 0:
                            return np.zeros((0,), dtype=np.float32)
                        if vals.size > n:
                            idx = np.random.choice(vals.size, n, replace=False)
                            vals = vals[idx]
                        return vals.astype(np.float32)
                    gap_rows.append(dict(
                        split=split,
                        gap_obj=_sample_gaps(obj),
                        gap_bg=_sample_gaps(bg),
                    ))

                    for ys, xs, area in inst_tokens:
                        if len(ys) == 0:
                            continue
                        labs = top1[ys, xs]
                        uniq = int(np.unique(labs).size)
                        counts = np.bincount(labs, minlength=n_exp)
                        majority = float(counts.max() / max(len(labs), 1))
                        neigh = neighbor_agreement(top1, ys, xs, valid)
                        inst_rows.append(dict(
                            split=split,
                            n_tokens=int(len(ys)),
                            n_unique_top1=uniq,
                            majority=majority,
                            neighbor=neigh,
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
                            top1=top1.astype(np.int16),
                            gap=gap.astype(np.float32),
                            obj=obj,
                            boxes=boxes_img.numpy(),
                            boxes_ori=boxes_ori.numpy(),
                        ))
                n_done += bsz
                if n_done % 20 == 0:
                    print(f'    {prefix}: {n_done} images')
            print(f'    done {prefix}: {n_done} images')

    catcher.restore()

    plt = set_style()
    n_images = int(sum(rec['n_images'] for rec in stats.values()))
    plot_object_vs_background(plt, stats, args.out_dir)
    if inst_rows:
        plot_fragmentation(
            plt, inst_rows, args.out_dir, args.min_inst_tokens,
            n_images=n_images)
    if img_rows:
        plot_load_fluctuation(
            plt, img_rows, args.out_dir, args.pseudo_batch_size,
            n_images=n_images)
    if gap_rows:
        plot_gap(plt, gap_rows, args.out_dir, args.gap_thr, n_images=n_images)
    draw_examples(examples, args.out_dir)

    summary = summarize(
        stats, inst_rows, img_rows, gap_rows, args.gap_thr, args.min_inst_tokens)
    summary_path = osp.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'checkpoint': osp.abspath(args.checkpoint),
            'config': osp.abspath(args.config),
            'top_k': int(k),
            'num_experts': int(n_exp),
            'stride': int(stride),
            'max_per_split': args.max_per_split,
            'gap_thr': args.gap_thr,
            'note': (
                'These diagnostics characterize intra-modality discrete Top-k. '
                'They do not claim HSMS-MoE is ineffective: conference results '
                'already show it helps. The intended reading is that expert '
                'specialization coexists with background-dominated load, '
                'instance-internal switching, image-level usage fluctuation, '
                'and low-margin hard assignments.'
            ),
            'splits': summary,
        }, f, indent=2)

    print('\nSaved figures to', osp.abspath(args.out_dir))
    print('Summary:', summary_path)
    for split, rec in summary.items():
        print(f"  {split}: images={rec['n_images']}, "
              f"obj_share={rec['object_token_share']:.3f}, "
              f"uniq_top1={rec['mean_unique_top1_per_instance']}, "
              f"cv={rec['mean_image_usage_cv']}, "
              f"low_gap_obj={rec['low_gap_rate_object']}")


if __name__ == '__main__':
    main()
