# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.logging import MMLogger
from mmengine.runner import Runner

from mmdet.registry import HOOKS


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def _as_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().mean().item())
    return float(value)


@HOOKS.register_module()
class HardnessSamplerHook(Hook):
    """Collect per-source box loss and update HardnessMultiDataSampler ratios.

    Grouping is over datasets (default 3), then split evenly across modalities
    of the same dataset. Ratios are updated only at epoch boundaries.

    Args:
        source_names: Names of ConcatDataset children, length M.
        group_ids: Group id for each of the M sources.
        group_sizes: Unique-scene counts per group, used in the size term.
        init_ratio: Initial six-way (or M-way) sampling ratios.
        loss_keys: Keys read from ``outputs`` and summed as hardness.
        warmup_epochs: 1-based epochs that keep ``init_ratio``.
        ema_momentum: EMA on grouped losses. Larger is smoother.
        clip_ratio: Clip relative to ``init_ratio``, then renormalize.
        min_count: Skip a source this epoch if global batch count is below this.
        enable_update: If False, only log losses (static sqrt-N mixture).
        freeze_after_epoch: 1-based epoch after which ratios stay fixed.
            ``6`` means the last update is after epoch 5.
    """

    priority = 'NORMAL'

    def __init__(self,
                 source_names: Sequence[str],
                 group_ids: Sequence[int],
                 group_sizes: Sequence[int],
                 init_ratio: Sequence[float],
                 loss_keys: Sequence[str] = ('loss_bbox', 'loss_iou'),
                 warmup_epochs: int = 2,
                 ema_momentum: float = 0.7,
                 clip_ratio: float = 2.0,
                 min_count: int = 20,
                 enable_update: bool = False,
                 freeze_after_epoch: Optional[int] = 6) -> None:
        assert len(source_names) == len(group_ids) == len(init_ratio)
        self.source_names = list(source_names)
        self.group_ids = [int(g) for g in group_ids]
        self.num_sources = len(self.source_names)
        self.num_groups = max(self.group_ids) + 1
        assert len(group_sizes) == self.num_groups
        self.group_sizes = [int(s) for s in group_sizes]
        self.init_ratio = [float(r) for r in init_ratio]
        self.loss_keys = tuple(loss_keys)
        self.warmup_epochs = int(warmup_epochs)
        self.ema_momentum = float(ema_momentum)
        self.clip_ratio = float(clip_ratio)
        self.min_count = int(min_count)
        self.enable_update = bool(enable_update)
        self.freeze_after_epoch = (
            None if freeze_after_epoch is None else int(freeze_after_epoch))

        self.name_to_id = {n: i for i, n in enumerate(self.source_names)}
        self.current_ratio = list(self.init_ratio)
        self.frozen = False
        self.ema_L: Optional[List[float]] = None
        self._unknown_batches = 0
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.sum_L = [0.0] * self.num_sources
        self.cnt = [0] * self.num_sources
        self._unknown_batches = 0

    def _get_sampler(self, runner: Runner):
        loader = runner.train_loop.dataloader
        batch_sampler = getattr(loader, 'batch_sampler', None)
        if batch_sampler is not None and hasattr(batch_sampler, 'sampler'):
            return batch_sampler.sampler
        return getattr(loader, 'sampler', None)

    def _apply_ratio(self, runner: Runner, ratio: Sequence[float]) -> None:
        sampler = self._get_sampler(runner)
        if sampler is None or not hasattr(sampler, 'update_ratio'):
            MMLogger.get_current_instance().warning(
                'HardnessSamplerHook: train sampler has no update_ratio(); '
                'skip applying new dataset_ratio.')
            return
        sampler.update_ratio(ratio)

    def _infer_source_id(self, sample) -> Optional[int]:
        name = getattr(sample, 'dataset_name', None)
        if not name:
            meta = getattr(sample, 'metainfo', None) or {}
            name = meta.get('dataset_name')
        if name in self.name_to_id:
            return self.name_to_id[name]

        source = getattr(sample, 'source', None)
        if not source:
            meta = getattr(sample, 'metainfo', None) or {}
            source = meta.get('source')
        path = ''
        meta = getattr(sample, 'metainfo', None) or {}
        path = str(meta.get('img_path', '') or getattr(sample, 'img_path', ''))
        path_l = path.lower()

        if source == 'visible' and 'dronevehicle' in path_l:
            return self.name_to_id.get('DroneVehicle_v')
        if source == 'thermal' and 'dronevehicle' in path_l:
            return self.name_to_id.get('DroneVehicle_t')
        if source == 'fusion' and 'dronevehicle' in path_l:
            return self.name_to_id.get('DroneVehicle_f')
        if source == 'visible' and 'visdrone' in path_l:
            return self.name_to_id.get('visdrone')
        if source == 'thermal' and ('rgbt' in path_l or 'droneperson' in path_l):
            return self.name_to_id.get('DronePerson_t')
        if source == 'fusion' and ('rgbt' in path_l or 'droneperson' in path_l):
            return self.name_to_id.get('DronePerson_f')
        return None

    def _batch_source_id(self, data_batch) -> Optional[int]:
        if not isinstance(data_batch, dict):
            return None
        samples = data_batch.get('data_samples', None)
        if not samples:
            return None
        sample = samples[0]
        return self._infer_source_id(sample)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if not outputs:
            return
        sid = self._batch_source_id(data_batch)
        if sid is None:
            self._unknown_batches += 1
            return
        loss = 0.0
        found = False
        for key in self.loss_keys:
            if key in outputs:
                loss += _as_float(outputs[key])
                found = True
        if not found:
            return
        self.sum_L[sid] += loss
        self.cnt[sid] += 1

    def _reduce_stats(self) -> Tuple[List[float], List[int]]:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        packed = torch.tensor(
            self.sum_L + [float(c) for c in self.cnt],
            dtype=torch.float64,
            device=device)
        packed = _all_reduce_sum(packed)
        vals = packed.detach().cpu().tolist()
        sum_L = vals[:self.num_sources]
        cnt = [int(round(x)) for x in vals[self.num_sources:]]
        return sum_L, cnt

    def _group_losses(self, src_L: List[Optional[float]]) -> List[Optional[float]]:
        grouped: List[List[float]] = [[] for _ in range(self.num_groups)]
        for sid, gid in enumerate(self.group_ids):
            if src_L[sid] is not None:
                grouped[gid].append(src_L[sid])
        out: List[Optional[float]] = []
        for g in grouped:
            out.append(sum(g) / len(g) if g else None)
        return out

    def _compute_ratio(self, group_L: List[float]) -> Tuple[List[float], List[float], bool]:
        min_L = min(group_L)
        max_S = max(self.group_sizes)
        w = []
        for L_g, S_g in zip(group_L, self.group_sizes):
            w.append((L_g / min_L) * ((max_S / S_g) ** 0.5))
        raw = [0.0] * self.num_sources
        for sid, gid in enumerate(self.group_ids):
            members = sum(1 for g in self.group_ids if g == gid)
            raw[sid] = w[gid] / members
        raw_sum = sum(raw)
        raw = [x / raw_sum for x in raw]

        clipped = []
        hit_clip = False
        tau = self.clip_ratio
        for r, r0 in zip(raw, self.init_ratio):
            lo, hi = r0 / tau, r0 * tau
            if r < lo or r > hi:
                hit_clip = True
            clipped.append(min(max(r, lo), hi))
        clip_sum = sum(clipped)
        clipped = [x / clip_sum for x in clipped]
        return clipped, w, hit_clip

    def after_train_epoch(self, runner: Runner) -> None:
        logger = MMLogger.get_current_instance()
        finished = runner.epoch + 1
        sum_L, cnt = self._reduce_stats()

        src_L: List[Optional[float]] = [None] * self.num_sources
        for i in range(self.num_sources):
            if cnt[i] >= self.min_count:
                src_L[i] = sum_L[i] / cnt[i]

        group_L_now = self._group_losses(src_L)
        if self.ema_L is None:
            self.ema_L = [
                g if g is not None else 1.0 for g in group_L_now
            ]
        else:
            beta = self.ema_momentum
            for g in range(self.num_groups):
                if group_L_now[g] is None:
                    continue
                self.ema_L[g] = beta * self.ema_L[g] + (1.0 - beta) * group_L_now[g]

        allow_update = (
            self.enable_update and not self.frozen
            and finished >= self.warmup_epochs
            and (self.freeze_after_epoch is None
                 or finished < self.freeze_after_epoch))

        hit_clip = False
        group_w = [float('nan')] * self.num_groups
        formula_ratio: Optional[List[float]] = None
        if all(x is not None and x > 0 for x in self.ema_L):
            formula_ratio, group_w, hit_clip = self._compute_ratio(self.ema_L)
            if allow_update:
                self.current_ratio = formula_ratio
                if (self.freeze_after_epoch is not None
                        and finished + 1 >= self.freeze_after_epoch):
                    self.frozen = True

        if (self.freeze_after_epoch is not None
                and finished >= self.freeze_after_epoch):
            self.frozen = True

        status = []
        if not self.enable_update:
            status.append('enable_update=False')
        if finished < self.warmup_epochs:
            status.append('warmup')
        if self.frozen:
            status.append('frozen')
        if allow_update:
            status.append('updated')
        if hit_clip:
            status.append('clipped')
        if self._unknown_batches:
            status.append(f'skip={self._unknown_batches}')
        status_str = ','.join(status) if status else 'log_only'

        cnt_msg = ', '.join(
            f'{n}:{c}' for n, c in zip(self.source_names, cnt))
        src_msg = ', '.join(
            f'{n}:{"na" if v is None else f"{v:.4f}"}'
            for n, v in zip(self.source_names, src_L))
        ema_msg = ', '.join(f'g{g}:{v:.4f}' for g, v in enumerate(self.ema_L))
        w_msg = ', '.join(
            f'g{g}:{v:.4f}' if v == v else f'g{g}:na'
            for g, v in enumerate(group_w))
        r_msg = ', '.join(
            f'{n}:{r:.4f}' for n, r in zip(self.source_names, self.current_ratio))
        formula_msg = 'na' if formula_ratio is None else ', '.join(
            f'{n}:{r:.4f}' for n, r in zip(self.source_names, formula_ratio))
        logger.info(
            'HardnessSampler epoch %d [%s] cnt=[%s] rawL=[%s] emaL=[%s] '
            'w=[%s] formula=[%s] ratio=[%s]',
            finished, status_str, cnt_msg, src_msg, ema_msg, w_msg,
            formula_msg, r_msg)
        if self._unknown_batches:
            logger.warning(
                'HardnessSampler skipped %d batches without dataset_name',
                self._unknown_batches)

        self._reset_buffers()

    def before_train(self, runner: Runner) -> None:
        self._apply_ratio(runner, self.current_ratio)

    def before_train_epoch(self, runner: Runner) -> None:
        self._apply_ratio(runner, self.current_ratio)

    def before_save_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        checkpoint['hardness_sampler'] = dict(
            current_ratio=self.current_ratio,
            ema_L=self.ema_L,
            frozen=self.frozen,
            enable_update=self.enable_update)

    def after_load_checkpoint(self, runner: Runner, checkpoint: dict) -> None:
        state = checkpoint.get('hardness_sampler')
        if not state:
            return
        self.current_ratio = list(state.get('current_ratio', self.current_ratio))
        ema_L = state.get('ema_L', None)
        self.ema_L = list(ema_L) if ema_L is not None else None
        self.frozen = bool(state.get('frozen', False))
        self._apply_ratio(runner, self.current_ratio)
