# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterator, Optional, Sequence, Sized, Union

import torch
from mmengine.dist import get_dist_info, sync_random_seed
from torch.utils.data import Sampler

from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class HardnessMultiDataSampler(Sampler):
    """Epoch-based multi-source sampler with updatable dataset ratios.

    Copied from :class:`MultiDataSampler` so the original class stays
    unchanged. Per-index weights are:

    ``w(i in m) = max(S) / S_m * r_m / sum(r)``

    therefore ``P(source m) ∝ r_m``. Call :meth:`update_ratio` at epoch
    boundaries; :meth:`__iter__` reads the current weights.
    """

    def __init__(self,
                 dataset: Sized,
                 dataset_ratio: Sequence[Union[int, float]],
                 seed: Optional[int] = None,
                 round_up: bool = True) -> None:
        assert hasattr(dataset, 'datasets'), (
            'HardnessMultiDataSampler requires ConcatDataset, '
            f'but got {type(dataset)}')
        rank, world_size = get_dist_info()
        self.rank = rank
        self.world_size = world_size

        self.dataset = dataset
        if seed is None:
            seed = sync_random_seed()
        self.seed = seed
        self.epoch = 0
        self.round_up = round_up

        if self.round_up:
            self.num_samples = math.ceil(len(self.dataset) / world_size)
            self.total_size = self.num_samples * self.world_size
        else:
            self.num_samples = math.ceil(
                (len(self.dataset) - rank) / world_size)
            self.total_size = len(self.dataset)

        self.sizes = [len(ds) for ds in self.dataset.datasets]
        assert len(dataset_ratio) == len(self.sizes), (
            'dataset_ratio length must equal the number of concatenated '
            f'datasets, got {len(dataset_ratio)} vs {len(self.sizes)}')
        self.update_ratio(dataset_ratio)

    def update_ratio(self, dataset_ratio: Sequence[Union[int, float]]) -> None:
        """Rebuild per-index sampling weights from source ratios."""
        ratio = [float(r) for r in dataset_ratio]
        assert len(ratio) == len(self.sizes)
        assert all(r > 0 for r in ratio), f'ratios must be positive, got {ratio}'
        ratio_sum = sum(ratio)
        max_size = max(self.sizes)
        dataset_weight = [
            torch.ones(s, dtype=torch.double) * max_size / s * r / ratio_sum
            for r, s in zip(ratio, self.sizes)
        ]
        self.dataset_ratio = ratio
        self.weights = torch.cat(dataset_weight)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights, len(self.weights), generator=g,
            replacement=True).tolist()
        if self.round_up:
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
        indices = indices[self.rank:self.total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
