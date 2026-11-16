# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class AddDatasetName(BaseTransform):
    """Write a dataset identity string into ``results`` for packing.

    Used by hardness sampling so a batch can be attributed to one of the
    concatenated training sources without changing ``PackDetInputs_Fusion``.
    Put this transform immediately before pack, and include ``dataset_name``
    in the pack ``meta_keys``.
    """

    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name

    def transform(self, results: dict) -> dict:
        results['dataset_name'] = self.dataset_name
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(dataset_name={self.dataset_name})'
