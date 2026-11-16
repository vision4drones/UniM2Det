
# Third-Party Notices

UniM2Det and UniM2Det-S are research projects for unified multi-dataset and multi-modal UAV object detection.

This repository contains original implementations developed by the authors, as well as portions of third-party open-source code that have been incorporated and modified. The following notices identify the third-party projects and their applicable licenses.

## 1. MMDetection

- **Project:** MMDetection
- **Copyright:** Copyright 2018-2023 OpenMMLab. All rights reserved.
- **Repository:** https://github.com/open-mmlab/mmdetection
- **Version:** 3.3.0
- **License:** Apache License 2.0
- **License URL:** https://www.apache.org/licenses/LICENSE-2.0

This project is based on MMDetection. Portions of its source code have been modified and extended to implement the proposed unified multi-dataset and multi-modal UAV object detection framework.

The modifications include, but are not limited to, custom backbone components, detector implementations, multi-modal feature processing, and dataset-related components.

The original copyright and license notices from MMDetection are retained in the corresponding source files where applicable. Modified files include notices identifying the changes made for this project.

The original MMDetection project is distributed under the Apache License 2.0.

## 2. Soft-MoE

- **Project:** Soft-MoE
- **Repository:** https://github.com/bwconrad/soft-moe
- **Related Paper:** From Sparse to Soft Mixtures of Experts
- **Paper URL:** https://arxiv.org/abs/2308.00951
- **License:** Apache License 2.0
- **License URL:** https://www.apache.org/licenses/LICENSE-2.0

UniM2Det-S adapts and modifies portions of the original Soft-MoE implementation, including `SoftMoELayerWrapper` and related slot-based expert dispatch and combination mechanisms.

These components have been adapted to the proposed hybrid shared and modality-specific soft mixture-of-experts architecture.

The original copyright and license notices are retained where applicable. Modified files include notices identifying the adaptations and modifications made for UniM2Det-S.

The original Soft-MoE implementation is distributed under the Apache License 2.0.

## 3. PyTorch

- **Project:** PyTorch
- **Repository:** https://github.com/pytorch/pytorch
- **License:** BSD-style License
- **License URL:** https://github.com/pytorch/pytorch/blob/main/LICENSE

PyTorch is used as the underlying deep learning framework and runtime dependency of this project. No PyTorch source code is intentionally incorporated into this repository.

PyTorch remains subject to its original license and applicable third-party notices.

## 4. License of Original Code

Unless otherwise indicated, the original code developed specifically for UniM2Det and UniM2Det-S is released under the Apache License 2.0.

Third-party source code incorporated into this repository remains subject to its respective original license terms. The original copyright, attribution, and license notices are preserved where applicable.

For the complete license terms, please refer to the `LICENSE` file and the respective upstream repositories.
