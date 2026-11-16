# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
from .swin_multi import SwinTransformer_Multi
from .swin_multi_same_pe import SwinTransformer_Multi_SamePE
from .pix_moe import TaskMoE_MultiModal
from .pix_moe_p import Taskprompt, TaskMoE_MultiModal_prompt
from .swin_multi_same_pe_average import SwinTransformer_Multi_SamePE_avg
from .swin_multi_same_pe_pixmoe_p import SwinTransformer_Multi_SamePE_PixMoE_p
from .swin_multi_same_pe_softmoe import SwinTransformer_Multi_SamePE_SoftMoE
from .soft_moe import TaskSoftMoE_MultiModal


__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'CSPNeXt',
    'SwinTransformer_Multi', 'SwinTransformer_Multi_SamePE',
    'SwinTransformer_Multi_SamePE_avg', 
    'SwinTransformer_Multi_SamePE_PixMoE_p',
    'SwinTransformer_Multi_SamePE_SoftMoE', 'TaskSoftMoE_MultiModal'
]
