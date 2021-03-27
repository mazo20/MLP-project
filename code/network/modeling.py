from .network_utils import IntermediateLayerGetter
from .deeplab       import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .              import resnet
from .              import resnet_depth_conv
from .IATPTPSAATPS  import Encoder

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, depth_mode,
                 first_aware, all_bottlenenck):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate                  = [12, 24, 36]
    elif output_stride == 16:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate                  = [6, 12, 18]
    else:
        replace_stride_with_dilation = [False, False, False]
        aspp_dilate                  = [3, 6, 9]

    if depth_mode == 'dconv':
        backbone = resnet_depth_conv.__dict__[backbone_name](
            pretrained                   = pretrained_backbone,
            replace_stride_with_dilation = replace_stride_with_dilation,
            depth_mode                   = depth_mode,
            first_aware                  = first_aware,
            all_bottlenenck              = all_bottlenenck)
    else:
        assert backbone_name in ['resnet50', 'resnet101'], 'backbone_name must be resnet50 or resnet101'

        if backbone_name == 'resnet50':
            blocks = [3, 4, 6, 3]
        elif backbone_name == 'resnet101':
            blocks = [3, 4, 23, 3]

        backbone = Encoder(blocks=blocks, replace_stride_with_dilation=replace_stride_with_dilation)
    
    inplanes         = 2048
    low_level_planes = 256

    return_layers = {'layer4': 'out', 'layer1': 'low_level'}

    # classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate, depth_mode)

    classifier = DeepLabHead(inplanes , num_classes, aspp_dilate, depth_mode)
    
    # backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    model = DeepLabV3(backbone, classifier)
    
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, depth_mode,
                first_aware, all_bottlenenck):
    model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, 
                         pretrained_backbone=pretrained_backbone, depth_mode=depth_mode, 
                         first_aware=first_aware, all_bottlenenck=all_bottlenenck)
    return model

# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=7, output_stride=8, pretrained_backbone=True, 
                           depth_mode='none', first_aware=False, all_bottlenenck=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, depth_mode=depth_mode,
                       first_aware=first_aware, all_bottlenenck=all_bottlenenck)


def deeplabv3plus_resnet101(num_classes=7, output_stride=8, pretrained_backbone=True, 
                            depth_mode='none', first_aware=False, all_bottlenenck=False):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, depth_mode=depth_mode)

def deeplabv3_resnet50(num_classes=7, output_stride=8, pretrained_backbone=True, depth_mode='none',
                       first_aware=False, all_bottlenenck=False):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride, 
                        pretrained_backbone=pretrained_backbone, depth_mode=depth_mode)

def deeplabv3_resnet101(num_classes=7, output_stride=8, pretrained_backbone=True, depth_mode='none', 
                        first_aware=False, all_bottlenenck=False):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride, 
                       pretrained_backbone=pretrained_backbone, depth_mode=depth_mode, 
                       first_aware=first_aware, all_bottlenenck=all_bottlenenck)