import torch
import torch.nn as nn
import re

from torch.nn              import Module, ModuleList, Conv2d, ReLU, BatchNorm2d, AdaptiveAvgPool2d, Sigmoid, MaxPool2d
from collections           import OrderedDict
from torch.nn.init         import kaiming_normal_, constant_
from torch.utils.model_zoo import load_url

model_urls = {
    'resnet50':  'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
}

class SqueezeAndExcitation(Module):
    
    def __init__(self, in_channels):
        super(SqueezeAndExcitation, self).__init__()
        self.avgglobalpool = AdaptiveAvgPool2d(output_size=1)
        self.conv1         = Conv2d(in_channels=in_channels, out_channels=in_channels // 16, kernel_size=1)
        self.relu          = ReLU(inplace=True)
        self.conv2         = Conv2d(in_channels=in_channels // 16, out_channels=in_channels, kernel_size=1)
        self.sigmoid       = Sigmoid()
        
    def forward(self, x):
        out = self.avgglobalpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        
        return x * out

class RGBDFusion(Module):
    
    def __init__(self, in_channels):
        super(RGBDFusion, self).__init__()
        self.rgbsqueeze   = SqueezeAndExcitation(in_channels=in_channels)
        self.depthsqueeze = SqueezeAndExcitation(in_channels=in_channels)
        
    def forward(self, rgb, depth):
        rgb_out   = self.rgbsqueeze(rgb)
        depth_out = self.depthsqueeze(depth)
        
        return rgb_out + depth_out

class Downsample(Module):
    
    def __init__(self, in_channels, out_channels, stride):
        super(Downsample, self).__init__()
        
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels * 4, kernel_size=1, stride=stride, bias=False)
        self.norm = BatchNorm2d(num_features=out_channels * 4)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        
        return out

class Bottleneck(Module):
    
    def  __init__(self, in_channels, out_channels, stride=1, dilation=1, downsample=False):
        super(Bottleneck, self).__init__()
        
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
        self.norm1 = BatchNorm2d(num_features=out_channels)
        self.relu1 = ReLU(inplace=True)
        
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,
                            bias=False, dilation=dilation)
        self.norm2 = BatchNorm2d(num_features=out_channels)
        self.relu2 = ReLU(inplace=True)
        
        self.conv3 = Conv2d(in_channels=out_channels, out_channels=out_channels * 4, kernel_size=1, stride=1, bias=False)
        self.norm3 = BatchNorm2d(num_features=out_channels * 4)
        
        if downsample:
            self.downsample = Downsample(in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            self.downsample = None
        
        self.relu3 = ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        
        out = out + identity
        out = self.relu3(out)
        
        return out

class ResNetLayer(Module):
    
    def  __init__(self, in_channels, out_channels, blocks, dilation, stride=1, dilate=1, replace_stride_with_dilation=False):
        super(ResNetLayer, self).__init__()
        downsample        = False
        previous_dilation = dilation
        
        if replace_stride_with_dilation:
            dilation *= stride
            stride    = 1
            
        if dilate > 1:
            dilation           = dilate
            dilate_first_block = dilate
        else:
            dilate_first_block = previous_dilation
            
        if stride != 1 or in_channels != out_channels * 4:
            downsample = True

        layers  = []
        layers += [Bottleneck(in_channels=in_channels, out_channels=out_channels, stride=stride, 
                              dilation=dilate_first_block, downsample=downsample)]
        
        in_channels = out_channels * 4

        for i in range(1, blocks):
            layers += [Bottleneck(in_channels=in_channels, out_channels=out_channels, dilation=dilation)]
            
        self.dilation = dilation
        self.layers   = ModuleList(layers)
        
        
    def forward(self, x):
        out = x
        
        for layer in self.layers:
            out = layer(out)
            
        return out

class Encoder(Module):
    def __init__(self, blocks, 
                 fusion_type='all', zero_init_residual=False, replace_stride_with_dilation=None, 
                 dilation=None):
        super(Encoder, self).__init__()

        self.fusion_type = fusion_type
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * 3
            
        if dilation is None:
            dilation = [1] * 4
        
        # STAGE 0
        self.rgbconv = Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.rgbnorm = BatchNorm2d(num_features=64)
        self.rgbrelu = ReLU(inplace=True)
        
        self.depthconv = Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.depthnorm = BatchNorm2d(num_features=64)
        self.depthrelu = ReLU(inplace=True)
        
        if fusion_type != 'late':
            self.fusion1 = RGBDFusion(in_channels=64)
        
        # STAGE 1
        self.rgbmaxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.rgbresnet1 = ResNetLayer(in_channels=64, out_channels=64, blocks=blocks[0], dilation=1, dilate=dilation[0])
        rgbdilation     = self.rgbresnet1.dilation
        
        if fusion_type != 'early':
            self.depthmaxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.depthresnet1 = ResNetLayer(in_channels=64, out_channels=64, blocks=blocks[0], dilation=1, dilate=dilation[0])
            depthdilation     = self.depthresnet1.dilation

        if fusion_type == 'all':
            self.fusion2 = RGBDFusion(in_channels=256)
        
        # STAGE 2
        self.rgbresnet2 = ResNetLayer(in_channels=256, out_channels=128, blocks=blocks[1], dilation=rgbdilation, stride=2, 
                                      dilate=dilation[1], replace_stride_with_dilation=replace_stride_with_dilation[0])
        rgbdilation     = self.rgbresnet2.dilation
        

        if fusion_type != 'early':
            self.depthresnet2 = ResNetLayer(in_channels=256, out_channels=128, blocks=blocks[1], 
                                            dilation=depthdilation, stride=2, dilate=dilation[1], 
                                            replace_stride_with_dilation=replace_stride_with_dilation[0])
            depthdilation = self.depthresnet2.dilation
        
        if fusion_type == 'all':
            self.fusion3 = RGBDFusion(in_channels=512)
        
        # STAGE 3
        self.rgbresnet3 = ResNetLayer(in_channels=512, out_channels=256, blocks=blocks[2], 
                                      dilation=rgbdilation, stride=2, dilate=dilation[2],
                                      replace_stride_with_dilation=replace_stride_with_dilation[1])
        rgbdilation     = self.rgbresnet3.dilation

        if fusion_type != 'early':
            self.depthresnet3 = ResNetLayer(in_channels=512, out_channels=256, blocks=blocks[2], 
                                            dilation=depthdilation, stride=2, dilate=dilation[2],
                                            replace_stride_with_dilation=replace_stride_with_dilation[1])
            depthdilation     = self.depthresnet3.dilation
        
        if fusion_type == 'all':
            self.fusion4 = RGBDFusion(in_channels=1024)
        
        # STAGE 3
        self.rgbresnet4 = ResNetLayer(in_channels=1024, out_channels=512, blocks=blocks[3], dilation=rgbdilation, stride=2, 
                                      dilate=dilation[3], replace_stride_with_dilation=replace_stride_with_dilation[2])
        
        if fusion_type != 'early':
            self.depthresnet4 = ResNetLayer(in_channels=1024, out_channels=512, blocks=blocks[3], 
                                            dilation=depthdilation, stride=2, dilate=dilation[3], 
                                            replace_stride_with_dilation=replace_stride_with_dilation[2])
            self.fusion5      = RGBDFusion(in_channels=2048)
        
        # WEIGHT INIT
        for module in self.modules():
            if isinstance(module, Conv2d):
                kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, BatchNorm2d):
                constant_(module.weight, 1)
                constant_(module.bias,   0)
                
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    constant_(module.norm2.weight, 0)
        
    def forward(self, rgb, depth):
        
        # STAGE 0
        rgb_out = self.rgbconv(rgb)
        rgb_out = self.rgbnorm(rgb_out)
        rgb_out = self.rgbrelu(rgb_out)

        depth_out = self.depthconv(depth)
        depth_out = self.depthnorm(depth_out)
        depth_out = self.depthrelu(depth_out)
        
        if self.fusion_type not in ['late', 'aspp']:
            rgb_out = self.fusion1(rgb_out, depth_out)
        
        # STAGE 1
        rgb_out = self.rgbmaxpool(rgb_out)
        rgb_out = self.rgbresnet1(rgb_out)
        
        if self.fusion_type != 'early':
            depth_out = self.depthmaxpool(depth_out)
            depth_out = self.depthresnet1(depth_out)
        
        if self.fusion_type == 'all':
            rgb_out = self.fusion2(rgb_out, depth_out)
        
        # STAGE 2
        rgb_out = self.rgbresnet2(rgb_out)

        if self.fusion_type != 'early':
            depth_out = self.depthresnet2(depth_out)

        if self.fusion_type == 'all':
            rgb_out = self.fusion3(rgb_out, depth_out)
        
        # STAGE 3
        rgb_out = self.rgbresnet3(rgb_out)

        if self.fusion_type != 'early':
            depth_out = self.depthresnet3(depth_out)

        if self.fusion_type == 'all':
            rgb_out = self.fusion4(rgb_out, depth_out)
        
        # STAGE 4
        rgb_out = self.rgbresnet4(rgb_out)

        if self.fusion_type != 'early':
            depth_out = self.depthresnet4(depth_out)
        
        if self.fusion_type not in ['early', 'aspp']:
            rgb_out = self.fusion5(rgb_out, depth_out)
        
        return rgb_out, depth_out

def name_to_res(layer_name):
    matches = [match.group() for match in re.finditer('(depth|rgb)resnet[1-4]\.layers', layer_name)]
    
    if len(matches) != 0:
        name       = matches[0]
        num        = [match.group() for match in re.finditer('[1-4]', name)][0]
        layer_name = layer_name.replace(name, 'layer' + num)
        
    layer_name = layer_name.replace('norm', 'bn')
    
    if 'downsample' in layer_name:
        layer_name = layer_name.replace('conv', '0')
        layer_name = layer_name.replace('bn',   '1')
        
    layer_name = layer_name.replace('depthconv', 'conv1')
    layer_name = layer_name.replace('depthbn',   'bn1')
    layer_name = layer_name.replace('rgbconv',   'conv1')
    layer_name = layer_name.replace('rgbbn',     'bn1')
    
    return layer_name

def load_encoder(backbone_name='resnet50', pretrained=False, fusion_type='all', 
                 replace_stride_with_dilation=None):
    if backbone_name == 'resnet50':
        blocks = [3, 4, 6, 3]
    elif backbone_name == 'resnet101':
        blocks = [3, 4, 23, 3]

    encoder = Encoder(blocks=blocks, fusion_type=fusion_type, 
                      replace_stride_with_dilation=replace_stride_with_dilation)

    if pretrained:
        weights        = load_url(model_urls[backbone_name], model_dir='./')
        new_state_dict = OrderedDict()

        for name, value in encoder.state_dict().items():
            if name_to_res(name) in weights:
                new_state_dict[name] = weights[name_to_res(name)]
            else:
                new_state_dict[name] = value
        
        new_state_dict['depthconv.weight'] = torch.sum(new_state_dict['depthconv.weight'], axis=1, keepdim=True)

        encoder.load_state_dict(new_state_dict)

    return encoder
