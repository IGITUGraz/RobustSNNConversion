import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models
from models.layers import *

__all__ = ['ResNet', 'resnet20', 'resnet32']


class BasicResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, T, in_planes, out_planes, stride=1, default_leak=1.0, default_vth=1.0,
                 learn_vth=False, soft_reset=False, surrogate='PCW', gamma=1.0):
        super(BasicResNetBlock, self).__init__()
        self.T = T
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act1 = LIFSpike(self.T, leak=default_leak, v_th=default_vth, soft_reset=soft_reset,
                             learn_vth=learn_vth, surrogate=surrogate, gamma=gamma)
        self.act2 = LIFSpike(self.T, leak=default_leak, v_th=default_vth, soft_reset=soft_reset,
                             learn_vth=learn_vth, surrogate=surrogate, gamma=gamma)
        self.convex = ConvexCombination(2)
        self.find_max_mem = False
        self.max_mems = []

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_planes)
            )

    def forward(self, x):
        h1 = self.bn1(self.conv1(x))
        out = self.act1(h1)
        out = self.bn2(self.conv2(out))
        out = self.convex(self.shortcut(x), out)
        self.max_mems = [h1, out] if self.find_max_mem else []
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, T, num_classes, norm, init_channels=3, learn_vth=False, use_bias=False,
                 soft_reset=False, default_leak=1.0, default_threshold=1.0, surrogate='PCW', gamma=1.0):
        super(ResNet, self).__init__()
        self.T = T
        self.init_channels = init_channels
        self.default_leak = default_leak
        self.default_vth = default_threshold
        self.learn_vth = learn_vth
        self.use_bias = use_bias
        self.soft = soft_reset
        self.surrogate = surrogate
        self.gamma = gamma

        self.norm = TensorNormalization(*norm)
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

        self.in_planes = 16
        self.features = self._make_layers(block, num_blocks)
        self.classifier = self._make_classifier(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.T, self.in_planes, planes, stride, default_leak=self.default_leak,
                                default_vth=self.default_vth, learn_vth=self.learn_vth, soft_reset=self.soft,
                                surrogate=self.surrogate, gamma=self.gamma))
            self.in_planes = planes * block.expansion
        return layers

    def _make_layers(self, block, num_blocks):
        layers = [nn.Conv2d(self.init_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
                  nn.BatchNorm2d(16),
                  LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                           learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma)]
        layers.extend(self._make_layer(block, 16, num_blocks[0], stride=1))
        layers.extend(self._make_layer(block, 32, num_blocks[1], stride=2))
        layers.extend(self._make_layer(block, 64, num_blocks[2], stride=2))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)

    def _make_classifier(self, dim_in, dim_out):
        layer = [nn.Flatten(),
                 nn.Linear(dim_in, dim_out, bias=self.use_bias)]
        return nn.Sequential(*layer)

    def set_surrogate_gradient(self, surrogate, gamma, mode='bptt'):
        for module in self.modules():
            if isinstance(module, LIFSpike):
                module.mode = mode
                module.surrogate = surrogate
                module.gamma = gamma

    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode

    def threshold_update(self, scaling_factor=1.0, thresholds=[]):
        self.scaling_factor = scaling_factor
        for pos in range(len(self.features)):
            if isinstance(self.features[pos], LIFSpike):
                if thresholds:
                    self.features[pos].v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))
            if isinstance(self.features[pos], BasicResNetBlock):
                if thresholds:
                    self.features[pos].act1.v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))
                if thresholds:
                    self.features[pos].act2.v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))

    def percentile(self, t, q=99.7):
        k = 1 + round(.01 * float(q) * (t.numel() - 1))
        result = t.view(-1).kthvalue(k).values.item()
        return result

    def forward(self, input, find_max_mem=False, max_mem_layer=0, percentile=True):
        out = self.norm(input)
        if self.T > 0:
            out = add_dimension(out, self.T)
            out = self.merge(out)
        if find_max_mem:
            for l in range(len(self.features)):
                if l == max_mem_layer:
                    if isinstance(self.features[l], LIFSpike):
                        return [self.percentile(out.view(-1))] if percentile else [out.max().item()]
                    if isinstance(self.features[l], BasicResNetBlock) or isinstance(self.features[l], nn.AdaptiveAvgPool2d):
                        out1, out2 = self.features[l - 1].max_mems
                        if percentile:
                            return [self.percentile(out1.view(-1)), self.percentile(out2.view(-1))]
                        else:
                            return [out1.max().item(), out2.max().item()]
                out = self.features[l](out)
        else:
            out = self.features(out)
            out = self.classifier(out)
            if self.T > 0:
                out = self.expand(out)
            return out


def resnet20(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
             surrogate='PCW', gamma=1.0, **kwargs):
    return ResNet(BasicResNetBlock, [3, 3, 3], T=timesteps, num_classes=num_classes, norm=norm, learn_vth=learn_vth,
                  use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)


def resnet32(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
             surrogate='PCW', gamma=1.0, **kwargs):
    return ResNet(BasicResNetBlock, [5, 5, 5], T=timesteps, num_classes=num_classes, norm=norm, learn_vth=learn_vth,
                  use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)
