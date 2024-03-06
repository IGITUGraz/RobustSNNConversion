import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import models
from models.layers import *


__all__ = ['WideResNet', 'wrn16_4', 'wrn28_2', 'wrn28_4']


class BasicBlock(nn.Module):
    def __init__(self, T, in_planes, out_planes, stride, dropout=0.3, default_leak=1.0, default_vth=1.0,
                 learn_vth=False, soft_reset=False, surrogate='PCW', gamma=1.0):
        super(BasicBlock, self).__init__()
        self.T = T
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.act1 = LIFSpike(self.T, leak=default_leak, v_th=default_vth, soft_reset=soft_reset,
                             learn_vth=learn_vth, surrogate=surrogate, gamma=gamma)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.act2 = LIFSpike(self.T, leak=default_leak, v_th=default_vth, soft_reset=soft_reset,
                             learn_vth=learn_vth,  surrogate=surrogate, gamma=gamma)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        self.convex = ConvexCombination(2)
        self.find_max_mem = False
        self.max_mems = []

    def forward(self, x):
        h1 = self.bn1(x)
        if not self.equalInOut:
            x = self.act1(h1)
        else:
            out = self.act1(h1)
        h2 = self.bn2(self.conv1(out if self.equalInOut else x))
        self.max_mems = [h1, h2] if self.find_max_mem else []
        out = self.act2(h2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.conv2(out)
        return self.convex(x if self.equalInOut else self.convShortcut(x), out)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, T, num_classes, norm, init_channels=3, dropout=0.0, learn_vth=False,
                 use_bias=False, soft_reset=False, default_leak=1.0, default_threshold=1.0, surrogate='PCW', gamma=1.0):
        super(WideResNet, self).__init__()
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        block = BasicBlock
        self.T = T
        self.init_channels = init_channels
        self.dropout = dropout
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

        self.features = self._make_layers(block, n, nChannels)
        self.classifier = self._make_classifier(nChannels[3], num_classes)

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

    def _make_block(self, block, in_planes, out_planes, nb_layers, stride, dropout):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(self.T, i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropout, 
                                default_leak=self.default_leak, default_vth=self.default_vth, learn_vth=self.learn_vth,
                                soft_reset=self.soft, surrogate=self.surrogate, gamma=self.gamma))
        return layers

    def _make_layers(self, block, n, nChannels):
        layers = [nn.Conv2d(self.init_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)]
        layers.extend(self._make_block(block, nChannels[0], nChannels[1], n, 1, self.dropout))
        layers.extend(self._make_block(block, nChannels[1], nChannels[2], n, 2, self.dropout))
        layers.extend(self._make_block(block, nChannels[2], nChannels[3], n, 2, self.dropout))
        layers.append(nn.BatchNorm2d(nChannels[3]))
        layers.append(LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                               learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma))
        layers.append(nn.AvgPool2d(8))
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
            if isinstance(self.features[pos], BasicBlock):
                if thresholds:
                    self.features[pos].act1.v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))
                if thresholds:
                    self.features[pos].act2.v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))
            if isinstance(self.features[pos], LIFSpike):
                if thresholds:
                    self.features[pos].v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))

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
                    if isinstance(self.features[l], BasicBlock) or isinstance(self.features[l], nn.BatchNorm2d):
                        out1, out2 = self.features[l - 1].max_mems
                        if percentile:
                            return [self.percentile(out1.view(-1)), self.percentile(out2.view(-1))]
                        else:
                            return [out1.max().item(), out2.max().item()]
                    if isinstance(self.features[l], LIFSpike):
                        return [self.percentile(out.view(-1))] if percentile else [out.max().item()]
                out = self.features[l](out)
        else:
            out = self.features(out)
            out = self.classifier(out)
            if self.T > 0:
                out = self.expand(out)
            return out


def wrn16_4(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
            surrogate='PCW', gamma=1.0, **kwargs):
    return WideResNet(depth=16, widen_factor=4, T=timesteps, num_classes=num_classes, norm=norm, learn_vth=learn_vth,
                      use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)


def wrn28_2(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
            surrogate='PCW', gamma=1.0, **kwargs):
    return WideResNet(depth=28, widen_factor=2, T=timesteps, num_classes=num_classes, norm=norm, learn_vth=learn_vth,
                      use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)


def wrn28_4(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
            surrogate='PCW', gamma=1.0, **kwargs):
    return WideResNet(depth=28, widen_factor=4, T=timesteps, num_classes=num_classes, norm=norm, learn_vth=learn_vth,
                      use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)
