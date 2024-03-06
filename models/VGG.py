import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import models
from models.layers import *


__all__ = ['VGG', 'VGG_TIN', 'vgg11_bn', 'vgg16_bn', 'vgg11_tin']


cfg_conv = {'vgg11': [64, 'A', 128, 256, 'A', 512, 512, 512, 'A', 512, 512],
            'vgg16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512, 'A']}


class VGG(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, init_channels=3, use_bias=False, dropout=0.2, default_leak=1.0,
                 default_threshold=1.0, learn_vth=False, soft_reset=False, surrogate='PCW', gamma=1.0):
        super().__init__()
        self.vgg_name = vgg_name
        self.T = T
        self.init_channels = init_channels
        self.W = 16 if vgg_name == 'vgg11' else 1
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

        self.features = self._make_layers(cfg_conv[self.vgg_name])
        self.classifier = self._make_classifier(num_class)

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

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.use_bias))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                                       learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [nn.Flatten(),
                 nn.Linear(512 * self.W, 4096, bias=self.use_bias),
                 LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                          learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma),
                 nn.Dropout(self.dropout),
                 nn.Linear(4096, 4096, bias=self.use_bias),
                 LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                          learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma),
                 nn.Dropout(self.dropout),
                 nn.Linear(4096, num_class, bias=self.use_bias)]
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

        for pos in range(len(self.classifier)):
            if isinstance(self.classifier[pos], LIFSpike):
                if thresholds:
                    self.classifier[pos].v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))

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
                if isinstance(self.features[l], LIFSpike) and l == max_mem_layer:
                    return self.percentile(out.view(-1)) if percentile else out.max().item()
                out = self.features[l](out)
            for c in range(len(self.classifier)):
                if isinstance(self.classifier[c], LIFSpike) and ((len(self.features) + c) == max_mem_layer):
                    return self.percentile(out.view(-1)) if percentile else out.max().item()
                out = self.classifier[c](out)
        else:
            out = self.features(out)
            out = self.classifier(out)
            if self.T > 0:
                out = self.expand(out)
            return out


class VGG_TIN(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, init_channels=3, use_bias=False, default_leak=1.0,
                 default_threshold=1.0, learn_vth=False, soft_reset=False, surrogate='PCW', gamma=1.0):
        super().__init__()
        self.vgg_name = vgg_name
        self.T = T
        self.init_channels = init_channels
        self.W = 1
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

        self.features = self._make_layers([64, 'A', 128, 'A', 256, 256, 'A', 512, 512, 'A', 512, 512])
        self.classifier = self._make_classifier(num_class)

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

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1, bias=self.use_bias))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                                       learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [nn.AdaptiveAvgPool2d((1, 1)),
                 nn.Flatten(),
                 nn.Linear(512 * self.W, 4096, bias=self.use_bias),
                 LIFSpike(self.T, leak=self.default_leak, v_th=self.default_vth, soft_reset=self.soft,
                          learn_vth=self.learn_vth, surrogate=self.surrogate, gamma=self.gamma),
                 nn.Linear(4096, num_class, bias=self.use_bias)]
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

        for pos in range(len(self.classifier)):
            if isinstance(self.classifier[pos], LIFSpike):
                if thresholds:
                    self.classifier[pos].v_th = nn.Parameter(torch.tensor(thresholds.pop(0) * self.scaling_factor))

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
                if isinstance(self.features[l], LIFSpike) and l == max_mem_layer:
                    return self.percentile(out.view(-1)) if percentile else out.max().item()
                out = self.features[l](out)
            for c in range(len(self.classifier)):
                if isinstance(self.classifier[c], LIFSpike) and ((len(self.features) + c) == max_mem_layer):
                    return self.percentile(out.view(-1)) if percentile else out.max().item()
                out = self.classifier[c](out)
        else:
            out = self.features(out)
            out = self.classifier(out)
            if self.T > 0:
                out = self.expand(out)
            return out


def vgg11_tin(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
              surrogate='PCW', gamma=1.0, **kwargs):
    return VGG_TIN(vgg_name='vgg11_tin', T=timesteps, num_class=num_classes, norm=norm, learn_vth=learn_vth,
                   use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)


def vgg11_bn(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
             surrogate='PCW', gamma=1.0, **kwargs):
    return VGG(vgg_name='vgg11', T=timesteps, num_class=num_classes, norm=norm, learn_vth=learn_vth,
               use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)


def vgg16_bn(timesteps, num_classes, norm, learn_vth=False, use_bias=False, soft_reset=False,
             surrogate='PCW', gamma=1.0, **kwargs):
    return VGG(vgg_name='vgg16', T=timesteps, num_class=num_classes, norm=norm, learn_vth=learn_vth,
               use_bias=use_bias, soft_reset=soft_reset, surrogate=surrogate, gamma=gamma, **kwargs)
