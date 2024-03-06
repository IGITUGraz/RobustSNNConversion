import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import numpy as np


def add_dimension(x, T):
    x.unsqueeze_(1)
    x = x.repeat(T, 1, 1, 1, 1)
    return x


class TensorNormalization(nn.Module):
    def __init__(self, mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std

    def normalizex(self, tensor, mean, std):
        mean = mean[None, :, None, None]
        std = std[None, :, None, None]
        if mean.device != tensor.device:
            mean = mean.to(tensor.device)
            std = std.to(tensor.device)
        return tensor.sub(mean).div(std)

    def forward(self, X):
        return self.normalizex(X, self.mean, self.std)


class MergeTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        return x_seq.flatten(0, 1).contiguous()


class ExpandTemporalDim(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq: torch.Tensor):
        y_shape = [self.T, int(x_seq.shape[0] / self.T)]
        y_shape.extend(x_seq.shape[1:])
        return x_seq.view(y_shape)


class ConvexCombination(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.comb = nn.Parameter(torch.ones(n) / n)

    def forward(self, *args):
        assert (len(args) == self.n)
        out = 0.
        for i in range(self.n):
            out += args[i] * self.comb[i]
        return out


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = torch.zeros_like(input).cuda()
        out[input >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class ExpSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        alpha = torch.tensor([gamma[0]])
        beta = torch.tensor([gamma[1]])
        ctx.save_for_backward(input, alpha, beta)
        out = torch.zeros_like(input).cuda()
        out[input >= 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, alpha, beta) = ctx.saved_tensors
        alpha = alpha[0].item()
        beta = beta[0].item()
        grad_input = grad_output.clone()
        grad = alpha * torch.exp(-beta * torch.abs(input))
        return grad * grad_input, None


class Rectangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input >= 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        grad = (input.abs() < gamma/2).float() / gamma
        return grad_input * grad, None


class PCW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gamma):
        out = (input >= 0).float()
        L = torch.tensor([gamma])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gamma = others[0].item()
        grad_input = grad_output.clone()
        grad = (1 / gamma) * (1 / gamma) * ((gamma - input.abs()).clamp(min=0))
        return grad_input * grad, None


class RateBp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, params):
        leak, v_th, soft_reset = params
        mem = 0.
        spike_pot = []
        T = x.shape[0]
        for t in range(T):
            mem = mem * leak + x[t, ...]
            spike = ((mem - v_th) >= 0).float()
            mem = mem - spike * v_th if soft_reset else (1 - spike) * mem
            spike_pot.append(spike)
        out = torch.stack(spike_pot, dim=0)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        out = out.mean(0).unsqueeze(0)
        grad_input = grad_output * (out > 0).float()
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, T, leak=1.0, v_th=1.0, soft_reset=False, learn_vth=False, surrogate='PCW', gamma=1.0):
        super(LIFSpike, self).__init__()
        self.T = T
        self.expand = ExpandTemporalDim(T)
        self.merge = MergeTemporalDim(T)
        self.mode = 'bptt'
        self.surrogate = surrogate
        self.gamma = gamma
        self.act_pcw = PCW.apply
        self.act_exp = ExpSpike.apply
        self.act_rect = Rectangular.apply
        self.act_ste = STE.apply
        self.ratebp = RateBp.apply
        self.relu = nn.ReLU(inplace=True)
        self.soft_reset = soft_reset
        self.leak_mem = leak
        self.learn_vth = learn_vth
        self.v_th = nn.Parameter(torch.tensor(v_th)) if learn_vth else v_th

    def forward(self, x):
        if self.learn_vth:
            self.v_th.data.clamp_(min=0.03)   # set minimum of v_th=0.03 just in case
        if self.mode == 'bptr' and self.T > 0:
            x = self.expand(x)
            x = self.ratebp(x, (self.leak_mem, self.v_th, self.soft_reset))
            x = self.merge(x)
        elif self.T > 0:
            x = self.expand(x)
            v_mem = 0
            spike_pot = []
            for t in range(self.T):
                v_mem = v_mem * self.leak_mem + x[t, ...]

                if self.surrogate == 'PCW':
                    spike = self.act_pcw(v_mem - self.v_th, self.gamma)
                elif self.surrogate == 'EXP':
                    spike = self.act_exp(v_mem - self.v_th, (1.0, self.gamma))
                elif self.surrogate == 'EXP-D':
                    spike = self.act_exp(v_mem - self.v_th, (0.3, self.gamma))
                elif self.surrogate == 'RECT':
                    spike = self.act_rect(v_mem - self.v_th, self.gamma)
                elif self.surrogate == 'STE':
                    spike = self.act_ste(v_mem - self.v_th, self.gamma)
                else:
                    raise NotImplementedError

                v_mem = v_mem - spike * self.v_th if self.soft_reset else (1 - spike) * v_mem
                spike_pot.append(spike)
            x = torch.stack(spike_pot, dim=0)
            x = self.merge(x)
        else:
            x = self.relu(x)
        return x
