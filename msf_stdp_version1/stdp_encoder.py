import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LIFState:
    def __init__(self):
        self.mem = None
        self.pre_trace = None
        self.post_trace = None

    def reset(self):
        self.mem = None
        self.pre_trace = None
        self.post_trace = None


class STDPConvLIFLayer(nn.Module):
    """
    Convolution + LIF + pair-based STDP.

    输入:  x      [B, C_in, H, W]
    输出:  spike  [B, C_out, H_out, W_out]

    STDP 更新:
        dW = lr * (a_plus * post_spike x pre_trace
                   - a_minus * post_trace x pre_input)

    这里的 x 表示局部卷积 patch 的相关性，不是逐点乘法。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: Optional[int] = None,
        tau_mem: float = 0.90,
        tau_trace: float = 0.95,
        threshold: float = 1.0,
        stdp_lr: float = 1e-4,
        a_plus: float = 1.0,
        a_minus: float = 0.7,
        w_min: float = 0.0,
        w_max: float = 1.0,
        normalize: bool = True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.tau_mem = tau_mem
        self.tau_trace = tau_trace
        self.threshold = threshold
        self.stdp_lr = stdp_lr
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.normalize = normalize

        # 正权重更适合做无监督 STDP 的局部特征学习。
        weight = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        weight = weight / (in_channels * kernel_size * kernel_size) ** 0.5
        self.weight = nn.Parameter(weight, requires_grad=False)

        self.state = LIFState()

    @torch.no_grad()
    def reset_state(self):
        self.state.reset()

    def forward(self, x: torch.Tensor, learn_stdp: bool = False) -> torch.Tensor:
        # x: [B,C,H,W]
        current = F.conv2d(x, self.weight, None, self.stride, self.padding)

        if self.state.mem is None or self.state.mem.shape != current.shape:
            self.state.mem = torch.zeros_like(current)
        self.state.mem = self.tau_mem * self.state.mem + current

        spike = (self.state.mem >= self.threshold).to(x.dtype)
        # hard reset
        self.state.mem = self.state.mem * (1.0 - spike)

        if learn_stdp:
            self.stdp_update(x.detach(), spike.detach())

        return spike

    @torch.no_grad()
    def stdp_update(self, pre: torch.Tensor, post_spike: torch.Tensor):
        # 初始化 trace
        if self.state.pre_trace is None or self.state.pre_trace.shape != pre.shape:
            self.state.pre_trace = torch.zeros_like(pre)
        if self.state.post_trace is None or self.state.post_trace.shape != post_spike.shape:
            self.state.post_trace = torch.zeros_like(post_spike)

        self.state.pre_trace = self.tau_trace * self.state.pre_trace + pre
        self.state.post_trace = self.tau_trace * self.state.post_trace + post_spike

        # unfold pre traces/current input
        # [B, C*k*k, L]
        pre_trace_patches = F.unfold(
            self.state.pre_trace,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )
        pre_current_patches = F.unfold(
            pre,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )

        b, _, l = pre_trace_patches.shape
        post_flat = post_spike.reshape(b, self.out_channels, l)
        post_trace_flat = self.state.post_trace.reshape(b, self.out_channels, l)

        # [C_out, C_in*k*k]
        ltp = torch.einsum("bol,bkl->ok", post_flat, pre_trace_patches)
        ltd = torch.einsum("bol,bkl->ok", post_trace_flat, pre_current_patches)

        scale = max(float(b * l), 1.0)
        dw = self.stdp_lr * (self.a_plus * ltp - self.a_minus * ltd) / scale
        dw = dw.reshape_as(self.weight)

        self.weight.add_(dw)
        self.weight.clamp_(self.w_min, self.w_max)

        if self.normalize:
            # 每个输出卷积核单独归一化，防止某些 filter 吸收全部响应。
            flat = self.weight.view(self.out_channels, -1)
            denom = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
            flat.div_(denom)
            self.weight.copy_(flat.view_as(self.weight))


class STDPFeatureEncoder(nn.Module):
    """
    两层 Conv-LIF-STDP 编码器。

    默认输出维度:
        c1 + c2 = 128 + 384 = 512

    返回:
        feat   [B, D]，用于 MSF 的 clip feature
        spikes dict，可用于动力学分析
    """

    def __init__(
        self,
        in_channels: int = 2,
        c1: int = 128,
        c2: int = 384,
        k1: int = 5,
        k2: int = 5,
        tau_mem: float = 0.90,
        tau_trace: float = 0.95,
        threshold1: float = 0.6,
        threshold2: float = 0.6,
        stdp_lr1: float = 2e-4,
        stdp_lr2: float = 1e-4,
    ):
        super().__init__()
        self.layer1 = STDPConvLIFLayer(
            in_channels, c1, kernel_size=k1, tau_mem=tau_mem,
            tau_trace=tau_trace, threshold=threshold1, stdp_lr=stdp_lr1
        )
        self.layer2 = STDPConvLIFLayer(
            c1, c2, kernel_size=k2, tau_mem=tau_mem,
            tau_trace=tau_trace, threshold=threshold2, stdp_lr=stdp_lr2
        )

        self.out_dim = c1 + c2

    @torch.no_grad()
    def reset_state(self):
        self.layer1.reset_state()
        self.layer2.reset_state()

    def forward(self, x: torch.Tensor, learn_stdp: bool = False):
        s1 = self.layer1(x, learn_stdp=learn_stdp)
        # 降采样后送入第二层，降低计算量，同时扩大感受野。
        s1_pool = F.avg_pool2d(s1, kernel_size=2, stride=2)
        s2 = self.layer2(s1_pool, learn_stdp=learn_stdp)

        f1 = s1.mean(dim=(2, 3))
        f2 = s2.mean(dim=(2, 3))
        feat = torch.cat([f1, f2], dim=1)

        spikes = {
            "s1": s1,
            "s2": s2,
            "rate1": f1.mean(dim=1),
            "rate2": f2.mean(dim=1),
        }
        return feat, spikes
